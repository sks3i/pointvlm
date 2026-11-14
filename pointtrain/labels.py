"""Training targets: masks -> interior points, and points -> the canonical <point> string.

Why not the centroid: the centroid of a non-convex or multi-part mask can lie outside it (a person
with an arm out, a C-shaped sofa, a donut, an object split in two by an occluder). On RefCOCO that
is a few percent of objects, and every one of them is a training label that the frozen eval would
score as a MISS.

Deepest point: the pixel farthest from the mask boundary, i.e. argmax of the Euclidean distance
transform. It is the centre of the largest inscribed circle, so it maximizes the error margin: a
prediction within radius d_max of it is still inside the mask. That margin is exactly what
point-in-mask accuracy rewards, and it also absorbs annotation noise at the polygon boundary.

Depth-weighted sampling (optional): pick a pixel with probability proportional to depth**gamma,
among pixels at least ``min_depth_frac * d_max`` (and ``min_depth_px``) from the edge. This gives
several valid targets per object across epochs/expressions, a spatial analogue of label smoothing,
while still keeping all targets well away from the boundary.

Pixel convention (frozen in pointeval/geometry.py): targets use pixel centres,
x = (col + 0.5) / W * 1000, and the eval maps back with floor(x / 1000 * W).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from pointeval.geometry import point_in_mask
from pointeval.parse import ParsedPoint

MAX_COORD = 999  # 1000 is legal in the eval, but LFM2 tokenizes "1000" as "100"+"0"; 999 is the same last pixel bin for W <= 1000.


def depth_map(mask: np.ndarray) -> np.ndarray:
	"""Euclidean distance (px) from each mask pixel to the nearest non-mask pixel.

	The mask is zero-padded first so the image border counts as a boundary: a point on the border
	of a truncated object is as fragile as one on its real edge.
	"""
	return ndimage.distance_transform_edt(np.pad(mask.astype(bool), 1))[1:-1, 1:-1]


def interior_pixel(
	mask: np.ndarray,
	mode: str = "deepest",
	rng: np.random.Generator | None = None,
	min_depth_frac: float = 0.5,
	min_depth_px: float = 2.0,
	gamma: float = 2.0,
) -> tuple[int, int, float, float] | None:
	"""Pick an interior pixel. Returns (col, row, depth_px, max_depth_px), or None for an empty mask."""
	d = depth_map(mask)
	dmax = float(d.max())
	if dmax <= 0:
		return None
	if mode == "deepest":
		# Ridge-shaped masks (bars, poles) have many pixels at d_max; argmax would pick the first in
		# raster order, biasing toward the top-left end. Take the tied pixel nearest their mean.
		rows, cols = np.nonzero(d >= dmax - 0.5)
		k = int(np.argmin((rows - rows.mean()) ** 2 + (cols - cols.mean()) ** 2))
		r, c = int(rows[k]), int(cols[k])
	elif mode == "depth_sampled":
		rng = rng or np.random.default_rng()
		thresh = max(min_depth_px, min_depth_frac * dmax)
		if dmax < thresh:  # tiny / thin object: every pixel is near an edge, fall back to the deepest
			return interior_pixel(mask, "deepest")
		rows, cols = np.nonzero(d >= thresh)
		w = d[rows, cols] ** gamma
		k = int(rng.choice(len(rows), p=w / w.sum()))
		r, c = int(rows[k]), int(cols[k])
	else:
		raise ValueError(f"unknown mode {mode}")
	return c, r, float(d[r, c]), dmax


def _q(v: float) -> int:
	"""Round half up to an integer in [0, MAX_COORD]. (Python's round() rounds half to even.)"""
	return min(max(int(math.floor(v + 0.5)), 0), MAX_COORD)


def pixel_to_canonical(col: int, row: int, width: int, height: int) -> tuple[int, int]:
	return _q((col + 0.5) / width * 1000), _q((row + 0.5) / height * 1000)


def frac_to_canonical(x: float, y: float) -> tuple[int, int]:
	"""Continuous coordinates in [0, 1] (RoboPoint; PixMo after /100) -> canonical integers."""
	return _q(x * 1000), _q(y * 1000)


def mask_target(mask: np.ndarray, mode: str = "deepest", rng=None, **kw) -> dict | None:
	"""Canonical target for a mask, verified with the eval's own point_in_mask.

	Rounding to integers moves a point by up to 0.5/1000 of the image, which for W > 1000 can cross
	into the neighbouring pixel. The check makes sure no label would score as a miss.
	"""
	h, w = mask.shape
	got = interior_pixel(mask, mode, rng, **kw)
	if got is None:
		return None
	col, row, depth, dmax = got
	x, y = pixel_to_canonical(col, row, w, h)
	if not point_in_mask(ParsedPoint(x, y), mask):
		if mode != "deepest":
			return mask_target(mask, "deepest")
		return None
	return {"x": x, "y": y, "depth_px": depth, "max_depth_px": dmax}


def format_points(points: list[tuple[int, int]]) -> str:
	"""One or more canonical tags, separated by a space."""
	return " ".join(f'<point x="{x}" y="{y}">' for x, y in points)


def order_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
	"""A fixed order for multi-point answers (left to right, then top to bottom).

	The set of points is unordered, but the model emits a sequence. With a random order the target
	of the first point is any of the N instances, which spreads probability over all of them; a
	canonical order makes each position's target (nearly) deterministic given the image.
	"""
	return sorted(points, key=lambda p: (p[0], p[1]))
