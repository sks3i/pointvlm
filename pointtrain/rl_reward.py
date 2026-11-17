"""GRPO reward for pointing: 1 if the answer is one well-formed point inside the target region, else 0.

It uses the frozen eval's own parser and point_in_mask, so the training signal is exactly the
quantity the benchmarks score (on RefCOCO-style rows). Anything else would be a proxy the policy
could learn to satisfy without improving the eval.

Format is enforced inside the same 0/1, not as a separate bonus: the answer must parse with the
canonical parser AND contain exactly one point (the prompt says "exactly one point", and the eval
scores only the first one). A separate format bonus would pay for well-formed misses, and once
every sample is well formed it is a constant that only adds noise to the group statistics.

Targets (``row["target"]``):
	{"type": "mask", "mask_rle": ...}
	    RefCOCO family: the COCO segmentation. Exact.
	{"type": "points", "points": [[x, y], ...], "width": W, "height": H}
	    RoboPoint: the dataset ships point SETS sampled inside the target region, not the region
	    itself. Proxy region = convex hull of the points, grown by ``margin_frac`` of the image
	    diagonal, plus a disc of that radius around every point. A hull is too generous for
	    non-convex regions (an L-shaped free space) and the margin is a guess, so this reward is
	    noisier than the mask reward; RoboPoint rows are tagged so their accuracy is logged apart.
"""

from __future__ import annotations

import numpy as np

from pointeval.geometry import point_in_mask, rle_to_mask
from pointeval.parse import ParseStatus, parse_canonical

MARGIN_FRAC = 0.02  # of the image diagonal: ~16 px on a 640x480 RoboPoint render


def _in_hull(p: np.ndarray, pts: np.ndarray, margin: float) -> bool:
	"""p inside the convex hull of pts grown by ``margin`` (all in pixels)."""
	if len(pts) >= 3:
		from scipy.spatial import ConvexHull, QhullError

		try:
			hull = ConvexHull(pts)
			# Facet equations n.x + d <= 0 inside, with unit normals: n.x + d is the signed distance.
			if (hull.equations[:, :2] @ p + hull.equations[:, 2] <= margin).all():
				return True
		except QhullError:  # collinear points: fall through to the discs
			pass
	return bool((np.linalg.norm(pts - p, axis=1) <= margin).any())


def region_hit(target: dict, x: float, y: float) -> bool:
	from pointeval.parse import ParsedPoint

	if target["type"] == "mask":
		return point_in_mask(ParsedPoint(x, y), rle_to_mask(target["mask_rle"]))
	if target["type"] == "points":
		w, h = target["width"], target["height"]
		sx, sy = w / 1000.0, h / 1000.0
		pts = np.asarray(target["points"], dtype=float) * [sx, sy]
		return _in_hull(np.array([x * sx, y * sy]), pts, MARGIN_FRAC * float(np.hypot(w, h)))
	raise ValueError(f"unknown target type {target['type']!r}")


def point_reward(row: dict, text: str) -> dict:
	pr = parse_canonical(text)
	ok = pr.status is ParseStatus.OK and len(pr.points) == 1
	hit = ok and region_hit(row["target"], pr.points[0].x, pr.points[0].y)
	return {"reward": float(hit), "format_ok": float(ok), "unparseable": float(pr.status is not ParseStatus.OK), "multi_point": float(pr.status is ParseStatus.OK and len(pr.points) > 1)}


SHAPE_TAU = 0.03  # of the image diagonal: misses farther than this get no partial credit
SHAPE_MAX = 0.5  # best possible reward for a miss; a hit is always 1


def miss_distance(target: dict, x: float, y: float) -> float:
	"""Distance from a (missed) point to the target region, as a fraction of the image diagonal."""
	from scipy import ndimage

	if target["type"] == "mask":
		m = rle_to_mask(target["mask_rle"])
		h, w = m.shape
		col, row = min(int(x / 1000 * w), w - 1), min(int(y / 1000 * h), h - 1)
		d = ndimage.distance_transform_edt(~m)[row, col]
		return float(d / np.hypot(w, h))
	w, h = target["width"], target["height"]
	pts = np.asarray(target["points"], dtype=float) * [w / 1000.0, h / 1000.0]
	diag = float(np.hypot(w, h))
	d = float(np.linalg.norm(pts - [x * w / 1000.0, y * h / 1000.0], axis=1).min()) - MARGIN_FRAC * diag
	return max(d, 0.0) / diag


def point_reward_shaped(row: dict, text: str) -> dict:
	"""point_reward plus partial credit for near misses: SHAPE_MAX * (1 - d / SHAPE_TAU) for a miss at
	distance d (fraction of the diagonal), 0 beyond SHAPE_TAU.

	Why: 19% of the SFT model's Point-Bench misses land within 2% of the diagonal of the mask, mostly
	on small targets. With a 0/1 reward a group of 8 near misses is unanimous (advantage 0), so GRPO
	gets no signal about which near miss was nearer. A dense term inside a small radius ranks them,
	while a hit (1) always beats any miss (< 0.5), so the optimum is unchanged. The radius is kept
	small so that pointing at a neighbouring wrong object (median miss: 10% of the diagonal) earns
	nothing. ``hit`` is returned separately; dev accuracy uses it, not the shaped reward.
	"""
	pr = parse_canonical(text)
	ok = pr.status is ParseStatus.OK and len(pr.points) == 1
	out = {"format_ok": float(ok), "unparseable": float(pr.status is not ParseStatus.OK), "multi_point": float(pr.status is ParseStatus.OK and len(pr.points) > 1), "hit": 0.0}
	if not ok:
		return {"reward": 0.0, **out}
	p = pr.points[0]
	if region_hit(row["target"], p.x, p.y):
		return {"reward": 1.0, **out, "hit": 1.0}
	d = miss_distance(row["target"], p.x, p.y)
	return {"reward": SHAPE_MAX * max(0.0, 1.0 - d / SHAPE_TAU), **out}
