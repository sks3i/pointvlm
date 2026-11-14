"""Coordinate conventions and masks.

Convention (frozen):
	* Canonical units u in [0, 1000] map to the ORIGINAL image (before any model
	  resizing), using a continuous coordinate frame where pixel column i covers
	  [i, i+1). So x_cont = x_norm / 1000 * W and the pixel hit is floor(x_cont),
	  clamped to W-1 (x_norm = 1000 is the right edge, which belongs to column W-1).
	* Denominator is 1000, never 999. A 999 denominator shifts every point by up
	  to 0.1% of the image, which is a ~0.6 px bias on a 640 px COCO image: small,
	  but systematic, and it is exactly the kind of thing that leaks a free point
	  of accuracy between two runs that look identical.
	* Training targets use the inverse: x_norm = (x_px + 0.5) / W * 1000, i.e. the
	  pixel CENTRE, so encode/decode round-trips to the same pixel.
"""

from __future__ import annotations

import math

import numpy as np

from pointeval.parse import NORM, ParsedPoint


def norm_to_pixel(p: ParsedPoint, width: int, height: int) -> tuple[int, int]:
	"""Canonical point -> integer (col, row) in the original image."""
	cx = p.x / NORM * width
	cy = p.y / NORM * height
	col = min(max(int(math.floor(cx)), 0), width - 1)
	row = min(max(int(math.floor(cy)), 0), height - 1)
	return col, row


def pixel_to_norm(col: float, row: float, width: int, height: int) -> ParsedPoint:
	"""Integer pixel (col,row) -> canonical units at the pixel centre."""
	return ParsedPoint((col + 0.5) / width * NORM, (row + 0.5) / height * NORM)


def polygons_to_mask(segmentation, height: int, width: int) -> np.ndarray:
	"""COCO polygon(s) or RLE -> bool mask (H, W). Accepts a flat list for one polygon."""
	from pycocotools import mask as mask_utils

	seg = segmentation
	if isinstance(seg, dict):  # RLE
		rle = seg
		if isinstance(rle.get("counts"), list):
			rle = mask_utils.frPyObjects(rle, height, width)
	else:
		if seg and not isinstance(seg[0], (list, tuple)):
			seg = [list(seg)]  # flat single polygon
		seg = [list(map(float, poly)) for poly in seg if len(poly) >= 6]
		if not seg:
			raise ValueError("no valid polygon")
		rle = mask_utils.merge(mask_utils.frPyObjects(seg, height, width))
	return mask_utils.decode(rle).astype(bool)


def mask_to_rle(mask: np.ndarray) -> dict:
	"""bool (H, W) -> compact COCO RLE with string counts, JSON serializable."""
	from pycocotools import mask as mask_utils

	rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
	return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def rle_to_mask(rle: dict) -> np.ndarray:
	from pycocotools import mask as mask_utils

	return mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")}).astype(bool)


def point_in_mask(p: ParsedPoint | None, mask: np.ndarray) -> bool:
	if p is None:
		return False
	h, w = mask.shape
	col, row = norm_to_pixel(p, w, h)
	return bool(mask[row, col])


def point_in_box(p: ParsedPoint | None, box_xywh, width: int, height: int) -> bool:
	"""Lenient diagnostic only (not a headline metric). box in COCO xywh pixels."""
	if p is None:
		return False
	cx, cy = p.x / NORM * width, p.y / NORM * height
	x, y, bw, bh = box_xywh
	return x <= cx <= x + bw and y <= cy <= y + bh
