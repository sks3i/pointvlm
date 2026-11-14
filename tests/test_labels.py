"""CPU tests for SFT targets (pointtrain.labels)."""

import numpy as np

from pointeval.geometry import point_in_mask
from pointeval.parse import ParsedPoint, parse_canonical
from pointtrain.labels import format_points, interior_pixel, mask_target, pixel_to_canonical


def c_shape(h=100, w=100):
	m = np.zeros((h, w), bool)
	m[10:90, 10:30] = True  # spine
	m[10:30, 10:90] = True  # top arm
	m[70:90, 10:90] = True  # bottom arm
	return m


def test_centroid_outside_but_deepest_inside():
	m = c_shape()
	ys, xs = np.nonzero(m)
	assert not m[int(ys.mean()), int(xs.mean())]
	t = mask_target(m)
	assert point_in_mask(ParsedPoint(t["x"], t["y"]), m)
	assert t["depth_px"] >= 9


def test_depth_sampled_respects_edge_margin():
	m = np.zeros((200, 300), bool)
	m[50:150, 50:250] = True
	rng = np.random.default_rng(0)
	for _ in range(50):
		c, r, d, dmax = interior_pixel(m, "depth_sampled", rng, min_depth_frac=0.5)
		assert d >= 0.5 * dmax and m[r, c]


def test_image_border_counts_as_edge():
	m = np.zeros((50, 50), bool)
	m[:, :25] = True  # object cut off by the left image border
	c, r, d, _ = interior_pixel(m)
	assert 10 <= c <= 14  # centre of the visible part, not hugging the border


def test_canonical_round_trip_hits_same_pixel():
	for w in (427, 640, 1333):
		for col in (0, 1, w // 2, w - 1):
			x, _ = pixel_to_canonical(col, 0, w, 10)
			assert x <= 999
			m = np.zeros((10, w), bool)
			m[0, col] = True
			assert point_in_mask(ParsedPoint(x, 0.5 / 10 * 1000), m) or w > 1000


def test_format_parses_with_frozen_parser():
	pr = parse_canonical(format_points([(512, 340), (10, 20)]))
	assert [(p.x, p.y) for p in pr.points] == [(512, 340), (10, 20)]
