"""CPU-only tests for the frozen eval logic. Run: pytest -q"""

from __future__ import annotations

import numpy as np
import pytest

from pointeval.geometry import (
	mask_to_rle,
	norm_to_pixel,
	pixel_to_norm,
	point_in_box,
	point_in_mask,
	polygons_to_mask,
	rle_to_mask,
)
from pointeval.manifests import _strip_format_suffix, hash_rank
from pointeval.metrics import mcnemar_exact, summarize, wilson_ci
from pointeval.parse import ParsedPoint, ParseStatus, parse_canonical, parse_molmo, parse_qwen_json
from pointeval.run import score_item

# --- canonical parser ------------------------------------------------------------


@pytest.mark.parametrize(
	"text,xy",
	[
		('<point x="512" y="340">', (512, 340)),
		('The mug: <point x="512" y="340"/> done', (512, 340)),
		('<point x="0" y="1000"></point>', (0, 1000)),
		('<point x="12.5" y="7">', (12.5, 7)),
	],
)
def test_canonical_ok(text, xy):
	r = parse_canonical(text)
	assert r.status is ParseStatus.OK and (r.first.x, r.first.y) == xy


@pytest.mark.parametrize(
	"text",
	["", "512, 340", "(512, 340)", '<point y="340" x="512">', '<point x="a" y="1">', '<point x=512 y=340>', '[{"point_2d": [1, 2]}]'],
)
def test_canonical_unparseable(text):
	assert parse_canonical(text).status is ParseStatus.UNPARSEABLE


@pytest.mark.parametrize("text", ['<point x="1001" y="3">', '<point x="-1" y="3">'])
def test_canonical_out_of_range_is_invalid(text):
	r = parse_canonical(text)
	assert r.status is ParseStatus.INVALID and r.first is None


def test_canonical_first_point_is_scored():
	r = parse_canonical('<point x="10" y="10"> or maybe <point x="900" y="900">')
	assert len(r.points) == 2 and r.first == ParsedPoint(10, 10)


# --- native adapters -------------------------------------------------------------


def test_molmo_percent_to_canonical():
	r = parse_molmo('<point x="61.5" y="40.4" alt="mug">mug</point>')
	assert r.first == ParsedPoint(615.0, 404.0)


def test_molmo_points_multi_in_text_order():
	r = parse_molmo('<points x1="10" y1="20" x2="30" y2="40" alt="cups">cups</points>')
	assert [(p.x, p.y) for p in r.points] == [(100, 200), (300, 400)]


def test_molmo_no_point():
	assert parse_molmo("There is no mug.").status is ParseStatus.UNPARSEABLE


def test_qwen_json_fenced():
	r = parse_qwen_json('```json\n[{"point_2d": [250, 750], "label": "mug"}]\n```')
	assert r.first == ParsedPoint(250, 750)


def test_qwen_json_bad():
	assert parse_qwen_json('[{"bbox_2d": [1,2,3,4]}]').status is ParseStatus.UNPARSEABLE


# --- geometry --------------------------------------------------------------------


def test_norm_pixel_edges():
	assert norm_to_pixel(ParsedPoint(0, 0), 640, 480) == (0, 0)
	assert norm_to_pixel(ParsedPoint(1000, 1000), 640, 480) == (639, 479)
	assert norm_to_pixel(ParsedPoint(500, 500), 640, 480) == (320, 240)


def test_pixel_centre_roundtrip():
	for w, h in [(640, 480), (427, 640), (1, 1), (1000, 3)]:
		for c in range(0, w, max(1, w // 7)):
			for r in range(0, h, max(1, h // 7)):
				assert norm_to_pixel(pixel_to_norm(c, r, w, h), w, h) == (c, r)


def test_roundtrip_survives_integer_quantization_on_coco_sizes():
	"""Writing integers 0-1000 quantizes to 1/1000 of the side: at <=1000 px every pixel survives."""
	w, h = 640, 480
	for c in range(w):
		p = pixel_to_norm(c, 0, w, h)
		q = ParsedPoint(round(p.x), round(p.y))
		assert abs(norm_to_pixel(q, w, h)[0] - c) <= 1


def test_aspect_ratio_independent_axes():
	# The same normalized point lands on different pixel offsets on non-square images.
	assert norm_to_pixel(ParsedPoint(500, 500), 1000, 100) == (500, 50)


def test_polygon_mask_and_rle_roundtrip():
	m = polygons_to_mask([10, 10, 50, 10, 50, 40, 10, 40], 60, 80)
	assert m.shape == (60, 80) and 1100 < m.sum() < 1300
	assert (rle_to_mask(mask_to_rle(m)) == m).all()


def test_multi_polygon():
	m = polygons_to_mask([[0, 0, 10, 0, 10, 10, 0, 10], [20, 20, 30, 20, 30, 30, 20, 30]], 40, 40)
	assert m[5, 5] and m[25, 25] and not m[15, 15]


def test_point_in_mask_none_is_miss():
	assert point_in_mask(None, np.ones((4, 4), bool)) is False


def test_point_in_box():
	assert point_in_box(ParsedPoint(500, 500), [300, 200, 100, 100], 700, 500)


# --- scoring ---------------------------------------------------------------------


def _item(mask):
	return {"mask_rle": mask_to_rle(mask), "width": mask.shape[1], "height": mask.shape[0], "bbox_xywh": [0, 0, 1, 1]}


def test_unparseable_scores_as_miss():
	m = np.ones((10, 10), bool)
	assert score_item("refcoco", _item(m), parse_canonical("no idea"))["hit"] is False


def test_where2place_fraction():
	m = np.zeros((10, 10), bool)
	m[:, :5] = True
	pr = parse_canonical('<point x="100" y="500"><point x="900" y="500">')
	s = score_item("where2place", _item(m), pr)
	assert s["hit"] is True and s["frac_points_in_mask"] == 0.5


def test_pointbench_counting():
	m = np.zeros((10, 20), bool)
	m[2:5, 2:5] = True
	m[2:5, 15:18] = True
	it = {**_item(m), "category": "counting", "count": 2}
	two = parse_canonical('<point x="150" y="350"><point x="800" y="350">')
	one = parse_canonical('<point x="150" y="350">')
	same = parse_canonical('<point x="150" y="350"><point x="160" y="350">')
	assert score_item("pointbench", it, two)["hit"] is True
	assert score_item("pointbench", it, one)["hit"] is False
	assert score_item("pointbench", it, same)["hit"] is False


# --- metrics & sampling ------------------------------------------------------------


def test_wilson_ci_examples():
	lo, hi = wilson_ci(1000, 2000)
	assert 0.477 < lo < 0.479 and 0.521 < hi < 0.523
	lo, hi = wilson_ci(50, 100)
	assert hi - lo > 0.19  # Where2Place: ~ +/- 10 pp
	assert wilson_ci(0, 100)[0] == 0.0 and wilson_ci(0, 100)[1] > 0.03


def test_summarize():
	assert summarize([True, False, True, True])["acc"] == 0.75


def test_mcnemar():
	a = [True] * 50 + [False] * 50
	assert mcnemar_exact(a, a)["p_value"] == 1.0
	b = [True] * 50 + [True] * 20 + [False] * 30  # 20 fixes, 0 regressions
	assert mcnemar_exact(a, b)["p_value"] < 1e-5


def test_hash_rank_is_stable():
	# Golden value: if this ever changes, the frozen sample would change.
	assert hash_rank("pointvlm-refcoco-val-v1", "x.jpg|1.00,2.00,3.00,4.00")[:12] == "52f7a29819a7"


def test_strip_w2p_suffix():
	t = "Find free space left of the mug. Your answer should be formatted as a list of tuples"
	assert _strip_format_suffix(t) == "Find free space left of the mug."
