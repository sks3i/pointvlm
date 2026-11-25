"""Lenient point parser: a SECONDARY score only. The headline numbers stay on the frozen v1 parsers.

The frozen parsers (pointeval/parse.py) score a format slip as a miss, which is the right rule
for our own checkpoints but punishes baselines for formatting rather than pointing. This parser
asks "is there a point anywhere in the answer, in any common notation?" so a lenient column can
be reported next to the strict one for every model, from the same stored raw text.

Accepted notations, extracted in textual order (first point = first in the text):
	tier 1 (structured): "point_2d"/"point": [x, y]; "bbox_2d": [x1, y1, x2, y2] -> box centre;
	        x=".." y=".." attribute pairs anywhere, quoted or not, '=' or ':' (covers <point x y>,
	        a tag with repeated x/y attributes, Molmo <points x1 y1 x2 y2>, {"x": .., "y": ..}).
	tier 2: (x, y) or [x, y] tuples, used only if tier 1 found nothing.
	tier 3: bare "x, y" number pairs, used only if tiers 1-2 found nothing.
Units, per point:
	* x/y attribute pairs from a Molmo model are percent (x10), as in parse_molmo.
	* both values written with a decimal point and <= 1  -> fraction of the image (x1000).
	* either value > 1000 -> pixels of the original image, rescaled by its width/height.
	* otherwise 0-1000 (the canonical / Qwen3-VL convention).
	Pixel answers that happen to be <= 1000 cannot be told apart from 0-1000 and are read as 0-1000.
Out of range after conversion -> invalid (a miss), as in the frozen rules. Single-target
benchmarks still score only the first point; counting uses all of them (score_item is unchanged).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pointeval.parse import NORM, ParsedPoint, ParseResult, ParseStatus

LENIENT_VERSION = "lenient-v1"
LENIENT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

_N = r"[-+]?(?:\d+\.?\d*|\.\d+)"
_Q = r"""["']?"""
_POINT_KEY = re.compile(rf"""["']?(?:point_2d|point)["']?\s*:\s*\[\s*({_N})\s*,\s*({_N})\s*\]""")
_BBOX_KEY = re.compile(rf"""["']?bbox_2d["']?\s*:\s*\[\s*({_N})\s*,\s*({_N})\s*,\s*({_N})\s*,\s*({_N})\s*\]""")
_XY_ATTR = re.compile(rf"""\bx\d*{_Q}\s*[=:]\s*{_Q}({_N}){_Q}\s*,?\s*{_Q}y\d*{_Q}\s*[=:]\s*{_Q}({_N}){_Q}""")
_TUPLE = re.compile(rf"[(\[]\s*({_N})\s*,\s*({_N})\s*[)\]]")
_BARE = re.compile(rf"(?<![\d.])({_N})\s*,\s*({_N})(?![\d.])")


def _extract(text: str) -> list[tuple[str, str, str]]:
	"""(x_str, y_str, kind) in textual order, from the first tier that finds anything."""
	found: list[tuple[int, str, str, str]] = []
	for m in _POINT_KEY.finditer(text):
		found.append((m.start(), m[1], m[2], "point_key"))
	for m in _BBOX_KEY.finditer(text):
		x1, y1, x2, y2 = (float(v) for v in m.groups())
		found.append((m.start(), str((x1 + x2) / 2), str((y1 + y2) / 2), "bbox_centre"))
	for m in _XY_ATTR.finditer(text):
		found.append((m.start(), m[1], m[2], "xy_attr"))
	if not found:
		found = [(m.start(), m[1], m[2], "tuple") for m in _TUPLE.finditer(text)]
	if not found:
		found = [(m.start(), m[1], m[2], "bare_pair") for m in _BARE.finditer(text)]
	found.sort(key=lambda t: t[0])
	return [(x, y, k) for _, x, y, k in found]


def _to_canonical(xs: str, ys: str, kind: str, width: int, height: int, percent_attrs: bool) -> tuple[float, float, str]:
	x, y = float(xs), float(ys)
	if kind == "xy_attr" and percent_attrs:
		return x * 10.0, y * 10.0, "percent"
	if "." in xs and "." in ys and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
		return x * NORM, y * NORM, "fraction"
	if x > NORM or y > NORM:
		return x / width * NORM, y / height * NORM, "pixels"
	return x, y, "0-1000"


def parse_lenient(text: str, width: int, height: int, model_family: str | None = None) -> ParseResult:
	"""``model_family="molmo"`` reads x/y attribute pairs as percent; any other value as 0-1000."""
	raw = _extract(text)
	if not raw:
		return ParseResult(ParseStatus.UNPARSEABLE, note="lenient: no point")
	pts, units, kinds = [], set(), set()
	for xs, ys, kind in raw:
		x, y, unit = _to_canonical(xs, ys, kind, width, height, model_family == "molmo")
		if not (0.0 <= x <= NORM and 0.0 <= y <= NORM):
			return ParseResult(ParseStatus.INVALID, note=f"lenient: out of range ({x:.1f}, {y:.1f})")
		pts.append(ParsedPoint(x, y))
		units.add(unit)
		kinds.add(kind)
	return ParseResult(ParseStatus.OK, pts, note=f"lenient {'+'.join(sorted(kinds))} {'+'.join(sorted(units))}")
