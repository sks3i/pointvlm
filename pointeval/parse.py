"""Parsing model text into points.

There is exactly ONE canonical parser, :func:`parse_canonical`, for the project
format ``<point x="512" y="340">`` with coordinates normalized to [0, 1000].
Every checkpoint we train is scored with it and nothing else.

Baseline models that were never trained on our format get a *native* adapter
(:func:`parse_molmo`, :func:`parse_qwen_json`) that converts their own format
into the same canonical :class:`ParsedPoint`. Native adapters exist only so the
baselines are not scored as zero for formatting reasons; they are never used
for our own checkpoints.

Strict policy (frozen, see docs/EVAL_SPEC.md):
	* No tag found, or a tag with a missing/non-numeric attribute  -> unparseable.
	* A coordinate outside [0, 1000] (canonical units)             -> invalid.
	* Unparseable and invalid both score as a MISS (never dropped from the
	  denominator) and are counted separately so format drift is visible.
	* When several points are present, single-target benchmarks score only the
	  FIRST point. We never search for "the best" point.
	* No fallback hunting for bare numbers such as "(512, 340)".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum

NORM = 1000.0


class ParseStatus(str, Enum):
	OK = "ok"
	UNPARSEABLE = "unparseable"
	INVALID = "invalid"  # parsed, but out of range / non-finite


@dataclass(frozen=True)
class ParsedPoint:
	"""A point in canonical units: x, y in [0, 1000], origin top-left, x right, y down."""

	x: float
	y: float


@dataclass
class ParseResult:
	status: ParseStatus
	points: list[ParsedPoint] = field(default_factory=list)
	note: str = ""

	@property
	def first(self) -> ParsedPoint | None:
		return self.points[0] if self.status is ParseStatus.OK and self.points else None


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)"

# <point x="512" y="340">  also accepts  <point x="512" y="340"/>,  </point> after, and
# extra attributes after y (e.g. alt="mug"). Attribute order is fixed (x then y) because
# that is what we train; anything else is a format error we want to see, not silently forgive.
_CANON_RE = re.compile(r'<point\s+x="(?P<x>[^"]*)"\s+y="(?P<y>[^"]*)"[^>]*>')


def _to_float(s: str) -> float | None:
	if not re.fullmatch(_NUM, s.strip()):
		return None
	return float(s)


def _validate(points: list[tuple[float, float]], note: str = "") -> ParseResult:
	out = []
	for x, y in points:
		if not (0.0 <= x <= NORM and 0.0 <= y <= NORM):
			return ParseResult(ParseStatus.INVALID, note=f"out of range: ({x}, {y})")
		out.append(ParsedPoint(x, y))
	return ParseResult(ParseStatus.OK, out, note)


def parse_canonical(text: str) -> ParseResult:
	"""Parse ``<point x=".." y="..">`` tags (0-1000). The only parser for our checkpoints."""
	matches = list(_CANON_RE.finditer(text))
	if not matches:
		return ParseResult(ParseStatus.UNPARSEABLE, note="no <point> tag")
	pts = []
	for m in matches:
		x, y = _to_float(m["x"]), _to_float(m["y"])
		if x is None or y is None:
			return ParseResult(ParseStatus.UNPARSEABLE, note=f"non-numeric attr in {m.group(0)!r}")
		pts.append((x, y))
	return _validate(pts)


# --- Native adapters for baselines -------------------------------------------------

# Molmo: <point x="61.5" y="40.4" alt="mug">mug</point>   (percent, 0-100)
#        <points x1="10.0" y1="20.0" x2="30.0" y2="40.0" alt="...">...</points>
_MOLMO_POINT_RE = re.compile(r'<point\s+x="(?P<x>[^"]*)"\s+y="(?P<y>[^"]*)"')
# The closing ">" is optional: long lists can hit the token cap mid-tag. Only complete
# xN/yN pairs are kept, so a cut-off last pair is dropped.
_MOLMO_POINTS_RE = re.compile(r"<points\s+(?P<attrs>[^>]*)(?:>|$)")
_ATTR_RE = re.compile(r'(?P<k>[xy])(?P<i>\d+)="(?P<v>[^"]*)"')


def parse_molmo(text: str) -> ParseResult:
	"""Molmo emits percentages (0-100). Convert to canonical 0-1000 by multiplying by 10."""
	tags: list[tuple[int, list[tuple[float, float]]]] = []
	for m in _MOLMO_POINT_RE.finditer(text):
		x, y = _to_float(m["x"]), _to_float(m["y"])
		if x is None or y is None:
			return ParseResult(ParseStatus.UNPARSEABLE, note="non-numeric molmo attr")
		tags.append((m.start(), [(x, y)]))
	for m in _MOLMO_POINTS_RE.finditer(text):
		xs: dict[int, float] = {}
		ys: dict[int, float] = {}
		for a in _ATTR_RE.finditer(m["attrs"]):
			v = _to_float(a["v"])
			if v is None:
				return ParseResult(ParseStatus.UNPARSEABLE, note="non-numeric molmo attr")
			(xs if a["k"] == "x" else ys)[int(a["i"])] = v
		idx = sorted(set(xs) & set(ys))
		if not idx:
			return ParseResult(ParseStatus.UNPARSEABLE, note="empty <points>")
		tags.append((m.start(), [(xs[i], ys[i]) for i in idx]))
	if not tags:
		return ParseResult(ParseStatus.UNPARSEABLE, note="no molmo point tag")
	tags.sort(key=lambda t: t[0])  # keep textual order so "first point" is well defined
	pts = [(x * 10.0, y * 10.0) for _, group in tags for x, y in group]
	return _validate(pts, note="molmo percent x10")


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.S)


def parse_qwen_json(text: str) -> ParseResult:
	"""Qwen3-VL's native grounding JSON: ``[{"point_2d": [x, y], "label": ".."}]``, 0-1000.

	Qwen2.5-VL used ABSOLUTE PIXELS of the resized input here; Qwen3-VL switched to
	0-1000 relative. Using this adapter on Qwen2.5-VL would be silently wrong.
	"""
	m = _JSON_BLOCK_RE.search(text)
	body = m["body"] if m else text
	start = body.find("[") if "[" in body else body.find("{")
	if start < 0:
		return ParseResult(ParseStatus.UNPARSEABLE, note="no json")
	try:
		obj, _ = json.JSONDecoder().raw_decode(body[start:])
	except json.JSONDecodeError:
		return ParseResult(ParseStatus.UNPARSEABLE, note="bad json")
	items = obj if isinstance(obj, list) else [obj]
	pts = []
	for it in items:
		p = it.get("point_2d") if isinstance(it, dict) else None
		if not (isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p)):
			return ParseResult(ParseStatus.UNPARSEABLE, note="missing point_2d")
		pts.append((float(p[0]), float(p[1])))
	if not pts:
		return ParseResult(ParseStatus.UNPARSEABLE, note="empty json list")
	return _validate(pts, note="qwen point_2d")


PARSERS = {
	"canonical": parse_canonical,
	"molmo": parse_molmo,
	"qwen_json": parse_qwen_json,
}
