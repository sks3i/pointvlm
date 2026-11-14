"""Diagnostic only (not part of eval v1): how much of LFM2-VL's score is lost to coordinate scale?

LFM2-VL often answers in 0-1 fractions or (0,0) when asked for 0-1000 canonical points. This
re-reads the stored raw outputs, extracts points (canonical regex first, then a lenient fallback
used ONLY here), classifies each item, rescales fractions x1000 and scores against the frozen
manifest masks. The frozen parser, scoring and official summary.json files are not touched.

	python scripts/diag_lfm2_rescale.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pointeval.metrics import summarize
from pointeval.parse import _CANON_RE, NORM, ParsedPoint, ParseResult, ParseStatus
from pointeval.run import ROOT, load_manifest, score_item

RUN = "lfm2vl-1.6b-canonical"
BENCHES = ["refcoco", "where2place", "pointbench"]
CLASSES = ["unparseable", "zero", "fraction", "scale_1000"]
_N = r"([-+]?(?:\d+\.?\d*|\.\d+))"
LENIENT = [re.compile(rf'x="{_N}"\s+y="{_N}"'), re.compile(rf"\(\s*{_N}\s*,\s*{_N}\s*\)"), re.compile(rf"\[\s*{_N}\s*,\s*{_N}\s*\]")]


def extract(raw: str) -> tuple[list[tuple[float, float]], bool]:
	"""All (x, y) pairs from the first method that finds any. Returns (pairs, used_fallback)."""
	pts = []
	for m in _CANON_RE.finditer(raw):
		try:
			pts.append((float(m["x"]), float(m["y"])))
		except ValueError:
			pts = []
			break
	if pts:
		return pts, False
	for rx in LENIENT:
		pts = [(float(a), float(b)) for a, b in rx.findall(raw)]
		if pts:
			return pts, True
	return [], False


def classify(pts) -> str:
	if not pts:
		return "unparseable"
	x, y = pts[0]
	if x == 0 and y == 0:
		return "zero"
	if x <= 1 and y <= 1:
		return "fraction"
	return "scale_1000"


def as_result(pts) -> ParseResult:
	if not pts or not all(0 <= x <= NORM and 0 <= y <= NORM for x, y in pts):
		return ParseResult(ParseStatus.INVALID if pts else ParseStatus.UNPARSEABLE)
	return ParseResult(ParseStatus.OK, [ParsedPoint(x, y) for x, y in pts])


def run_bench(bench: str) -> tuple[dict, dict]:
	items = {it["id"]: it for it in load_manifest(bench)[0]}
	preds = [json.loads(l) for l in (ROOT / "results" / bench / RUN / "predictions.jsonl").read_text().splitlines()]
	counts = dict.fromkeys(CLASSES, 0)
	examples: dict[str, list[str]] = {c: [] for c in CLASSES}
	hits, frac_hits, frac_v1_hits, n_fallback = [], [], 0, 0
	for p in preds:
		pts, fb = extract(p["raw"])
		n_fallback += fb
		c = classify(pts)
		counts[c] += 1
		if len(examples[c]) < 3:
			examples[c].append(p["raw"])
		if c == "fraction":
			pts = [(x * NORM, y * NORM) for x, y in pts]
		hit = c in ("fraction", "scale_1000") and score_item(bench, items[p["id"]], as_result(pts))["hit"]
		hits.append(hit)
		if c == "fraction":
			frac_hits.append(hit)
			frac_v1_hits += bool(p["hit"])
	n = len(preds)
	out = {
		"n": n,
		"classes": {c: {"count": counts[c], "pct": 100 * counts[c] / n} for c in CLASSES},
		"rescaled_acc_all": summarize(hits),
		"rescaled_acc_fraction_only": summarize(frac_hits),
		"fraction_hits_under_v1_as_is": frac_v1_hits,
		"v1_official_acc": summarize([bool(p["hit"]) for p in preds]),
		"n_lenient_fallback": n_fallback,
		"scoring_note": "first point vs mask; pointbench counting items use the frozen counting rule via score_item on all rescaled points"
		if bench == "pointbench"
		else "first point vs mask",
	}
	return out, examples


def main() -> None:
	res, ex = {}, {}
	for b in BENCHES:
		res[b], ex[b] = run_bench(b)
	res["examples"] = ex
	out = ROOT / "results" / "diagnostics" / "lfm2_rescale.json"
	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_text(json.dumps(res, indent=2) + "\n")
	for b in BENCHES:
		r = res[b]
		a, f = r["rescaled_acc_all"], r["rescaled_acc_fraction_only"]
		cls = " ".join(f"{c}={r['classes'][c]['count']}" for c in CLASSES)
		print(f"{b}: n={r['n']} {cls} | rescaled all {100 * a['acc']:.1f}% [{100 * a['ci95'][0]:.1f},{100 * a['ci95'][1]:.1f}]"
			f" | fraction-only {100 * f['acc']:.1f}% [{100 * f['ci95'][0]:.1f},{100 * f['ci95'][1]:.1f}] (n={f['n']})"
			f" | frac hits v1={r['fraction_hits_under_v1_as_is']} | fallback={r['n_lenient_fallback']}")


if __name__ == "__main__":
	main()
