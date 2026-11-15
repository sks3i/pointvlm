"""Paired comparison of two eval runs on the same frozen bench (exact McNemar, per category).

	python scripts/compare_runs.py pointbench sft-qwen3vl-2b-v1-e2 grpo-v1

Reads results/<bench>/<run>/predictions.jsonl for both runs; nothing is re-scored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pointeval.metrics import mcnemar_exact, summarize

ROOT = Path(__file__).resolve().parent.parent


def load(bench: str, run: str) -> dict[str, dict]:
	path = ROOT / "results" / bench / run / "predictions.jsonl"
	return {p["id"]: p for p in map(json.loads, path.read_text().splitlines())}


def main() -> None:
	bench, a_run, b_run = sys.argv[1:4]
	a, b = load(bench, a_run), load(bench, b_run)
	ids = sorted(set(a) & set(b))
	if len(ids) != len(a) or len(ids) != len(b):
		print(f"warning: {len(a)} vs {len(b)} items, comparing the {len(ids)} in common")
	cats = {"all": ids}
	for i in ids:
		if a[i].get("category"):
			cats.setdefault(a[i]["category"], []).append(i)
	print(f"{bench}: {a_run} (A) vs {b_run} (B)")
	print(f"{'subset':<12}{'n':>5}{'A acc':>8}{'B acc':>8}{'delta':>8}{'A only':>8}{'B only':>8}{'p':>9}")
	for name, sub in cats.items():
		ha, hb = [bool(a[i]["hit"]) for i in sub], [bool(b[i]["hit"]) for i in sub]
		sa, sb, t = summarize(ha), summarize(hb), mcnemar_exact(ha, hb)
		print(f"{name:<12}{len(sub):>5}{sa['acc']:>8.3f}{sb['acc']:>8.3f}{sb['acc'] - sa['acc']:>+8.3f}{t['b10']:>8}{t['b01']:>8}{t['p_value']:>9.4f}")


if __name__ == "__main__":
	main()
