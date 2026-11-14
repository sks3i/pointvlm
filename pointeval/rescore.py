"""Re-score a finished run from its stored raw outputs, with the current parser and scoring.

	python -m pointeval.rescore results/refcoco/qwen3vl-2b-canonical [more run dirs ...]

Rewrites predictions.jsonl and summary.json in place. Raw text, latency and token counts
are kept as recorded; only parse/score fields and the metrics derived from them change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pointeval.parse import PARSERS
from pointeval.run import load_manifest, score_predictions, write_predictions


def rescore(run_dir: Path) -> dict:
	summary = json.loads((run_dir / "summary.json").read_text())
	items, meta = load_manifest(summary["bench"])
	if meta["manifest_sha256"] != summary["manifest_sha256"]:
		raise SystemExit(f"{run_dir}: run used manifest {summary['manifest_sha256']}, current is {meta['manifest_sha256']}")
	preds = [json.loads(l) for l in (run_dir / "predictions.jsonl").read_text().splitlines()]
	summary.update(score_predictions(summary["bench"], items, preds, PARSERS[summary["parser"]]))
	write_predictions(run_dir, preds)
	(run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
	return summary


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("run_dirs", nargs="+", type=Path)
	for d in ap.parse_args().run_dirs:
		s = rescore(d)
		print(f"{d}: acc={s['point_in_mask']['acc']:.4f} parse={s['parse_status']}")


if __name__ == "__main__":
	main()
