"""Merge a subset rerun into a finished run and re-score with the frozen parser (secondary result).

	python scripts/merge_rerun.py pointbench qwen3vl-32b-awq-vllm-native qwen3vl-32b-awq-vllm-native-cap512

Takes results/<bench>/<base>/predictions.jsonl, swaps in every item from results/<bench>/<rerun>/,
re-scores all items with the run's frozen parser and scoring v1, and writes
results/<bench>/<rerun>-merged/{predictions.jsonl,summary.json}. Neither input folder is touched.
Run scripts/rescore_lenient.py on the merged folder for the lenient column.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pointeval.metrics import latency_stats  # noqa: E402
from pointeval.parse import PARSERS  # noqa: E402
from pointeval.run import load_manifest, score_predictions, write_predictions  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
KEEP = ("id", "query", "raw", "n_new_tokens", "latency_ms", "resized_to")


def main() -> None:
	bench, base_run, rerun = sys.argv[1:4]
	base_dir, rerun_dir = ROOT / "results" / bench / base_run, ROOT / "results" / bench / rerun
	base_sum, rerun_sum = (json.loads((d / "summary.json").read_text()) for d in (base_dir, rerun_dir))
	if base_sum["parser"] != rerun_sum["parser"] or base_sum["prompt_sha256"] != rerun_sum["prompt_sha256"]:
		raise SystemExit("base and rerun differ in parser or prompt")
	items, meta = load_manifest(bench)
	if meta["manifest_sha256"] != base_sum["manifest_sha256"]:
		raise SystemExit("manifest changed since the base run")
	load = lambda d: {p["id"]: p for p in map(json.loads, (d / "predictions.jsonl").read_text().splitlines())}  # noqa: E731
	base, new = load(base_dir), load(rerun_dir)
	preds = [{k: v for k, v in (new.get(it["id"]) or base[it["id"]]).items() if k in KEEP} for it in items]
	scores = score_predictions(bench, items, preds, PARSERS[base_sum["parser"]])
	out_dir = ROOT / "results" / bench / f"{rerun}-merged"
	out_dir.mkdir(exist_ok=True)
	write_predictions(out_dir, preds)
	summary = {
		**{k: base_sum[k] for k in ("bench", "model", "mode", "parser", "prompt_template", "prompt_sha256", "manifest_sha256", "dataset_revision")},
		"partial": False,
		"merged_from": {"base": base_run, "rerun": rerun, "n_swapped": len(new), "base_max_new_tokens": base_sum["model_config"]["max_new_tokens"], "rerun_max_new_tokens": rerun_sum["model_config"]["max_new_tokens"]},
		"note": "secondary result, not frozen-spec: swapped items were decoded with a higher token cap",
		**scores,
		"latency": {**latency_stats([p["latency_ms"] for p in preds]), "engine": base_sum["latency"]["engine"]},
		"mean_new_tokens": sum(p["n_new_tokens"] for p in preds) / len(preds),
		"runner": "scripts/merge_rerun.py",
	}
	(out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
	print(json.dumps({k: summary[k] for k in ("point_in_mask", "parse_status", "by_category")}, indent=1))


if __name__ == "__main__":
	main()
