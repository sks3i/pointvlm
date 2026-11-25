"""Secondary lenient score for finished runs, from their stored raw text (pointeval/lenient.py).

	python scripts/rescore_lenient.py results/pointbench/grpo-v4 results/refcoco/molmo-7b-d-native ...

Writes <run_dir>/lenient.json (summary + per-item first point/hit). predictions.jsonl and
summary.json, i.e. the frozen headline score, are not touched. Scoring rules are pointeval.run's
score_item, unchanged; only the text -> points step differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pointeval.lenient import LENIENT_SHA256, LENIENT_VERSION, parse_lenient  # noqa: E402
from pointeval.metrics import summarize, wilson_ci  # noqa: E402
from pointeval.run import load_manifest, score_item  # noqa: E402


def rescore(run_dir: Path) -> dict:
	summary = json.loads((run_dir / "summary.json").read_text())
	bench = summary["bench"]
	items, meta = load_manifest(bench)
	if meta["manifest_sha256"] != summary["manifest_sha256"]:
		raise SystemExit(f"{run_dir}: manifest changed since the run")
	by_id = {it["id"]: it for it in items}
	family = "molmo" if summary["parser"] == "molmo" else None
	preds = [json.loads(l) for l in (run_dir / "predictions.jsonl").read_text().splitlines()]
	hits, status, cats, per_item, rescued = [], {}, {}, [], 0
	for p in preds:
		it = by_id[p["id"]]
		pr = parse_lenient(p["raw"], it["width"], it["height"], family)
		hit = bool(score_item(bench, it, pr)["hit"])
		hits.append(hit)
		status[pr.status.value] = status.get(pr.status.value, 0) + 1
		rescued += hit and not p["hit"]
		if it.get("category"):
			cats.setdefault(it["category"], []).append(hit)
		per_item.append({"id": p["id"], "status": pr.status.value, "note": pr.note, "first": [pr.first.x, pr.first.y] if pr.first else None, "n_points": len(pr.points), "hit": hit, "strict_hit": p["hit"]})
	out = {
		"run": str(run_dir),
		"bench": bench,
		"parser": LENIENT_VERSION,
		"parser_sha256": LENIENT_SHA256,
		"strict_parser": summary["parser"],
		"strict_acc": summary["point_in_mask"]["acc"],
		"point_in_mask": summarize(hits),
		"parse_status": status,
		"rescued_vs_strict": rescued,
		"lost_vs_strict": sum((not h) and p["hit"] for h, p in zip(hits, preds)),
	}
	if cats:
		out["by_category"] = {c: {"n": len(h), "acc": sum(h) / len(h), "ci95": wilson_ci(sum(h), len(h))} for c, h in sorted(cats.items())}
	(run_dir / "lenient.json").write_text(json.dumps({**out, "items": per_item}, indent=1) + "\n")
	return out


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("run_dirs", nargs="+", type=Path)
	for d in ap.parse_args().run_dirs:
		s = rescore(d)
		print(f"{d}: strict {s['strict_acc']:.4f} lenient {s['point_in_mask']['acc']:.4f} (+{s['rescued_vs_strict']} / -{s['lost_vs_strict']}) {s['parse_status']}")


if __name__ == "__main__":
	main()
