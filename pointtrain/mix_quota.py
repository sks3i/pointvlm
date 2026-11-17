"""Assemble an RL train set from scored pools with a fixed number of learnable rows per group.

	python -m pointtrain.mix_quota --out data/rl/v3 \
		--source data/rl/func_v1/scored.jsonl:func=3500 \
		--source data/rl/v1/train.jsonl:rp_region_ref=3000,rp_object_ref=1000 \
		--dev data/rl/v2/dev.jsonl --dev data/rl/func_v1/dev.jsonl

Rows count as learnable when 0 < sft_pass < 1. A group with fewer learnable rows than its quota
contributes all of them (the shortfall is reported). Picks are hash-based, so the mix is reproducible.
Why quotas: v2 diluted RoboPoint free space from 30% to 10% of the prompts and gave back part of
v1's Where2Place gain; fixing each group's count makes the mix an explicit choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def _rows(p: Path) -> list[dict]:
	return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _h(s: str) -> str:
	return hashlib.sha256(s.encode()).hexdigest()


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--source", action="append", required=True, help="path:GROUP=N[,GROUP=N...]")
	ap.add_argument("--dev", action="append", type=Path, default=[])
	a = ap.parse_args()
	a.out.mkdir(parents=True, exist_ok=True)
	train, report = [], {}
	for spec in a.source:
		path, _, q = spec.rpartition(":")
		rows = [r for r in _rows(Path(path)) if 0 < r.get("sft_pass", 0.5) < 1]
		for item in q.split(","):
			g, n = item.split("=")
			pick = sorted((r for r in rows if r["group"] == g), key=lambda r: _h("mix3:" + r["key"]))
			train += pick[: int(n)]
			report[g] = {"quota": int(n), "available": len(pick), "taken": min(len(pick), int(n))}
	train.sort(key=lambda r: _h("order3:" + r["key"]))
	dev = [r for p in a.dev for r in _rows(p)]
	assert len({r["key"] for r in train}) == len(train), "duplicate rows across sources"
	assert not {r["image"] for r in train} & {r["image"] for r in dev}, "train and dev share images"
	for name, rows in (("train", train), ("dev", dev)):
		with (a.out / f"{name}.jsonl").open("w") as f:
			for r in rows:
				f.write(json.dumps(r) + "\n")
	meta = {"sources": a.source, "dev": [str(p) for p in a.dev], "n_train": len(train), "n_dev": len(dev), "groups": report, "dev_by_group": dict(Counter(r["group"] for r in dev))}
	(a.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
	print(json.dumps(meta, indent=2))


if __name__ == "__main__":
	main()
