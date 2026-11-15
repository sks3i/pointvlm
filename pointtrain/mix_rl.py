"""Assemble an RL train/dev set from already-scored pools.

	python -m pointtrain.mix_rl --out data/rl/v2 --main data/rl/synth_v1 --extra data/rl/v1 --extra-n 4000

``main`` contributes every learnable row (0 < SFT pass rate < 1, from its scored.jsonl); ``extra``
contributes ``extra_n`` learnable rows from its train.jsonl, chosen by hash, to keep what the earlier
pool taught (RoboPoint free space moved Where2Place +14). dev.jsonl is the union of both dev sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def _rows(p: Path) -> list[dict]:
	return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--main", type=Path, required=True)
	ap.add_argument("--extra", type=Path, required=True)
	ap.add_argument("--extra-n", type=int, default=4000)
	a = ap.parse_args()
	a.out.mkdir(parents=True, exist_ok=True)
	main_rows = [r for r in _rows(a.main / "scored.jsonl") if 0 < r["sft_pass"] < 1]
	extra = sorted(_rows(a.extra / "train.jsonl"), key=lambda r: hashlib.sha256(("mix:" + r["key"]).encode()).hexdigest())[: a.extra_n]
	train = sorted(main_rows + extra, key=lambda r: hashlib.sha256(("order:" + r["key"]).encode()).hexdigest())
	dev = _rows(a.main / "dev.jsonl") + _rows(a.extra / "dev.jsonl")
	train_imgs, dev_imgs = {r["image"] for r in train}, {r["image"] for r in dev}
	assert not train_imgs & dev_imgs, "train and dev share images"
	for name, rows in (("train", train), ("dev", dev)):
		with (a.out / f"{name}.jsonl").open("w") as f:
			for r in rows:
				f.write(json.dumps(r) + "\n")
	meta = {"main": str(a.main), "extra": str(a.extra), "n_train": len(train), "n_dev": len(dev), "train_by_group": dict(Counter(r["group"] for r in train)), "dev_by_group": dict(Counter(r["group"] for r in dev))}
	(a.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
	print(json.dumps(meta, indent=2))


if __name__ == "__main__":
	main()
