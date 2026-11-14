"""Build the SFT mixture: train.jsonl, dev.jsonl and meta.json.

	python -m pointtrain.build --out data/sft/v1

Leakage rules (checked here, then audited on pixels by pointtrain.leakcheck):
	* RefCOCO family: an image is excluded from training if ANY of RefCOCO / RefCOCO+ / RefCOCOg
	  holds it out (val, testA, testB, test), and if it appears in the frozen eval manifest.
	* dev.jsonl comes from train-split images chosen by hash and shares no image with train.jsonl.
	  It is what we tune and select checkpoints on; the frozen eval is for final numbers only.
	* Sampling and shuffling are hash-based (sha256 of a salted key), so the mixture is reproducible
	  and does not depend on RNG versions or row order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from pointeval.run import load_manifest
from pointtrain.sources import (
	PIXMO_REPO,
	PIXMO_REV,
	REFCOCO_FAMILY,
	ROBOPOINT_REPO,
	ROBOPOINT_REV,
	build_pixmo,
	build_refcoco_family,
	build_robopoint,
	coco_id_from_name,
	refcoco_heldout_image_ids,
)

SALT = "pointvlm-sft-mix-v1"


def _rank(key: str) -> str:
	return hashlib.sha256(f"{SALT}:{key}".encode()).hexdigest()


def _cap(rows: list[dict], n: int | None) -> list[dict]:
	return rows if n is None or len(rows) <= n else sorted(rows, key=lambda r: _rank(r["key"]))[:n]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
	with path.open("w") as f:
		for r in rows:
			f.write(json.dumps(r) + "\n")


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--data-root", type=Path, default=Path.home() / "data")
	ap.add_argument("--label-mode", choices=["deepest", "depth_sampled"], default="deepest")
	ap.add_argument("--max-sents-per-ref", type=int, default=2)
	ap.add_argument("--dev-frac", type=float, default=0.015, help="fraction of RefCOCO-family train IMAGES held out as dev")
	ap.add_argument("--dev-max", type=int, default=500, help="dev objects kept (one expression each)")
	ap.add_argument("--refcoco-max", type=int, default=120000)
	ap.add_argument("--pixmo-max", type=int, default=60000)
	ap.add_argument("--robopoint-max", type=int, default=40000, help="per RoboPoint kind (region_ref, object_ref)")
	ap.add_argument("--skip", nargs="*", default=[], choices=["pixmo", "robopoint"])
	ap.add_argument("--exclude-images", type=Path, default=None, help="flagged_images.txt from pointtrain.leakcheck")
	a = ap.parse_args()
	a.out.mkdir(parents=True, exist_ok=True)

	# 1. Exclusion set.
	heldout = refcoco_heldout_image_ids()
	items, _ = load_manifest("refcoco")
	eval_ids = {coco_id_from_name(it["file_name"]) for it in items}
	exclude = set().union(*heldout.values()) | eval_ids
	print({k: len(v) for k, v in heldout.items()}, "eval images:", len(eval_ids), "excluded total:", len(exclude))

	# 2. Sources.
	stats: dict = {}
	rc_train, rc_dev, stats["refcoco_family"] = build_refcoco_family(exclude, a.dev_frac, a.label_mode, a.max_sents_per_ref)
	train = _cap(rc_train, a.refcoco_max)
	if "pixmo" not in a.skip:
		px, stats["pixmo"] = build_pixmo(a.data_root)
		train += _cap(px, a.pixmo_max)
	if "robopoint" not in a.skip:
		rp, stats["robopoint"] = build_robopoint(a.data_root)
		for kind in ("robopoint_region_ref", "robopoint_object_ref"):
			train += _cap([r for r in rp if r["source"] == kind], a.robopoint_max)
	n_flag_dropped = 0
	if a.exclude_images:
		flagged = set(a.exclude_images.read_text().split())
		n_before = len(train)
		train = [r for r in train if r["image"] not in flagged]
		rc_dev = [r for r in rc_dev if r["image"] not in flagged]
		n_flag_dropped = n_before - len(train)
	train.sort(key=lambda r: _rank("order:" + r["key"]))

	# Dev: one expression per object, canonical prompt only (that is how the eval asks).
	by_ann: dict[int, dict] = {}
	for r in sorted(rc_dev, key=lambda r: _rank(r["key"])):
		by_ann.setdefault(r["ann_id"], r)
	dev = sorted(by_ann.values(), key=lambda r: _rank("dev:" + r["key"]))[: a.dev_max]

	# 3. Leak checks (hard failures).
	train_coco = {int(r["image_id"].split(":")[1]) for r in train if r["image_id"].startswith("coco:")}
	dev_coco = {int(r["image_id"].split(":")[1]) for r in dev}
	assert not train_coco & exclude, "train contains a held-out / eval COCO image"
	assert not train_coco & eval_ids, "train contains an eval image"
	assert not dev_coco & exclude, "dev contains a held-out / eval COCO image"
	assert not train_coco & dev_coco, "train and dev share images"

	# 4. Write.
	_write_jsonl(a.out / "train.jsonl", train)
	_write_jsonl(a.out / "dev.jsonl", dev)
	rc_depth = np.array([r["depth_px"] for r in train if r["source"].startswith("refcoco")])
	meta = {
		"salt": SALT,
		"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
		"revisions": {
			**{k: {"repo": r, "revision": v} for k, (r, v) in REFCOCO_FAMILY.items()},
			"pixmo": {"repo": PIXMO_REPO, "revision": PIXMO_REV},
			"robopoint": {"repo": ROBOPOINT_REPO, "revision": ROBOPOINT_REV},
		},
		"n_train": len(train),
		"n_dev": len(dev),
		"train_by_source": dict(Counter(r["source"] for r in train)),
		"multi_point_rows": sum(r["answer"].count("<point") > 1 for r in train),
		"leakage": {
			"heldout_image_ids": {k: len(v) for k, v in heldout.items()},
			"eval_refcoco_images": len(eval_ids),
			"excluded_images": len(exclude),
			"train_coco_images": len(train_coco),
			"dev_coco_images": len(dev_coco),
			"rows_dropped_pixel_near_duplicates": n_flag_dropped,
			"checks_passed": True,
		},
		"label_stats_refcoco": {
			"mode": a.label_mode,
			"depth_px_p10_p50_p90": np.percentile(rc_depth, [10, 50, 90]).round(1).tolist() if len(rc_depth) else None,
		},
		"source_stats": stats,
	}
	(a.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
	print(json.dumps({k: meta[k] for k in ("n_train", "n_dev", "train_by_source", "leakage")}, indent=2))


if __name__ == "__main__":
	main()
