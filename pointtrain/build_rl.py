"""Build the GRPO prompt pool: data/rl/<name>/{pool.jsonl, dev.jsonl, meta.json}.

	python -m pointtrain.build_rl --out data/rl/v1

Where the SFT model goes wrong is reading the phrase (ordinals like "second from left",
relations like "left of" / "behind" / "far", long RefCOCOg descriptions), not aiming. So the pool
is weighted toward expressions with spatial or ordinal words, and every row carries a target the
reward can check:
	* RefCOCO / RefCOCO+ / RefCOCOg train: exact COCO masks (type "mask").
	* RoboPoint object_ref / region_ref: point sets inside the target (type "points"), relation
	  heavy by construction ("the free space left of the bowl"). Rows with < 3 points are dropped:
	  one or two points say too little about the region for the hull proxy in rl_reward.

All prompts are the frozen eval prompt (CANONICAL_PROMPT): RL should sharpen the behaviour the
eval asks for, and paraphrase robustness was the SFT data's job.

Leakage: the same rules as pointtrain.build (any RefCOCO-family held-out image, any eval-manifest
image, and the pixel-audit flagged list are excluded). dev.jsonl takes the RefCOCO dev-split images
(never in SFT or RL train) plus RoboPoint images held out by hash, so it shares no image with the
pool. The pool is then filtered by pointtrain.rl_difficulty (pass rate of the SFT model).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from pointeval.adapters import CANONICAL_PROMPT
from pointeval.run import load_manifest
from pointtrain.build import _write_jsonl
from pointtrain.sources import build_refcoco_family, build_robopoint, coco_id_from_name, refcoco_heldout_image_ids

SALT = "pointvlm-rl-pool-v1"

SPATIAL = re.compile(
	r"\b(left|right|middle|center|centre|between|behind|front|back|far|farthest|furthest|closest|nearest|near|"
	r"next to|beside|above|below|under|top|bottom|corner|edge|first|second|third|fourth|last|leftmost|rightmost|"
	r"\d+(st|nd|rd|th))\b",
	re.I,
)


def _rank(key: str) -> str:
	return hashlib.sha256(f"{SALT}:{key}".encode()).hexdigest()


def _u(key: str) -> float:
	return int(_rank(key)[:12], 16) / 16**12


def _take(rows: list[dict], n: int, tag: str) -> list[dict]:
	return sorted(rows, key=lambda r: _rank(tag + r["key"]))[:n]


def _rc_row(r: dict, group: str) -> dict:
	return {
		"key": r["key"],
		"source": r["source"],
		"group": group,
		"image": r["image"],
		"image_id": r["image_id"],
		"query": r["query"],
		"prompt": CANONICAL_PROMPT.format(query=r["query"]),
		"target": {"type": "mask", "mask_rle": r["mask_rle"]},
	}


def _rp_rows(data_root: Path) -> list[dict]:
	"""RoboPoint single-target rows with their full point set and image size."""
	from PIL import Image

	rows, _ = build_robopoint(data_root, multi_frac=0.0)
	out = [r for r in rows if r["n_points"] >= 3]
	sizes: dict[str, tuple[int, int] | None] = {}
	# The single-target row kept only ONE of the points; re-read the full set from the source.
	full = _robopoint_point_sets(data_root)
	res = []
	for r in out:
		pts = full.get(r["key"])
		if not pts:
			continue
		if r["image"] not in sizes:
			try:
				with Image.open(data_root / r["image"]) as im:
					sizes[r["image"]] = im.size
			except Exception:  # truncated files from the partial tar shard
				sizes[r["image"]] = None
		if sizes[r["image"]] is None:
			continue
		w, h = sizes[r["image"]]
		kind = r["source"].removeprefix("robopoint_")
		res.append(
			{
				"key": r["key"],
				"source": r["source"],
				"group": f"rp_{kind}",
				"image": r["image"],
				"image_id": r["image_id"],
				"query": r["query"],
				"prompt": CANONICAL_PROMPT.format(query=r["query"]),
				"target": {"type": "points", "points": pts, "width": w, "height": h},
			}
		)
	return res


def _robopoint_point_sets(data_root: Path) -> dict[str, list[list[int]]]:
	import ijson

	from pointtrain.labels import frac_to_canonical
	from pointtrain.sources import _TUPLE_RE, _robopoint_refs

	sets = {}
	with (data_root / "robopoint" / "robopoint_1432k.json").open("rb") as fh:
		for e in _robopoint_refs(ijson.items(fh, "item")):
			conv = e["conversations"]
			for t in range(0, len(conv) - 1, 2):
				pts = [list(frac_to_canonical(float(a), float(b))) for a, b in _TUPLE_RE.findall(conv[t + 1]["value"])]
				if pts:
					sets[f"robopoint:{e['id']}:{t}"] = pts
	return sets


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--data-root", type=Path, default=Path.home() / "data")
	ap.add_argument("--exclude-images", type=Path, default=Path("data/sft/flagged_v1.txt"))
	ap.add_argument("--rc-spatial", type=int, default=12000, help="RefCOCO-family rows with a spatial/ordinal word")
	ap.add_argument("--rc-other", type=int, default=4000, help="other RefCOCO-family rows (guards against forgetting)")
	ap.add_argument("--rp-per-kind", type=int, default=6000, help="RoboPoint rows per kind (object_ref, region_ref)")
	ap.add_argument("--dev-per-group", type=int, default=100)
	ap.add_argument("--rp-dev-frac", type=float, default=0.02, help="RoboPoint images held out for dev, by hash")
	a = ap.parse_args()
	a.out.mkdir(parents=True, exist_ok=True)

	heldout = refcoco_heldout_image_ids()
	items, _ = load_manifest("refcoco")
	eval_ids = {coco_id_from_name(it["file_name"]) for it in items}
	exclude = set().union(*heldout.values()) | eval_ids
	flagged = set(a.exclude_images.read_text().split()) if a.exclude_images and a.exclude_images.exists() else set()

	# RefCOCO family. Same dev_frac / hash as the SFT build, so the dev images are the same ones the
	# SFT run never trained on.
	rc_train, rc_dev, rc_stats = build_refcoco_family(exclude, 0.015, "deepest", 2, keep_masks=True)
	rc_train = [r for r in rc_train if r["image"] not in flagged]
	rc_dev = [r for r in rc_dev if r["image"] not in flagged]
	sp = [r for r in rc_train if SPATIAL.search(r["query"])]
	other = [r for r in rc_train if not SPATIAL.search(r["query"])]
	pool = [_rc_row(r, "rc_spatial") for r in _take(sp, a.rc_spatial, "sp")]
	pool += [_rc_row(r, "rc_other") for r in _take(other, a.rc_other, "ot")]

	# One expression per dev object.
	by_ann: dict[int, dict] = {}
	for r in sorted(rc_dev, key=lambda r: _rank(r["key"])):
		by_ann.setdefault(r["ann_id"], r)
	dev_sp = [r for r in by_ann.values() if SPATIAL.search(r["query"])]
	dev_ot = [r for r in by_ann.values() if not SPATIAL.search(r["query"])]
	dev = [_rc_row(r, "rc_spatial") for r in _take(dev_sp, a.dev_per_group, "dsp")]
	dev += [_rc_row(r, "rc_other") for r in _take(dev_ot, a.dev_per_group, "dot")]

	# RoboPoint, split by image.
	rp = [r for r in _rp_rows(a.data_root) if r["image"] not in flagged]
	for kind in ("rp_object_ref", "rp_region_ref"):
		rows = [r for r in rp if r["group"] == kind]
		tr = [r for r in rows if _u("rpdev:" + r["image"]) >= a.rp_dev_frac]
		dv = [r for r in rows if _u("rpdev:" + r["image"]) < a.rp_dev_frac]
		pool += _take(tr, a.rp_per_kind, kind)
		dev += _take(dv, a.dev_per_group, "d" + kind)

	pool.sort(key=lambda r: _rank("order:" + r["key"]))

	# Leak checks (hard failures).
	pool_coco = {int(r["image_id"].split(":")[1]) for r in pool if r["image_id"].startswith("coco:")}
	dev_coco = {int(r["image_id"].split(":")[1]) for r in dev if r["image_id"].startswith("coco:")}
	assert not pool_coco & exclude and not dev_coco & exclude, "held-out / eval COCO image in RL data"
	assert not pool_coco & dev_coco, "pool and dev share COCO images"
	assert not {r["image"] for r in pool} & {r["image"] for r in dev}, "pool and dev share images"
	assert not {r["image"] for r in pool + dev} & flagged, "flagged near-duplicate image in RL data"

	_write_jsonl(a.out / "pool.jsonl", pool)
	_write_jsonl(a.out / "dev.jsonl", dev)
	meta = {
		"salt": SALT,
		"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
		"n_pool": len(pool),
		"n_dev": len(dev),
		"pool_by_group": dict(Counter(r["group"] for r in pool)),
		"dev_by_group": dict(Counter(r["group"] for r in dev)),
		"refcoco_spatial_available": len(sp),
		"refcoco_other_available": len(other),
		"leakage": {"excluded_images": len(exclude), "flagged": len(flagged), "pool_coco_images": len(pool_coco), "dev_coco_images": len(dev_coco), "checks_passed": True},
		"refcoco_stats": rc_stats,
	}
	(a.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
	print(json.dumps({k: meta[k] for k in ("n_pool", "n_dev", "pool_by_group", "dev_by_group", "leakage")}, indent=2))


if __name__ == "__main__":
	main()
