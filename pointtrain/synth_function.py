"""Function-style referring prompts from COCO masks: "the utensil used to cut food" -> the only knife.

	python -m pointtrain.synth_function --out data/rl/func_v1

The pass@8 check (pointtrain.affordance_check) showed the SFT model usually knows these objects by
name but misreads function descriptions, with 46% of prompts solved 1-7 times out of 8: the band GRPO
learns from. Each category has two descriptions (affordance_check.FUNCTIONS). The pool uses the
FIRST and dev.jsonl the SECOND, so dev measures transfer to a new wording on unseen images rather
than memorizing a template.

Same rules as the check: the object is the only instance of its category in the image (no crowd
annotation of it either) and covers >= 0.5% of the image. Leakage as in synth_coco: train2014 only,
minus RefCOCO-family held-out, eval-manifest and flagged images; RefCOCO dev-split images go to dev.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from pointeval.adapters import CANONICAL_PROMPT
from pointeval.geometry import mask_to_rle
from pointeval.run import load_manifest
from pointtrain.affordance_check import FUNCTIONS
from pointtrain.build import _write_jsonl
from pointtrain.prompts import _u as dev_u
from pointtrain.sources import coco_id_from_name, refcoco_heldout_image_ids
from pointtrain.synth_coco import _h


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--ann", type=Path, default=Path.home() / "data/coco/annotations/annotations/instances_train2014.json")
	ap.add_argument("--exclude-images", type=Path, default=Path("data/sft/flagged_v1.txt"))
	ap.add_argument("--min-area-frac", type=float, default=0.005)
	ap.add_argument("--per-category", type=int, default=120, help="pool rows per category (common classes would dominate otherwise)")
	ap.add_argument("--dev-per-category", type=int, default=3)
	ap.add_argument("--skip-dev-images", type=Path, default=Path("outputs/affordance_check/prompts.jsonl"), help="images used by the pass@k check")
	a = ap.parse_args()
	a.out.mkdir(parents=True, exist_ok=True)
	from pycocotools import mask as mask_utils

	heldout = refcoco_heldout_image_ids()
	exclude = set().union(*heldout.values()) | {coco_id_from_name(it["file_name"]) for it in load_manifest("refcoco")[0]}
	flagged = set(a.exclude_images.read_text().split()) if a.exclude_images.exists() else set()
	used = {json.loads(l)["image"] for l in a.skip_dev_images.read_text().splitlines()} if a.skip_dev_images.exists() else set()
	d = json.loads(a.ann.read_text())
	cats = {c["id"]: c["name"] for c in d["categories"]}
	anns = defaultdict(list)
	for x in d["annotations"]:
		anns[x["image_id"]].append(x)
	pool, dev = [], []
	n_pool, n_dev = Counter(), Counter()
	for im in sorted(d["images"], key=lambda im: _h(f"func:{im['id']}")):
		iid, W, H = im["id"], im["width"], im["height"]
		path = f"coco/train2014/{im['file_name']}"
		if iid in exclude or path in flagged:
			continue
		is_dev = dev_u(f"dev:coco:{iid}") < 0.015
		if is_dev and path in used:
			continue
		count = Counter(cats[x["category_id"]] for x in anns[iid])
		crowd = {cats[x["category_id"]] for x in anns[iid] if x["iscrowd"]}
		cands = [x for x in anns[iid] if cats[x["category_id"]] in FUNCTIONS and count[cats[x["category_id"]]] == 1 and cats[x["category_id"]] not in crowd and x["area"] / (W * H) >= a.min_area_frac]
		if not cands:
			continue
		# One object per image, the least represented category so far, so rare classes fill up.
		bucket = n_dev if is_dev else n_pool
		x = min(cands, key=lambda x: (bucket[cats[x["category_id"]]], _h(f"pick:{x['id']}")))
		c = cats[x["category_id"]]
		if bucket[c] >= (a.dev_per_category if is_dev else a.per_category):
			continue
		mask = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(x["segmentation"], H, W))).astype(bool)
		q = FUNCTIONS[c][1 if is_dev else 0]
		row = {
			"key": f"func:{iid}:{x['id']}",
			"source": "synth_function",
			"group": "func",
			"image": path,
			"image_id": f"coco:{iid}",
			"category": c,
			"query": q,
			"prompt": CANONICAL_PROMPT.format(query=q),
			"target": {"type": "mask", "mask_rle": mask_to_rle(mask)},
		}
		(dev if is_dev else pool).append(row)
		bucket[c] += 1
	pool.sort(key=lambda r: _h("order:" + r["key"]))
	assert not {r["image"] for r in pool} & {r["image"] for r in dev}
	_write_jsonl(a.out / "pool.jsonl", pool)
	_write_jsonl(a.out / "dev.jsonl", dev)
	meta = {"n_pool": len(pool), "n_dev": len(dev), "categories_pool": len(n_pool), "categories_dev": len(n_dev), "min_per_category_pool": min(n_pool.values()), "args": {k: str(v) for k, v in vars(a).items()}}
	(a.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
	print(json.dumps(meta, indent=2))


if __name__ == "__main__":
	main()
