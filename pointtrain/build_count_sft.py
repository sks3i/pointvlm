"""Counting SFT data: "Point to all the X" -> one point per instance. data/sft/count_v1/{train,meta}.

	python -m pointtrain.build_count_sft --out data/sft/count_v1

Why SFT before RL: on held-out "all the X" prompts with 2+ objects the SFT model was right 0/8 times
(COCO) and 4% (PixMo); v3 12% at best (pointtrain.count_check). GRPO needs prompts the policy
sometimes solves, so the multi-point behaviour has to be put in by supervision first.

Rows (the image sets never overlap data/rl/count_dev):
	* COCO train2014 images outside the RefCOCO dev split, minus held-out / eval / flagged images,
	  same category filter as the dev set (no crowd, 1-10 instances, every instance >= 0.2% of the
	  image and >= 0.1x the largest). Answer: the deepest interior point of every instance mask,
	  left to right (labels.order_points), so position k of the answer has a stable meaning.
	* PixMo-Points multi-instance rows whose images ARE in the SFT train set (the count dev set uses
	  PixMo images outside it), 1-10 points, the human points in the same order.
	* Single-target replay from data/sft/v1/train.jsonl (one-point canonical rows), so the format
	  "exactly one point" holds on every other prompt.
Prompts copy the frozen Point-Bench wording for counting items (count_check.count_prompt).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from pointeval.run import load_manifest
from pointtrain.build import _write_jsonl
from pointtrain.count_check import count_prompt, plural_label
from pointtrain.labels import format_points, frac_to_canonical, mask_target, order_points
from pointtrain.prompts import _u as dev_u
from pointtrain.sources import PIXMO_REPO, PIXMO_REV, _parquets, coco_id_from_name, refcoco_heldout_image_ids
from pointtrain.synth_coco import _h, name, plural


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--ann", type=Path, default=Path.home() / "data/coco/annotations/annotations/instances_train2014.json")
	ap.add_argument("--coco-per-n", type=int, default=800, help="rows per instance count 1..10 (capped by availability)")
	ap.add_argument("--pixmo-max", type=int, default=5000)
	ap.add_argument("--replay", type=int, default=8000)
	a = ap.parse_args()
	a.out.mkdir(parents=True, exist_ok=True)
	from pycocotools import mask as mask_utils

	heldout = refcoco_heldout_image_ids()
	exclude = set().union(*heldout.values()) | {coco_id_from_name(it["file_name"]) for it in load_manifest("refcoco")[0]}
	flagged = set(Path("data/sft/flagged_v1.txt").read_text().split())
	dev_imgs = {json.loads(l)["image"] for l in open("data/rl/count_dev/dev.jsonl")}
	rows, stats = [], Counter()

	d = json.loads(a.ann.read_text())
	cats = {c["id"]: c["name"] for c in d["categories"]}
	anns = defaultdict(list)
	for x in d["annotations"]:
		anns[x["image_id"]].append(x)
	by_n = Counter()
	for im in sorted(d["images"], key=lambda im: _h(f"csft:{im['id']}")):
		iid, W, H = im["id"], im["width"], im["height"]
		path = f"coco/train2014/{im['file_name']}"
		if dev_u(f"dev:coco:{iid}") < 0.015 or iid in exclude or path in flagged or path in dev_imgs:
			continue
		groups = defaultdict(list)
		for x in anns[iid]:
			groups[cats[x["category_id"]]].append(x)
		ok = []
		for c, xs in groups.items():
			if any(x["iscrowd"] for x in xs) or len(xs) > 10:
				continue
			areas = [x["area"] / (W * H) for x in xs]
			if min(areas) >= 0.002 and min(areas) >= 0.1 * max(areas):
				ok.append((c, xs))
		if not ok:
			continue
		c, xs = min(ok, key=lambda o: (by_n[len(o[1])], _h(f"csp:{iid}:{o[0]}")))
		if by_n[len(xs)] >= a.coco_per_n:
			continue
		pts = []
		for x in xs:
			m = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(x["segmentation"], H, W))).astype(bool)
			t = mask_target(m)
			if t is None:
				break
			pts.append((t["x"], t["y"]))
		if len(pts) != len(xs):
			stats["coco_skipped_no_interior"] += 1
			continue
		label = plural(name(c))
		rows.append({"source": "count_coco", "key": f"csft:coco:{iid}:{c}", "image": path, "image_id": f"coco:{iid}", "query": f"all the {label}", "n_points": len(pts), "prompt": count_prompt(label), "answer": format_points(order_points(pts))})
		by_n[len(xs)] += 1
	stats["coco_by_n"] = dict(sorted(by_n.items()))

	import pandas as pd

	sft_rows = [json.loads(l) for l in open("data/sft/v1/train.jsonl")]
	sft_pixmo_imgs = {r["image"] for r in sft_rows if r["source"] == "pixmo"}
	have = {}
	for line in (Path.home() / "data/pixmo/index.jsonl").read_text().splitlines():
		e = json.loads(line)
		if e.get("ok"):
			have[e["image_sha256"]] = e["path"]
	df = pd.concat([pd.read_parquet(f) for f in _parquets(PIXMO_REPO, PIXMO_REV)])
	px = []
	for r in df.itertuples(index=False):
		path = have.get(r.image_sha256)
		n = len(r.points)
		if path is None or path not in sft_pixmo_imgs or path in flagged or path in dev_imgs or not 1 <= n <= 10:
			continue
		label = str(r.label).strip()
		if not label:
			continue
		label = plural_label(label[:1].lower() + label[1:])
		pts = order_points([frac_to_canonical(p["x"] / 100, p["y"] / 100) for p in r.points])
		px.append({"source": "count_pixmo", "key": f"csft:pixmo:{r.image_sha256}:{r.label}", "image": path, "image_id": f"pixmo:{r.image_sha256}", "query": f"all the {label}", "n_points": n, "prompt": count_prompt(label), "answer": format_points(pts)})
	px = sorted(px, key=lambda r: _h("px:" + r["key"]))[: a.pixmo_max]
	rows += px
	stats["pixmo_by_n"] = dict(sorted(Counter(r["n_points"] for r in px).items()))

	single = [r for r in sft_rows if r["answer"].count("<point") == 1 and "exactly one point" in r["prompt"] and r["image"] not in dev_imgs]
	replay = sorted(single, key=lambda r: _h("replay:" + r["key"]))[: a.replay]
	rows += replay
	stats["replay_by_source"] = dict(Counter(r["source"] for r in replay))
	rows.sort(key=lambda r: _h("order:" + r["key"]))
	assert not {r["image"] for r in rows} & dev_imgs, "count SFT rows share an image with the count dev set"
	_write_jsonl(a.out / "train.jsonl", rows)
	meta = {"n_train": len(rows), "by_source": dict(Counter(r["source"] for r in rows)), "stats": stats, "args": {k: str(v) for k, v in vars(a).items()}}
	(a.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
	print(json.dumps(meta, indent=2))


if __name__ == "__main__":
	main()
