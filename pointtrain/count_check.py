"""Counting dev set + pass@k check: can GRPO learn "point to all the X", or does it need SFT first?

	python -m pointtrain.count_check --out data/rl/count_dev            # build + sample SFT
	python -m pointtrain.count_check --out data/rl/count_dev --build-only

Scoring follows the PointArena counting rule the frozen Point-Bench uses: the number of points must
equal the number of targets, and every target must get at least one point.

Prompts copy Point-Bench's exact wording for these items: the frozen harness formats its query
"Point to all the X." into CANONICAL_PROMPT, which itself starts with "Point to", so the model sees
"Point to Point to all the X.. Answer with exactly one point ...". Training and dev use that same
string so the format matches the eval (a format choice, not item leakage).

Sources (none touch any eval item):
	* COCO train2014 images in the RefCOCO dev split (never trained on by SFT or GRPO), minus held-out,
	  eval-manifest and flagged images. Category with 1-10 instances, no crowd region of it, every
	  instance >= 0.2% of the image and >= 0.1x the largest (COCO often leaves tiny background
	  instances unlabeled, which would make the count wrong). Target: every instance mask.
	* PixMo-Points rows whose image is on disk but NOT in the SFT training set, minus flagged images,
	  1-10 points. Target: the labeled points; a predicted point matches a labeled one within
	  ``radius_frac`` of the image diagonal (one-to-one, Hungarian).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pointeval.adapters import CANONICAL_PROMPT
from pointeval.geometry import mask_to_rle, norm_to_pixel, rle_to_mask
from pointeval.parse import ParsedPoint, ParseStatus, parse_canonical
from pointeval.run import load_manifest
from pointtrain.labels import frac_to_canonical
from pointtrain.prompts import _u as dev_u
from pointtrain.sources import PIXMO_REPO, PIXMO_REV, _parquets, coco_id_from_name, refcoco_heldout_image_ids
from pointtrain.synth_coco import _h, name, plural

RADIUS_FRAC = 0.05
IRREGULAR = {"man": "men", "woman": "women", "child": "children", "foot": "feet", "tooth": "teeth", "leaf": "leaves", "goose": "geese", "fish": "fish", "deer": "deer"}


def plural_label(label: str) -> str:
	*head, last = label.split(" ")
	if last in IRREGULAR:
		return " ".join(head + [IRREGULAR[last]])
	return label if label.endswith("s") else plural(label)


def count_prompt(label: str) -> str:
	return CANONICAL_PROMPT.format(query=f"Point to all the {label}.")


def count_score(row: dict, text: str) -> dict:
	"""PointArena counting rule. Returns hit (0/1), n_pred, n_true, matched targets."""
	pr = parse_canonical(text)
	pts = pr.points if pr.status is ParseStatus.OK else []
	t = row["target"]
	if t["type"] == "masks":
		masks = [rle_to_mask(m) for m in t["masks"]]
		h, w = masks[0].shape
		covered = set()
		for p in pts:
			c, r = norm_to_pixel(p, w, h)
			covered |= {i for i, m in enumerate(masks) if m[r, c]}
		n_true, matched = len(masks), len(covered)
	else:
		from scipy.optimize import linear_sum_assignment

		w, h = t["width"], t["height"]
		gt = np.asarray(t["points"], float) * [w / 1000, h / 1000]
		n_true = len(gt)
		matched = 0
		if pts:
			pp = np.asarray([[p.x, p.y] for p in pts], float) * [w / 1000, h / 1000]
			d = np.linalg.norm(pp[:, None] - gt[None], axis=2)
			ri, ci = linear_sum_assignment(d)
			matched = int((d[ri, ci] <= RADIUS_FRAC * np.hypot(w, h)).sum())
	hit = len(pts) == n_true and matched == n_true
	return {"reward": float(hit), "hit": float(hit), "n_pred": len(pts), "n_true": n_true, "count_ok": float(len(pts) == n_true), "matched_frac": matched / n_true}


def build(a) -> list[dict]:
	from PIL import Image
	from pycocotools import mask as mask_utils

	heldout = refcoco_heldout_image_ids()
	exclude = set().union(*heldout.values()) | {coco_id_from_name(it["file_name"]) for it in load_manifest("refcoco")[0]}
	flagged = set(Path("data/sft/flagged_v1.txt").read_text().split())
	rows = []

	d = json.loads(a.ann.read_text())
	cats = {c["id"]: c["name"] for c in d["categories"]}
	anns = defaultdict(list)
	for x in d["annotations"]:
		anns[x["image_id"]].append(x)
	by_n = Counter()
	for im in sorted(d["images"], key=lambda im: _h(f"count:{im['id']}")):
		iid, W, H = im["id"], im["width"], im["height"]
		path = f"coco/train2014/{im['file_name']}"
		if dev_u(f"dev:coco:{iid}") >= 0.015 or iid in exclude or path in flagged:
			continue
		groups = defaultdict(list)
		for x in anns[iid]:
			groups[cats[x["category_id"]]].append(x)
		ok = []
		for c, xs in groups.items():
			if any(x["iscrowd"] for x in xs) or len(xs) > 10:
				continue
			areas = [x["area"] / (W * H) for x in xs]
			if min(areas) < 0.002 or min(areas) < 0.1 * max(areas):
				continue
			ok.append((c, xs))
		if not ok:
			continue
		c, xs = min(ok, key=lambda o: (by_n[min(len(o[1]), 5)], _h(f"cpick:{iid}:{o[0]}")))
		n = len(xs)
		if by_n[min(n, 5)] >= a.coco_per_bucket:
			continue
		masks = [mask_to_rle(mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(x["segmentation"], H, W))).astype(bool)) for x in xs]
		label = plural(name(c))
		rows.append({"key": f"count:coco:{iid}:{c}", "source": "count_coco", "group": "count_coco", "image": path, "image_id": f"coco:{iid}", "query": f"all the {label}", "n": n, "prompt": count_prompt(label), "target": {"type": "masks", "masks": masks}})
		by_n[min(n, 5)] += 1

	import pandas as pd

	sft_imgs = {json.loads(l)["image"] for l in open("data/sft/v1/train.jsonl") if '"pixmo"' in l}
	have = {}
	for line in (Path.home() / "data/pixmo/index.jsonl").read_text().splitlines():
		e = json.loads(line)
		if e.get("ok"):
			have[e["image_sha256"]] = e["path"]
	df = pd.concat([pd.read_parquet(f) for f in _parquets(PIXMO_REPO, PIXMO_REV)])
	by_n = Counter()
	for r in sorted(df.itertuples(index=False), key=lambda r: _h(f"px:{r.image_sha256}:{r.label}")):
		path = have.get(r.image_sha256)
		n = len(r.points)
		if path is None or path in sft_imgs or path in flagged or not 1 <= n <= 10 or by_n[min(n, 5)] >= a.pixmo_per_bucket:
			continue
		try:
			with Image.open(Path.home() / "data" / path) as im:
				W, H = im.size
		except Exception:
			continue
		label = str(r.label).strip()
		label = label[:1].lower() + label[1:]
		label = plural_label(label)
		pts = [list(frac_to_canonical(p["x"] / 100, p["y"] / 100)) for p in r.points]
		rows.append({"key": f"count:pixmo:{r.image_sha256}:{r.label}", "source": "count_pixmo", "group": "count_pixmo", "image": path, "image_id": f"pixmo:{r.image_sha256}", "query": f"all the {label}", "n": n, "prompt": count_prompt(label), "target": {"type": "points", "points": pts, "width": W, "height": H}})
		by_n[min(n, 5)] += 1
	return rows


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--ann", type=Path, default=Path.home() / "data/coco/annotations/annotations/instances_train2014.json")
	ap.add_argument("--coco-per-bucket", type=int, default=30, help="per instance-count bucket 1,2,3,4,5+")
	ap.add_argument("--pixmo-per-bucket", type=int, default=30)
	ap.add_argument("--k", type=int, default=8)
	ap.add_argument("--build-only", action="store_true")
	ap.add_argument("--model", default="outputs/sft-qwen3vl-2b-v1-e2/merged")
	ap.add_argument("--tag", default="sft")
	a = ap.parse_args()
	a.out.mkdir(parents=True, exist_ok=True)
	dev_path = a.out / "dev.jsonl"
	if not dev_path.exists():
		rows = build(a)
		with dev_path.open("w") as f:
			for r in rows:
				f.write(json.dumps(r) + "\n")
	rows = [json.loads(l) for l in dev_path.read_text().splitlines()]
	print(f"{len(rows)} prompts:", dict(Counter((r['group'], min(r['n'], 5)) for r in rows)))
	if a.build_only:
		return

	from train_utils.configs.model_config import LoRAConfig, ModelConfig, QuantizationConfig
	from train_utils.vlm import build_adapter

	cfg = ModelConfig(model_id=str(Path(a.model).resolve()), vlm_adapter="hf_chat", dtype="bfloat16", attn_implementation="sdpa", lora=LoRAConfig(use_lora=False), quantization=QuantizationConfig(load_in_4bit=False))
	adapter = build_adapter(cfg, max_image_side=1024, image_root=str(Path.home() / "data"))
	model, _ = adapter.load()
	model.eval()
	res = []
	for i in range(0, len(rows), 8):
		chunk = rows[i : i + 8]
		comps = adapter.sample(model, [r for r in chunk for _ in range(a.k)], 128, True, 1.0, 1.0, 0)
		greedy = adapter.sample(model, chunk, 128, do_sample=False)
		for j, r in enumerate(chunk):
			ss = [count_score(r, adapter.decode(c)) for c in comps[j * a.k : (j + 1) * a.k]]
			g = count_score(r, adapter.decode(greedy[j]))
			res.append({"key": r["key"], "group": r["group"], "n": r["n"], "hits": int(sum(s["hit"] for s in ss)), "greedy": g["hit"], "greedy_n_pred": g["n_pred"], "mean_n_pred": float(np.mean([s["n_pred"] for s in ss])), "count_ok_rate": float(np.mean([s["count_ok"] for s in ss]))})
	(a.out / f"results_{a.tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res))
	summary = {}
	for g in sorted({r["group"] for r in res}):
		for bucket in ("1", "2+", "all"):
			rs = [r for r in res if r["group"] == g and (bucket == "all" or (bucket == "1") == (r["n"] == 1))]
			if not rs:
				continue
			summary[f"{g}/n={bucket}"] = {
				"n": len(rs),
				"greedy": round(float(np.mean([r["greedy"] for r in rs])), 3),
				"pass@1": round(float(np.mean([r["hits"] / a.k for r in rs])), 3),
				"pass@8": round(float(np.mean([r["hits"] > 0 for r in rs])), 3),
				"greedy_one_point": round(float(np.mean([r["greedy_n_pred"] == 1 for r in rs])), 3),
				"count_ok_rate": round(float(np.mean([r["count_ok_rate"] for r in rs])), 3),
				"hits_out_of_8": dict(sorted(Counter(r["hits"] for r in rs).items())),
			}
	(a.out / f"summary_{a.tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
