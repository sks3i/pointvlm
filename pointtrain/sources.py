"""Per-source converters to {image, prompt, answer, ...} training rows.

Image paths are relative to the data root (default ~/data):
	coco/train2014/COCO_train2014_*.jpg        RefCOCO / RefCOCO+ / RefCOCOg (UNC / UMD train splits)
	pixmo/images/<sha256>.<ext>                 PixMo-Points (fetched by pointtrain.download_pixmo)
	robopoint/images/{region_ref,object_ref}/*  RoboPoint (the shards we extracted)
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from pointeval.geometry import mask_to_rle, polygons_to_mask
from pointtrain.labels import format_points, frac_to_canonical, mask_target, order_points
from pointtrain.prompts import MULTI_PROMPT, MULTI_REGION_PROMPT, _u, single_prompt

HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"

# Pinned annotation revisions. jxu124/refcoco is the same revision the eval manifest used.
REFCOCO_FAMILY = {
	"refcoco": ("jxu124/refcoco", "9fba8200c5326e996f789191f095bd464ef1d09e"),
	"refcoco+": ("jxu124/refcocoplus", "12e20b0b6039fbf656e89e2f26597e84c1037847"),
	"refcocog": ("jxu124/refcocog", "55319436ad54b9480cefda6b9d64397de92456dd"),
}
PIXMO_REPO, PIXMO_REV = "allenai/pixmo-points", "2b5c6931e790e00ae00d4a2857e5f95d88f09a66"
ROBOPOINT_REPO, ROBOPOINT_REV = "wentao-yuan/robopoint-data", "7b06a4831afe7d5d46f19db0f0042ab4f58388e8"


def _parquets(repo: str, rev: str, split_prefix: str | None = None) -> list[str]:
	from huggingface_hub import snapshot_download

	root = Path(snapshot_download(repo, repo_type="dataset", revision=rev))
	files = sorted(glob.glob(str(root / "data" / "*.parquet")))
	return [f for f in files if split_prefix is None or Path(f).name.startswith(split_prefix + "-")]


def coco_id_from_name(name: str) -> int | None:
	m = re.search(r"COCO_(?:train|val)2014_0*(\d+)", name)
	return int(m.group(1)) if m else None


# --- RefCOCO family ------------------------------------------------------------------


def refcoco_heldout_image_ids() -> dict[str, set[int]]:
	"""COCO image ids in ANY non-train split of RefCOCO, RefCOCO+ and RefCOCOg.

	The three datasets were split independently over the same COCO train2014 pool. RefCOCOg's UMD
	train split in particular contains images that are in RefCOCO val/testA/testB, so filtering each
	dataset by its own split flag is not enough: an image is excluded from training if it is held
	out by any of them.
	"""
	import pandas as pd

	out: dict[str, set[int]] = {}
	for name, (repo, rev) in REFCOCO_FAMILY.items():
		for f in _parquets(repo, rev):
			split = Path(f).name.split("-")[0]
			if split == "train":
				continue
			ids = set(pd.read_parquet(f, columns=["image_id"])["image_id"].astype(int))
			out[f"{name}/{split}"] = ids
	return out


def build_refcoco_family(
	exclude_image_ids: set[int],
	dev_frac: float,
	label_mode: str,
	max_sents_per_ref: int,
	seed: int = 0,
) -> tuple[list[dict], list[dict], dict]:
	"""Rows for the train splits of RefCOCO/+/g, minus excluded images, plus a held-out dev split.

	Dev = train-split images chosen by hash (never the eval's UNC val), used for checkpoint selection
	and in-training monitoring so the frozen eval set is only looked at for final numbers.
	"""
	import pandas as pd

	rng = np.random.default_rng(seed)
	train, dev = [], []
	stats: dict = defaultdict(int)
	mask_cache: dict[int, dict] = {}  # ann_id -> target (RefCOCO and RefCOCO+ share annotations)
	for name, (repo, rev) in REFCOCO_FAMILY.items():
		(f,) = _parquets(repo, rev, "train")
		df = pd.read_parquet(f, columns=["ann_id", "ref_id", "image_id", "split", "sentences", "raw_anns", "raw_image_info", "image_path"])
		assert set(df["split"]) == {"train"}
		for r in df.itertuples(index=False):
			stats[f"{name}/refs"] += 1
			img_id = int(r.image_id)
			if img_id in exclude_image_ids:
				stats[f"{name}/refs_dropped_heldout_image"] += 1
				continue
			info = json.loads(r.raw_image_info)
			w, h = int(info["width"]), int(info["height"])
			ann = json.loads(r.raw_anns)
			if ann.get("iscrowd"):
				stats[f"{name}/refs_dropped_crowd"] += 1
				continue
			is_dev = _u(f"dev:coco:{img_id}") < dev_frac
			ann_id = int(r.ann_id)
			if ann_id not in mask_cache or label_mode != "deepest":
				try:
					mask = polygons_to_mask(ann["segmentation"], h, w)
				except ValueError:
					stats[f"{name}/refs_dropped_bad_polygon"] += 1
					continue
				tgt = mask_target(mask, label_mode, rng)
				if tgt is None:
					stats[f"{name}/refs_dropped_no_interior"] += 1
					continue
				tgt["mask_rle"] = mask_to_rle(mask) if is_dev else None
				mask_cache[ann_id] = tgt
			tgt = mask_cache[ann_id]
			sents = sorted({s["sent"].strip() for s in r.sentences if s["sent"].strip()})
			sents = sorted(sents, key=lambda s: _u(f"sent:{name}:{r.ref_id}:{s}"))[:max_sents_per_ref]
			for s in sents:
				key = f"{name}:{r.ref_id}:{s}"
				row = {
					"source": name,
					"key": key,
					"image": r.image_path,  # coco/train2014/COCO_train2014_*.jpg
					"image_id": f"coco:{img_id}",
					"width": w,
					"height": h,
					"query": s,
					"prompt": single_prompt(s, key) if not is_dev else single_prompt(s, key, canonical_frac=1.0),
					"answer": format_points([(tgt["x"], tgt["y"])]),
					"depth_px": round(tgt["depth_px"], 2),
					"max_depth_px": round(tgt["max_depth_px"], 2),
				}
				if is_dev:
					row["mask_rle"] = tgt["mask_rle"]
					row["ann_id"] = ann_id
					dev.append(row)
				else:
					train.append(row)
				stats[f"{name}/{'dev' if is_dev else 'train'}_rows"] += 1
	return train, dev, dict(stats)


# --- PixMo-Points --------------------------------------------------------------------


def build_pixmo(data_root: Path, max_points: int = 20) -> tuple[list[dict], dict]:
	"""Rows for PixMo-Points images we have on disk (see pointtrain.download_pixmo).

	count == 1 -> single-point example with a single-target prompt.
	count > 1  -> all points (in canonical order) under the multi-instance prompt, if <= max_points.
	count == 0 -> skipped (the eval has no "not present" case, and the canonical format has no
	              way to say "none").
	PixMo stores coordinates in percent (0-100) of the image.
	"""
	import pandas as pd

	index_path = data_root / "pixmo" / "index.jsonl"
	have = {}
	if index_path.exists():
		for line in index_path.read_text().splitlines():
			e = json.loads(line)
			if e.get("ok"):
				have[e["image_sha256"]] = e
	df = pd.concat([pd.read_parquet(f) for f in _parquets(PIXMO_REPO, PIXMO_REV)])
	rows, stats = [], defaultdict(int)
	for r in df.itertuples(index=False):
		e = have.get(r.image_sha256)
		if e is None:
			stats["skipped_no_image"] += 1
			continue
		pts = [frac_to_canonical(p["x"] / 100, p["y"] / 100) for p in r.points]
		label = str(r.label).strip()
		label = label[:1].lower() + label[1:]  # "Book" -> "book" inside a sentence
		key = f"pixmo:{r.image_sha256}:{label}"
		if not pts:
			stats["skipped_zero_count"] += 1
			continue
		if len(pts) == 1:
			prompt = single_prompt(label, key)
			stats["single"] += 1
		elif len(pts) <= max_points:
			prompt = MULTI_PROMPT.format(query=label)
			pts = order_points(pts)
			stats["multi"] += 1
		else:
			stats["skipped_too_many_points"] += 1
			continue
		rows.append(
			{
				"source": "pixmo",
				"key": key,
				"image": e["path"],
				"image_id": f"pixmo:{r.image_sha256}",
				"query": label,
				"prompt": prompt,
				"answer": format_points(pts),
				"n_points": len(pts),
			}
		)
	return rows, dict(stats)


# --- RoboPoint -----------------------------------------------------------------------

_RP_SUFFIX = ("Your answer should", "Your answer must")
_TUPLE_RE = re.compile(r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)")


def _strip_rp_question(text: str) -> str:
	text = text.replace("<image>", "").strip()
	cut = min((text.find(m) for m in _RP_SUFFIX if m in text), default=-1)
	return (text[:cut] if cut > 0 else text).strip()


def _robopoint_refs(items):
	for e in items:
		if e.get("id", "").split("/")[0] in ("region_ref", "object_ref"):
			yield e


def build_robopoint(data_root: Path, multi_frac: float = 0.3) -> tuple[list[dict], dict]:
	"""Rows for RoboPoint region_ref (free space) and object_ref (objects) on extracted images.

	The question text is stripped of RoboPoint's own answer-format suffix (as the frozen Where2Place
	manifest does) and wrapped in our prompt. RoboPoint answers are sets of points sampled inside the
	target region. Most rows take ONE of them under the single-target prompt, the form Where2Place is
	evaluated in; ``multi_frac`` of rows keep the full set under the multi-point prompt.
	"""
	import ijson

	img_root = data_root / "robopoint" / "images"
	rows, stats = [], defaultdict(int)
	# Streamed: json.loads of the 1.7 GB file holds ~10+ GB of Python objects at once.
	with (data_root / "robopoint" / "robopoint_1432k.json").open("rb") as fh:
		entries = list(_robopoint_refs(ijson.items(fh, "item")))
	# Membership from one readdir per folder instead of a stat per entry: on this box a path lookup
	# of certain missing names in the extracted object_ref folder kills the process (likely a damaged
	# directory index from the truncated tar shard), and listing is faster anyway.
	have = {f"{d.name}/{n}" for d in img_root.iterdir() if d.is_dir() for n in os.listdir(d)}
	for e in entries:
		kind = e.get("id", "").split("/")[0]
		if kind not in ("region_ref", "object_ref") or "image" not in e:
			continue
		if e["image"] not in have:
			stats[f"{kind}/skipped_no_image"] += 1
			continue
		conv = e["conversations"]
		# Some entries are multi-turn; take each (human, gpt) pair.
		for t in range(0, len(conv) - 1, 2):
			q = _strip_rp_question(conv[t]["value"])
			pts = [frac_to_canonical(float(a), float(b)) for a, b in _TUPLE_RE.findall(conv[t + 1]["value"])]
			if not q or not pts:
				stats[f"{kind}/skipped_unparsed"] += 1
				continue
			key = f"robopoint:{e['id']}:{t}"
			if len(pts) > 1 and _u("multi:" + key) < multi_frac:
				prompt, ans = MULTI_REGION_PROMPT.format(query=q), format_points(order_points(pts))
			else:
				pick = pts[int(_u("pick:" + key) * len(pts))]
				prompt, ans = single_prompt(q, key), format_points([pick])
			rows.append(
				{
					"source": f"robopoint_{kind}",
					"key": key,
					"image": f"robopoint/images/{e['image']}",
					"image_id": f"robopoint:{e['image']}",
					"query": q,
					"prompt": prompt,
					"answer": ans,
					"n_points": len(pts),
				}
			)
			stats[f"{kind}/rows"] += 1
	return rows, dict(stats)
