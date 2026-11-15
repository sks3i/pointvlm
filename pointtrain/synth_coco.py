"""Synthetic referring prompts from full COCO instance masks, aimed at the errors GRPO v1 did not fix.

	python -m pointtrain.synth_coco --out data/rl/synth_v1

The SFT model mostly misses by misreading the phrase: ordinals ("second from left"), relations
("left of", "between"), near/far, and typos. RefCOCO has few hard cases of these (59% of our
"spatial" RefCOCO prompts were solved 8/8). COCO annotates EVERY instance of its 80 categories, so
the answer to "the third cup from the left" can be computed from the masks, and the target mask is
exact, which is what the point-in-mask reward needs.

Every template is only emitted when its answer is unambiguous by a margin:
	* ordinal / extreme / middle: all instances of the category in the image are ranked by mask
	  centroid; every gap between consecutive instances up to the target must exceed
	  ``min_gap_frac`` of the image side, the category must have no crowd annotation (a crowd
	  region hides an unknown number of instances, so counting would be wrong), and every instance
	  must be a fair size (>= ``min_area_frac`` of the image and >= ``min_rel_size`` x the largest),
	  because readers do not count tiny background instances
	* relation (left of / right of / above / below an anchor): the anchor category has exactly one
	  instance; exactly ONE instance of the target category satisfies the strict relation (its box
	  clears the anchor's box and overlaps it on the other axis) and no other instance even has its
	  centre on that side, so a loose human reading picks the same object
	* nearest: the target is the instance closest to a unique anchor by at least ``nearest_ratio``
	* between: the anchor category has exactly two non-overlapping instances, exactly one target
	  instance has its centre between theirs (by a margin) and it does not sit on either anchor
	* near / far (proxy): COCO has no depth; we use mask area, only when the largest instance is
	  ``depth_ratio`` times the next AND sits lowest in the image (perspective cue). Tagged
	  separately because the proxy can be wrong.
	* typo: a fraction of rows get one character edit in the category word (swap, drop, double),
	  words of >= 4 letters only

Every target must cover >= ``min_area_frac`` of the image so a point inside it is a fair ask.

Leakage: train2014 only, minus any image held out by RefCOCO/+/g, any eval-manifest image, and the
pixel-audit flagged list; also checked against Point-Bench by Flickr photo id (COCO and Point-Bench
both come from Flickr). Images in the RefCOCO dev split (same hash as the SFT build) go to dev.jsonl,
never to the pool, so the synthetic dev set shares no image with any training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pointeval.adapters import CANONICAL_PROMPT
from pointeval.geometry import mask_to_rle
from pointeval.run import load_manifest
from pointtrain.build import _write_jsonl
from pointtrain.prompts import _u as dev_u
from pointtrain.sources import coco_id_from_name, refcoco_heldout_image_ids

SALT = "pointvlm-synth-coco-v1"
NAMES = {"dining table": "table", "sports ball": "ball", "potted plant": "potted plant", "cell phone": "phone", "tv": "TV", "hair drier": "hair dryer", "motorcycle": "motorbike"}
PLURALS = {"person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep", "skis": "pairs of skis", "scissors": "pairs of scissors", "bus": "buses", "couch": "couches", "glass": "glasses", "wine glass": "wine glasses", "sandwich": "sandwiches", "toothbrush": "toothbrushes", "bench": "benches", "giraffe": "giraffes"}
ORD = ["", "second", "third", "fourth", "fifth"]
SURFACES = {"dining table", "bed", "couch", "bench", "chair", "toilet", "sink", "oven", "refrigerator"}


def _h(key: str) -> float:
	return int(hashlib.sha256(f"{SALT}:{key}".encode()).hexdigest()[:12], 16) / 16**12


def name(cat: str) -> str:
	return NAMES.get(cat, cat)


def plural(word: str) -> str:
	if word in PLURALS:
		return PLURALS[word]
	last = word.split()[-1]
	if re.search(r"(s|x|ch|sh)$", last):
		return word + "es"
	return word + "s"


def typo(word: str, key: str) -> str | None:
	"""One character edit inside the longest token of ``word``; None if it is too short."""
	toks = word.split()
	i = max(range(len(toks)), key=lambda k: len(toks[k]))
	w = toks[i]
	if len(w) < 4:
		return None
	j = 1 + int(_h("pos:" + key) * (len(w) - 2))  # never the first letter
	kind = int(_h("kind:" + key) * 3)
	if kind == 0 and j < len(w) - 1:
		w = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
	elif kind == 1:
		w = w[:j] + w[j + 1 :]
	else:
		w = w[:j] + w[j] + w[j:]
	toks[i] = w
	return " ".join(toks)


class Inst:
	__slots__ = ("ann", "cat", "x0", "y0", "x1", "y1", "cx", "cy", "area")

	def __init__(self, ann, cat, W, H):
		x, y, w, h = ann["bbox"]
		self.ann, self.cat = ann, cat
		self.x0, self.y0, self.x1, self.y1 = x / W, y / H, (x + w) / W, (y + h) / H
		self.area = ann["area"] / (W * H)
		# Box centre as a cheap stand-in for the mask centroid (masks are decoded only for targets).
		self.cx, self.cy = (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2


def templates(insts: list[Inst], crowd_cats: set[str], a) -> list[tuple[str, str, Inst]]:
	"""(type, query, target) for one image."""
	out = []
	by = defaultdict(list)
	for it in insts:
		by[it.cat].append(it)
	unique = {c: v[0] for c, v in by.items() if len(v) == 1 and v[0].area >= a.min_anchor_frac and c not in crowd_cats}
	big = lambda it: it.area >= a.min_area_frac

	for cat, v in by.items():
		if cat in crowd_cats or len(v) < 2 or len(v) > a.max_instances:
			continue
		nm, n = name(cat), len(v)
		# People count the instances they notice: a crowd of tiny background cars changes "the fourth
		# car from the left" for COCO but not for a reader. Counting templates need every instance to be
		# a fair size, both absolutely and relative to the largest one.
		countable = all(it.area >= a.min_area_frac and it.area >= a.min_rel_size * max(x.area for x in v) for it in v)
		for axis in (("x", "y") if countable else ()):
			key = (lambda it: it.cx) if axis == "x" else (lambda it: it.cy)
			s = sorted(v, key=key)
			gaps = [key(s[i + 1]) - key(s[i]) for i in range(n - 1)]
			for k in range(n):
				# Every gap between the target and the end it is counted from must be clear.
				clear_lo = all(g >= a.min_gap_frac for g in gaps[:k]) and (k == n - 1 or gaps[k] >= a.min_gap_frac)
				clear_hi = all(g >= a.min_gap_frac for g in gaps[k:]) and (k == 0 or gaps[k - 1] >= a.min_gap_frac)
				t = s[k]
				if not big(t):
					continue
				if axis == "x":
					if k == 0 and clear_lo:
						out.append(("extreme", f"the leftmost {nm}", t))
					if k == n - 1 and clear_hi:
						out.append(("extreme", f"the rightmost {nm}", t))
					if 1 <= k <= 4 and k < n - 1 and clear_lo and n >= 3:
						out.append(("ordinal", f"the {ORD[k]} {nm} from the left", t))
					r = n - 1 - k
					if 1 <= r <= 4 and r < n - 1 and clear_hi and n >= 3:
						out.append(("ordinal", f"the {ORD[r]} {nm} from the right", t))
					if n == 3 and k == 1 and clear_lo and clear_hi:
						out.append(("middle", f"the {nm} in the middle", t))
				else:
					if k == 0 and clear_lo:
						out.append(("extreme", f"the {nm} at the top", t))
					if k == n - 1 and clear_hi:
						out.append(("extreme", f"the {nm} at the bottom", t))

		# Near / far proxy: much larger AND lowest in the frame.
		s = sorted(v, key=lambda it: -it.area)
		if s[0].area >= a.depth_ratio * s[1].area and s[0].y1 >= max(it.y1 for it in v) - 1e-6 and big(s[0]):
			out.append(("depth", f"the {nm} closest to the camera", s[0]))
		if s[-2].area >= a.depth_ratio * s[-1].area and s[-1].y1 <= min(it.y1 for it in v) + 1e-6 and big(s[-1]):
			out.append(("depth", f"the {nm} farthest away", s[-1]))

		# Relations to a unique anchor of another category.
		for acat, anc in unique.items():
			if acat == cat:
				continue
			an = name(acat)
			m = a.rel_margin
			rels = {
				"to the left of": lambda it: it.x1 <= anc.x0 + m and _voverlap(it, anc),
				"to the right of": lambda it: it.x0 >= anc.x1 - m and _voverlap(it, anc),
				"above": lambda it: it.y1 <= anc.y0 + m and _hoverlap(it, anc),
				"below": lambda it: it.y0 >= anc.y1 - m and _hoverlap(it, anc),
			}
			# People read "below the frisbee" loosely (anything lower), so the target must pass the
			# strict test and every other instance must fail the LOOSE one (centre on that side).
			loose = {
				"to the left of": lambda it: it.cx < anc.cx,
				"to the right of": lambda it: it.cx > anc.cx,
				"above": lambda it: it.cy < anc.cy,
				"below": lambda it: it.cy > anc.cy,
			}
			for phrase, ok in rels.items():
				# Image-space "above" with a surface ("the table above the pizza" = a table further back)
				# reads as a 3-D claim and is usually false; keep vertical relations to non-surfaces.
				if phrase in ("above", "below") and {cat, acat} & SURFACES:
					continue
				yes = [it for it in v if ok(it)]
				near = [it for it in v if loose[phrase](it)]
				if len(yes) == 1 and len(near) == 1 and big(yes[0]):
					out.append(("relation", f"the {nm} {phrase} the {an}", yes[0]))
			d = sorted(v, key=lambda it: (it.cx - anc.cx) ** 2 + (it.cy - anc.cy) ** 2)
			d0 = ((d[0].cx - anc.cx) ** 2 + (d[0].cy - anc.cy) ** 2) ** 0.5
			d1 = ((d[1].cx - anc.cx) ** 2 + (d[1].cy - anc.cy) ** 2) ** 0.5
			if d1 >= a.nearest_ratio * max(d0, 1e-3) and big(d[0]):
				out.append(("nearest", f"the {nm} closest to the {an}", d[0]))

		# Between the two instances of another category.
		for acat, pair in by.items():
			if acat == cat or acat in crowd_cats or len(pair) != 2:
				continue
			l, r = sorted(pair, key=lambda it: it.cx)
			if r.cx - l.cx < 3 * a.min_gap_frac:
				continue
			y0, y1 = min(l.y0, r.y0), max(l.y1, r.y1)
			if _inter(l, r) > 0 or min(l.area, r.area) < a.min_anchor_frac:
				continue
			mid = [it for it in v if l.cx + a.min_gap_frac < it.cx < r.cx - a.min_gap_frac and y0 < it.cy < y1]
			loose = [it for it in v if l.cx < it.cx < r.cx]
			# "Between" means in the gap: the target must not sit on either anchor (a person holding
			# the two toothbrushes is not between them).
			if len(mid) == 1 and len(loose) == 1 and big(mid[0]) and max(_inter(mid[0], l), _inter(mid[0], r)) < 0.1 * _box(mid[0]):
				out.append(("between", f"the {nm} between the two {plural(name(acat))}", mid[0]))
	return out


def _box(a) -> float:
	return (a.x1 - a.x0) * (a.y1 - a.y0)


def _inter(a, b) -> float:
	return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))


def _voverlap(a, b) -> bool:
	return min(a.y1, b.y1) - max(a.y0, b.y0) > 0


def _hoverlap(a, b) -> bool:
	return min(a.x1, b.x1) - max(a.x0, b.x0) > 0


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--ann", type=Path, default=Path.home() / "data/coco/annotations/annotations/instances_train2014.json")
	ap.add_argument("--exclude-images", type=Path, default=Path("data/sft/flagged_v1.txt"))
	ap.add_argument("--per-type", type=int, default=3000, help="pool rows per template type (typos counted within)")
	ap.add_argument("--dev-per-type", type=int, default=60)
	ap.add_argument("--typo-frac", type=float, default=0.15)
	ap.add_argument("--min-area-frac", type=float, default=0.004)
	ap.add_argument("--min-anchor-frac", type=float, default=0.01)
	ap.add_argument("--min-gap-frac", type=float, default=0.04)
	ap.add_argument("--max-instances", type=int, default=8)
	ap.add_argument("--min-rel-size", type=float, default=0.15, help="counting templates: every instance >= this x the largest")
	ap.add_argument("--rel-margin", type=float, default=0.0)
	ap.add_argument("--nearest-ratio", type=float, default=1.6)
	ap.add_argument("--depth-ratio", type=float, default=2.5)
	ap.add_argument("--max-per-image", type=int, default=3)
	a = ap.parse_args()
	a.out.mkdir(parents=True, exist_ok=True)

	from pycocotools import mask as mask_utils

	heldout = refcoco_heldout_image_ids()
	items, _ = load_manifest("refcoco")
	exclude = set().union(*heldout.values()) | {coco_id_from_name(it["file_name"]) for it in items}
	flagged = set(a.exclude_images.read_text().split()) if a.exclude_images.exists() else set()
	pb_flickr = {m.group(1) for r in load_manifest("pointbench")[0] if (m := re.match(r"(\d+)_", r["file_name"].split("/")[-1]))}

	d = json.loads(a.ann.read_text())
	cats = {c["id"]: c["name"] for c in d["categories"]}
	anns = defaultdict(list)
	for an in d["annotations"]:
		anns[an["image_id"]].append(an)
	stats = Counter()
	cand = {"pool": defaultdict(list), "dev": defaultdict(list)}
	for im in d["images"]:
		iid, W, H = im["id"], im["width"], im["height"]
		path = f"coco/train2014/{im['file_name']}"
		m = re.search(r"/(\d+)_[0-9a-f]+", im.get("flickr_url") or "")
		if iid in exclude or path in flagged or (m and m.group(1) in pb_flickr):
			stats["images_excluded"] += 1
			continue
		crowd = {cats[x["category_id"]] for x in anns[iid] if x["iscrowd"]}
		insts = [Inst(x, cats[x["category_id"]], W, H) for x in anns[iid] if not x["iscrowd"]]
		found = templates(insts, crowd, a)
		if not found:
			continue
		split = "dev" if dev_u(f"dev:coco:{iid}") < 0.015 else "pool"
		# A few per image, chosen by hash, so no single busy image dominates.
		found = sorted(found, key=lambda f: _h(f"{iid}:{f[1]}"))[: a.max_per_image]
		for typ, q, t in found:
			cand[split][typ].append((im, q, t))
			stats[f"{split}/{typ}"] += 1

	rows = {"pool": [], "dev": []}
	for split, cap in (("pool", a.per_type), ("dev", a.dev_per_type)):
		for typ, lst in cand[split].items():
			lst = sorted(lst, key=lambda x: _h(f"take:{x[0]['id']}:{x[1]}"))[:cap]
			for im, q, t in lst:
				W, H = im["width"], im["height"]
				seg = t.ann["segmentation"]
				rle = mask_utils.merge(mask_utils.frPyObjects(seg, H, W)) if isinstance(seg, list) else seg
				mask = mask_utils.decode(rle).astype(bool)
				if not mask.any():
					continue
				key = f"synth:{im['id']}:{t.ann['id']}:{q}"
				group = f"syn_{typ}"
				if split == "pool" and _h("typo:" + key) < a.typo_frac:
					nm = name(t.cat)
					tw = typo(nm, key)
					if tw and nm in q:
						q = q.replace(nm, tw, 1)
						group = "syn_typo"
				rows[split].append(
					{
						"key": key,
						"source": "synth_coco",
						"group": group,
						"image": f"coco/train2014/{im['file_name']}",
						"image_id": f"coco:{im['id']}",
						"query": q,
						"prompt": CANONICAL_PROMPT.format(query=q),
						"target": {"type": "mask", "mask_rle": mask_to_rle(mask)},
					}
				)
	pool, dev = rows["pool"], rows["dev"]
	pool.sort(key=lambda r: _h("order:" + r["key"]))
	pc = {r["image_id"] for r in pool}
	dc = {r["image_id"] for r in dev}
	assert not pc & dc, "pool and dev share images"
	assert not {int(x.split(":")[1]) for x in pc | dc} & exclude, "held-out / eval COCO image in synthetic data"
	_write_jsonl(a.out / "pool.jsonl", pool)
	_write_jsonl(a.out / "dev.jsonl", dev)
	meta = {
		"salt": SALT,
		"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
		"n_pool": len(pool),
		"n_dev": len(dev),
		"pool_by_group": dict(Counter(r["group"] for r in pool)),
		"dev_by_group": dict(Counter(r["group"] for r in dev)),
		"candidates": dict(stats),
		"leakage": {"excluded_ids": len(exclude), "flagged": len(flagged), "pointbench_flickr_ids": len(pb_flickr), "checks_passed": True},
	}
	(a.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
	print(json.dumps({k: meta[k] for k in ("n_pool", "n_dev", "pool_by_group", "dev_by_group", "candidates")}, indent=2))
	for r in pool[:25]:
		print(r["group"], "|", r["query"])


if __name__ == "__main__":
	main()
