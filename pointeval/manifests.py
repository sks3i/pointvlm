"""Build the FROZEN eval manifests (run once in week 1, then commit and never rebuild).

A manifest is a JSONL file that holds everything needed to score an item except
the image pixels: the query text, the ground-truth mask as RLE, the image size,
and a sha256 of the image bytes. Scoring therefore never depends on dataset
library behaviour, row order, or a re-uploaded dataset. Images are re-fetched
from the pinned dataset revision and checked against the stored hash.

Sampling uses hash ranking, not an RNG: rank every candidate by
sha256(SALT + stable_key) and take the first N. This is independent of numpy /
python RNG versions and of the order rows come back in, and adding or removing
an unrelated row never reshuffles the rest. SALT plays the role of the seed.

Usage (on the GPU box, needs network to huggingface.co):
	python -m pointeval.manifests refcoco
	python -m pointeval.manifests where2place
	python -m pointeval.manifests pointbench   # optional
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

from pointeval.geometry import mask_to_rle, polygons_to_mask

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = ROOT / "manifests"

REFCOCO_REPO = "lmms-lab/RefCOCO"
REFCOCO_SPLIT = "val"
REFCOCO_N = 2000
REFCOCO_SALT = "pointvlm-refcoco-val-v1"
# lmms-lab "val" mixes UNC val with train/testA/testB refs, and its `segmentation` keeps only the
# first polygon. The raw UNC annotations (question_id == COCO ann_id) give the true split and the
# full multi-polygon segmentation.
UNC_REPO = "jxu124/refcoco"
UNC_SPLIT = "validation"  # rows here all have split == "val"

WHERE2PLACE_REPO = "wentao-yuan/where2place"
POINTBENCH_REPO = "PointArena/pointarena"


def hash_rank(salt: str, key: str) -> str:
	return hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()


def sha256_bytes(b: bytes) -> str:
	return hashlib.sha256(b).hexdigest()


def image_bytes(pil_image) -> bytes:
	"""Canonical bytes for hashing a decoded image: raw RGB pixels + size."""
	im = pil_image.convert("RGB")
	return f"{im.width}x{im.height}:".encode() + im.tobytes()


def resolve_revision(repo: str) -> str:
	from huggingface_hub import HfApi

	return HfApi().dataset_info(repo).sha


def _write(name: str, rows: list[dict], meta: dict) -> None:
	MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
	path = MANIFEST_DIR / f"{name}.jsonl"
	if path.exists():
		sys.exit(f"{path} exists. Manifests are frozen; delete it by hand only if you really mean to.")
	with path.open("w") as f:
		for r in rows:
			f.write(json.dumps(r, sort_keys=True) + "\n")
	meta["manifest_sha256"] = sha256_bytes(path.read_bytes())
	meta["n"] = len(rows)
	(MANIFEST_DIR / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
	print(f"wrote {path} ({len(rows)} items), sha256={meta['manifest_sha256']}")


# --- RefCOCO val 2k ------------------------------------------------------------------


def _expressions(row) -> list[str]:
	ans = row.get("answer")
	if isinstance(ans, list):
		exprs = [a.strip() for a in ans if isinstance(a, str) and a.strip()]
	elif isinstance(ans, str) and ans.strip():
		exprs = [ans.strip()]
	else:
		exprs = []
	return exprs


def unc_segmentation(raw_anns: str) -> list:
	"""Full COCO segmentation (all polygons) from a jxu124/refcoco `raw_anns` JSON string."""
	return json.loads(raw_anns)["segmentation"]


def load_unc_val(revision: str) -> dict[int, list]:
	from datasets import load_dataset

	unc = load_dataset(UNC_REPO, split=UNC_SPLIT, revision=revision).select_columns(["ann_id", "split", "raw_anns"])
	assert set(unc["split"]) == {"val"}
	return {int(r["ann_id"]): unc_segmentation(r["raw_anns"]) for r in unc}


def build_refcoco(revision: str | None, unc_revision: str | None = None) -> None:
	from datasets import load_dataset

	revision = revision or resolve_revision(REFCOCO_REPO)
	unc_revision = unc_revision or resolve_revision(UNC_REPO)
	unc_val = load_unc_val(unc_revision)
	ds = load_dataset(REFCOCO_REPO, split=REFCOCO_SPLIT, revision=revision)
	cols = [c for c in ds.column_names if c != "image"]
	meta_ds = ds.select_columns(cols)
	n_not_unc_val = 0

	# One candidate per OBJECT, not per sentence: several sentences describe the same
	# object, and sampling them separately would make the 2k items correlated.
	objects: dict[str, list[int]] = defaultdict(list)
	for i, row in enumerate(meta_ds):
		if row.get("iscrowd"):
			continue
		if int(row["question_id"]) not in unc_val:
			n_not_unc_val += 1
			continue
		key = f"{row['file_name']}|{','.join(f'{v:.2f}' for v in row['bbox'])}"
		objects[key].append(i)

	ranked = sorted(objects, key=lambda k: hash_rank(REFCOCO_SALT, k))
	rows, skipped = [], defaultdict(int)
	for key in ranked:
		if len(rows) == REFCOCO_N:
			break
		idxs = sorted(objects[key])
		exprs = sorted({e for i in idxs for e in _expressions(meta_ds[i])})
		if not exprs:
			skipped["no_expression"] += 1
			continue
		expr = exprs[int(hash_rank(REFCOCO_SALT + ":expr", key), 16) % len(exprs)]
		row_idx = idxs[0]
		row = ds[row_idx]
		img = row["image"]
		w, h = img.size
		try:
			mask = polygons_to_mask(unc_val[int(row["question_id"])], h, w)
		except ValueError:
			skipped["bad_polygon"] += 1
			continue
		area = int(mask.sum())
		if area == 0:
			skipped["empty_mask"] += 1
			continue
		rows.append(
			{
				"id": f"refcoco-val-{len(rows):04d}",
				"object_key": key,
				"row_index": row_idx,
				"question_id": str(row.get("question_id")),
				"file_name": row["file_name"],
				"width": w,
				"height": h,
				"query": expr,
				"n_candidate_expressions": len(exprs),
				"bbox_xywh": [float(v) for v in row["bbox"]],
				"mask_rle": mask_to_rle(mask),
				"mask_area_frac": area / (w * h),
				"image_sha256": sha256_bytes(image_bytes(img)),
			}
		)
	if len(rows) < REFCOCO_N:
		sys.exit(f"only {len(rows)} usable objects")
	_write(
		"refcoco_val_2k",
		rows,
		{
			"repo": REFCOCO_REPO,
			"split": REFCOCO_SPLIT,
			"revision": revision,
			"salt": REFCOCO_SALT,
			"unc_repo": UNC_REPO,
			"unc_split": UNC_SPLIT,
			"unc_revision": unc_revision,
			"n_objects_total": len(objects),
			"n_rows_total": len(ds),
			"n_rows_not_unc_val": n_not_unc_val,
			"skipped": dict(skipped),
		},
	)


# --- Where2Place ---------------------------------------------------------------------

# RoboPoint's questions end with its own answer-format instructions (normalized 0-1
# tuples). We strip that suffix and apply our own frozen prompt templates instead,
# keeping the raw text in the manifest for traceability.
_W2P_SUFFIX_MARKERS = ("Your answer should", "Your answer must")


def _strip_format_suffix(text: str) -> str:
	cut = min((text.find(m) for m in _W2P_SUFFIX_MARKERS if m in text), default=-1)
	return (text[:cut] if cut > 0 else text).strip()


def _find_mask(mask_dir: Path, stem: str) -> Path:
	cands = sorted(p for p in mask_dir.iterdir() if p.stem == stem)
	# Some stems ship both a lossless .png and a .jpg copy (same binary mask); prefer the .png.
	if len(cands) > 1:
		cands = [p for p in cands if p.suffix == ".png"] or cands
	if len(cands) != 1:
		raise FileNotFoundError(f"mask for {stem}: {cands}")
	return cands[0]


def build_where2place(revision: str | None) -> None:
	from huggingface_hub import snapshot_download
	from PIL import Image

	revision = revision or resolve_revision(WHERE2PLACE_REPO)
	root = Path(snapshot_download(WHERE2PLACE_REPO, repo_type="dataset", revision=revision))
	qs = [json.loads(l) for l in (root / "point_questions.jsonl").read_text().splitlines() if l.strip()]
	print("where2place question keys:", sorted(qs[0].keys()))
	rows = []
	for q in qs:
		img_field = q.get("image") or q.get("image_path") or q.get("img")
		text = q.get("text") or q.get("question") or q.get("prompt")
		if img_field is None or text is None:
			sys.exit(f"unexpected where2place schema: {sorted(q.keys())}")
		img_path = root / "images" / Path(img_field).name
		img = Image.open(img_path)
		w, h = img.size
		m = np.array(Image.open(_find_mask(root / "masks", img_path.stem)).convert("L")) > 127
		if m.shape != (h, w):
			sys.exit(f"mask/image size mismatch for {img_path.name}")
		rows.append(
			{
				"id": f"w2p-{len(rows):03d}",
				"question_id": str(q.get("question_id", q.get("id", len(rows)))),
				"file_name": img_path.name,
				"width": w,
				"height": h,
				"query": _strip_format_suffix(text),
				"raw_text": text,
				"category": q.get("category"),
				"mask_rle": mask_to_rle(m),
				"mask_area_frac": float(m.mean()),
				"image_sha256": sha256_bytes(image_bytes(img)),
			}
		)
	_write("where2place", rows, {"repo": WHERE2PLACE_REPO, "revision": revision})


# --- Point-Bench (optional) ---------------------------------------------------------


def pointbench_root(revision: str) -> Path:
	"""Download the pinned snapshot and unpack its image/mask zips (the repo ships zips, not files)."""
	import zipfile

	from huggingface_hub import snapshot_download

	snap = Path(snapshot_download(POINTBENCH_REPO, repo_type="dataset", revision=revision))
	root = Path.home() / ".cache" / "pointvlm" / "pointbench" / revision
	for z in sorted(snap.glob("*.zip")):
		done = root / f".{z.stem}.done"
		if not done.exists():
			with zipfile.ZipFile(z) as zf:
				for info in zf.infolist():
					# macOS zips: UTF-8 (NFD) names without the UTF-8 flag, so zipfile decodes them
					# as cp437. Recover the real name and NFC-normalize it to match data.json.
					if not info.flag_bits & 0x800:
						info.filename = info.filename.encode("cp437").decode("utf-8")
					info.filename = unicodedata.normalize("NFC", info.filename)
					if not info.filename.startswith("__MACOSX"):
						zf.extract(info, root)
			done.touch()
	(root / "data.json").write_bytes((snap / "data.json").read_bytes())
	return root


def build_pointbench(revision: str | None) -> None:
	from PIL import Image

	revision = revision or resolve_revision(POINTBENCH_REPO)
	base = pointbench_root(revision)
	data_json = base / "data.json"
	entries = json.loads(data_json.read_text())
	rows, skipped = [], []
	for e in entries:
		# data.json mixes NFC and NFD spellings; extracted files are NFC.
		img_path = next(base.rglob(unicodedata.normalize("NFC", e["image_filename"])))
		mask_path = next(base.rglob(unicodedata.normalize("NFC", e["mask_filename"])))
		img = Image.open(img_path)
		w, h = img.size
		mask_img = np.array(Image.open(mask_path).convert("L"))
		if mask_img.shape != (h, w):
			# A few upstream entries pair an image with a mask of another size; drop them, recorded in meta.
			skipped.append({"image_filename": e["image_filename"], "image_wh": [w, h], "mask_wh": list(mask_img.shape[::-1])})
			continue
		# Counting items can have several targets in one mask PNG; we keep connected
		# components at scoring time (see run.py), here we just store the union.
		m = mask_img > 127
		rows.append(
			{
				"id": f"pb-{len(rows):04d}",
				"file_name": str(img_path.relative_to(base)),
				"width": w,
				"height": h,
				"query": e["user_input"],
				"category": e["category"],
				"count": e.get("count"),
				"mask_rle": mask_to_rle(m),
				"mask_area_frac": float(m.mean()),
				"image_sha256": sha256_bytes(image_bytes(img)),
			}
		)
	_write("pointbench", rows, {"repo": POINTBENCH_REPO, "revision": revision, "n_entries_total": len(entries), "skipped_size_mismatch": skipped})


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("bench", choices=["refcoco", "where2place", "pointbench"])
	ap.add_argument("--revision", default=None, help="dataset commit sha; default resolves current main and records it")
	ap.add_argument("--unc-revision", default=None, help=f"refcoco only: {UNC_REPO} commit sha")
	a = ap.parse_args()
	if a.bench == "refcoco":
		build_refcoco(a.revision, a.unc_revision)
	else:
		{"where2place": build_where2place, "pointbench": build_pointbench}[a.bench](a.revision)


if __name__ == "__main__":
	main()
