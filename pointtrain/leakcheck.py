"""Pixel-level leak audit: are any training images (near-)duplicates of eval images?

	python -m pointtrain.leakcheck data/sft/v1

ID-based filtering (pointtrain.build) only catches the same COCO image id. The same photo can also
appear under another id (COCO has near-duplicates), in another dataset (a PixMo web image that is
also a COCO or Where2Place photo), or re-encoded / resized. So we compare 64-bit difference hashes
(dHash) of every training image against every eval image (RefCOCO 2k, Where2Place, Point-Bench):

	dHash: grayscale, resize to 9x8, bit = (pixel > right neighbour). Robust to resizing, JPEG
	re-encoding and small colour shifts; Hamming distance <= 6 of 64 bits is a near-duplicate.

Writes <dir>/leak_report.json and <dir>/flagged_images.txt (train image paths to exclude; pass it to
pointtrain.build --exclude-images and rebuild).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

DATA_ROOT = Path.home() / "data"
THRESH = 6


def dhash(img) -> int:
	from PIL import Image

	g = img.convert("L").resize((9, 8), Image.BILINEAR)
	a = np.asarray(g, dtype=np.int16)
	bits = (a[:, 1:] > a[:, :-1]).flatten()
	return int(np.packbits(bits).view(">u8")[0])


def _hash_path(rel: str):
	from PIL import Image

	try:
		with Image.open(DATA_ROOT / rel) as im:
			im.load()  # full decode: catches truncated files that open lazily
			return rel, dhash(im)
	except Exception:
		return rel, None


def eval_hashes() -> dict[str, int]:
	from pointeval.run import ImageSource, load_manifest

	out = {}
	for bench in ("refcoco", "where2place", "pointbench"):
		try:
			items, meta = load_manifest(bench)
		except FileNotFoundError:
			continue
		src = ImageSource(bench, meta["revision"])
		for it in items:
			out[f"{bench}:{it['id']}"] = dhash(src.get(it))
	return out


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("sft_dir", type=Path)
	ap.add_argument("--workers", type=int, default=16)
	a = ap.parse_args()

	ev = eval_hashes()
	ev_keys = list(ev)
	ev_arr = np.array([ev[k] for k in ev_keys], dtype=np.uint64)
	print(f"{len(ev)} eval images hashed")

	paths = set()
	for name in ("train.jsonl", "dev.jsonl"):
		for line in (a.sft_dir / name).read_text().splitlines():
			paths.add(json.loads(line)["image"])
	paths = sorted(paths)
	# spawn, not fork: eval_hashes() above started datasets/pyarrow threads, and a forked child can
	# inherit one of their locks held and deadlock.
	with mp.get_context("spawn").Pool(a.workers) as pool:
		hashed = pool.map(_hash_path, paths, chunksize=256)
	bad = [p for p, h in hashed if h is None]
	ok = [(p, h) for p, h in hashed if h is not None]
	tr_arr = np.array([h for _, h in ok], dtype=np.uint64)
	print(f"{len(ok)} train/dev images hashed, {len(bad)} unreadable")

	flagged = []
	for i in range(0, len(ev_arr)):
		d = np.bitwise_count(tr_arr ^ ev_arr[i])
		for j in np.nonzero(d <= THRESH)[0]:
			flagged.append({"train_image": ok[j][0], "eval_item": ev_keys[i], "hamming": int(d[j])})
	report = {
		"threshold_bits": THRESH,
		"n_eval_images": len(ev),
		"n_train_images": len(ok),
		"unreadable": bad[:50],
		"n_unreadable": len(bad),
		"n_flagged_pairs": len(flagged),
		"n_flagged_train_images": len({f["train_image"] for f in flagged}),
		"flagged": sorted(flagged, key=lambda f: f["hamming"])[:500],
	}
	(a.sft_dir / "leak_report.json").write_text(json.dumps(report, indent=2) + "\n")
	# Unreadable / truncated files are excluded too: one bad PNG kills a DataLoader worker mid-run.
	(a.sft_dir / "flagged_images.txt").write_text("".join(sorted({f["train_image"] + "\n" for f in flagged} | {b + "\n" for b in bad})))
	print(json.dumps({k: v for k, v in report.items() if k != "flagged"}, indent=2))


if __name__ == "__main__":
	main()
