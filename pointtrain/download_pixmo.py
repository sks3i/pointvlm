"""Fetch a deterministic subset of PixMo-Points images (the dataset ships URLs, not pixels).

	python -m pointtrain.download_pixmo --n-images 40000

Images are ranked by sha256(salt + image_sha256) and fetched in that order until n succeed, so the
subset is reproducible and growing --n-images only adds images. Every file is checked against the
dataset's image_sha256 (dead links and changed images are dropped). Results append to
<data_root>/pixmo/index.jsonl, so the command can be re-run to resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pointtrain.sources import PIXMO_REPO, PIXMO_REV, _parquets

SALT = "pointvlm-pixmo-subset-v1"
_EXT = {b"\xff\xd8": ".jpg", b"\x89P": ".png", b"GI": ".gif", b"RI": ".webp"}


def _fetch(url: str, sha: str, out_dir: Path, timeout: float) -> dict:
	import requests

	try:
		r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (pointvlm research)"})
		r.raise_for_status()
		data = r.content
	except Exception as e:  # dead link, timeout, TLS error...
		return {"image_sha256": sha, "ok": False, "err": type(e).__name__}
	if hashlib.sha256(data).hexdigest() != sha:
		return {"image_sha256": sha, "ok": False, "err": "sha_mismatch"}
	ext = _EXT.get(data[:2], ".img")
	path = out_dir / f"{sha}{ext}"
	path.write_bytes(data)
	return {"image_sha256": sha, "ok": True, "path": f"pixmo/images/{path.name}", "bytes": len(data)}


def main() -> None:
	import pandas as pd

	ap = argparse.ArgumentParser()
	ap.add_argument("--n-images", type=int, default=40000)
	ap.add_argument("--data-root", type=Path, default=Path.home() / "data")
	ap.add_argument("--workers", type=int, default=48)
	ap.add_argument("--timeout", type=float, default=15.0)
	a = ap.parse_args()

	out_dir = a.data_root / "pixmo" / "images"
	out_dir.mkdir(parents=True, exist_ok=True)
	index = a.data_root / "pixmo" / "index.jsonl"
	done = {}
	if index.exists():
		for line in index.read_text().splitlines():
			e = json.loads(line)
			done[e["image_sha256"]] = e["ok"]

	df = pd.concat([pd.read_parquet(f, columns=["image_url", "image_sha256"]) for f in _parquets(PIXMO_REPO, PIXMO_REV)])
	urls = df.drop_duplicates("image_sha256").set_index("image_sha256")["image_url"].to_dict()
	ranked = sorted(urls, key=lambda s: hashlib.sha256(f"{SALT}:{s}".encode()).hexdigest())
	n_ok = sum(done.values())
	todo = [s for s in ranked if s not in done]
	print(f"{len(urls)} images in PixMo-Points, {n_ok} already fetched, target {a.n_images}", flush=True)

	with index.open("a") as log, ThreadPoolExecutor(a.workers) as pool:
		i = 0
		while n_ok < a.n_images and i < len(todo):
			# Over-submit by the observed failure rate (~30-40% of web links are dead).
			batch = todo[i : i + min(4 * (a.n_images - n_ok), 4000)]
			i += len(batch)
			futs = [pool.submit(_fetch, urls[s], s, out_dir, a.timeout) for s in batch]
			for f in as_completed(futs):
				e = f.result()
				log.write(json.dumps(e) + "\n")
				n_ok += e["ok"]
			log.flush()
			print(f"tried {i}, ok {n_ok}", flush=True)


if __name__ == "__main__":
	main()
