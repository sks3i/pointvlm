"""Grids of hits and misses for one eval run: mask (green), model point (red), baseline point (white ring).

	python scripts/viz_predictions.py --run sft-qwen3vl-2b-v1-e2 --baseline qwen3vl-2b-canonical --bench refcoco
"""

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pointeval.geometry import rle_to_mask
from pointeval.run import ImageSource, load_manifest

ROOT = Path(__file__).resolve().parent.parent


def load_preds(bench, run):
	return {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "results" / bench / run / "predictions.jsonl").read_text().splitlines()}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--run", required=True)
	ap.add_argument("--baseline", default=None)
	ap.add_argument("--bench", default="refcoco")
	ap.add_argument("--n", type=int, default=5)
	ap.add_argument("--seed", default="viz-v1")
	a = ap.parse_args()

	items, meta = load_manifest(a.bench)
	by_id = {it["id"]: it for it in items}
	preds = load_preds(a.bench, a.run)
	base = load_preds(a.bench, a.baseline) if a.baseline else {}
	src = ImageSource(a.bench, meta["revision"])
	rank = lambda i: hashlib.sha256(f"{a.seed}:{i}".encode()).hexdigest()
	out_dir = ROOT / "results" / "figures"
	out_dir.mkdir(parents=True, exist_ok=True)

	for kind, want in (("correct", True), ("wrong", False)):
		ids = sorted((i for i, p in preds.items() if p["hit"] == want), key=rank)[: a.n]
		fig, axes = plt.subplots(1, a.n, figsize=(4.2 * a.n, 4.8), dpi=150)
		for ax, i in zip(axes, ids):
			it, p = by_id[i], preds[i]
			img = np.asarray(src.get(it))
			mask = rle_to_mask(it["mask_rle"])
			ax.imshow(img)
			overlay = np.zeros((*mask.shape, 4))
			overlay[mask] = (0.1, 0.9, 0.2, 0.45)
			ax.imshow(overlay)
			h, w = mask.shape
			b = base.get(i)
			if b and b["points"]:
				bx, by = b["points"][0]
				ax.scatter([bx / 1000 * w], [by / 1000 * h], s=160, facecolors="none", edgecolors="white", linewidths=2.2)
			if p["points"]:
				x, y = p["points"][0]
				ax.scatter([x / 1000 * w], [y / 1000 * h], s=90, c="red", edgecolors="black", linewidths=1.2, marker="o")
			tag = "" if not b else ("  base: hit" if b["hit"] else "  base: miss")
			ax.set_title("\n".join(textwrap.wrap(f'"{it["query"]}"', 32)) + f"\n{i}{tag}", fontsize=9)
			ax.axis("off")
		fig.suptitle(f"{a.run} on {a.bench}: {kind}  (green = GT mask, red = SFT point, white ring = base model point)", fontsize=11)
		fig.tight_layout()
		path = out_dir / f"{a.bench}-{a.run}-{kind}.png"
		fig.savefig(path, bbox_inches="tight")
		print(path)


if __name__ == "__main__":
	main()
