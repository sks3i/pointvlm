"""Blog figure: one stored GRPO v4 prediction per frozen benchmark (mask green, point red).

	python scripts/fig_eval_examples.py --items refcoco:refcoco-val-0001 where2place:w2p-000 pointbench:pb-0161 \
		--out results/figures/blog_eval_examples.png

Reads results/<bench>/<run>/predictions.jsonl only; no model is run.
"""

import argparse
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
NAMES = {"refcoco": "RefCOCO val", "where2place": "Where2Place", "pointbench": "Point-Bench"}


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--items", nargs="+", required=True, help="bench:item_id")
	ap.add_argument("--run", default="grpo-v4")
	ap.add_argument("--out", required=True)
	ap.add_argument("--height-in", type=float, default=5.2)
	a = ap.parse_args()

	panels = []
	for spec in a.items:
		bench, iid = spec.split(":")
		items, meta = load_manifest(bench)
		item = next(it for it in items if it["id"] == iid)
		pred = next(json.loads(l) for l in (ROOT / "results" / bench / a.run / "predictions.jsonl").read_text().splitlines() if json.loads(l)["id"] == iid)
		img = np.asarray(ImageSource(bench, meta["revision"]).get(item))
		panels.append((bench, item, pred, img, rle_to_mask(item["mask_rle"])))

	ratios = [p[3].shape[1] / p[3].shape[0] for p in panels]
	fig_w = 24.0
	fig, axes = plt.subplots(1, len(panels), figsize=(fig_w, a.height_in + 2.2), gridspec_kw={"width_ratios": ratios, "wspace": 0.06})
	for ax, (bench, item, pred, img, mask) in zip(np.atleast_1d(axes), panels):
		h, w = mask.shape
		ax.imshow(img)
		overlay = np.zeros((h, w, 4))
		overlay[mask] = [0.1, 0.85, 0.2, 0.45]
		ax.imshow(overlay)
		ax.contour(mask, levels=[0.5], colors=["#1aa832"], linewidths=1.5)
		for x, y in pred["points"]:
			ax.scatter([x / 1000 * w], [y / 1000 * h], s=260, c="#e8202a", edgecolors="white", linewidths=2.5, zorder=5)
		cat = f" · {item['category']}" if bench == "pointbench" else ""
		q = "\n".join(textwrap.wrap(f"“{item['query']}”", 52))
		ax.set_title(f"{NAMES[bench]}{cat}\n{q}", fontsize=15, loc="left", pad=10, fontweight="normal")
		raw = "\n".join(textwrap.wrap(pred["raw"], 60))
		ax.text(0.0, -0.03, f"GRPO v4 output: {raw}", transform=ax.transAxes, va="top", ha="left", fontsize=12.5, family="monospace", color="#222")
		ax.set_xticks([])
		ax.set_yticks([])
		for s in ax.spines.values():
			s.set_visible(False)
	fig.savefig(a.out, dpi=125, bbox_inches="tight", pad_inches=0.25, facecolor="white")
	print(a.out)


if __name__ == "__main__":
	main()
