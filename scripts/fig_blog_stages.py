"""Blog figures, one per training stage: 4 stored predictions in a 2x2 grid.

	python scripts/fig_blog_stages.py --spec results/figures/blog_figures.json --only blog_sft
	python scripts/fig_blog_stages.py --sheet pointbench:pb-0001,pointbench:pb-0002 --run sft-qwen3vl-2b-v1-e2 --out /tmp/sheet.png

Spec: {"blog_sft": {"title": ..., "panels": [{"bench", "id", "run", "prev", "verdict", "caption"}, ...]}, ...}.
Mask green, current run's points red with white rim, previous run's points as white rings.
Reads results/<bench>/<run>/predictions.jsonl only; no model is run.
"""

import argparse
import os
import json
import textwrap
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pointeval.geometry import rle_to_mask
from pointeval.run import ImageSource, load_manifest

ROOT = Path(__file__).resolve().parent.parent
NAMES = {"refcoco": "RefCOCO val", "where2place": "Where2Place", "pointbench": "Point-Bench"}
LABELS = {
	"qwen3vl-2b-canonical": "Base",
	"sft-qwen3vl-2b-v1-e2": "SFT",
	"grpo-v1": "GRPO v1",
	"grpo-v2": "GRPO v2",
	"grpo-v3": "GRPO v3",
	"sft-count-v1": "Count SFT",
	"grpo-v4": "GRPO v4",
}


@lru_cache(None)
def manifest(bench):
	items, meta = load_manifest(bench)
	return {it["id"]: it for it in items}, meta


@lru_cache(None)
def preds(bench, run):
	rows = (json.loads(l) for l in (ROOT / "results" / bench / run / "predictions.jsonl").read_text().splitlines())
	return {r["id"]: r for r in rows}


def verdict(p):
	return "hit" if p["hit"] else "miss"


def draw(ax, panel, small=False):
	bench, iid, run, prev = panel["bench"], panel["id"], panel["run"], panel.get("prev")
	items, meta = manifest(bench)
	item = items[iid]
	pred = preds(bench, run)[iid]
	img = np.asarray(ImageSource(bench, meta["revision"]).get(item))
	mask = rle_to_mask(item["mask_rle"])
	h, w = mask.shape
	ax.imshow(img)
	overlay = np.zeros((h, w, 4))
	overlay[mask] = [0.1, 0.85, 0.2, 0.45]
	ax.imshow(overlay)
	ax.contour(mask, levels=[0.5], colors=["#1aa832"], linewidths=1.5)
	s = 120 if small else 260
	if prev:
		for x, y in preds(bench, prev)[iid]["points"]:
			ax.scatter([x / 1000 * w], [y / 1000 * h], s=s * 1.6, facecolors="none", edgecolors="black", linewidths=5, zorder=4)
			ax.scatter([x / 1000 * w], [y / 1000 * h], s=s * 1.6, facecolors="none", edgecolors="white", linewidths=3, zorder=4)
	for x, y in pred["points"]:
		ax.scatter([x / 1000 * w], [y / 1000 * h], s=s, c="#e8202a", edgecolors="white", linewidths=2.5, zorder=5)
	cat = f" · {item['category']}" if item.get("category") else ""
	fs = 9 if small else 15
	q = "\n".join(textwrap.wrap(f"“{item['query']}”", 40 if small else 60))
	ax.set_title(f"{NAMES[bench]}{cat} · {iid}\n{q}", fontsize=fs, loc="left", pad=8)
	name = LABELS.get(run, run)
	legend = f"red = {name} ({verdict(pred)})"
	if prev:
		legend += f" · white ring = {LABELS.get(prev, prev)} ({verdict(preds(bench, prev)[iid])})"
	bb = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
	chars = max(30, int(bb.width * 72 / ((fs - 2) * 0.6)) - 2)  # monospace glyph ≈ 0.6 em
	lines = ["\n".join(textwrap.wrap(legend, chars)), "\n".join(textwrap.wrap(f"{name} output: " + pred["raw"], chars))]
	if panel.get("caption"):
		lines.insert(0, "\n".join(textwrap.wrap(panel["caption"], chars)))
	ax.text(0.0, -0.025, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=fs - 2, family="monospace", color="#222")
	ax.set_xticks([])
	ax.set_yticks([])
	for sp in ax.spines.values():
		sp.set_visible(False)
	return img.shape[1] / img.shape[0]


def aspect(panel):
	item = manifest(panel["bench"])[0][panel["id"]]
	return item["width"] / item["height"] if "width" in item else 4 / 3


def figure(spec, out, cols=2, width_in=24.0, dpi=100, small=False):
	panels = spec["panels"]
	rows = [panels[i : i + cols] for i in range(0, len(panels), cols)]
	row_h = [width_in / sum(aspect(p) for p in r) for r in rows]
	top_in, bot_in = (0.5, 0.7) if small else (0.85, 1.15)
	extra = top_in + bot_in
	fig = plt.figure(figsize=(width_in, sum(row_h) + extra * len(rows) + (0.8 if spec.get("title") else 0)))
	subs = fig.subfigures(len(rows), 1, height_ratios=[rh + extra for rh in row_h])
	for sub, r in zip(np.atleast_1d(subs), rows):
		axes = sub.subplots(1, len(r), gridspec_kw={"width_ratios": [aspect(p) for p in r], "wspace": 0.08})
		hh = width_in / sum(aspect(p) for p in r) + extra
		sub.subplots_adjust(left=0.01, right=0.99, top=1 - top_in / hh, bottom=bot_in / hh)
		for ax, p in zip(np.atleast_1d(axes), r):
			draw(ax, p, small)
	if spec.get("title"):
		fig.suptitle(spec["title"], fontsize=20, x=0.01, ha="left", y=1.0, va="bottom")
	fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.25, facecolor="white")
	if not small and os.path.getsize(out) > 3e6:  # drop the 2 low bits per channel: invisible, compresses ~2x
		rgb = np.asarray(Image.open(out).convert("RGB")) & 0xFC
		Image.fromarray(rgb).save(out, optimize=True)
	print(out)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--spec")
	ap.add_argument("--only", nargs="*")
	ap.add_argument("--sheet", help="comma list of bench:id for a quick contact sheet")
	ap.add_argument("--run", default="grpo-v4")
	ap.add_argument("--prev")
	ap.add_argument("--out")
	a = ap.parse_args()
	if a.sheet:
		panels = [dict(bench=s.split(":")[0], id=s.split(":")[1], run=a.run, prev=a.prev) for s in a.sheet.split(",")]
		figure({"panels": panels}, a.out, cols=4, width_in=20, dpi=60, small=True)
		return
	specs = json.loads(Path(a.spec).read_text())
	for name, spec in specs.items():
		if a.only and name not in a.only:
			continue
		figure(spec, ROOT / "results" / "figures" / f"{name}.png")


if __name__ == "__main__":
	main()
