"""Run one model on one frozen benchmark and write predictions + summary.

	python -m pointeval.run --model qwen3vl-2b --mode native --bench refcoco
	python -m pointeval.run --model qwen3vl-2b --mode canonical --bench where2place
	python -m pointeval.run --checkpoint outputs/sft-001 --bench refcoco   # our models: canonical only

Outputs go to results/<bench>/<run_name>/{predictions.jsonl,summary.json}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from pointeval.geometry import norm_to_pixel, point_in_box, point_in_mask, rle_to_mask
from pointeval.manifests import (
	MANIFEST_DIR,
	REFCOCO_REPO,
	REFCOCO_SPLIT,
	WHERE2PLACE_REPO,
	image_bytes,
	pointbench_root,
	sha256_bytes,
)
from pointeval.metrics import latency_stats, summarize, wilson_ci
from pointeval.parse import PARSERS, ParseResult, ParseStatus

ROOT = Path(__file__).resolve().parent.parent
BENCH_FILES = {"refcoco": "refcoco_val_2k", "where2place": "where2place", "pointbench": "pointbench"}
N_WARMUP = 5


def load_manifest(bench: str) -> tuple[list[dict], dict]:
	name = BENCH_FILES[bench]
	path = MANIFEST_DIR / f"{name}.jsonl"
	meta = json.loads((MANIFEST_DIR / f"{name}.meta.json").read_text())
	if sha256_bytes(path.read_bytes()) != meta["manifest_sha256"]:
		raise SystemExit(f"{path} does not match its recorded sha256: the frozen manifest was modified")
	return [json.loads(l) for l in path.read_text().splitlines()], meta


class ImageSource:
	"""Fetch images from the pinned dataset revision and verify them against the manifest hash."""

	def __init__(self, bench: str, revision: str):
		self.bench = bench
		if bench == "refcoco":
			from datasets import load_dataset

			self.ds = load_dataset(REFCOCO_REPO, split=REFCOCO_SPLIT, revision=revision)
		else:
			from huggingface_hub import snapshot_download

			if bench == "where2place":
				root = Path(snapshot_download(WHERE2PLACE_REPO, repo_type="dataset", revision=revision))
				self.base = root / "images"
			else:
				self.base = pointbench_root(revision)

	def get(self, item: dict):
		from PIL import Image

		img = self.ds[item["row_index"]]["image"] if self.bench == "refcoco" else Image.open(self.base / item["file_name"])
		img = img.convert("RGB")
		if sha256_bytes(image_bytes(img)) != item["image_sha256"]:
			raise SystemExit(f"image hash mismatch for {item['id']}")
		return img


def score_item(bench: str, item: dict, pr: ParseResult) -> dict:
	"""Frozen scoring rules. Returns per-item metrics (bools/floats)."""
	mask = rle_to_mask(item["mask_rle"])
	first = pr.first
	out = {"hit": point_in_mask(first, mask)}
	if bench == "refcoco":
		out["hit_box"] = point_in_box(first, item["bbox_xywh"], item["width"], item["height"])
	elif bench == "where2place":
		# RoboPoint's published metric: fraction of ALL predicted points inside the mask.
		pts = pr.points if pr.status is ParseStatus.OK else []
		out["frac_points_in_mask"] = (sum(point_in_mask(p, mask) for p in pts) / len(pts)) if pts else 0.0
	elif bench == "pointbench" and item["category"] == "counting":
		# PointArena rule: #points must equal #targets and every target gets >= 1 point.
		from scipy import ndimage

		labels, n_targets = ndimage.label(mask)
		pts = pr.points if pr.status is ParseStatus.OK else []
		want = item.get("count") or n_targets
		hit_ids = set()
		for p in pts:
			c, r = norm_to_pixel(p, mask.shape[1], mask.shape[0])
			if labels[r, c]:
				hit_ids.add(int(labels[r, c]))
		out["hit"] = len(pts) == want and len(hit_ids) == n_targets
	return out


SCORING_VERSION = "v1"
PARSER_SHA256 = hashlib.sha256((ROOT / "pointeval" / "parse.py").read_bytes()).hexdigest()[:16]


def score_predictions(bench: str, items: list[dict], preds: list[dict], parse) -> dict:
	"""Parse each prediction's raw text, score it, write the per-item fields into ``preds`` in
	place, and return the summary metrics. Shared by run.py and rescore.py."""
	by_id = {it["id"]: it for it in items}
	hits, status_counts = [], {s.value: 0 for s in ParseStatus}
	extra: dict[str, list] = {}
	cats: dict[str, list[bool]] = {}
	for p in preds:
		item = by_id[p["id"]]
		pr = parse(p["raw"])
		sc = score_item(bench, item, pr)
		status_counts[pr.status.value] += 1
		hits.append(sc["hit"])
		for k, v in sc.items():
			if k != "hit":
				extra.setdefault(k, []).append(v)
		if item.get("category"):
			cats.setdefault(item["category"], []).append(sc["hit"])
		p.update({"status": pr.status.value, "note": pr.note, "points": [[q.x, q.y] for q in pr.points], **sc})
	out = {
		"scoring_version": SCORING_VERSION,
		"parser_sha256": PARSER_SHA256,
		"point_in_mask": summarize(hits),
		"parse_status": status_counts,
	}
	for k, vs in extra.items():
		out[k] = summarize(vs) if isinstance(vs[0], bool) else {"mean": sum(vs) / len(vs)}
	if cats:
		out["by_category"] = {
			c: {"n": len(h), "acc": sum(h) / len(h), "ci95": wilson_ci(sum(h), len(h))} for c, h in sorted(cats.items())
		}
	return out


def write_predictions(out_dir: Path, preds: list[dict]) -> None:
	with (out_dir / "predictions.jsonl").open("w") as f:
		for p in preds:
			f.write(json.dumps(p) + "\n")


def hardware_info() -> dict:
	import torch
	import transformers

	info = {
		"python": platform.python_version(),
		"torch": torch.__version__,
		"transformers": transformers.__version__,
		"cuda": torch.version.cuda,
		"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
		"cpu": platform.processor() or platform.machine(),
	}
	try:
		info["driver"] = subprocess.check_output(
			["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
		).strip()
	except Exception:
		pass
	try:
		info["code_git_sha"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
	except Exception:
		pass
	return info


def main() -> None:
	from pointeval.adapters import CANONICAL_PROMPT, MODELS, NATIVE_PROMPTS, load_local

	ap = argparse.ArgumentParser()
	g = ap.add_mutually_exclusive_group(required=True)
	g.add_argument("--model", choices=sorted(MODELS))
	g.add_argument("--checkpoint")
	ap.add_argument("--mode", choices=["canonical", "native"], default="canonical")
	ap.add_argument("--bench", choices=sorted(BENCH_FILES), required=True)
	ap.add_argument("--limit", type=int, default=None, help="smoke test only; summaries with --limit are marked partial")
	ap.add_argument("--run-name", default=None)
	a = ap.parse_args()

	if a.checkpoint:
		model, native = load_local(a.checkpoint), None
		if a.mode != "canonical":
			raise SystemExit("our checkpoints are scored with the canonical format only")
	else:
		ctor, native = MODELS[a.model]
		if a.mode == "native" and native is None:
			raise SystemExit(f"{a.model} has no native pointing format; use --mode canonical")
		model = ctor()
	parser_name = "canonical" if a.mode == "canonical" else native
	parse = PARSERS[parser_name]
	template = CANONICAL_PROMPT if a.mode == "canonical" else NATIVE_PROMPTS[native]

	items, meta = load_manifest(a.bench)
	if a.limit:
		items = items[: a.limit]
	src = ImageSource(a.bench, meta["revision"])
	run_name = a.run_name or f"{a.model or Path(a.checkpoint).name}-{a.mode}"
	out_dir = ROOT / "results" / a.bench / (run_name + ("-partial" if a.limit else ""))
	out_dir.mkdir(parents=True, exist_ok=True)

	# Warm-up: CUDA context, cuDNN autotune, allocator. Not timed, not scored.
	warm_img = src.get(items[0])
	for _ in range(N_WARMUP):
		model.generate(warm_img, template.format(query=items[0]["query"]))

	preds = []
	for i, item in enumerate(items):
		img = src.get(item)
		gen = model.generate(img, template.format(query=item["query"]))
		preds.append(
			{
				"id": item["id"],
				"query": item["query"],
				"raw": gen.text,
				"n_new_tokens": gen.n_new_tokens,
				"latency_ms": round(gen.latency_ms, 2),
			}
		)
		if (i + 1) % 100 == 0:
			print(f"{i + 1}/{len(items)}", flush=True)

	scores = score_predictions(a.bench, items, preds, parse)
	write_predictions(out_dir, preds)

	summary = {
		"bench": a.bench,
		"partial": bool(a.limit),
		"model": a.model or a.checkpoint,
		"model_config": getattr(model, "config", {}),
		"mode": a.mode,
		"parser": parser_name,
		"prompt_template": template,
		"prompt_sha256": hashlib.sha256(template.encode()).hexdigest()[:16],
		"manifest_sha256": meta["manifest_sha256"],
		"dataset_revision": meta["revision"],
		**scores,
		"latency": latency_stats([p["latency_ms"] for p in preds]),
		"mean_new_tokens": sum(p["n_new_tokens"] for p in preds) / len(preds),
		"hardware": hardware_info(),
	}
	(out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
	print(json.dumps({k: summary[k] for k in ("point_in_mask", "parse_status", "latency")}, indent=2))


if __name__ == "__main__":
	main()
