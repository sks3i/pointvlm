"""Qwen3-VL baselines served by vLLM on the frozen benchmarks, resumable. Runs in .venv-vllm.

	.venv-vllm/bin/python scripts/eval_vllm.py --model QuantTrio/Qwen3-VL-32B-Instruct-AWQ --revision c2d44b1 \
		--tag qwen3vl-32b-awq --mode native canonical --bench refcoco where2place pointbench

Kept from the frozen spec: prompts (pointeval.adapters), greedy decoding (temperature 0), 64 new
tokens, batch 1 (one request at a time), 5 untimed warm-ups, the model's default image
preprocessing, frozen parsers and scoring (pointeval.run.score_predictions). Raw text is stored.
Different from pointeval.run, and recorded in every summary:
  * inference engine is vLLM, so latency is "vLLM latency", not comparable 1:1 with HF numbers;
    timing = wall time of llm.chat() for one request (preprocess + prefill + decode + detokenize).
  * max_model_len is capped to fit a 24 GB card. Qwen3-VL keeps images up to 16.7 MP (~16k
    visual tokens), so a prompt longer than max_model_len is retried with the image downsized
    (aspect kept) to fit; those items carry "resized_to" and are counted in the summary.
Outputs: results/<bench>/<tag>-<mode>/{predictions.jsonl,summary.json}; predictions.partial.jsonl
is the per-item log a restart resumes from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pointeval.adapters import CANONICAL_PROMPT, MAX_NEW_TOKENS, NATIVE_PROMPTS  # noqa: E402
from pointeval.metrics import latency_stats  # noqa: E402
from pointeval.parse import PARSERS  # noqa: E402
from pointeval.run import N_WARMUP, ImageSource, hardware_info, load_manifest, score_predictions, write_predictions  # noqa: E402

PIXELS_PER_TOKEN = 32 * 32  # Qwen3-VL: 16 px patches, 2x2 merge
PROMPT_TEXT_TOKENS = 200  # headroom for template + query + answer


class VLLMQwen:
	def __init__(self, model: str, revision: str, max_model_len: int, gpu_mem: float, enforce_eager: bool, kv_cache_dtype: str, max_batched: int | None, profile_side: int | None, kv_cache_gib: float | None):
		import torch
		from vllm import LLM, SamplingParams

		self.llm = LLM(
			model=model,
			revision=None if revision == "local" else revision,
			max_model_len=max_model_len,
			gpu_memory_utilization=gpu_mem,
			**({"kv_cache_memory_bytes": int(kv_cache_gib * 2**30)} if kv_cache_gib else {}),
			max_num_seqs=1,
			# profile_side only sizes the dummy image vLLM profiles memory with; real images keep the
			# default preprocessing (and are resized by generate_fit only if they exceed max_model_len).
			limit_mm_per_prompt={"image": {"count": 1, "width": profile_side, "height": profile_side} if profile_side else 1, "video": 0},
			enforce_eager=enforce_eager,
			kv_cache_dtype=kv_cache_dtype,
			seed=0,
			# Batch 1 only: capturing graphs for larger batch sizes costs ~1.5 GiB the KV cache needs.
			compilation_config={"cudagraph_capture_sizes": [1]},
			**({"max_num_batched_tokens": max_batched} if max_batched else {}),
		)
		self.params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
		self.max_model_len = max_model_len
		import vllm

		self.config = {
			"engine": f"vllm {vllm.__version__}",
			"model": model,
			"revision": revision,
			"quantization": str(self.llm.llm_engine.vllm_config.model_config.quantization),
			"dtype": str(self.llm.llm_engine.vllm_config.model_config.dtype),
			"kv_cache_dtype": kv_cache_dtype,
			"max_model_len": max_model_len,
			"gpu_memory_utilization": gpu_mem,
			"enforce_eager": enforce_eager,
			"max_num_batched_tokens": max_batched,
			"profile_image_side": profile_side,
			"cudagraph_capture_sizes": [1],
			"kv_cache_memory_gib": kv_cache_gib,
			"max_new_tokens": MAX_NEW_TOKENS,
			"temperature": 0.0,
			"torch": torch.__version__,
		}

	def generate(self, image, prompt: str):
		msgs = [{"role": "user", "content": [{"type": "image_pil", "image_pil": image}, {"type": "text", "text": prompt}]}]
		t0 = time.perf_counter()
		out = self.llm.chat(msgs, self.params, use_tqdm=False)[0]
		ms = (time.perf_counter() - t0) * 1000
		return out.outputs[0].text, len(out.outputs[0].token_ids), ms


def _resize(img, max_pixels: int):
	w, h = img.size
	s = min(1.0, (max_pixels / (w * h)) ** 0.5)
	return img.resize((max(32, int(w * s)), max(32, int(h * s))), resample=3)  # PIL BICUBIC


def generate_fit(model: VLLMQwen, img, prompt: str, budget: int | None = None):
	"""Native image first (or ``budget`` px if given); if the prompt exceeds max_model_len, downsize to fit, then halve."""
	while True:
		x = img if budget is None else _resize(img, budget)
		try:
			text, n, ms = model.generate(x, prompt)
			return text, n, ms, (None if budget is None else f"{x.size[0]}x{x.size[1]}")
		except Exception as e:  # vLLM VLLMValidationError: "The decoder prompt (length N) is longer than the maximum model length"
			if "longer than the maximum model length" not in str(e) and "too long" not in str(e):
				raise
			budget = (model.max_model_len - PROMPT_TEXT_TOKENS) * PIXELS_PER_TOKEN if budget is None else budget // 2
			print(f"too long at {img.size} -> retry at <= {budget} px", flush=True)


def gpu_used_gb() -> float:
	import subprocess

	try:
		out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
		return round(int(out.strip().splitlines()[0]) / 1024, 2)
	except Exception:
		return -1.0


def run_bench(model: VLLMQwen, bench: str, mode: str, run_name: str, peak: list[float]) -> None:
	parser_name = "canonical" if mode == "canonical" else "qwen_json"
	template = CANONICAL_PROMPT if mode == "canonical" else NATIVE_PROMPTS["qwen_json"]
	items, meta = load_manifest(bench)
	src = ImageSource(bench, meta["revision"])
	out_dir = ROOT / "results" / bench / run_name
	out_dir.mkdir(parents=True, exist_ok=True)
	if (out_dir / "summary.json").exists():
		print(f"=== {bench} {run_name}: already done", flush=True)
		return
	log = out_dir / "predictions.partial.jsonl"
	inflight = out_dir / "inflight.json"
	done = {}
	if log.exists():
		for line in log.read_text().splitlines():
			try:
				p = json.loads(line)
			except json.JSONDecodeError:  # torn last line after a crash
				continue
			done[p["id"]] = p
	print(f"=== {bench} {run_name}: {len(done)}/{len(items)} already done", flush=True)

	warm = src.get(items[0])
	for _ in range(N_WARMUP):
		generate_fit(model, warm, template.format(query=items[0]["query"]))

	t0 = time.time()
	with log.open("a") as f:
		for item in items:
			if item["id"] in done:
				continue
			# A CUDA OOM kills the engine, so it is handled across restarts: the item in flight is
			# recorded first, and if the process died on it, the restarted run retries it at half
			# the pixel budget it was attempted with.
			img = src.get(item)
			budget = None
			if inflight.exists():
				st = json.loads(inflight.read_text())
				if st["id"] == item["id"]:
					budget = (st["budget"] or img.size[0] * img.size[1]) // 2
					print(f"{item['id']} crashed last time; retrying at <= {budget} px", flush=True)
			inflight.write_text(json.dumps({"id": item["id"], "budget": budget}))
			text, n, ms, resized = generate_fit(model, img, template.format(query=item["query"]), budget)
			p = {"id": item["id"], "query": item["query"], "raw": text, "n_new_tokens": n, "latency_ms": round(ms, 2)}
			if resized:
				p["resized_to"] = resized
			f.write(json.dumps(p) + "\n")
			f.flush()
			os.fsync(f.fileno())
			done[item["id"]] = p
			if len(done) % 50 == 0:
				peak[0] = max(peak[0], gpu_used_gb())
				print(f"{bench} {len(done)}/{len(items)}  gpu {peak[0]:.1f} GB  {(time.time() - t0) / 60:.0f} min", flush=True)

	inflight.unlink(missing_ok=True)
	preds = [done[it["id"]] for it in items]
	scores = score_predictions(bench, items, preds, PARSERS[parser_name])
	write_predictions(out_dir, preds)
	resized = [p for p in preds if "resized_to" in p]
	peak[0] = max(peak[0], gpu_used_gb())
	summary = {
		"bench": bench,
		"partial": False,
		"model": f"{model.config['model']}@{model.config['revision'][:12]}",
		"model_config": model.config,
		"mode": mode,
		"parser": parser_name,
		"prompt_template": template,
		"prompt_sha256": hashlib.sha256(template.encode()).hexdigest()[:16],
		"manifest_sha256": meta["manifest_sha256"],
		"dataset_revision": meta["revision"],
		**scores,
		"latency": {**latency_stats([p["latency_ms"] for p in preds]), "engine": model.config["engine"]},
		"mean_new_tokens": sum(p["n_new_tokens"] for p in preds) / len(preds),
		"resized_to_fit_max_model_len": {
			"n": len(resized),
			"acc_without_them": sum(p["hit"] for p in preds if "resized_to" not in p) / max(1, len(preds) - len(resized)),
		},
		"gpu_memory_used_gb_peak_sampled": {"value": peak[0], "note": "nvidia-smi memory.used, whole GPU incl. desktop (~1 GB), sampled every 50 items; vLLM preallocates gpu_memory_utilization"},
		"hardware": hardware_info(),
		"runner": "scripts/eval_vllm.py",
	}
	(out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
	print(json.dumps({k: summary[k] for k in ("point_in_mask", "parse_status", "latency", "resized_to_fit_max_model_len")}, indent=2), flush=True)


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--model", required=True)
	ap.add_argument("--revision", required=True, help='HF revision, or "local" for a checkpoint directory')
	ap.add_argument("--tag", required=True, help="run name prefix: results/<bench>/<tag>-<mode>")
	ap.add_argument("--mode", nargs="+", choices=["native", "canonical"], default=["native"])
	ap.add_argument("--bench", nargs="+", default=["refcoco", "where2place", "pointbench"])
	ap.add_argument("--max-model-len", type=int, default=8192)
	ap.add_argument("--gpu-mem", type=float, default=0.93)
	ap.add_argument("--kv-cache-dtype", default="auto")
	ap.add_argument("--kv-cache-gib", type=float, default=None, help="fixed KV cache size instead of gpu_memory_utilization")
	ap.add_argument("--enforce-eager", action="store_true")
	ap.add_argument("--profile-image-side", type=int, default=None, help="size of the dummy image used for memory profiling")
	ap.add_argument("--max-num-batched-tokens", type=int, default=None, help="prefill chunk size; smaller = lower activation peak, same outputs")
	ap.add_argument("--smoke", type=int, default=0, help="run N items of the first bench (plus its largest image), write nothing")
	a = ap.parse_args()
	model = VLLMQwen(a.model, a.revision, a.max_model_len, a.gpu_mem, a.enforce_eager, a.kv_cache_dtype, a.max_num_batched_tokens, a.profile_image_side, a.kv_cache_gib)
	print("loaded", model.config, "gpu", gpu_used_gb(), "GB", flush=True)
	if a.smoke:
		items, meta = load_manifest(a.bench[0])
		src = ImageSource(a.bench[0], meta["revision"])
		big = max(items, key=lambda it: it["width"] * it["height"])
		tpl = CANONICAL_PROMPT if a.mode[0] == "canonical" else NATIVE_PROMPTS["qwen_json"]
		for it in items[: a.smoke] + [big]:
			text, n, ms, r = generate_fit(model, src.get(it), tpl.format(query=it["query"]))
			print(it["id"], (it["width"], it["height"]), repr(text[:120]), n, f"{ms:.0f} ms", r, "gpu", gpu_used_gb(), flush=True)
		return
	peak = [gpu_used_gb()]
	for mode in a.mode:
		for bench in a.bench:
			run_bench(model, bench, mode, f"{a.tag}-{mode}", peak)
	print("=== all done", flush=True)


if __name__ == "__main__":
	main()
