"""Score the RL pool by the SFT model's pass rate, then keep the prompts GRPO can learn from.

	python -m pointtrain.rl_difficulty data/rl/v1                 # writes scored.jsonl (resumable)
	python -m pointtrain.rl_difficulty data/rl/v1 --select-only   # (re)writes train.jsonl

Why filter. A group whose G samples all get the same reward has advantage 0 for every sample: it
costs a full rollout and contributes no policy gradient. If the SFT model solves a prompt with
probability p, the chance a group of G is unanimous is p^G + (1-p)^G. At p = 0.9 and G = 8 that is
43%; at p = 0.97 it is 78%. RefCOCO is ~91% for this model, so an unfiltered batch would be mostly
wasted. We estimate p with K samples per prompt (same temperature as training) and keep
0 < p_hat < 1, i.e. prompts the model sometimes gets right and sometimes wrong.

Caveat: p_hat from K = 8 is coarse (a true p of 0.95 shows up as 8/8 two thirds of the time), and the
policy changes during training, so this is a cheap pre-filter. The trainer also drops unanimous
groups online (skip_zero_std_groups) and logs how often that happens.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def score(pool_dir: Path, model_id: str, k: int, batch: int, temperature: float, max_new_tokens: int, limit: int | None, reward: str = "pointtrain.rl_reward:point_reward") -> None:
	from train_utils.configs.model_config import LoRAConfig, ModelConfig, QuantizationConfig
	from train_utils.vlm import build_adapter

	from train_utils.trainers.vlm_grpo_trainer import load_reward_fn

	reward_fn = load_reward_fn(reward)

	rows = [json.loads(l) for l in (pool_dir / "pool.jsonl").read_text().splitlines()][:limit]
	out_path = pool_dir / "scored.jsonl"
	done = {json.loads(l)["key"] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
	todo = [r for r in rows if r["key"] not in done]
	print(f"{len(rows)} prompts, {len(done)} already scored, {len(todo)} to go", flush=True)
	cfg = ModelConfig(model_id=model_id, vlm_adapter="hf_chat", dtype="bfloat16", attn_implementation="sdpa", lora=LoRAConfig(use_lora=False), quantization=QuantizationConfig(load_in_4bit=False))
	adapter = build_adapter(cfg, max_image_side=1024, image_root=str(Path.home() / "data"))
	model, _ = adapter.load()
	model.eval()
	per = max(1, batch // k)  # prompts per generate call
	t0 = time.time()
	with out_path.open("a") as f:
		for i in range(0, len(todo), per):
			chunk = todo[i : i + per]
			exp = [r for r in chunk for _ in range(k)]
			try:
				comps = adapter.sample(model, exp, max_new_tokens, True, temperature, 1.0, 0)
			except OSError as e:  # unreadable image: skip the chunk's bad rows one by one
				print("image error, retrying rows singly:", e, flush=True)
				for r in chunk:
					try:
						cs = adapter.sample(model, [r] * k, max_new_tokens, True, temperature, 1.0, 0)
					except OSError:
						continue
					_write(f, r, [adapter.decode(c) for c in cs], reward_fn)
				continue
			texts = [adapter.decode(c) for c in comps]
			for j, r in enumerate(chunk):
				_write(f, r, texts[j * k : (j + 1) * k], reward_fn)
			n = i + len(chunk)
			if (i // per) % 20 == 0:
				el = time.time() - t0
				print(f"{n}/{len(todo)}  {n / el:.1f} prompts/s  eta {(len(todo) - n) / max(n / el, 1e-9) / 60:.0f} min", flush=True)


def _write(f, row, texts, reward_fn):
	# Pass rate counts full successes ("hit" when the reward reports it), not partial credit.
	rs = [(lambda o: o.get("hit", o["reward"]))(reward_fn(row, t)) for t in texts]
	row = dict(row, sft_pass=sum(rs) / len(rs), sft_k=len(rs), sft_example=texts[0])
	f.write(json.dumps(row) + "\n")
	f.flush()


def select(pool_dir: Path, lo: float, hi: float) -> None:
	rows = [json.loads(l) for l in (pool_dir / "scored.jsonl").read_text().splitlines()]
	hist: dict[str, Counter] = defaultdict(Counter)
	for r in rows:
		hist[r["group"]][round(r["sft_pass"] * r["sft_k"])] += 1
	keep = [r for r in rows if lo < r["sft_pass"] < hi]
	with (pool_dir / "train.jsonl").open("w") as f:
		for r in keep:
			f.write(json.dumps(r) + "\n")
	summary = {
		"n_scored": len(rows),
		"n_train": len(keep),
		"train_by_group": dict(Counter(r["group"] for r in keep)),
		"mean_pass_by_group": {g: round(sum(r["sft_pass"] for r in rows if r["group"] == g) / sum(hist[g].values()), 3) for g in hist},
		"hits_out_of_k_by_group": {g: dict(sorted(c.items())) for g, c in hist.items()},
	}
	(pool_dir / "difficulty.json").write_text(json.dumps(summary, indent=2) + "\n")
	print(json.dumps(summary, indent=2))


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("pool_dir", type=Path)
	ap.add_argument("--model", default=str(ROOT / "outputs" / "sft-qwen3vl-2b-v1-e2" / "merged"))
	ap.add_argument("--k", type=int, default=8)
	ap.add_argument("--batch", type=int, default=64, help="sequences per generate call")
	ap.add_argument("--temperature", type=float, default=1.0)
	ap.add_argument("--max-new-tokens", type=int, default=24)
	ap.add_argument("--limit", type=int, default=None)
	ap.add_argument("--select-only", action="store_true")
	ap.add_argument("--reward", default="pointtrain.rl_reward:point_reward")
	ap.add_argument("--lo", type=float, default=0.0)
	ap.add_argument("--hi", type=float, default=1.0)
	a = ap.parse_args()
	if not a.select_only:
		score(a.pool_dir, a.model, a.k, a.batch, a.temperature, a.max_new_tokens, a.limit, a.reward)
	select(a.pool_dir, a.lo, a.hi)


if __name__ == "__main__":
	main()
