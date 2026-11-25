# pointvlm

Teach a small VLM to point (`<point x="512" y="340">`, 0-1000 normalized): frozen evals, then SFT, GRPO and quantization.
Training uses [localtrainer](https://github.com/sks3i/localtrainer) (`train_utils`, branch `vlm-grpo`); this repo holds everything project specific.

* Blog post: [Teaching a 2B VLM to Point, Part 1](https://sabarishks.com/posts/teaching-2b-vlm-to-point-part-1/)
* Released model: [Sks3i/qwen3-vl-2b-pointer](https://huggingface.co/Sks3i/qwen3-vl-2b-pointer) (Qwen3-VL-2B after SFT + four GRPO runs; the blog's "run 4")

| Frozen harness | RefCOCO val | Where2Place | Point-Bench |
| --- | --- | --- | --- |
| Qwen3-VL-2B-Instruct (same prompt) | 80.6% | 40% | 48.9% |
| After SFT | 91.1% | 51% | 58.6% |
| After GRPO v4 (released) | 92.1% | 57% | 60.9% |

These come from this repo's own frozen protocol (prompt, 64-token cap, strict parser), not each benchmark's official one, so they are not directly comparable to published numbers. Per-item predictions for every run are in `results/`; compare two runs with `scripts/compare_runs.py` (paired exact McNemar).

Pipeline scripts: `scripts/run_sft_qwen_lfm2.sh` (SFT), `scripts/run_grpo.sh` … `scripts/run_grpo_v3.sh` (GRPO runs 1-3), `scripts/run_count_sft.sh` + `scripts/run_grpo_v4.sh` (counting SFT + GRPO run 4), `scripts/eval_vllm.py` (vLLM eval and latency). Configs in `configs/sft/` and `configs/grpo/`.

## Evals (frozen in week 1)

The rules are in [docs/EVAL_SPEC.md](docs/EVAL_SPEC.md). Do not change parser, prompts, manifests, or scoring after the freeze; add a new versioned eval instead.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[models,dev]"
pytest -q                                   # CPU tests for parser, geometry, metrics

# one-time: build frozen manifests (writes manifests/*.jsonl + *.meta.json; commit them)
python -m pointeval.manifests refcoco
python -m pointeval.manifests where2place
python -m pointeval.manifests pointbench    # optional

# baselines
python -m pointeval.run --model qwen3vl-2b  --mode native    --bench refcoco
python -m pointeval.run --model qwen3vl-2b  --mode canonical --bench refcoco
python -m pointeval.run --model lfm2vl-1.6b --mode canonical --bench refcoco
python -m pointeval.run --model molmo-7b-d  --mode native    --bench refcoco
# ...same with --bench where2place

# our checkpoints: canonical only
python -m pointeval.run --checkpoint outputs/sft-001 --bench refcoco
```

Molmo's remote code may need an older `transformers`. If it fails, use a separate env:
`python -m venv .venv-molmo && .venv-molmo/bin/pip install -e . "transformers==4.47.1" torch accelerate einops torchvision`.

## SFT (week 2)

Code in `pointtrain/`, configs in `configs/sft/`. Training runs on localtrainer's `vlm_sft` trainer (the `vlm-grpo` branch, which also contains `vlm_grpo`; `pip install -e ../localtrainer`).

```bash
# data (images under ~/data: coco/train2014, pixmo/images, robopoint/images)
python -m pointtrain.download_pixmo --n-images 40000
python -m pointtrain.build --out data/sft/v1            # train.jsonl, dev.jsonl, meta.json (ID-level leak checks)
python -m pointtrain.leakcheck data/sft/v1              # pixel near-duplicates vs all eval images
python -m pointtrain.build --out data/sft/v1 --exclude-images data/sft/v1/flagged_images.txt   # if anything was flagged

# train: one switch picks the model (configs/sft/presets/*.yaml)
python -m pointtrain.sft --model qwen3vl-2b             # LoRA, ~14 GB
python -m pointtrain.sft --model lfm2vl-1.6b            # full FT of LLM + projector, ~21 GB
.venv-molmo/bin/python -m pointtrain.sft --model molmo-7b-d   # QLoRA, ~15 GB
python -m pointtrain.sft --model qwen3vl-2b --set training.learning_rate=3e-5 --dry-run       # overrides

# eval (frozen harness, canonical only); LoRA runs write <run>/merged
python -m pointeval.run --checkpoint outputs/sft-qwen3vl-2b-v1/merged --bench refcoco --run-name sft-qwen3vl-2b-v1
python -m pointtrain.export outputs/sft-qwen3vl-2b-v1/checkpoint-2000   # intermediate LoRA checkpoint -> merged
```

* Labels: deepest interior point of each mask (distance-transform max), checked with the eval's own `point_in_mask`.
* Leakage: a COCO image is dropped if any of RefCOCO / RefCOCO+ / RefCOCOg holds it out or the eval manifest uses it (RefCOCOg train alone touches 729 of the 1,210 eval images).
* Model selection uses `dev.jsonl` (held-out RefCOCO-family train images); `dev_metrics.jsonl` in each run logs greedy dev accuracy every 500 steps.
* `pointeval/adapters.py::load_local` dispatches Molmo checkpoints to the Molmo wrapper (loading only; prompts, decoding and scoring unchanged).


## License

Apache-2.0 (see `LICENSE`). Datasets and base models keep their own licenses.
