# pointvlm

Teach a small VLM to point (`<point x="512" y="340">`, 0-1000 normalized): frozen evals, then SFT, GRPO and quantization.
Training uses [localtrainer](https://github.com/sks3i/localtrainer) (`train_utils`); this repo holds everything project specific.

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

Code in `pointtrain/`, configs in `configs/sft/`. Training runs on localtrainer's `vlm_sft` trainer (the `vlm-sft` branch; `pip install -e ../localtrainer`).
Notes on the math and design choices: [Pointing SFT: the math and intuition](https://claude.ai/artifact/FxTMmNKLHwsnMbHyCkTDVP).

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
