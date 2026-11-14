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
