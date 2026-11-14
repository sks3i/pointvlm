"""Pointing SFT entry point: config + model preset -> localtrainer vlm_sft run with a dev metric.

	python -m pointtrain.sft --config configs/sft/base.yaml --model qwen3vl-2b
	python -m pointtrain.sft --config configs/sft/base.yaml --model lfm2vl-1.6b --set training.learning_rate=3e-5
	.venv-molmo/bin/python -m pointtrain.sft --config configs/sft/base.yaml --model molmo-7b-d

Then evaluate with the frozen harness (canonical mode only):
	python -m pointeval.run --checkpoint outputs/<run>/merged --bench refcoco   # LoRA runs
	python -m pointeval.run --checkpoint outputs/<run> --bench refcoco          # full fine-tunes
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import dacite
import yaml

ROOT = Path(__file__).resolve().parent.parent
PRESET_DIR = ROOT / "configs" / "sft" / "presets"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", stream=sys.stdout)
logger = logging.getLogger("pointtrain.sft")


def deep_merge(base: dict, over: dict) -> dict:
	out = copy.deepcopy(base)
	for k, v in (over or {}).items():
		out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
	return out


def _set(d: dict, dotted: str, value: str) -> None:
	keys = dotted.split(".")
	for k in keys[:-1]:
		d = d.setdefault(k, {})
	d[keys[-1]] = yaml.safe_load(value)


def resolve_config(config_path: Path, model: str | None, overrides: list[str]) -> dict:
	raw = yaml.safe_load(config_path.read_text())
	preset = model or raw.get("model_preset")
	if not (PRESET_DIR / f"{preset}.yaml").exists():
		raise SystemExit(f"unknown model preset {preset}; have {sorted(p.stem for p in PRESET_DIR.glob('*.yaml'))}")
	raw = deep_merge(raw, yaml.safe_load((PRESET_DIR / f"{preset}.yaml").read_text()))
	raw["model_preset"] = preset
	for o in overrides:
		k, v = o.split("=", 1)
		_set(raw, k, v)
	data_name = Path(raw["vlm_sft"]["train_file"]).parent.name
	raw["run_name"] = raw.get("run_name") or f"sft-{preset}-{data_name}"
	raw["output_dir"] = raw.get("output_dir") or str(ROOT / "outputs" / raw["run_name"])
	for k in ("train_file", "eval_file"):
		if raw["vlm_sft"].get(k):
			raw["vlm_sft"][k] = str((ROOT / raw["vlm_sft"][k]).resolve())
	return raw


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--config", type=Path, default=ROOT / "configs" / "sft" / "base.yaml")
	ap.add_argument("--model", default=None, help="preset name in configs/sft/presets (overrides model_preset)")
	ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides, e.g. training.max_steps=50")
	ap.add_argument("--dry-run", action="store_true", help="print the resolved config and exit")
	a = ap.parse_args()

	raw = resolve_config(a.config, a.model, a.set)
	if a.dry_run:
		print(yaml.safe_dump(raw, sort_keys=False))
		return

	from train_utils.configs.base_config import RunConfig
	from train_utils.train import run_vlm

	from pointtrain.dev_eval import DevPointAccuracyCallback

	config = dacite.from_dict(RunConfig, raw, config=dacite.Config(strict=False))
	out = Path(config.output_dir)
	out.mkdir(parents=True, exist_ok=True)
	(out / "pointvlm_config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))

	pv = raw.get("pointvlm", {})

	def dev_callbacks(adapter):
		if not pv.get("dev_file") or pv.get("dev_n", 0) <= 0:
			return []
		dev_file = str((ROOT / pv["dev_file"]).resolve())
		return [DevPointAccuracyCallback(adapter, dev_file, pv["dev_n"], pv.get("dev_every_steps", 500), str(out / "dev_metrics.jsonl"))]

	logger.info("run=%s preset=%s out=%s", config.run_name, raw["model_preset"], out)
	run_vlm(config, callback_factory=dev_callbacks)
	ckpt = out / "merged" if config.model.lora.use_lora and config.vlm_sft.save_merged else out
	py = ".venv-molmo/bin/python" if config.model.vlm_adapter == "molmo" else "python"
	logger.info("Eval with:  %s -m pointeval.run --checkpoint %s --bench refcoco", py, ckpt)


if __name__ == "__main__":
	main()
