"""Pointing GRPO entry point: config -> localtrainer vlm_grpo run.

	python -m pointtrain.grpo --config configs/grpo/base.yaml
	python -m pointtrain.grpo --set vlm_grpo.max_steps=5 vlm_grpo.dev_n=32 run_name=grpo-smoke

Then evaluate the merged checkpoint with the frozen harness (canonical mode only):
	python -m pointeval.run --checkpoint outputs/<run>/merged --bench pointbench
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import dacite
import yaml

from pointtrain.sft import _set

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", stream=sys.stdout)
logger = logging.getLogger("pointtrain.grpo")


def resolve_config(config_path: Path, overrides: list[str]) -> dict:
	raw = yaml.safe_load(config_path.read_text())
	for o in overrides:
		k, v = o.split("=", 1)
		_set(raw, k, v)
	data_name = Path(raw["vlm_grpo"]["train_file"]).parent.name
	raw["run_name"] = raw.get("run_name") or f"grpo-{data_name}"
	raw["output_dir"] = raw.get("output_dir") or str(ROOT / "outputs" / raw["run_name"])
	for k in ("train_file", "dev_file"):
		if raw["vlm_grpo"].get(k):
			raw["vlm_grpo"][k] = str((ROOT / raw["vlm_grpo"][k]).resolve())
	if not Path(raw["model"]["model_id"]).is_absolute() and (ROOT / raw["model"]["model_id"]).exists():
		raw["model"]["model_id"] = str((ROOT / raw["model"]["model_id"]).resolve())
	return raw


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--config", type=Path, default=ROOT / "configs" / "grpo" / "base.yaml")
	ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides, e.g. vlm_grpo.beta=0")
	ap.add_argument("--dry-run", action="store_true")
	a = ap.parse_args()
	raw = resolve_config(a.config, a.set)
	if a.dry_run:
		print(yaml.safe_dump(raw, sort_keys=False))
		return

	from train_utils.configs.base_config import RunConfig
	from train_utils.train import run_vlm

	config = dacite.from_dict(RunConfig, raw, config=dacite.Config(strict=False))
	out = Path(config.output_dir)
	out.mkdir(parents=True, exist_ok=True)
	(out / "pointvlm_config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
	logger.info("run=%s out=%s", config.run_name, out)
	run_vlm(config)
	logger.info("Eval with:  python -m pointeval.run --checkpoint %s --bench pointbench", out / "merged")


if __name__ == "__main__":
	main()
