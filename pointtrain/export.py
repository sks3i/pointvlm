"""Turn an intermediate LoRA checkpoint into something pointeval can load.

	python -m pointtrain.export outputs/sft-qwen3vl-2b-v1/checkpoint-2000
	python -m pointeval.run --checkpoint outputs/sft-qwen3vl-2b-v1/checkpoint-2000/merged --bench refcoco

Full fine-tune checkpoints need no export: the checkpoint dir already holds full weights + processor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dacite
import yaml


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("checkpoint", type=Path)
	ap.add_argument("--out", type=Path, default=None)
	a = ap.parse_args()

	from train_utils.configs.base_config import RunConfig
	from train_utils.vlm import export_merged

	run_dir = a.checkpoint if (a.checkpoint / "pointvlm_config.yaml").exists() else a.checkpoint.parent
	raw = yaml.safe_load((run_dir / "pointvlm_config.yaml").read_text())
	config = dacite.from_dict(RunConfig, raw, config=dacite.Config(strict=False))
	if not config.model.lora.use_lora:
		raise SystemExit("full fine-tune checkpoint: evaluate the checkpoint dir directly")
	export_merged(a.checkpoint, a.out or a.checkpoint / "merged", config.model)


if __name__ == "__main__":
	main()
