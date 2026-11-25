"""Staged start-up test for the GRPO hang: each stage logs (and fsyncs) before and after.

	python scripts/stage_test.py --stage 1   # processor + model to GPU
	python scripts/stage_test.py --stage 2   # + one batch through the processor and one sampled generate
	python scripts/stage_test.py --stage 3   # + one full GRPO step
"""

import argparse
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--stage", type=int, required=True)
ap.add_argument("--log", default="outputs/stage_test.log")
a = ap.parse_args()
LOG = open(a.log, "a")
T0 = time.time()


def log(msg):
	line = f"{time.strftime('%T')} +{time.time() - T0:6.1f}s stage{a.stage} {msg}"
	print(line, flush=True)
	LOG.write(line + "\n")
	LOG.flush()
	os.fsync(LOG.fileno())


log("start; importing torch")
import torch

log(f"torch {torch.__version__} threads={torch.get_num_threads()}")
torch.cuda.init()
log(f"cuda init ok, free={torch.cuda.mem_get_info()[0] / 2**30:.1f} GB")
from train_utils.configs.model_config import LoRAConfig, ModelConfig, QuantizationConfig
from train_utils.vlm import build_adapter

log("imports done")
mid = str(Path("outputs/sft-qwen3vl-2b-v1-e2/merged").resolve())
cfg = ModelConfig(model_id=mid, vlm_adapter="hf_chat", dtype="bfloat16", attn_implementation="sdpa", lora=LoRAConfig(use_lora=False), quantization=QuantizationConfig(load_in_4bit=False))
adapter = build_adapter(cfg, max_image_side=1024, image_root=str(Path.home() / "data"))
log("loading processor + model")
model, proc = adapter.load()
log(f"model loaded, allocated={torch.cuda.memory_allocated() / 2**30:.2f} GB")
if a.stage >= 2:
	import json

	rows = [json.loads(l) for l, _ in zip(open("data/rl/v2/train.jsonl"), range(8))]
	log("processor batch")
	b = adapter._prompt_batch([r for r in rows for _ in range(8)], "left")
	log(f"batch ok {tuple(b['input_ids'].shape)}")
	model.eval()
	comps = adapter.sample(model, [r for r in rows for _ in range(8)], 24)
	log(f"generate ok, {len(comps)} completions, e.g. {adapter.decode(comps[0])!r}")
if a.stage >= 3:
	log("stage 3 runs through pointtrain.grpo instead (see run command)")
log("done")
sys.exit(0)
