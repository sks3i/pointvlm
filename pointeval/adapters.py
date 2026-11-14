"""Model wrappers: image + prompt -> raw text, with frozen decoding settings.

Decoding is greedy (do_sample=False) so the eval is deterministic up to GPU
kernel nondeterminism. Each model keeps its OWN default image preprocessing
(resolution, tiling), recorded in the run summary, because "how the model
sees the image" is part of the model being evaluated.

Prompt templates are frozen per (model, mode). mode="canonical" asks for our
<point x y> format and is the only mode used for our own checkpoints.
mode="native" asks in the model's own pointing format, for baselines only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

CANONICAL_PROMPT = (
	"Point to {query}. Answer with exactly one point in the format "
	'<point x="X" y="Y">, where X and Y are integers from 0 to 1000 giving the '
	"position as a fraction of the image width and height (0,0 is the top-left corner)."
)

NATIVE_PROMPTS = {
	# Molmo was trained on "Point to ..." and answers in its own <point> percent format.
	"molmo": "Point to {query}.",
	# Qwen3-VL grounding JSON; coordinates are 0-1000 relative in Qwen3-VL.
	"qwen_json": (
		"Locate {query} in the image and output its point coordinates in JSON format "
		'like [{{"point_2d": [x, y], "label": "..."}}].'
	),
}

MAX_NEW_TOKENS = 64


@dataclass
class Generation:
	text: str
	n_new_tokens: int
	latency_ms: float  # preprocessing + generate + decode, CUDA-synchronized


class HFChatVLM:
	"""Qwen3-VL and LFM2-VL both load via AutoModelForImageTextToText + chat template."""

	def __init__(self, model_id: str, dtype=torch.bfloat16, attn: str = "sdpa"):
		from transformers import AutoModelForImageTextToText, AutoProcessor

		self.model_id = model_id
		self.processor = AutoProcessor.from_pretrained(model_id)
		self.model = AutoModelForImageTextToText.from_pretrained(
			model_id, dtype=dtype, device_map="cuda", attn_implementation=attn
		).eval()
		self.config = {"dtype": str(dtype), "attn": attn, "max_new_tokens": MAX_NEW_TOKENS}

	@torch.inference_mode()
	def generate(self, image, prompt: str) -> Generation:
		torch.cuda.synchronize()
		t0 = time.perf_counter()
		messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
		inputs = self.processor.apply_chat_template(
			messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
		).to(self.model.device)
		out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
		new = out[0, inputs["input_ids"].shape[1] :]
		text = self.processor.decode(new, skip_special_tokens=True)
		torch.cuda.synchronize()
		return Generation(text, int(new.numel()), (time.perf_counter() - t0) * 1000)


class Molmo:
	"""allenai/Molmo-7B-D-0924 via trust_remote_code.

	fp32 weights (~32 GB) do not fit a 24 GB card, so we load bf16 weights. The
	model card notes this can shift outputs slightly vs fp32+autocast; recorded
	in the summary. Molmo's remote code is pinned to older transformers; if it
	breaks, run it from a separate venv (see README).
	"""

	def __init__(self, model_id: str = "allenai/Molmo-7B-D-0924"):
		from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

		self.model_id = model_id
		self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto")
		self.model = AutoModelForCausalLM.from_pretrained(
			model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
		).eval()
		self.gen_cfg = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS * 2, stop_strings="<|endoftext|>")
		self.config = {"dtype": "bfloat16 weights", "max_new_tokens": MAX_NEW_TOKENS * 2}

	@torch.inference_mode()
	def generate(self, image, prompt: str) -> Generation:
		torch.cuda.synchronize()
		t0 = time.perf_counter()
		inputs = self.processor.process(images=[image.convert("RGB")], text=prompt)
		inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
		inputs["images"] = inputs["images"].to(torch.bfloat16)
		out = self.model.generate_from_batch(inputs, self.gen_cfg, tokenizer=self.processor.tokenizer)
		new = out[0, inputs["input_ids"].size(1) :]
		text = self.processor.tokenizer.decode(new, skip_special_tokens=True)
		torch.cuda.synchronize()
		return Generation(text, int(new.numel()), (time.perf_counter() - t0) * 1000)


# name -> (constructor, native parser name or None)
MODELS = {
	"qwen3vl-2b": (lambda: HFChatVLM("Qwen/Qwen3-VL-2B-Instruct"), "qwen_json"),
	"lfm2vl-1.6b": (lambda: HFChatVLM("LiquidAI/LFM2-VL-1.6B"), None),
	"molmo-7b-d": (lambda: Molmo(), "molmo"),
}


def load_local(path: str) -> HFChatVLM:
	"""Our own checkpoints (SFT/RL outputs) are evaluated with mode='canonical' only."""
	return HFChatVLM(path)
