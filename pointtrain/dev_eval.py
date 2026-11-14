"""In-training task metric: greedy point-in-mask accuracy on the held-out dev split.

Loss is a poor proxy for pointing accuracy: most of the answer tokens are fixed format (<point x="),
and a units-digit error costs as much cross-entropy as a hundreds-digit error. So every
``every_steps`` optimizer steps we decode the dev set exactly as the eval does (greedy, canonical
parser, the eval's point_in_mask) and log:

	dev/point_acc         headline, comparable in spirit to the frozen eval
	dev/unparseable       format drift (should go to ~0 within a few hundred steps)
	dev/zero_zero         rate of (0, 0) answers, LFM2's failure mode at baseline
	dev/mean_l1_norm      mean |pred - target| in canonical units, a smooth error signal
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from transformers import TrainerCallback

from pointeval.geometry import point_in_mask, rle_to_mask
from pointeval.parse import ParseStatus, parse_canonical

logger = logging.getLogger(__name__)


def _target(answer: str):
	pr = parse_canonical(answer)
	return pr.first


class DevPointAccuracyCallback(TrainerCallback):
	def __init__(self, adapter, dev_file: str, n: int = 300, every_steps: int = 250, out_file: str | None = None):
		self.adapter = adapter
		self.items = [json.loads(l) for l in Path(dev_file).read_text().splitlines()][:n]
		self.every = every_steps
		self.out_file = Path(out_file) if out_file else None

	def evaluate(self, model) -> dict:
		was_training = model.training
		model.eval()
		hits = unparse = zz = 0
		l1 = []
		for it in self.items:
			text = self.adapter.generate(model, it["image"], it["prompt"])
			pr = parse_canonical(text)
			p = pr.first
			if pr.status is not ParseStatus.OK:
				unparse += 1
			if p is not None:
				zz += p.x == 0 and p.y == 0
				t = _target(it["answer"])
				l1.append(abs(p.x - t.x) + abs(p.y - t.y))
			hits += point_in_mask(p, rle_to_mask(it["mask_rle"]))
		if was_training:
			model.train()
		n = len(self.items)
		return {
			"dev/point_acc": hits / n,
			"dev/unparseable": unparse / n,
			"dev/zero_zero": zz / n,
			"dev/mean_l1_norm": sum(l1) / len(l1) if l1 else float("nan"),
		}

	def _run(self, args, state, model):
		m = self.evaluate(model)
		m["step"] = state.global_step
		logger.info("dev @ step %d: %s", state.global_step, m)
		state.log_history.append(dict(m))
		if self.out_file:
			with self.out_file.open("a") as f:
				f.write(json.dumps(m) + "\n")
		return m

	def on_train_begin(self, args, state, control, model=None, **kw):
		self._run(args, state, model)  # step-0 number = the base model in our training prompt

	def on_step_end(self, args, state, control, model=None, **kw):
		if state.global_step > 0 and state.global_step % self.every == 0:
			m = self._run(args, state, model)
			self._report(args, m, state)

	def on_train_end(self, args, state, control, model=None, **kw):
		if state.global_step % self.every:
			self._run(args, state, model)

	def _report(self, args, m, state):
		if "wandb" in (args.report_to or []):
			try:
				import wandb

				if wandb.run is not None:
					wandb.log({k: v for k, v in m.items() if k != "step"}, step=state.global_step)
			except ImportError:
				pass
