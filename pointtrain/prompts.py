"""Training prompts.

Single-target examples mostly use the frozen eval prompt itself (pointeval.adapters.CANONICAL_PROMPT):
the eval asks in exactly that wording, so training on it removes any prompt-format gap. That is not
leakage; the leakage rules are about images and objects, not the instruction template. A minority
use paraphrases so the model learns "point at the referent" rather than one exact string.

Multi-target examples (PixMo counting, RoboPoint point sets) use a DIFFERENT prompt that asks for
every instance. They must never use the canonical prompt, which says "exactly one point": mixing
"one point" wording with multi-point answers would teach the model to sometimes emit several points,
and the eval scores only the first.
"""

from __future__ import annotations

import hashlib

from pointeval.adapters import CANONICAL_PROMPT

_FORMAT = (
	'Answer with exactly one point in the format <point x="X" y="Y">, where X and Y are integers '
	"from 0 to 1000 giving the position as a fraction of the image width and height (0,0 is the top-left corner)."
)
SINGLE_PARAPHRASES = [
	"Where is {query}? " + _FORMAT,
	"Click on {query}. " + _FORMAT,
	"Show me {query} in the image. " + _FORMAT,
]
MULTI_PROMPT = (
	"Point to each instance of \"{query}\" in the image. Answer with one <point x=\"X\" y=\"Y\"> per instance, separated by spaces, where X "
	"and Y are integers from 0 to 1000 giving the position as a fraction of the image width and height "
	"(0,0 is the top-left corner)."
)
MULTI_REGION_PROMPT = (
	"{query} Answer with several points, each in the format <point x=\"X\" y=\"Y\">, separated by spaces, where X "
	"and Y are integers from 0 to 1000 giving the position as a fraction of the image width and height "
	"(0,0 is the top-left corner)."
)


def _u(key: str) -> float:
	"""Deterministic uniform in [0, 1) from a string key (no RNG state to carry around)."""
	return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) / 16**12


def single_prompt(query: str, key: str, canonical_frac: float = 0.7) -> str:
	if _u("prompt:" + key) < canonical_frac:
		return CANONICAL_PROMPT.format(query=query)
	i = int(_u("para:" + key) * len(SINGLE_PARAPHRASES))
	return SINGLE_PARAPHRASES[i].format(query=query)
