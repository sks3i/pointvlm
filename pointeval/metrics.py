"""Accuracy with honest error bars, and paired comparison between two runs.

Why error bars matter here: accuracy on n items is a binomial proportion with
standard error sqrt(p(1-p)/n). At p ~ 0.5:
	n = 2000 (RefCOCO sample) -> SE ~ 1.1 pp -> 95% CI ~ +/- 2.2 pp
	n = 100  (Where2Place)    -> SE ~ 5.0 pp -> 95% CI ~ +/- 9.8 pp
So a 3-point gain on Where2Place is noise; on RefCOCO it is (barely) signal.
Comparisons between two checkpoints are PAIRED (same items), so we use McNemar's
exact test on the discordant pairs, which is far tighter than comparing two CIs.
"""

from __future__ import annotations

import math


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
	"""Wilson score interval for k successes out of n. Well behaved near 0 and 1, unlike p +/- z*SE."""
	if n == 0:
		return (0.0, 0.0)
	p = k / n
	denom = 1 + z * z / n
	centre = (p + z * z / (2 * n)) / denom
	half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
	return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(hits: list[bool]) -> dict:
	n, k = len(hits), sum(hits)
	lo, hi = wilson_ci(k, n)
	return {"n": n, "hits": k, "acc": k / n if n else 0.0, "ci95": [lo, hi]}


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
	"""Two-sided exact McNemar test for paired binary outcomes (same items, two models).

	Only discordant items carry information: b01 = A wrong & B right, b10 = A right & B wrong.
	Under H0 (no difference) each discordant item is a fair coin, so b01 ~ Binomial(b01+b10, 0.5).
	"""
	if len(a) != len(b):
		raise ValueError("paired test needs the same items in the same order")
	b01 = sum((not x) and y for x, y in zip(a, b))
	b10 = sum(x and (not y) for x, y in zip(a, b))
	m = b01 + b10
	if m == 0:
		return {"b01": 0, "b10": 0, "p_value": 1.0}
	tail = sum(math.comb(m, i) for i in range(0, min(b01, b10) + 1)) / 2**m
	return {"b01": b01, "b10": b10, "p_value": min(1.0, 2 * tail)}


def latency_stats(ms: list[float]) -> dict:
	if not ms:
		return {}
	s = sorted(ms)

	def q(p: float) -> float:
		i = (len(s) - 1) * p
		lo, hi = math.floor(i), math.ceil(i)
		return s[lo] + (s[hi] - s[lo]) * (i - lo)

	return {"n": len(s), "mean_ms": sum(s) / len(s), "p50_ms": q(0.5), "p90_ms": q(0.9), "p99_ms": q(0.99)}
