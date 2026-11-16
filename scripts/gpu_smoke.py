"""Driver sanity check, independent of our code: bf16 matmuls on the GPU for a fixed time.

	python scripts/gpu_smoke.py --seconds 30

If this alone freezes the machine, the problem is the NVIDIA driver / kernel / power, not training.
"""

import argparse
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=30)
ap.add_argument("--n", type=int, default=4096)
a = ap.parse_args()
torch.set_num_threads(2)
t0 = time.time()
x = torch.randn(a.n, a.n, device="cuda", dtype=torch.bfloat16)
print(f"cuda init + alloc {time.time() - t0:.1f}s", flush=True)
it, t1 = 0, time.time()
while time.time() - t1 < a.seconds:
	y = x @ x
	torch.cuda.synchronize()
	it += 1
	if it % 200 == 0:
		print(f"{time.time() - t1:5.1f}s  {it * 2 * a.n**3 / (time.time() - t1) / 1e12:.0f} TFLOP/s  mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
print("ok", flush=True)
