#!/usr/bin/env bash
# vLLM baselines (scripts/eval_vllm.py) with the protections every GPU job on this box uses:
#   systemd-run --user --unit=pointvlm-vllm-32b --same-dir -p Nice=10 -p CPUQuota=1600% -p MemoryMax=40G \
#     -p Restart=on-failure -p RestartSec=60 scripts/run_vllm_eval.sh <eval_vllm.py args>
# CPUs 8-11 (6.0 GHz favored cores of the i9-14900KF) crash under load, so the job is pinned off them.
# Resumable: finished items are in results/<bench>/<run>/predictions.partial.jsonl.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false VLLM_ENABLE_V1_MULTIPROCESSING=0
LOG=${LOG:-outputs/run_vllm_eval.log}
echo "=== $(date) $*" >> "$LOG"
exec taskset -c 0-7,12-31 .venv-vllm/bin/python scripts/eval_vllm.py "$@" >> "$LOG" 2>&1
