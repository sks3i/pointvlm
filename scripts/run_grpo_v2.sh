#!/usr/bin/env bash
# GRPO v2: synthetic COCO error-targeted prompts (pointtrain.synth_coco) + 4k learnable v1 prompts,
# from the SFT checkpoint. Score -> mix -> train -> frozen evals -> paired comparisons.
#   systemd-run --user --unit=pointvlm-grpo-v2 --same-dir -p Nice=10 -p CPUQuota=1600% -p MemoryMax=40G scripts/run_grpo_v2.sh   (log: outputs/run_grpo_v2.log)
# The unit limits keep the desktop responsive and kill the job (not the PC) on a CPU or RAM runaway.
set -uo pipefail
cd "$(dirname "$0")/.."
exec > >(tee -a outputs/run_grpo_v2.log) 2>&1
PY=.venv/bin/python
# Cap CPU thread pools (torch/OpenMP, MKL, tokenizers) so preprocessing cannot saturate all 32 cores.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
RUN=grpo-v2
OUT=outputs/$RUN

if [ ! -f data/rl/synth_v1/train.jsonl ]; then
	echo "=== difficulty scoring synth $(date)"
	$PY -m pointtrain.rl_difficulty data/rl/synth_v1 || { echo "SCORING FAILED"; exit 1; }
fi
[ -f data/rl/v2/train.jsonl ] || $PY -m pointtrain.mix_rl --out data/rl/v2 --main data/rl/synth_v1 --extra data/rl/v1 --extra-n 4000 || exit 1

extra=""
last=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -n "$last" ] && extra="vlm_grpo.resume_from_checkpoint=$PWD/$last"
if [ ! -d "$OUT/merged" ]; then
	echo "=== train $RUN $(date) $extra"
	$PY -m pointtrain.grpo --set run_name=$RUN vlm_grpo.train_file=data/rl/v2/train.jsonl vlm_grpo.dev_file=data/rl/v2/dev.jsonl vlm_grpo.dev_n=704 $extra "$@" || { echo "TRAIN FAILED"; exit 1; }
fi

for bench in pointbench where2place refcoco; do
	echo "=== eval $bench $(date)"
	$PY -m pointeval.run --checkpoint "$OUT/merged" --bench $bench --run-name $RUN || echo "EVAL FAILED $bench"
	$PY scripts/compare_runs.py $bench sft-qwen3vl-2b-v1-e2 $RUN || true
	$PY scripts/compare_runs.py $bench grpo-v1 $RUN || true
done
echo "=== all done $(date)"
touch outputs/$RUN.done
