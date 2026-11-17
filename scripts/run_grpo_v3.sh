#!/usr/bin/env bash
# GRPO v3: function-style prompts (pointtrain.synth_function) + RoboPoint free space at v1's share +
# the v2 relation templates, with partial credit for near misses (rl_reward.point_reward_shaped).
# From the SFT checkpoint. Score -> mix -> train -> frozen evals -> paired comparisons.
#   systemd-run --user --unit=pointvlm-grpo-v3 --same-dir -p Nice=10 -p CPUQuota=1600% -p MemoryMax=40G scripts/run_grpo_v3.sh
set -uo pipefail
cd "$(dirname "$0")/.."
exec > >(tee -a outputs/run_grpo_v3.log) 2>&1
PY=.venv/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
RUN=grpo-v3
OUT=outputs/$RUN

[ -f data/rl/func_v1/pool.jsonl ] || $PY -m pointtrain.synth_function --out data/rl/func_v1 || exit 1
if [ ! -f data/rl/func_v1/train.jsonl ]; then
	echo "=== difficulty scoring func $(date)"
	$PY -m pointtrain.rl_difficulty data/rl/func_v1 || { echo "SCORING FAILED"; exit 1; }
fi
[ -f data/rl/v3/train.jsonl ] || $PY -m pointtrain.mix_quota --out data/rl/v3 \
	--source data/rl/func_v1/scored.jsonl:func=3500 \
	--source data/rl/v1/train.jsonl:rp_region_ref=3000,rp_object_ref=1000,rc_spatial=300,rc_other=200 \
	--source data/rl/synth_v1/scored.jsonl:syn_relation=800,syn_nearest=600,syn_extreme=400,syn_middle=300 \
	--dev data/rl/v2/dev.jsonl --dev data/rl/func_v1/dev.jsonl || exit 1

extra=""
last=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -n "$last" ] && extra="vlm_grpo.resume_from_checkpoint=$PWD/$last"
if [ ! -d "$OUT/merged" ]; then
	echo "=== train $RUN $(date) $extra"
	$PY -m pointtrain.grpo --set run_name=$RUN vlm_grpo.train_file=data/rl/v3/train.jsonl vlm_grpo.dev_file=data/rl/v3/dev.jsonl vlm_grpo.dev_n=1000 \
		vlm_grpo.reward_fn=pointtrain.rl_reward:point_reward_shaped vlm_grpo.micro_batch_size=8 $extra "$@" || { echo "TRAIN FAILED"; exit 1; }
fi

for bench in pointbench where2place refcoco; do
	echo "=== eval $bench $(date)"
	$PY -m pointeval.run --checkpoint "$OUT/merged" --bench $bench --run-name $RUN || echo "EVAL FAILED $bench"
	$PY scripts/compare_runs.py $bench sft-qwen3vl-2b-v1-e2 $RUN || true
	$PY scripts/compare_runs.py $bench grpo-v1 $RUN || true
done
echo "=== all done $(date)"
touch outputs/$RUN.done
