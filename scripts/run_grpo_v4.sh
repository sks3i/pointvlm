#!/usr/bin/env bash
# GRPO v4: counting ("Point to all the X", PointArena rule with F1 partial credit) on top of the counting
# SFT checkpoint, mixed with the v3 prompt groups so their gains are kept.
#   systemd-run --user --unit=pointvlm-grpo-v4 --same-dir -p Nice=10 -p CPUQuota=1600% -p MemoryMax=40G scripts/run_grpo_v4.sh
set -uo pipefail
cd "$(dirname "$0")/.."
exec > >(tee -a outputs/run_grpo_v4.log) 2>&1
PY=.venv/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
RUN=grpo-v4
OUT=outputs/$RUN
BASE=$PWD/outputs/sft-count-v1/merged
REWARD=pointtrain.rl_reward:point_or_count_reward

if [ ! -f data/rl/count_v1/train.jsonl ]; then
	echo "=== difficulty scoring counting pool $(date)"
	$PY -m pointtrain.rl_difficulty data/rl/count_v1 --model $BASE --reward $REWARD --max-new-tokens 64 --limit 6000 || { echo "SCORING FAILED"; exit 1; }
fi
[ -f data/rl/v4/train.jsonl ] || $PY -m pointtrain.mix_quota --out data/rl/v4 \
	--source data/rl/count_v1/scored.jsonl:count_coco=2000,count_pixmo=1000 \
	--source data/rl/func_v1/scored.jsonl:func=2000 \
	--source data/rl/v1/train.jsonl:rp_region_ref=2500,rp_object_ref=800,rc_spatial=200,rc_other=100 \
	--source data/rl/synth_v1/scored.jsonl:syn_relation=600,syn_nearest=400 \
	--dev data/rl/v3/dev.jsonl --dev data/rl/count_dev/dev.jsonl || exit 1

extra=""
last=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -n "$last" ] && extra="vlm_grpo.resume_from_checkpoint=$PWD/$last"
if [ ! -d "$OUT/merged" ]; then
	echo "=== train $RUN $(date) $extra"
	$PY -m pointtrain.grpo --set run_name=$RUN model.model_id=$BASE vlm_grpo.train_file=data/rl/v4/train.jsonl vlm_grpo.dev_file=data/rl/v4/dev.jsonl vlm_grpo.dev_n=1200 \
		vlm_grpo.reward_fn=$REWARD vlm_grpo.max_new_tokens=64 vlm_grpo.micro_batch_size=8 $extra "$@" || { echo "TRAIN FAILED"; exit 1; }
fi
$PY -m pointtrain.count_check --out data/rl/count_dev --model $OUT/merged --tag v4 || true
for bench in pointbench where2place refcoco; do
	echo "=== eval $bench $(date)"
	$PY -m pointeval.run --checkpoint "$OUT/merged" --bench $bench --run-name $RUN || echo "EVAL FAILED $bench"
	$PY scripts/compare_runs.py $bench sft-qwen3vl-2b-v1-e2 $RUN || true
	$PY scripts/compare_runs.py $bench grpo-v3 $RUN || true
done
echo "=== all done $(date)"
touch outputs/$RUN.done
