#!/usr/bin/env bash
# Counting SFT on top of grpo-v3 (best checkpoint), then the counting pass@8 check and the frozen evals.
#   systemd-run --user --unit=pointvlm-count-sft --same-dir -p Nice=10 -p CPUQuota=1600% -p MemoryMax=40G scripts/run_count_sft.sh
set -uo pipefail
cd "$(dirname "$0")/.."
exec > >(tee -a outputs/run_count_sft.log) 2>&1
PY=.venv/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
RUN=sft-count-v1
[ -f data/sft/count_v1/train.jsonl ] || $PY -m pointtrain.build_count_sft --out data/sft/count_v1 || exit 1
if [ ! -d outputs/$RUN/merged ]; then
	extra=""
	last=$(ls -d outputs/$RUN/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
	[ -n "$last" ] && extra="vlm_sft.resume_from_checkpoint=$PWD/$last"
	echo "=== train $RUN $(date) $extra"
	$PY -m pointtrain.sft --model qwen3vl-2b --set run_name=$RUN model.model_id=$PWD/outputs/grpo-v3/merged \
		vlm_sft.train_file=data/sft/count_v1/train.jsonl training.num_train_epochs=1 pointvlm.dev_n=0 \
		training.save_steps=500 training.extra_training_args.dataloader_persistent_workers=false training.extra_training_args.dataloader_pin_memory=false training.dataloader_num_workers=0 $extra || { echo "TRAIN FAILED"; exit 1; }
fi
echo "=== count check $(date)"
$PY -m pointtrain.count_check --out data/rl/count_dev --model outputs/$RUN/merged --tag count_sft || echo "CHECK FAILED"
for bench in pointbench where2place refcoco; do
	echo "=== eval $bench $(date)"
	$PY -m pointeval.run --checkpoint outputs/$RUN/merged --bench $bench --run-name $RUN || echo "EVAL FAILED $bench"
	$PY scripts/compare_runs.py $bench grpo-v3 $RUN || true
done
echo "=== all done $(date)"
