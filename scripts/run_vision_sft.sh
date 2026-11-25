#!/usr/bin/env bash
# SFT with the vision encoder + projector trained (preset qwen3vl-2b-vision), same data and schedule as
# SFT v1, then the frozen evals and paired comparisons against SFT v1. Resumes from the newest checkpoint.
#   systemd-run --user --unit=pointvlm-vision-sft --same-dir -p Nice=10 -p CPUQuota=1600% -p MemoryMax=40G \
#     -p Restart=on-failure -p RestartSec=60 -p StartLimitIntervalSec=0 taskset -c 0-7,12-31 scripts/run_vision_sft.sh
set -uo pipefail
cd "$(dirname "$0")/.."
exec > >(tee -a outputs/run_vision_sft.log) 2>&1
PY=.venv/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
RUN=sft-qwen3vl-2b-vis-v1-e2
if [ ! -d outputs/$RUN/merged ]; then
	extra="" ; last=$(ls -d outputs/$RUN/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
	[ -n "$last" ] && extra="vlm_sft.resume_from_checkpoint=$PWD/$last"
	echo "=== train $RUN $(date) $extra"
	$PY -m pointtrain.sft --model qwen3vl-2b-vision --set run_name=$RUN vlm_sft.max_num_samples=60000 training.num_train_epochs=2 \
		training.save_steps=500 training.dataloader_num_workers=0 $extra || { echo "TRAIN FAILED"; exit 1; }
fi
for bench in pointbench where2place refcoco; do
	echo "=== eval $bench $(date)"
	[ -f results/$bench/$RUN/summary.json ] || $PY -m pointeval.run --checkpoint outputs/$RUN/merged --bench $bench --run-name $RUN || { echo "EVAL FAILED $bench"; exit 1; }
	$PY scripts/compare_runs.py $bench sft-qwen3vl-2b-v1-e2 $RUN || true
done
echo "=== all done $(date)"
