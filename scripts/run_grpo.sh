#!/usr/bin/env bash
# Score the RL pool by SFT pass rate, train GRPO, then run the frozen evals and paired comparisons.
# Run detached from any terminal/app session:
#   systemd-run --user --unit=pointvlm-grpo --same-dir scripts/run_grpo.sh   (log: outputs/run_grpo.log)
# Re-running resumes: scoring appends to scored.jsonl, training resumes from the newest checkpoint.
set -uo pipefail
cd "$(dirname "$0")/.."
exec > >(tee -a outputs/run_grpo.log) 2>&1
PY=.venv/bin/python
DATA=data/rl/v1
RUN=${RUN:-grpo-v1}
OUT=outputs/$RUN

if [ ! -f $DATA/train.jsonl ]; then
	echo "=== difficulty scoring $(date)"
	$PY -m pointtrain.rl_difficulty $DATA || { echo "SCORING FAILED"; exit 1; }
fi

extra=""
last=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
[ -n "$last" ] && extra="vlm_grpo.resume_from_checkpoint=$PWD/$last"
if [ ! -d "$OUT/merged" ]; then
	echo "=== train $RUN $(date) $extra"
	$PY -m pointtrain.grpo --set run_name=$RUN $extra "$@" || { echo "TRAIN FAILED"; exit 1; }
fi

for bench in pointbench where2place refcoco; do
	echo "=== eval $bench $(date)"
	$PY -m pointeval.run --checkpoint "$OUT/merged" --bench $bench --run-name $RUN || echo "EVAL FAILED $bench"
	$PY scripts/compare_runs.py $bench sft-qwen3vl-2b-v1-e2 $RUN || true
done
echo "=== all done $(date)"
touch outputs/$RUN.done
