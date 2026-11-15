#!/usr/bin/env bash
# Train Qwen3-VL-2B then LFM2-VL-1.6B on data/sft/v1 (2 epochs, 60k-row subsample), then run the
# frozen evals on each final checkpoint. Run detached from any terminal/app session:
#   systemd-run --user --unit=pointvlm-sft --same-dir scripts/run_sft_qwen_lfm2.sh   (log: journalctl --user -u pointvlm-sft, and outputs/run_sft_qwen_lfm2.log)
set -uo pipefail
cd "$(dirname "$0")/.."
exec > >(tee -a outputs/run_sft_qwen_lfm2.log) 2>&1
PY=.venv/bin/python
COMMON="vlm_sft.max_num_samples=60000 training.num_train_epochs=2 training.save_steps=1000"

run_model () {
	local preset=$1 ckpt=$2
	local out="outputs/sft-$preset-v1-e2" extra=""
	# Resume from the newest checkpoint if a previous attempt left one.
	local last=$(ls -d "$out"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
	[ -n "$last" ] && extra="vlm_sft.resume_from_checkpoint=$PWD/$last"
	echo "=== train $preset $(date) ${extra}"
	$PY -m pointtrain.sft --model "$preset" --set $COMMON run_name="sft-$preset-v1-e2" $extra || { echo "TRAIN FAILED $preset"; return 1; }
	for bench in refcoco where2place pointbench; do
		echo "=== eval $preset $bench $(date)"
		$PY -m pointeval.run --checkpoint "$ckpt" --bench "$bench" --run-name "sft-$preset-v1-e2" || echo "EVAL FAILED $preset $bench"
	done
}

if [ ! -f data/sft/v1/leak_report.json ]; then
	if [ ! -f data/sft/v1/meta.json ]; then
		echo "=== build data $(date)"
		$PY -m pointtrain.build --out data/sft/v1 || { echo "BUILD FAILED"; exit 1; }
	fi
	$PY -m pointtrain.leakcheck data/sft/v1 || { echo "LEAKCHECK FAILED"; exit 1; }
	if [ -s data/sft/v1/flagged_images.txt ]; then
		echo "=== rebuild without $(wc -l < data/sft/v1/flagged_images.txt) flagged images"
		cp data/sft/v1/flagged_images.txt data/sft/flagged_v1.txt
		$PY -m pointtrain.build --out data/sft/v1 --exclude-images data/sft/flagged_v1.txt || exit 1
	fi
fi

run_model qwen3vl-2b outputs/sft-qwen3vl-2b-v1-e2/merged
run_model lfm2vl-1.6b outputs/sft-lfm2vl-1.6b-v1-e2
echo "=== all done $(date)"
touch outputs/run_sft_qwen_lfm2.done
