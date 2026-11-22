#!/usr/bin/env bash
# The Qwen recipe on LFM2-VL-1.6B, end to end. Every stage is skipped if its output exists, so re-running resumes.
#   A. GRPO "v3" recipe from the LFM2 SFT checkpoint: function prompts + free space + relations, shaped reward
#   B. counting SFT (LoRA) on top of A, then GRPO "v4" recipe with the counting reward
# Pass rates are model-specific, so every pool is rescored with LFM2 (6k prompts each) before mixing.
#   systemd-run --user --unit=pointvlm-lfm2 --same-dir -p Nice=10 -p CPUQuota=1600% -p MemoryMax=40G scripts/run_lfm2.sh
set -uo pipefail
cd "$(dirname "$0")/.."
exec > >(tee -a outputs/run_lfm2.log) 2>&1
PY=.venv/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
CFG=configs/grpo/lfm2.yaml
SFT=$PWD/outputs/sft-lfm2vl-1.6b-v1-e2
D=data/rl/lfm2

evals () {  # $1 run name, $2 checkpoint
	for bench in pointbench where2place refcoco; do
		echo "=== eval $1 $bench $(date)"
		[ -f results/$bench/$1/summary.json ] || $PY -m pointeval.run --checkpoint "$2" --bench $bench --run-name $1 || echo "EVAL FAILED $1 $bench"
		$PY scripts/compare_runs.py $bench sft-lfm2vl-1.6b-v1-e2 $1 || true
	done
}
grpo () {  # $1 run, then overrides
	local run=$1; shift
	local extra="" last=$(ls -d outputs/$run/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
	[ -n "$last" ] && extra="vlm_grpo.resume_from_checkpoint=$PWD/$last"
	if [ ! -d outputs/$run/merged ]; then
		echo "=== train $run $(date) $extra"
		$PY -m pointtrain.grpo --config $CFG --set run_name=$run vlm_grpo.micro_batch_size=8 vlm_grpo.generation_batch_size=24 "$@" $extra || { echo "TRAIN FAILED $run"; exit 1; }
	fi
}

# --- A: v3 recipe ---------------------------------------------------------------------------
for p in v1 func synth; do
	[ -f $D/$p/train.jsonl ] || { echo "=== score $p $(date)"; $PY -m pointtrain.rl_difficulty $D/$p --model $SFT --limit 6000 || { echo "SCORING FAILED $p"; exit 1; }; }
done
[ -f $D/a/train.jsonl ] || $PY -m pointtrain.mix_quota --out $D/a \
	--source $D/func/scored.jsonl:func=3500 \
	--source $D/v1/scored.jsonl:rp_region_ref=3000,rp_object_ref=1000,rc_spatial=300,rc_other=200 \
	--source $D/synth/scored.jsonl:syn_relation=800,syn_nearest=600,syn_extreme=400,syn_middle=300 \
	--dev data/rl/v2/dev.jsonl --dev data/rl/func_v1/dev.jsonl || exit 1
grpo lfm2-grpo-a model.model_id=$SFT vlm_grpo.train_file=$D/a/train.jsonl vlm_grpo.dev_file=$D/a/dev.jsonl vlm_grpo.dev_n=1000 vlm_grpo.reward_fn=pointtrain.rl_reward:point_reward_shaped
evals lfm2-grpo-a outputs/lfm2-grpo-a/merged

# --- B: counting SFT, then v4 recipe ------------------------------------------------------------
A=$PWD/outputs/lfm2-grpo-a/merged
if [ ! -d outputs/lfm2-sft-count/merged ]; then
	ex="" ; last=$(ls -d outputs/lfm2-sft-count/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
	[ -n "$last" ] && ex="vlm_sft.resume_from_checkpoint=$PWD/$last"
	echo "=== count SFT $(date) $ex"
	$PY -m pointtrain.sft --model lfm2vl-1.6b-lora --set run_name=lfm2-sft-count model.model_id=$A \
		vlm_sft.train_file=data/sft/count_v1/train.jsonl training.num_train_epochs=1 pointvlm.dev_n=0 training.save_steps=500 \
		training.extra_training_args.dataloader_persistent_workers=false training.extra_training_args.dataloader_pin_memory=false training.dataloader_num_workers=0 $ex || { echo "COUNT SFT FAILED"; exit 1; }
fi
C=$PWD/outputs/lfm2-sft-count/merged
$PY -m pointtrain.count_check --out data/rl/count_dev --model $A --tag lfm2_a || true
$PY -m pointtrain.count_check --out data/rl/count_dev --model $C --tag lfm2_count_sft || true
[ -f $D/count/train.jsonl ] || { echo "=== score count $(date)"; $PY -m pointtrain.rl_difficulty $D/count --model $C --reward pointtrain.rl_reward:point_or_count_reward --max-new-tokens 64 --limit 6000 || { echo "SCORING FAILED count"; exit 1; }; }
[ -f $D/b/train.jsonl ] || $PY -m pointtrain.mix_quota --out $D/b \
	--source $D/count/scored.jsonl:count_coco=2000,count_pixmo=1000 \
	--source $D/func/scored.jsonl:func=2000 \
	--source $D/v1/scored.jsonl:rp_region_ref=2500,rp_object_ref=800,rc_spatial=200,rc_other=100 \
	--source $D/synth/scored.jsonl:syn_relation=600,syn_nearest=400 \
	--dev $D/a/dev.jsonl --dev data/rl/count_dev/dev.jsonl || exit 1
grpo lfm2-grpo-b model.model_id=$C vlm_grpo.train_file=$D/b/train.jsonl vlm_grpo.dev_file=$D/b/dev.jsonl vlm_grpo.dev_n=1200 vlm_grpo.reward_fn=pointtrain.rl_reward:point_or_count_reward vlm_grpo.max_new_tokens=64
$PY -m pointtrain.count_check --out data/rl/count_dev --model outputs/lfm2-grpo-b/merged --tag lfm2_b || true
evals lfm2-grpo-b outputs/lfm2-grpo-b/merged
echo "=== all done $(date)"
