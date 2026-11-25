# Blog part 1: facts on disk (Qwen3-VL-2B pointing)

Collected 2026-11-24 from configs, logs, data meta files and frozen-eval summaries in ~/Projects/pointvlm and ~/Projects/localtrainer (branches `sft` / `vlm-grpo`). No new runs. Frozen eval = docs/EVAL_SPEC.md v1, scoring v1, parser sha af5a37222d02c46a, greedy, batch 1, max_new_tokens 64 (Molmo 128).

## 1. SFT recipe: outputs/sft-qwen3vl-2b-v1-e2

- Base: `Qwen/Qwen3-VL-2B-Instruct`, bf16, sdpa attention, gradient checkpointing.
- Trained modules: LoRA on the LLM only, r 16, alpha 32, dropout 0.05, on q/k/v/o and gate/up/down in every `model.language_model.layers.*`. Vision tower and merger/projector frozen, no modules_to_save. Merged to `outputs/sft-qwen3vl-2b-v1-e2/merged`.
- Data: `data/sft/v1/train.jsonl` (built by pointtrain.build, 241,292 rows; pinned HF revisions in data/sft/v1/meta.json). The run used `max_num_samples=60000`, i.e. the **first 60,000 rows** (the file is pre-shuffled by hash). Rows per source in that slice:

| Source | Rows in the 60k slice | of which multi-point | Full built file |
|---|---|---|---|
| PixMo-Points (allenai/pixmo-points) | 14,883 | 5,631 | 59,117 |
| RefCOCO (mask -> point) | 10,120 | 0 | 40,491 |
| RefCOCO+ (mask -> point) | 9,794 | 0 | 40,067 |
| RefCOCOg (mask -> point) | 9,728 | 0 | 39,233 |
| RoboPoint object_ref | 7,827 | 2,338 | 31,383 |
| RoboPoint region_ref | 7,648 | 2,219 | 31,007 |
| **Total** | **60,000** | 10,188 | 241,292 |

- RefCOCO-family masks -> points: label_mode "deepest" = the mask pixel farthest from the mask edge (distance-transform maximum; median depth 52 px), up to 2 sentences per ref. Target written as `<point x="X" y="Y">` with x,y in 0..1000, pixel centres.
- Prompts: single-target rows use the frozen canonical prompt 70% of the time, 3 paraphrases otherwise; multi-target rows (PixMo counting, RoboPoint sets) use a separate "point to each instance" prompt, never the canonical one.
- Leakage: all RefCOCO/+/g val/test images and the 1,210 eval-manifest images dropped (6,549 images); pixel near-duplicate audit dropped 1,130 rows; 481 flagged images (PixMo duplicates of Point-Bench etc.) excluded (data/sft/flagged_v1.txt). Dev = 500 rows from 1.5% of RefCOCO-family train images.
- Schedule: 2 epochs over the 60k rows = 7,500 optimizer steps; per-device batch 4 x grad-accum 4 = effective batch 16; LR 5e-5 cosine, warmup 3%, weight decay 0, grad clip 1.0, seed 42. max_image_side 1024 (caps PixMo web images), max_seq_length 2048.
- Wall time on the 4090: ~4.6 h of training (steps 0-2000 in 1 h 22 min in the first attempt, which was interrupted; resumed from checkpoint-2000, steps 2000-7500 took 11,650 s = 3.2 h). Final train loss 0.611. Peak memory ~14 GB (bs 4, measured in the smoke run; not logged per step for SFT).
- Dev point accuracy (300 dev rows, greedy): 87.3% at step 7500. Curve: outputs/sft-qwen3vl-2b-v1-e2/dev_metrics.jsonl.

## 2. GRPO recipe (localtrainer `trainer_type: vlm_grpo`)

Code: ~/Projects/localtrainer/train_utils/trainers/vlm_grpo_trainer.py, math in train_utils/grpo/math.py; pointvlm configs/grpo/base.yaml + scripts/run_grpo{,_v2,_v3,_v4}.sh.

Shared by v1-v4:
- Policy: fresh LoRA (r 16, alpha 32, **dropout 0**) on the LLM attention + MLP projections of the merged starting checkpoint; vision tower and merger frozen. Reference model for KL = same weights with the adapter disabled (i.e. exactly the starting checkpoint). Merged at the end.
- Group size G = 8 samples per prompt; 16 prompts per step = 128 completions per optimizer step; 600 steps; num_iterations 1 (one gradient step per batch, so the PPO ratio is 1 and clip_frac was 0.0 throughout).
- Sampling: temperature 1.0, top_p 1.0, top_k 0 (pure sampling).
- Advantages: group-normalised, (r - mean_group) / (std_group + 1e-4) with population std (`scale_rewards: group`); groups with zero reward std skipped (`skip_zero_std_groups`).
- Loss: clipped surrogate min(rho A, clip(rho, 1-0.2, 1+0.28) A) (asymmetric "clip-higher" eps_low 0.2 / eps_high 0.28) + beta x KL, beta = 0.02, KL estimator k3 = exp(x) - x - 1 with x = log pi_ref - log pi_theta, per token; token-level aggregation (`loss_aggregation: token`).
- Optimiser: LR 2e-5 constant after 10 warmup steps, weight decay 0, grad clip 1.0, bf16, gradient checkpointing, seed 42. Checkpoint every 50 steps (resumable), dev eval every 50 steps.
- max_image_side 1024.

| Run | Start checkpoint | Train prompts | Reward fn | max_new_tokens | gen batch / micro batch | Dev set | Wall time (train) | Mean s/step | Peak mem (torch max allocated) |
|---|---|---|---|---|---|---|---|---|---|
| v1 | SFT merged | data/rl/v1 train 15,483 | point_reward (binary) | 24 | 64 / 16 | 400 | 2.0 h (11:47-13:48, 2026-11-15) | 11.6 | 11.6 GB |
| v2 | SFT merged | data/rl/v2 train 11,491 | point_reward (binary) | 24 | 64 / 16 | 704 | 1.9 h (final launch 19:19-21:14; two earlier launches froze the PC at step ~1) | 10.7 | 11.5 GB |
| v3 | SFT merged | data/rl/v3 train 10,074 | point_reward_shaped | 24 | 64 / 8 | 890 | 2.2 h (00:14-02:24, 2026-11-17) | 11.9 | 11.6 GB |
| v4 | sft-count-v1 merged | data/rl/v4 train 8,550 | point_or_count_reward | 64 | 24 / 8 | 1,190 | 3.2 h (08:40-11:52, 2026-11-21, incl. count check; first launch OOM at step 19 with gen batch 64) | 16.5 | 10.7 GB |

(s/step is the trainer's own t_step mean and excludes dev evals. 600 steps x 16 prompts = 9,600 prompts drawn per run (76,800 completions), i.e. under one epoch of every train file above except v4's 8,550.)

Difficulty scoring (pass-rate filter) wall time: v1 pool 28k ~1.7 h (~3 prompts/s); counting pool 6k for v4 ~6.3 h at 64 tokens (02:07-08:27).

### Rewards, exactly as coded (pointtrain/rl_reward.py; uses the frozen eval's own parse_canonical and point_in_mask)

- **v1, v2 `point_reward`**: reward = 1 iff the answer parses with the canonical parser AND has exactly one point AND that point is inside the target region; else 0. No separate format bonus. Region: RefCOCO family = the COCO mask; RoboPoint (no masks, only GT point sets) = convex hull of the GT points grown by 2% of the image diagonal, plus a disc of that radius around each GT point (collinear sets: discs only).
- **v3 `point_reward_shaped`**: same gate (parse OK + exactly one point, else 0). Hit -> 1.0. Miss -> 0.5 x max(0, 1 - d / 0.03), where d = distance from the point to the region as a fraction of the image diagonal (mask: Euclidean distance transform of the mask complement at that pixel; RoboPoint: distance to the nearest GT point minus the 2% margin, floored at 0). So partial credit only within 3% of the diagonal, never more than 0.5. Dev accuracy uses the separate `hit` key, not the shaped reward.
- **v4 `point_or_count_reward`**: non-counting rows -> point_reward_shaped. Counting rows (target type masks / count_points) -> PointArena counting rule via count_check.count_score: hit iff number of predicted points == number of targets AND every target is matched. Matching: masks targets = a target counts as matched if any predicted point falls inside its mask (one point can cover overlapping masks); point targets (PixMo) = Hungarian assignment (scipy linear_sum_assignment) on pixel distance, matched if assigned distance <= 0.05 x image diagonal (RADIUS_FRAC). Reward: hit -> 1.0; no points or 0 matched -> 0; else 0.5 x F1 with precision = matched / n_pred, recall = matched / n_true.

### Pass-rate filter (pointtrain/rl_difficulty.py)

- k = 8 samples per prompt at temperature 1.0 (same as training) from the model each run starts from; pass = full hits only (`hit` key, not partial credit). Kept band 0 < p_hat < 1 (at least 1 and at most 7 of 8 correct). Mixing (pointtrain/mix_quota.py) applies the same 0 < p < 1 rule per group, then quotas.
- Scored with the SFT model: v1 pool, synth_v1, func_v1. Scored with sft-count-v1: count_v1 (only the first 6,000 of 10,774 pool rows, --limit 6000). v4 reused the SFT-scored v1/func/synth rows without rescoring.

| Pool | Scored | Kept (0<p<1) | Per group kept / scored, mean pass |
|---|---|---|---|
| data/rl/v1 (RefCOCO family + RoboPoint) | 28,000 | 15,483 (55%) | rc_spatial 4,483/12,000 (p 0.82; 7,044 were 8/8), rc_other 1,303/4,000 (0.86), rp_object_ref 5,050/6,000 (0.59), rp_region_ref 4,647/6,000 (0.48) |
| data/rl/synth_v1 (COCO synthetic relations etc.) | 15,307 | 7,491 (49%) | ordinal 1,700/2,587 (0.39), nearest 1,335/2,607 (0.62), extreme 1,151/2,575 (0.71), typo 999/2,019, depth 939/2,618, relation 774/1,569 (0.71), middle 560/1,271, between 33/61 |
| data/rl/func_v1 (COCO function prompts) | 9,033 | 4,017 (44%) | func mean pass 0.78; 4,448 were 8/8, 568 were 0/8 |
| data/rl/count_v1 (counting) | 6,000 | 2,172 (36%) | count_coco 950 (pass 0.28; 1,891 were 0/8), count_pixmo 1,222 (0.24) |

Per-run training sets:
- v1: data/rl/v1/train.jsonl, all 15,483 kept rows.
- v2: data/rl/v2/train.jsonl 11,491 = all 7,491 kept synth_v1 rows + 4,000 kept v1 rows (rc_spatial 1,118, rp_object 1,344, rp_region 1,206, rc_other 332).
- v3: 10,074 = func 3,500, rp_region_ref 3,000, rp_object_ref 1,000, rc_spatial 300, rc_other 200, syn_relation 774 (quota 800), syn_nearest 600, syn_extreme 400, syn_middle 300.
- v4: 8,550 = count_coco 950 (quota 2,000, only 950 learnable), count_pixmo 1,000, func 2,000, rp_region_ref 2,500, rp_object_ref 800, rc_spatial 200, rc_other 100, syn_relation 600, syn_nearest 400.

### Counting SFT (outputs/sft-count-v1, between v3 and v4)

LoRA preset as main SFT (r 16, alpha 32, dropout 0.05, LLM only, LR 5e-5 cosine, effective batch 16) on top of grpo-v3/merged, 1 epoch over data/sft/count_v1 (18,774 rows: count_coco 5,774, count_pixmo 5,000, single-target replay 8,000 = refcocog 1,624, refcoco 1,600, refcoco+ 1,540, pixmo 1,509, robopoint region 865 / object 862). ~36 min. Run with dataloader_pin_memory=false and num_workers=0 after a native crash.

### Training curves on disk

- Per step (reward, kl, frac_zero_std_groups, entropy, completion_len, clip_frac, grad_norm, lr, t_gen, t_step, mem_gb): outputs/grpo-v{1,2,3,4}/metrics.jsonl
- Dev accuracy every 50 steps (overall + per group): outputs/grpo-v{1,2,3,4}/dev_metrics.jsonl; SFT: outputs/sft-qwen3vl-2b-v1-e2/dev_metrics.jsonl
- Sampled completions: outputs/grpo-v{1,2,3,4}/samples.jsonl
- Counting check per checkpoint: data/rl/count_dev/summary_{sft,v3,count_sft,v4}.json; affordance pass@8: outputs/affordance_check/summary.json

Mean of first 20 vs last 20 steps:

| Run | reward | KL (k3) | frac zero-std groups | entropy | dev acc step 0 -> 600 |
|---|---|---|---|---|---|
| v1 | 0.59 -> 0.78 | 0.003 -> 0.171 | 0.20 -> 0.75 | 0.64 -> 0.47 | 80.0 -> 81.5 |
| v2 | 0.54 -> 0.75 | 0.002 -> 0.166 | 0.22 -> 0.78 | 0.62 -> 0.49 | 72.9 -> 76.4 |
| v3 | 0.63 -> 0.81 (shaped) | 0.003 -> 0.157 | 0.19 -> 0.70 | 0.62 -> 0.41 | 74.7 -> 77.8 (hit) |
| v4 | 0.66 -> 0.83 (shaped) | 0.002 -> 0.129 | 0.20 -> 0.77 | 0.62 -> 0.41 | 65.2 -> 67.6 (hit) |

Dev sets differ per run, so dev numbers are not comparable across rows.

## 3. Frozen eval results (point-in-mask accuracy, %)

| Model | RefCOCO | Where2Place | Point-Bench | PB affordable | PB counting | PB reasoning | PB spatial | PB steerable |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B base, native (point_2d JSON) | 86.2 | 44.0 | 51.9 | 71.2 | 21.5 | 54.9 | 62.6 | 49.0 |
| Qwen3-VL-2B base, canonical prompt | 80.6 | 40.0 | 48.9 | 73.2 | 13.3 | 55.4 | 57.9 | 44.4 |
| SFT (sft-qwen3vl-2b-v1-e2) | 91.1 | 51.0 | 58.6 | 82.3 | 16.4 | 66.8 | 72.8 | 54.5 |
| GRPO v1 | 92.8 | 65.0 | 57.5 | 79.3 | 17.9 | 66.8 | 71.8 | 51.5 |
| GRPO v2 | 92.4 | 61.0 | 58.7 | 81.8 | 17.9 | 65.8 | 70.8 | 57.1 |
| GRPO v3 | 92.7 | 63.0 | 59.9 | 82.3 | 21.5 | 67.9 | 72.8 | 54.5 |
| Counting SFT (sft-count-v1, on v3) | 91.1 | 56.0 | 59.4 | 79.3 | 29.2 | 64.2 | 69.7 | 54.5 |
| GRPO v4 | 92.1 | 57.0 | 60.9 | 85.9 | 30.3 | 66.3 | 70.3 | 51.5 |
| Molmo-7B-D, native | 89.0 | 31.0 | 56.6 | 81.8 | 26.2 | 67.9 | 66.2 | 40.9 |

Latency on the RTX 4090 (ms per query, batch 1, bf16, end to end), p50 / p90, and mean new tokens:

| Model | RefCOCO p50/p90 | Where2Place p50/p90 | Point-Bench p50/p90 | tokens RC/W2P/PB | parse fails RC/W2P/PB |
|---|---|---|---|---|---|
| Qwen3-VL-2B base, native (point_2d JSON) | 314 / 350 | 431 / 474 | 443 / 2838 | 33.5/43.4/39.5 | 21/0/113 |
| Qwen3-VL-2B base, canonical prompt | 272 / 346 | 482 / 568 | 407 / 2778 | 27.4/42.3/34.4 | 149/2/34 |
| SFT (sft-qwen3vl-2b-v1-e2) | 160 / 166 | 161 / 168 | 203 / 2617 | 15.0/15.0/15.1 | 0/0/0 |
| GRPO v1 | 155 / 159 | 156 / 160 | 195 / 2468 | 14.9/15.0/15.5 | 0/0/0 |
| GRPO v2 | 157 / 164 | 159 / 167 | 201 / 2560 | 14.9/15.0/15.5 | 0/0/0 |
| GRPO v3 | 157 / 162 | 158 / 159 | 206 / 2636 | 14.9/15.0/16.5 | 0/0/0 |
| Counting SFT (sft-count-v1, on v3) | 164 / 169 | 163 / 165 | 221 / 2635 | 15.1/15.0/19.7 | 0/0/0 |
| GRPO v4 | 164 / 171 | 171 / 177 | 215 / 2594 | 15.0/15.0/19.0 | 0/0/0 |
| Molmo-7B-D, native | 915 / 1129 | 3205 / 3234 | 1430 / 2421 | 32.7/122.2/51.7 | 39/0/59 |

Ns: RefCOCO 2000, Where2Place 100, Point-Bench 979 (affordable 198, counting 195, reasoning 193, spatial 195, steerable 198).

- qwen3vl-2b-native: model `qwen3vl-2b`, mode native, parser qwen_json, parser_sha af5a37222d02c46a, scoring v1, driver (not recorded), max_new_tokens 64
- qwen3vl-2b-canonical: model `qwen3vl-2b`, mode canonical, parser canonical, parser_sha af5a37222d02c46a, scoring v1, driver (not recorded), max_new_tokens 64
- sft-qwen3vl-2b-v1-e2: model `outputs/sft-qwen3vl-2b-v1-e2/merged`, mode canonical, parser canonical, parser_sha af5a37222d02c46a, scoring v1, driver (not recorded), max_new_tokens 64
- grpo-v1: model `outputs/grpo-v1/merged`, mode canonical, parser canonical, parser_sha af5a37222d02c46a, scoring v1, driver (not recorded), max_new_tokens 64
- grpo-v2: model `outputs/grpo-v2/merged`, mode canonical, parser canonical, parser_sha af5a37222d02c46a, scoring v1, driver 595.99.02, max_new_tokens 64
- grpo-v3: model `outputs/grpo-v3/merged`, mode canonical, parser canonical, parser_sha af5a37222d02c46a, scoring v1, driver 595.99.02, max_new_tokens 64
- sft-count-v1: model `outputs/sft-count-v1/merged`, mode canonical, parser canonical, parser_sha af5a37222d02c46a, scoring v1, driver 595.99.02, max_new_tokens 64
- grpo-v4: model `outputs/grpo-v4/merged`, mode canonical, parser canonical, parser_sha af5a37222d02c46a, scoring v1, driver 595.99.02, max_new_tokens 64
- molmo-7b-d-native: model `molmo-7b-d`, mode native, parser molmo, parser_sha af5a37222d02c46a, scoring v1, driver (not recorded), max_new_tokens 128

Notes:
- Molmo-7B-D: bf16 weights, its own prompt "Point to {query}.", native Molmo parser, max_new_tokens 128.
- Point-Bench p90 is ~2.5 s for every Qwen model even with ~15-token answers; p50 is ~200 ms. The tail is not from long outputs but from large images: Qwen3-VL keeps up to 16.7 MP (~16k visual tokens). Confirmed in section 4.4: under vLLM, where images above ~6 MP are resized, grpo-v4's Point-Bench p90 drops to 802 ms.
- Paired comparisons (exact McNemar, scripts/compare_runs.py), from the run logs: v1 vs SFT W2P p=0.004, RefCOCO p<1e-4, Point-Bench p=0.25; v3 vs SFT Point-Bench p=0.19, W2P p=0.023; v4 vs SFT Point-Bench p=0.043 (first significant Point-Bench gain), v4 vs v3 Point-Bench p=0.33, counting 17 gained / 0 lost.
- Raw outputs for rescoring: results/<bench>/<run>/predictions.jsonl (`raw` field); summaries in summary.json next to them.

## 4. Qwen3-VL-32B out of the box (vLLM), lenient parsing, vLLM latency

Added 2026-11-24. Frozen v1 rules unchanged; the new numbers are new result folders plus a secondary lenient score.

### 4.1 Setup

- Engine: vllm 0.34.0 in a separate venv (.venv-vllm: torch 2.14.0+cu130, transformers 5.22.2); the training env is untouched.
- Checkpoint: `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` revision `c2d44b1376c73fa120fdb4e216c2ef61c1945498`. Qwen publishes no official 4-bit Qwen3-VL-32B (only FP8, 33 GB, too big), so this is the most-downloaded community AWQ: 4-bit weights, group size 128, GEMM, zero point, vision tower left in 16-bit (`modules_to_not_convert: [visual]`). vLLM quantization `auto_awq`, activation dtype `torch.float16` (AWQ kernels run in fp16).
- Fit on the 24 GB 4090: weights 19.5 GiB after load. KV cache `fp8` with a fixed 0.8 GiB, max_model_len 6144, max_num_seqs 1, max_num_batched_tokens 2048 for RefCOCO and Where2Place native, 1024 for everything after (chunked prefill; chunk size changes memory, not the computation), CUDA graphs captured for batch 1 only, memory profiling with a 2480x2480 dummy image. bf16 KV could not hold 8k tokens next to the weights (needs 1.97 GiB, ~1.2 GiB left), and an 8k-token image OOMed the vision encoder, hence fp8 KV and 6144.
- Decoding as frozen: greedy (temperature 0), max 64 new tokens, batch 1, 5 untimed warm-ups, each model's default image preprocessing (Qwen3-VL keeps images up to 16.7 MP).
- Deviation: an image whose prompt exceeds max_model_len is downsized (aspect kept, PIL bicubic) to fit (6.1 MP) and flagged `resized_to` in predictions.jsonl. Only Point-Bench has such images; counts and accuracy without them are below.
- Peak VRAM (nvidia-smi memory.used for the whole GPU incl. ~0.8 GB desktop, sampled): see table. vLLM preallocates, so this is the reservation, not the working set.
- Protections: systemd-run --user, Nice 10, CPUQuota 1600%, MemoryMax 40G, Restart=on-failure; taskset off CPUs 8-11; OMP/MKL threads 8; TOKENIZERS_PARALLELISM=false; per-item fsynced predictions.partial.jsonl, resume on restart (an item that crashed the engine is retried at half its pixel budget). History: native Point-Bench hit two transient CUDA OOMs in prefill at items pb-0011 and pb-0020 (4 MP images that fit the context; 0.4-0.6 GB of the allocator's memory was fragmented). I relaunched with 1024-token prefill chunks and PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, cleared the retry marker so those items were not shrunk, and the rest ran without errors. No item was resized because of an OOM. Runner: scripts/eval_vllm.py, scripts/run_vllm_eval.sh; log outputs/run_vllm_32b.log.

### 4.2 Qwen3-VL-32B results (strict frozen parser = headline)

| Mode | RefCOCO | Where2Place | Point-Bench | PB affordable | PB counting | PB reasoning | PB spatial | PB steerable | PB resized | PB acc w/o resized | vLLM p50/p90 ms RC / W2P / PB | peak GPU GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| native | 93.2 | 65.0 | 59.0 | 87.4 | 15.4 | 71.0 | 75.9 | 45.5 | 172 | 58.6 | 885/988 / 1130/1242 / 1281/4469 | 23.4 |
| canonical | 89.3 | 64.0 | 60.0 | 87.9 | 13.3 | 72.0 | 75.4 | 51.0 | 172 | 59.4 | 540/892 / 541/1292 / 1281/4470 | 23.4 |

- native: parse failures (unparseable+invalid) RC/W2P/PB = 21/0/130; mean new tokens 33.5/43.5/36.6; prompt sha 8d49532482f95cdb.
- canonical: parse failures (unparseable+invalid) RC/W2P/PB = 78/0/75; mean new tokens 21.9/25.3/33.8; prompt sha b829bffc420e1ec0.
- Prompts. Native (sha 8d49532482f95cdb): `Locate {query} in the image and output its point coordinates in JSON format like [{"point_2d": [x, y], "label": "..."}].` It does not ask for exactly one point. Only the canonical prompt (shared with our 2B models) says "exactly one point".
- The native Point-Bench row above is the frozen run at max_new_tokens 64. A secondary 512-token rerun of the items that hit the cap is in section 4.6 (59.0 -> 60.2).

### 4.3 Strict vs lenient score (all models, same stored raw text)

Lenient parser: pointeval/lenient.py (`lenient-v1`), rescoring script scripts/rescore_lenient.py, per-run output results/<bench>/<run>/lenient.json. Accepts <point x y> (also malformed / multi-attribute tags), point_2d JSON (also truncated lists), bbox_2d centres, (x, y) tuples, bare 'x, y' pairs, 0-1 fractions (x1000) and pixel coordinates (rescaled by image size when a value exceeds 1000); Molmo attributes read as percent; first point only on single-target benchmarks, all points for Point-Bench counting (same score_item). Frozen v1 parser untouched (sha af5a37222d02c46a).

| Model | Engine | RefCOCO strict / lenient | Where2Place strict / lenient | Point-Bench strict / lenient |
|---|---|---|---|---|
| Qwen3-VL-2B base, native | HF | 86.2 / 87.1 | 44.0 / 44.0 | 51.9 / 52.5 |
| Qwen3-VL-2B base, canonical | HF | 80.6 / 86.9 | 40.0 / 42.0 | 48.9 / 50.7 |
| SFT | HF | 91.1 / 91.1 | 51.0 / 51.0 | 58.6 / 58.6 |
| GRPO v1 | HF | 92.8 / 92.8 | 65.0 / 65.0 | 57.5 / 57.5 |
| GRPO v2 | HF | 92.4 / 92.4 | 61.0 / 61.0 | 58.7 / 58.7 |
| GRPO v3 | HF | 92.7 / 92.7 | 63.0 / 63.0 | 59.9 / 59.9 |
| Counting SFT | HF | 91.1 / 91.1 | 56.0 / 56.0 | 59.4 / 59.4 |
| GRPO v4 | HF | 92.1 / 92.1 | 57.0 / 57.0 | 60.9 / 60.9 |
| GRPO v4 (vLLM, 8k ctx, bf16 KV) | vLLM | 92.4 / 92.4 | 59.0 / 59.0 | 61.1 / 61.1 |
| GRPO v4 (vLLM, 32B settings) | vLLM | 92.2 / 92.2 | 62.0 / 62.0 | 61.5 / 61.5 |
| Molmo-7B-D, native | HF | 89.0 / 89.0 | 31.0 / 31.0 | 56.6 / 56.6 |
| Qwen3-VL-32B AWQ, native | vLLM | 93.2 / 93.7 | 65.0 / 65.0 | 59.0 / 59.6 |
| Qwen3-VL-32B AWQ, canonical | vLLM | 89.3 / 90.0 | 64.0 / 64.0 | 60.0 / 61.8 |

Point-Bench per category, lenient:

| Model | affordable | counting | reasoning | spatial | steerable |
|---|---|---|---|---|---|
| Qwen3-VL-2B base, native | 72.2 | 22.6 | 56.0 | 62.6 | 49.0 |
| Qwen3-VL-2B base, canonical | 74.7 | 16.4 | 56.0 | 60.0 | 46.0 |
| SFT | 82.3 | 16.4 | 66.8 | 72.8 | 54.5 |
| GRPO v1 | 79.3 | 17.9 | 66.8 | 71.8 | 51.5 |
| GRPO v2 | 81.8 | 17.9 | 65.8 | 70.8 | 57.1 |
| GRPO v3 | 82.3 | 21.5 | 67.9 | 72.8 | 54.5 |
| Counting SFT | 79.3 | 29.2 | 64.2 | 69.7 | 54.5 |
| GRPO v4 | 85.9 | 30.3 | 66.3 | 70.3 | 51.5 |
| GRPO v4 (vLLM, 8k ctx, bf16 KV) | 84.3 | 30.8 | 66.8 | 71.3 | 52.0 |
| GRPO v4 (vLLM, 32B settings) | 83.3 | 31.3 | 67.9 | 70.3 | 54.5 |
| Molmo-7B-D, native | 81.8 | 26.2 | 67.9 | 66.2 | 40.9 |
| Qwen3-VL-32B AWQ, native | 88.4 | 16.4 | 71.5 | 75.9 | 45.5 |
| Qwen3-VL-32B AWQ, canonical | 87.9 | 20.5 | 74.1 | 75.4 | 51.0 |

### 4.4 Latency: HF vs vLLM (ms per query, batch 1, end to end, RTX 4090, driver 595.99.02)

HF = pointeval.run (transformers 5.19, bf16, sdpa). Compare speeds within one engine: the "10x" 2B-vs-32B figure is vLLM vs vLLM (GRPO v4 at 32B settings 90 ms vs 32B native 885 ms, RefCOCO p50), never HF 2B vs vLLM 32B. vLLM = scripts/eval_vllm.py (vLLM 0.34.0), wall time of one llm.chat() call (preprocess + prefill + decode + detokenize). vLLM rows resize images above max_model_len, which also trims the Point-Bench tail.

| Model | Engine / setting | RefCOCO p50 / p90 | Where2Place p50 / p90 | Point-Bench p50 / p90 | PB resized | mean tokens RC/W2P/PB |
|---|---|---|---|---|---|---|
| Qwen3-VL-2B base, native | HF | 314 / 350 | 431 / 474 | 443 / 2838 | 0 | 33.5/43.4/39.5 |
| GRPO v4 | HF | 164 / 171 | 171 / 177 | 215 / 2594 | 0 | 15.0/15.0/19.0 |
| GRPO v4 (vLLM, 8k ctx, bf16 KV) | vLLM, 8192 ctx, KV auto | 91 / 94 | 93 / 93 | 136 / 1219 | 149 | 15.0/15.0/19.0 |
| GRPO v4 (vLLM, 32B settings) | vLLM, 6144 ctx, KV fp8 | 90 / 92 | 91 / 92 | 139 / 802 | 172 | 15.0/15.0/19.0 |
| Molmo-7B-D, native | HF | 915 / 1129 | 3205 / 3234 | 1430 / 2421 | 0 | 32.7/122.2/51.7 |
| Qwen3-VL-32B AWQ, native | vLLM, 6144 ctx, KV fp8 | 885 / 988 | 1130 / 1242 | 1281 / 4469 | 172 | 33.5/43.5/36.6 |
| Qwen3-VL-32B AWQ, canonical | vLLM, 6144 ctx, KV fp8 | 540 / 892 | 541 / 1292 | 1281 / 4470 | 172 | 21.9/25.3/33.8 |

### 4.5 Paired comparisons with Qwen3-VL-32B AWQ native (exact McNemar, scripts/compare_runs.py, same items)

| Bench | A | A acc | 32B acc | A only right | 32B only right | p |
|---|---|---|---|---|---|---|
| RefCOCO | Qwen3-VL-2B base native | 86.2 | 93.2 | 53 | 194 | <1e-4 |
| RefCOCO | GRPO v1 | 92.8 | 93.2 | 88 | 96 | 0.61 |
| RefCOCO | GRPO v4 | 92.1 | 93.2 | 88 | 111 | 0.12 |
| RefCOCO | Molmo-7B-D | 89.0 | 93.2 | 73 | 157 | <1e-4 |
| Where2Place | Qwen3-VL-2B base native | 44 | 65 | 11 | 32 | 0.002 |
| Where2Place | GRPO v1 | 65 | 65 | 15 | 15 | 1.0 |
| Where2Place | GRPO v4 | 57 | 65 | 13 | 21 | 0.23 |
| Where2Place | Molmo-7B-D | 31 | 65 | 8 | 42 | <1e-4 |
| Point-Bench | Qwen3-VL-2B base native | 51.9 | 59.0 | 82 | 152 | <1e-4 |
| Point-Bench | GRPO v1 | 57.5 | 59.0 | 91 | 106 | 0.32 |
| Point-Bench | GRPO v4 | 60.9 | 59.0 | 116 | 98 | 0.25 |
| Point-Bench | Molmo-7B-D | 56.6 | 59.0 | 110 | 134 | 0.14 |

GRPO v4 vs 32B canonical on Point-Bench: 60.9 vs 60.0, 103 vs 94, p = 0.57.

Note: the Point-Bench rows above use the frozen 64-token native run. With the 512-token rerun merged in (section 4.6), GRPO v4 vs 32B native is 60.9 vs 60.2, 109 vs 102, p = 0.68; counting 30.3 vs 19.5, 24 vs 3, p < 0.0001. Conclusions unchanged.

Takeaways (facts only):
- Scale alone, 2B to 32B with the same native prompt, gains +7.1 RefCOCO, +21 Where2Place and +7.2 Point-Bench (all p < 0.01).
- Our tuned 2B (GRPO v1/v4) is statistically indistinguishable from the 4-bit 32B on all three benchmarks. It is about 10x faster, vLLM vs vLLM on the same 4090 (RefCOCO p50 ~90 ms for GRPO v4 at 32B settings vs ~885 ms for the 32B) and needs about 4.5x less weight memory (bf16 2B ~4.1 GiB vs 4-bit 32B 19.5 GiB).
- By category, the 32B is ahead on reasoning (71.0 vs 66.3), spatial (75.9 vs 70.3) and affordable (87.4 vs 85.9). It is behind on counting (15.4 vs 30.3; 19.5 after the 512-token rerun, section 4.6) and steerable (45.5 vs 51.5). Native Point-Bench has 130 strict parse failures (93 left after lenient parsing). 51 of the 93 are refusals ("There are none."), and a dozen more say the referenced point is not visible. Those are mostly steerable items that refer to a point missing from the image.
- The canonical prompt costs the 32B 3.9 points on RefCOCO (89.3 vs 93.2). It has 78 strict parse failures there (63 after lenient parsing), for example a tag with no y or `<point x="588" in the image.`. On Point-Bench, canonical is slightly ahead of native (60.0 vs 59.0). The gain is in steerable (51.0 vs 45.5) and reasoning (72.0 vs 71.0); counting is lower (13.3 vs 15.4).
- Raw outputs: results/<bench>/qwen3vl-32b-awq-vllm-{native,canonical}/predictions.jsonl (`raw`, plus `resized_to` where it applies), lenient.json next to them.
- Engine caveat for tables that pool runs: our 2B rows in section 3 and most of 4.3 are HF, the 32B rows are vLLM (with images above 6144 tokens resized). Any table or count that puts them side by side (strict/lenient, parse failures, out-of-bounds / unparseable counts) mixes engines; the like-for-like 2B reference for the 32B is "GRPO v4 (vLLM, 32B settings)".

### 4.6 Qwen3-VL-32B native Point-Bench rerun at 512 tokens (secondary, 2026-11-24)

Why: the frozen 64-token cap cut off 62 of the 32B's native Point-Bench answers (status unparseable at exactly 64 new tokens), 37 of them in counting (of 195). Long "point to all the X" answers in point_2d JSON need ~13 tokens per point, so the cap, not the model, decided those items.

Method:
- Same vLLM settings as 4.1 (same checkpoint/revision, 6144 ctx, fp8 KV, greedy, batch 1, same resize rule), only those 62 items, max_new_tokens 512.
- Rerun folder: results/pointbench/qwen3vl-32b-awq-vllm-native-cap512 (rerun_ids.txt lists the 62 ids). Merged folder: results/pointbench/qwen3vl-32b-awq-vllm-native-cap512-merged (the 62 swapped into the original run; summary.json records `merged_from`, n_swapped 62, base cap 64, rerun cap 512, and marks it "secondary result, not frozen-spec").
- Code: scripts/eval_vllm.py gained new flags to rerun a list of ids with a different token cap into a new folder; scripts/merge_rerun.py does the swap and rescoring. Frozen results (qwen3vl-32b-awq-vllm-native) are untouched; the frozen headline stays 59.0.

Results (strict frozen parser; lenient = strict, 0 rescued / 0 lost):

| Run | Point-Bench | affordable | counting | reasoning | spatial | steerable |
|---|---|---|---|---|---|---|
| 32B native, frozen (cap 64) | 59.0 | 87.4 | 15.4 | 71.0 | 75.9 | 45.5 |
| 32B native, cap-512 merged | 60.2 (589/979, CI 57.1-63.2) | 88.4 | 19.5 | 71.5 | 75.9 | 45.5 |

- Of the 62 rerun items, 12 are now hits (8 of them counting); one of the 12 was already a hit at 64 tokens, so the net gain is +11 (578 -> 589).
- The other answers: 22 refusals ("There are none." style), 1 degenerate loop that ran to 512 tokens, 2 limited by the 6144 context; the rest parse but miss.
- Merged run: 95 strict unparseable, mean new tokens 39.9, latency p50/p90 1281/4476 ms.

Paired with GRPO v4 (exact McNemar, same items): Point-Bench 60.9 vs 60.2, v4 only right 109 vs 32B only right 102, p = 0.68; counting 30.3 vs 19.5, 24 vs 3, p < 0.0001. The tuned 2B is still level with the 32B overall and still well ahead on counting.


### 4.7 Base Qwen3-VL-2B on vLLM with the 32B settings (RefCOCO only, 2026-11-24)

Run dirs: results/refcoco/qwen3vl-2b-vllm32set-{canonical,native} (scripts/eval_vllm.py, commit 0ade4d2).
Qwen/Qwen3-VL-2B-Instruct @ 89644892; vLLM 0.34.0, fp8 KV (0.8 GiB), max_model_len 6144, max_num_batched_tokens 1024,
profile side 2480, CUDA graphs [1], max_num_seqs 1, greedy, 64 new tokens, 5 warm-ups (same flags as grpo-v4-vllm32set).

| Mode | Strict acc (95% CI) | Lenient | HF strict / lenient | p50 / p90 ms (vLLM) | HF p50 ms |
|---|---|---|---|---|---|
| canonical | 85.0 (83.4-86.5) | 86.8 | 80.6 / 86.9 | 146 / 181 | 272 |
| native | 86.4 (84.8-87.8) | 87.3 | 86.2 / 87.05 | 166 / 185 | 314 |

- Format slips depend on the engine: HF canonical had 149 outputs ending in `]` (strict-unparseable), vLLM 39.
  Paired: 105 items HF-unparseable -> vLLM hit; among items both parsed, 28 vLLM-only hits vs 33 HF-only hits.
  Under the lenient parser the engines agree, so localization is the same and only format adherence differs.
  Only 482/2000 (canonical) and 570/2000 (native) raw outputs are byte-identical between engines (bf16 numerics, fp8 KV).
- Molmo-7B-D on vLLM 0.34 was tried and dropped: needs --native molmo --trust-remote-code, max_model_len <= 4096
  (max_position_embeddings) and skipping a dead `import tensorflow`; with fp8 KV the output is garbled (`y="="="36.5"`),
  with bf16 KV the format is clean but points on the first 5 items differ widely from HF (HF hits all 5). Molmo latency
  therefore stays HF (915 ms p50).
