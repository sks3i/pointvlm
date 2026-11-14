# Eval spec v1 (frozen)

Frozen on the day the manifests are committed. Anything below changes only by creating `v2` alongside v1, never by editing v1.

## Benchmarks

| Bench | Source (pinned revision in `manifests/*.meta.json`) | Items | Headline metric |
|---|---|---|---|
| RefCOCO val 2k | UNC val objects (`jxu124/refcoco` split `validation`): masks from its `raw_anns`, images from `lmms-lab/RefCOCO` split `val` | 2000 objects | point-in-mask accuracy |
| Where2Place | `wentao-yuan/where2place` | all (~100) | point-in-mask accuracy (first point) |
| Point-Bench (optional) | `PointArena/pointarena` | all (~982) | PointArena success rule |

Secondary numbers (reported, never headline): RefCOCO point-in-box; Where2Place fraction of all predicted points in mask (RoboPoint's metric); Point-Bench per category.

## Sampling (RefCOCO)

* Unit is an object (`file_name` + `bbox`), not a sentence, so items are not near-duplicates.
* Only true UNC val objects are eligible. lmms-lab's `val` also has UNC train/testA/testB refs, so rows are kept only if `question_id` (= COCO `ann_id`) is in `jxu124/refcoco` `validation`.
* Masks come from the full COCO segmentation in `jxu124/refcoco` `raw_anns` (all polygons). lmms-lab's `segmentation` keeps only the first polygon. Both dataset revisions are pinned in `manifests/refcoco_val_2k.meta.json`.
* Rank objects by `sha256("pointvlm-refcoco-val-v1:" + key)`, take the first 2000 that have an expression and a non-empty mask. One expression per object, chosen by the same hash.
* The manifest stores query, mask RLE, image size and an image sha256. Scoring reads only the manifest; images are verified against the hash.

## Output format and parsing

* Canonical: `<point x="X" y="Y">` (also `/>` or `</point>`), x before y, numbers in [0, 1000], origin top-left. Extra attributes after y are allowed (`<point x="593" y="405" alt="shirt">`); x must still come first and both values must be quoted numbers.
* Unparseable or out of range scores as a miss and stays in the denominator. Rates are reported per run.
* Only the first point is scored on single-target benchmarks.
* Baselines may use a native adapter (Molmo percent x10; Qwen3-VL `point_2d`, 0-1000). Our checkpoints use the canonical parser only.
* Molmo `<points x1=.. y1=..>` cut off by the token cap before its closing `>` keeps its complete xN/yN pairs; an incomplete last pair is dropped.
* Every summary records `scoring_version` (v1) and `parser_sha256`. `python -m pointeval.rescore <run_dir>` re-scores stored raw outputs with the current rules (latency and tokens unchanged).

## Coordinates

* `x_px = floor(x / 1000 * W)`, clamped to `W - 1` (same for y), on the original image. Denominator 1000, never 999.
* Training targets use pixel centres: `x = (col + 0.5) / W * 1000`.

## Decoding and prompts

* Greedy, `max_new_tokens = 64` (Molmo 128 for its alt text), batch size 1, each model's default image preprocessing.
* Prompt templates are in `pointeval/adapters.py`; each run records the template and its hash.

## Latency

* End-to-end ms per query (preprocess + generate + decode), CUDA synchronized, batch size 1, after 5 untimed warm-up calls. Report mean, p50, p90 and mean output tokens, with GPU, driver, torch, transformers and dtype.

## Statistics

* Wilson 95% CI on every accuracy. Checkpoint comparisons use exact McNemar on the same items.
