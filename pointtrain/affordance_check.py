"""Can GRPO fix function-style references, or does the model lack the knowledge? A pass@k check.

	python -m pointtrain.affordance_check --out outputs/affordance_check

GRPO only gets gradient from prompts the policy sometimes solves: if a prompt is solved 0 of K
times, every group is unanimous and the advantage is 0. So we sample the SFT model K = 8 times on
function-style prompts ("the thing used to cut food") and, as a control, on the same objects named
plainly ("the knife"). Reading the result:
	* function hits ~0/8 while name hits ~8/8 -> grounding is fine, the function -> object knowledge is
	  missing: needs SFT data (RL has nothing to reinforce)
	* function hits spread over 1..7/8           -> the knowledge is there but unreliable: GRPO can fix it
Images: COCO train2014 images in the RefCOCO dev split (never trained on by SFT or GRPO), minus every
held-out / eval / flagged image. The object must be the only instance of its category in the image and
cover >= 0.5% of it, so the reference is unambiguous. Point-Bench is not used.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pointeval.adapters import CANONICAL_PROMPT
from pointeval.geometry import mask_to_rle
from pointeval.run import load_manifest
from pointtrain.prompts import _u as dev_u
from pointtrain.sources import coco_id_from_name, refcoco_heldout_image_ids
from pointtrain.synth_coco import name

# Two function descriptions per category; none names the object or a close synonym.
FUNCTIONS = {
	"knife": ["the utensil used to cut food", "the thing you would slice bread with"],
	"fork": ["the utensil with prongs for eating", "the thing you would spear food with"],
	"spoon": ["the utensil for eating soup", "the thing you would stir coffee with"],
	"cup": ["the thing you would drink coffee from", "the container for a hot drink"],
	"wine glass": ["the thing you would drink wine from", "the stemmed drinking vessel"],
	"bottle": ["the container you would pour a drink from", "the thing that holds liquid and has a cap"],
	"bowl": ["the dish you would eat cereal from", "the container for serving soup"],
	"chair": ["something a single person can sit on", "the seat with a backrest"],
	"couch": ["the furniture several people can sit on together", "where you would lie down to watch TV"],
	"bed": ["where people sleep", "the furniture you would lie on at night"],
	"dining table": ["the surface people eat meals at", "where you would put plates for dinner"],
	"toilet": ["the bathroom fixture you flush", "what you would use when you need the restroom"],
	"sink": ["where you would wash your hands", "the basin with a faucet"],
	"refrigerator": ["the appliance that keeps food cold", "where you would store milk"],
	"oven": ["the appliance used for baking", "where you would roast a chicken"],
	"microwave": ["the appliance for reheating food quickly", "the small box that heats leftovers"],
	"toaster": ["the appliance that browns bread", "what you would use to make toast crisp"],
	"tv": ["the screen you would watch shows on", "what you would turn on to see the news"],
	"laptop": ["the portable computer", "what you would type an email on while traveling"],
	"keyboard": ["what you would type on", "the input device with letter keys"],
	"mouse": ["what you would use to move the cursor", "the device you click with"],
	"remote": ["what you would use to change the channel", "the handheld controller for the TV"],
	"cell phone": ["the device you would make a call with", "what you would use to send a text message"],
	"clock": ["what tells the time", "the thing you would check to see if you are late"],
	"book": ["something you would read", "the thing with pages and a cover"],
	"scissors": ["the tool used to cut paper", "what you would use to cut hair"],
	"vase": ["the container for displaying flowers", "where you would put a bouquet"],
	"umbrella": ["what keeps you dry in the rain", "the thing you open when it starts to pour"],
	"backpack": ["the bag worn on the back", "what a student carries books in"],
	"handbag": ["the bag carried by hand or on the shoulder", "where someone would keep a wallet"],
	"suitcase": ["the luggage you would pack for a trip", "what you would wheel through an airport"],
	"bicycle": ["the two-wheeled vehicle you pedal", "what you would ride to work without a motor"],
	"motorcycle": ["the two-wheeled motor vehicle", "what a rider with a helmet would drive on the highway"],
	"car": ["the vehicle a family would drive to the store", "the four-wheeled passenger vehicle"],
	"bus": ["the vehicle that carries many passengers on city streets", "public transport on the road"],
	"truck": ["the vehicle for hauling cargo", "what would deliver heavy goods by road"],
	"train": ["the vehicle that runs on rails", "what you would board at a station platform"],
	"airplane": ["the vehicle that flies passengers", "what takes off from a runway"],
	"boat": ["what you would use to cross a lake", "the vessel that floats on water"],
	"traffic light": ["what tells drivers when to stop and go", "the signal at an intersection"],
	"stop sign": ["the sign that tells drivers to halt", "the red octagonal sign"],
	"fire hydrant": ["what firefighters connect a hose to", "the water outlet on the sidewalk for emergencies"],
	"bench": ["where you could sit in a park", "the long outdoor seat"],
	"kite": ["what you would fly on a windy day", "the thing on a string up in the sky"],
	"frisbee": ["the flying disc", "what you would throw to a dog to catch"],
	"skateboard": ["the board with wheels you ride", "what you would do an ollie on"],
	"surfboard": ["what you would ride a wave on", "the board used at the beach in the surf"],
	"skis": ["what you would wear to go down a snowy slope", "the long narrow runners for snow"],
	"snowboard": ["the single board for riding down snow", "what you strap both feet to on a slope"],
	"tennis racket": ["what you would hit a tennis ball with", "the strung paddle used on a court"],
	"baseball bat": ["what a batter swings", "the wooden club used to hit a pitch"],
	"baseball glove": ["what a fielder catches the ball with", "the leather mitt"],
	"sports ball": ["the ball being played with", "what the players are trying to hit or kick"],
	"teddy bear": ["the stuffed toy a child would cuddle", "the soft plush animal"],
	"toothbrush": ["what you would clean your teeth with", "the thing used with toothpaste"],
	"hair drier": ["what you would use to dry your hair", "the appliance that blows hot air"],
	"potted plant": ["the plant in a container", "what you would water in a pot"],
	"pizza": ["the round food with cheese and toppings", "what you would order and cut into slices"],
	"cake": ["the dessert for a birthday", "what you would put candles on"],
	"sandwich": ["the food made with two slices of bread", "what you would pack for lunch"],
	"banana": ["the yellow fruit you peel", "what a monkey would eat"],
	"apple": ["the fruit that keeps the doctor away", "the round fruit you could bite into"],
	"orange": ["the citrus fruit", "the fruit you would squeeze for juice"],
	"donut": ["the fried pastry with a hole", "what you might have with coffee at a bakery"],
	"hot dog": ["the sausage in a bun", "what you would buy at a ballpark"],
	"horse": ["the animal you could ride", "the animal that pulls a carriage"],
	"dog": ["the pet that barks", "the animal you would take for a walk on a leash"],
	"cat": ["the pet that meows", "the animal that chases mice"],
	"cow": ["the animal that gives milk", "the farm animal that says moo"],
	"sheep": ["the animal that gives wool", "the farm animal you would shear"],
	"elephant": ["the animal with a trunk", "the largest land animal"],
	"giraffe": ["the animal with a very long neck", "the tallest animal"],
	"zebra": ["the striped animal", "the black and white horse-like animal"],
	"bird": ["the animal that can fly", "the animal with feathers"],
	"tie": ["what a man would wear with a suit around his neck", "the formal neckwear"],
	"parking meter": ["where you would pay for street parking", "the coin-operated post by the curb"],
}


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--ann", type=Path, default=Path.home() / "data/coco/annotations/annotations/instances_train2014.json")
	ap.add_argument("--exclude-images", type=Path, default=Path("data/sft/flagged_v1.txt"))
	ap.add_argument("--min-area-frac", type=float, default=0.005)
	ap.add_argument("--per-category", type=int, default=4)
	ap.add_argument("--k", type=int, default=8)
	ap.add_argument("--build-only", action="store_true")
	a = ap.parse_args()
	a.out.mkdir(parents=True, exist_ok=True)
	from pycocotools import mask as mask_utils

	heldout = refcoco_heldout_image_ids()
	exclude = set().union(*heldout.values()) | {coco_id_from_name(it["file_name"]) for it in load_manifest("refcoco")[0]}
	flagged = set(a.exclude_images.read_text().split()) if a.exclude_images.exists() else set()
	d = json.loads(a.ann.read_text())
	cats = {c["id"]: c["name"] for c in d["categories"]}
	anns = defaultdict(list)
	for x in d["annotations"]:
		anns[x["image_id"]].append(x)
	rows, per_cat = [], Counter()
	for im in sorted(d["images"], key=lambda im: dev_u(f"aff:{im['id']}")):
		iid, W, H = im["id"], im["width"], im["height"]
		path = f"coco/train2014/{im['file_name']}"
		if dev_u(f"dev:coco:{iid}") >= 0.015 or iid in exclude or path in flagged:
			continue
		count = Counter(cats[x["category_id"]] for x in anns[iid])
		crowd = {cats[x["category_id"]] for x in anns[iid] if x["iscrowd"]}
		for x in anns[iid]:
			c = cats[x["category_id"]]
			if c not in FUNCTIONS or count[c] != 1 or c in crowd or x["area"] / (W * H) < a.min_area_frac or per_cat[c] >= a.per_category:
				continue
			seg = x["segmentation"]
			mask = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(seg, H, W))).astype(bool)
			fn = FUNCTIONS[c][per_cat[c] % 2]
			base = {"image": path, "image_id": f"coco:{iid}", "category": c, "target": {"type": "mask", "mask_rle": mask_to_rle(mask)}}
			rows.append({**base, "kind": "function", "query": fn, "prompt": CANONICAL_PROMPT.format(query=fn)})
			rows.append({**base, "kind": "name", "query": f"the {name(c)}", "prompt": CANONICAL_PROMPT.format(query=f"the {name(c)}")})
			per_cat[c] += 1
			break  # one object per image
	with (a.out / "prompts.jsonl").open("w") as f:
		for r in rows:
			f.write(json.dumps(r) + "\n")
	print(f"{len(rows) // 2} objects, {len(per_cat)} categories")
	if a.build_only:
		return

	from train_utils.configs.model_config import LoRAConfig, ModelConfig, QuantizationConfig
	from train_utils.vlm import build_adapter

	from pointtrain.rl_reward import point_reward

	cfg = ModelConfig(model_id=str(Path("outputs/sft-qwen3vl-2b-v1-e2/merged").resolve()), vlm_adapter="hf_chat", dtype="bfloat16", attn_implementation="sdpa", lora=LoRAConfig(use_lora=False), quantization=QuantizationConfig(load_in_4bit=False))
	adapter = build_adapter(cfg, max_image_side=1024, image_root=str(Path.home() / "data"))
	model, _ = adapter.load()
	model.eval()
	per = 8
	res = []
	for i in range(0, len(rows), per):
		chunk = rows[i : i + per]
		comps = adapter.sample(model, [r for r in chunk for _ in range(a.k)], 24, True, 1.0, 1.0, 0)
		greedy = adapter.sample(model, chunk, 24, do_sample=False)
		for j, r in enumerate(chunk):
			texts = [adapter.decode(c) for c in comps[j * a.k : (j + 1) * a.k]]
			hits = sum(point_reward(r, t)["reward"] for t in texts)
			res.append({k: r[k] for k in ("image", "category", "kind", "query")} | {"hits": int(hits), "greedy": point_reward(r, adapter.decode(greedy[j]))["reward"], "example": texts[0]})
	with (a.out / "results.jsonl").open("w") as f:
		for r in res:
			f.write(json.dumps(r) + "\n")
	summary = {}
	for kind in ("function", "name"):
		rs = [r for r in res if r["kind"] == kind]
		h = Counter(r["hits"] for r in rs)
		summary[kind] = {
			"n": len(rs),
			"greedy_acc": round(float(np.mean([r["greedy"] for r in rs])), 3),
			"pass@1": round(float(np.mean([r["hits"] / a.k for r in rs])), 3),
			"pass@8": round(float(np.mean([r["hits"] > 0 for r in rs])), 3),
			"hits_out_of_8": dict(sorted(h.items())),
		}
	fn = {(r["image"], r["category"]): r for r in res if r["kind"] == "function"}
	nm = {(r["image"], r["category"]): r for r in res if r["kind"] == "name"}
	both = [(fn[k], nm[k]) for k in fn if k in nm]
	summary["function_misses_with_name_greedy_hit"] = sum(1 for f, n in both if not f["greedy"] and n["greedy"])
	summary["function_greedy_misses"] = sum(1 for f, _ in both if not f["greedy"])
	(a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()
