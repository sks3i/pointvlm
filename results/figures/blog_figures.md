# Blog figures (stored predictions only)

Made by `scripts/fig_blog_stages.py --spec results/figures/blog_figures.json` from `results/<bench>/<run>/predictions.jsonl`; no model was run. Green = GT mask, red dot = the stage's points, white ring = the previous checkpoint's points. Each figure is 2x2, 2402 px wide, under 3 MB.

Notes: synthetic-dev predictions were never stored per item (only aggregate `dev_metrics.jsonl`), so the v2 wins come from RefCOCO and Point-Bench spatial. v1/v2/v3 are compared with SFT, the checkpoint each was trained from. v4 is compared with v3, because the count SFT that v4 starts from was built on v3.

## blog_sft.png

SFT vs base Qwen3-VL-2B (canonical prompt): 2 wins, 2 phrase misreadings

| # | bench | item | query | verdict | caption |
|---|---|---|---|---|---|
| 1 | Point-Bench · affordable | pb-0197 | Point to what could ignite things. | win | Win: base points at the flame; SFT points at the lighter, the thing that ignites. |
| 2 | Where2Place · seen | w2p-011 | Locate several points within a vacant area on the front portion of the stove. | win | Win: base lands on the floor; SFT lands on the free cooktop zone at the front of the stove. |
| 3 | RefCOCO val | refcoco-val-0342 | 2nd guy from the left | failure (ordinal) | Failure: SFT points at the leftmost person in the dugout, not the 2nd from the left; base got it. |
| 4 | Point-Bench · spatial | pb-0439 | Point to the building that is to the left of the rightmost building. | failure (relation) | Failure: SFT points at the anchor (rightmost building) instead of the one to its left; base got it. |

## blog_grpo_v1.png

GRPO v1 vs SFT: free-space wins on Where2Place, relational breaks on Point-Bench

| # | bench | item | query | verdict | caption |
|---|---|---|---|---|---|
| 1 | Where2Place · unseen | w2p-052 | Locate several spots within the free region under the dresser. | win | Win: SFT sits on the dresser edge; v1 moves into the free region underneath. |
| 2 | Where2Place · seen | w2p-079 | Find a few locations in the empty space that is to the right of the toy tower. | win | Win: SFT lands off the table edge; v1 lands in the empty tabletop right of the toy tower. |
| 3 | Point-Bench · spatial | pb-0566 | Point to the object to the left of the sink. | failure (broke) | Broke: SFT found the object left of the sink; v1 lands on the bare counter between it and the sink. |
| 4 | Point-Bench · spatial | pb-0390 | Point to the object to the left of the pink vase. | failure (broke) | Broke: SFT found the vase left of the pink vase; v1 drifts onto the pink vase (the anchor). |

## blog_grpo_v2.png

GRPO v2 vs SFT: ordinal/relation fixes (RefCOCO, Point-Bench spatial) vs the reasoning transfer gap

| # | bench | item | query | verdict | caption |
|---|---|---|---|---|---|
| 1 | RefCOCO val | refcoco-val-0596 | 2nd sink from the left | win (ordinal) | Ordinal fixed: SFT picks the 1st sink; v2 picks the 2nd from the left. |
| 2 | Point-Bench · spatial | pb-0265 | Point to the window to the right of the red door. | win (relation) | Relation fixed: SFT picks the window above the door; v2 picks the window right of the red door. |
| 3 | Point-Bench · reasoning | pb-0126 | Point to the object the car is stopping for. | failure (reasoning) | Transfer gap: both point at a parked bicycle; the answer is the traffic light the car stops for. |
| 4 | Point-Bench · affordable | pb-0204 | Point to the tool that creates music. | failure (affordance) | Transfer gap: both point at the strap on the player's shoulder, not the drum that makes the music. |

## blog_grpo_v3.png

GRPO v3 vs SFT: function-style wins, and the counting drift (several points despite 'exactly one point')

| # | bench | item | query | verdict | caption |
|---|---|---|---|---|---|
| 1 | Point-Bench · affordable | pb-0467 | Point to the object people can use to paint. | win (function) | Function win: SFT picks the sponge stick; v3 picks a paintbrush (the GT objects). |
| 2 | Point-Bench · affordable | pb-0703 | Point to the tool used to save pencils. | win (near-miss fixed) | Near miss fixed: SFT lands just above the pencil case; v3 lands on it. |
| 3 | Point-Bench · counting | pb-0387 | Point to all the apples. | counting drift (scored hit) | Drift: prompt asks for exactly one point; v3 emits three, one per apple, and the counting item is scored a hit. |
| 4 | Point-Bench · spatial | pb-0222 | Point to the object in front of the black bus. | failure (broke) | Broke: SFT found the bus in front; v3 points at the black bus itself (the anchor). |

## blog_grpo_v4.png

GRPO v4 vs GRPO v3: counting with several correct points, plus one non-counting win and one loss

| # | bench | item | query | verdict | caption |
|---|---|---|---|---|---|
| 1 | Point-Bench · counting | pb-0511 | Point to all the glasses of drinks. | win (counting) | Counting win: v3 gives one point; v4 gives one point per glass, all three inside. |
| 2 | Point-Bench · counting | pb-0795 | Point to all the cars. | win (counting) | Counting win: v3 gives one point; v4 gives one per car. |
| 3 | Point-Bench · reasoning | pb-0704 | Point to the item that people can drink. | win (non-counting) | Non-counting win: v3 picks a dish on the table; v4 picks the glass of drink. |
| 4 | Where2Place · seen | w2p-046 | Find some spots in the free space above the drawers. | failure (lost) | Lost: v3 hit the free counter space above the drawers; v4 jumps to the top-right corner. |
