"""Pairwise LLM-as-judge (a PROXY for human preference, never reported as human preference).

Protocol: for each item and model pair (X, Y) the judge sees the instruction and two anonymized
responses "A"/"B". Every item is judged in BOTH orders (X as A, then Y as A); the first order
is chosen by a seeded RNG. Verdicts are mapped back to models; if the two orders disagree the
item is a tie (this cancels position bias).
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from typing import Any

RUBRIC = """You are an expert chef and food scientist judging two AI responses to the same \
cooking request. Judge on:
1. Coherence - clear, complete, well-organized.
2. Feasibility - a home cook could actually make it; quantities/steps are sensible and safe.
3. Flavor harmony - the ingredients and pairings taste good together.
4. Instruction-following - does exactly what was asked (requested ingredients, cuisine, format, \
number of suggestions).
Do not prefer a response just because it is longer. Ignore which response comes first.

### Request
{instruction}

### Response A
{a}

### Response B
{b}

Reply with ONLY a JSON object:
{{"coherence": "A"|"B"|"tie", "feasibility": "A"|"B"|"tie", "flavor_harmony": "A"|"B"|"tie", \
"instruction_following": "A"|"B"|"tie", "overall": "A"|"B"|"tie", "reason": "<one sentence>"}}"""

CRITERIA = ["coherence", "feasibility", "flavor_harmony", "instruction_following", "overall"]


def judge_orders(item_id: str, pair: tuple[str, str], seed: int) -> list[tuple[str, str]]:
    """Both presentation orders, the first chosen by a seeded RNG keyed on the item."""
    x, y = pair
    rng = random.Random(f"{seed}|{item_id}|{x}|{y}")
    orders = [(x, y), (y, x)]
    if rng.random() < 0.5:
        orders.reverse()
    return orders


def parse_verdict(text: str) -> dict[str, str]:
    """Extract the JSON verdict; unknown/invalid fields become ``"tie"``."""
    m = re.search(r"\{.*\}", text or "", re.S)
    data: dict[str, Any] = {}
    if m:
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            data = {}
    out = {}
    for c in CRITERIA:
        v = str(data.get(c, "tie")).strip().upper()
        out[c] = "A" if v == "A" else "B" if v == "B" else "tie"
    return out


def combine(first: str, second: str) -> str:
    """Combine two model-level verdicts ("win"/"tie"/"loss" for X); disagreement -> tie."""
    return first if first == second else "tie"


def judge_item(
    judge_fn: Callable[[str], str],
    instruction: str,
    outputs: dict[str, str],
    item_id: str,
    pair: tuple[str, str],
    seed: int,
) -> dict[str, Any]:
    """Judge one item for model pair ``pair=(X, Y)``; outcomes are from X's perspective."""
    x, _y = pair
    per_order = []
    for a_model, b_model in judge_orders(item_id, pair, seed):
        prompt = RUBRIC.format(instruction=instruction, a=outputs[a_model], b=outputs[b_model])
        verdict = parse_verdict(judge_fn(prompt))
        mapped = {}
        for c, v in verdict.items():
            if v == "tie":
                mapped[c] = "tie"
            else:
                winner = a_model if v == "A" else b_model
                mapped[c] = "win" if winner == x else "loss"
        per_order.append({"a": a_model, "b": b_model, "raw": verdict, "outcome": mapped})
    outcome = {c: combine(per_order[0]["outcome"][c], per_order[1]["outcome"][c]) for c in CRITERIA}
    return {"item_id": item_id, "pair": f"{pair[0]}|{pair[1]}", "orders": per_order, **outcome}
