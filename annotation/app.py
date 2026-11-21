"""Gradio UI for blind pairwise A/B annotation. All logic lives in :mod:`annotation.core`.

Launch from the repo root::

    python -m annotation.app --generations-dir results/generations
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gradio as gr

from annotation.core import (
    CRITERIA,
    append_record,
    build_comparisons,
    done_keys,
    load_generations,
    make_record,
    remaining,
)

CHOICE_LABELS: dict[str, str] = {"A better": "A", "Tie": "tie", "B better": "B"}
CRITERION_LABELS: dict[str, str] = {
    "coherence": "Coherence",
    "feasibility": "Feasibility",
    "flavor_harmony": "Flavor harmony",
    "instruction_following": "Instruction-following",
}
RATING_CHOICES: list[str] = ["1", "2", "3", "4", "5"]


def _view(state: dict[str, Any] | None) -> tuple[str, str, str, str, str]:
    """(progress, task, instruction, response A, response B) for the current comparison."""
    if not state:
        return ("Enter your annotator id and press **Start**.", "", "", "", "")
    total, queue = state["total"], state["queue"]
    done = total - len(queue)
    progress = f"**Progress:** {done} / {total} done"
    if total == 0:
        return ("No comparisons available (need >= 2 generation files).", "", "", "", "")
    if not queue:
        return (progress + " - all done, thank you!", "", "", "", "")
    view = queue[0].public_view()
    return (
        progress,
        f"**Task:** {view['task']}",
        view["instruction"],
        view["response_a"],
        view["response_b"],
    )


def build_app(generations_dir: str | Path, out_path: str | Path) -> gr.Blocks:
    """Build the Blocks app. Model identities stay server-side in ``gr.State``."""
    generations = load_generations(generations_dir)

    def start(annotator_id: str) -> tuple[Any, ...]:
        aid = (annotator_id or "").strip()
        if not aid:
            gr.Warning("Please enter an annotator id.")
            return (None, *_view(None))
        comps = build_comparisons(generations, aid)
        state = {
            "annotator": aid,
            "total": len(comps),
            "queue": remaining(comps, done_keys(out_path, aid)),
        }
        return (state, *_view(state))

    def submit(state: dict[str, Any] | None, choice: str | None, *rest: Any) -> tuple[Any, ...]:
        n = len(CRITERIA)
        ratings_a, ratings_b, comment = rest[:n], rest[n : 2 * n], rest[2 * n]
        keep_inputs = (gr.skip(),) * (2 * n + 2)
        if not state or not state["queue"]:
            gr.Warning("Nothing to submit - press Start first.")
            return (state, *_view(state), *keep_inputs)
        if choice not in CHOICE_LABELS:
            gr.Warning("Please choose an overall preference.")
            return (state, *_view(state), *keep_inputs)
        comp = state["queue"][0]
        record = make_record(
            comp,
            state["annotator"],
            CHOICE_LABELS[choice],
            dict(zip(CRITERIA, ratings_a, strict=True)),
            dict(zip(CRITERIA, ratings_b, strict=True)),
            comment or "",
        )
        append_record(out_path, record)
        state = {**state, "queue": state["queue"][1:]}
        cleared = (None,) * (2 * n + 1) + ("",)
        return (state, *_view(state), *cleared)

    with gr.Blocks(title="Recipe preference annotation") as demo:
        gr.Markdown(
            "# Blind recipe / pairing preference\n"
            "Compare two anonymous responses to the same instruction. Judge the food, not the "
            "formatting or length. See `annotation/README.md` for the rubric."
        )
        state = gr.State(None)
        with gr.Row():
            annotator = gr.Textbox(label="Annotator id", placeholder="e.g. alice", scale=3)
            start_btn = gr.Button("Start / resume", variant="primary", scale=1)
        progress = gr.Markdown(_view(None)[0])
        task_md = gr.Markdown()
        gr.Markdown("### Instruction")
        instruction = gr.Markdown()
        rating_inputs: dict[str, list[gr.Radio]] = {"a": [], "b": []}
        with gr.Row(equal_height=False):
            outputs_md = {}
            for side in ("a", "b"):
                with gr.Column():
                    gr.Markdown(f"### Response {side.upper()}")
                    outputs_md[side] = gr.Markdown()
                    for crit in CRITERIA:
                        rating_inputs[side].append(
                            gr.Radio(
                                RATING_CHOICES,
                                value=None,
                                label=f"{CRITERION_LABELS[crit]} ({side.upper()}, optional 1-5)",
                            )
                        )
        choice = gr.Radio(list(CHOICE_LABELS), value=None, label="Overall preference")
        comment = gr.Textbox(label="Comment (optional)", lines=2)
        submit_btn = gr.Button("Submit & next", variant="primary")

        view_outputs = [progress, task_md, instruction, outputs_md["a"], outputs_md["b"]]
        rating_list = rating_inputs["a"] + rating_inputs["b"]
        start_btn.click(start, inputs=[annotator], outputs=[state, *view_outputs])
        annotator.submit(start, inputs=[annotator], outputs=[state, *view_outputs])
        submit_btn.click(
            submit,
            inputs=[state, choice, *rating_list, comment],
            outputs=[state, *view_outputs, choice, *rating_list, comment],
        )
    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--generations-dir", default="results/generations")
    p.add_argument("--out", default="annotation/data/annotations.csv")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--share", action="store_true", help="create a public Gradio link")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    args = parse_args(argv)
    demo = build_app(args.generations_dir, args.out)
    demo.queue().launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
