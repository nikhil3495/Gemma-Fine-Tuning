"""Prompt templates (Gemma chat format), training targets, and model-output parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from recipe_gemma.flavor.normalize import display_name

START_USER = "<start_of_turn>user\n"
START_MODEL = "<start_of_turn>model\n"
END_TURN = "<end_of_turn>"

RECIPE_INSTRUCTION = "Create a recipe using: {ingredients}. Cuisine: {cuisine}"
RECIPE_FORMAT_HINT = (
    "\nRespond with sections 'Title:', 'Ingredients:' (with quantities), 'Steps:' (numbered) "
    "and 'Flavor notes:'."
)
PAIRING_INSTRUCTION = (
    "Suggest 5 ingredients that pair well with {ingredient}, with a one-line reason each."
)


def chat_prompt(user_message: str) -> str:
    """Wrap a user message in Gemma's chat template, ending where the model turn begins."""
    return f"{START_USER}{user_message}{END_TURN}\n{START_MODEL}"


def recipe_user_message(ingredients: list[str], cuisine: str) -> str:
    """Task A user message. ``ingredients`` are canonical or display names."""
    names = ", ".join(display_name(i) for i in ingredients)
    return RECIPE_INSTRUCTION.format(ingredients=names, cuisine=cuisine) + RECIPE_FORMAT_HINT


def pairing_user_message(ingredient: str) -> str:
    """Task B user message."""
    return PAIRING_INSTRUCTION.format(ingredient=display_name(ingredient))


def format_recipe_target(
    title: str, ingredients: list[str], steps: list[str], flavor_notes: str
) -> str:
    """Task A target text (what the fine-tuned model learns to emit)."""
    ing = "\n".join(f"- {i}" for i in ingredients)
    stp = "\n".join(f"{n}. {s}" for n, s in enumerate(steps, 1))
    return f"Title: {title}\n\nIngredients:\n{ing}\n\nSteps:\n{stp}\n\nFlavor notes: {flavor_notes}"


def format_pairing_target(anchor: str, partners: list[tuple[str, str]]) -> str:
    """Task B target: numbered ``partner - reason`` lines."""
    lines = [f"Ingredients that pair well with {display_name(anchor)}:"]
    for n, (partner, reason) in enumerate(partners, 1):
        lines.append(f"{n}. {display_name(partner).capitalize()} - {reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# Parsing model outputs (used by evaluation; tolerant of markdown and formatting drift)
# ---------------------------------------------------------------------------------------
_HDR = r"^\s*(?:#+\s*|\*\*)?\s*{name}\s*(?:\*\*)?\s*:?\s*(?:\*\*)?\s*"
_SECTIONS = {
    "title": re.compile(_HDR.format(name="(?:recipe\\s+)?(?:title|name)"), re.I | re.M),
    "ingredients": re.compile(_HDR.format(name="ingredients"), re.I | re.M),
    "steps": re.compile(
        _HDR.format(name="(?:steps|instructions|directions|method|preparation)"), re.I | re.M
    ),
    "notes": re.compile(_HDR.format(name="(?:flavor|flavour)\\s+notes?"), re.I | re.M),
}
_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$")


@dataclass
class ParsedRecipe:
    """Sections recovered from generated text."""

    title: str = ""
    ingredients: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def well_formed(self) -> bool:
        """Has a title, >=1 ingredient and >=1 step."""
        return bool(self.title) and bool(self.ingredients) and bool(self.steps)


def strip_chat(text: str) -> str:
    """Remove the prompt echo and turn markers from a generation."""
    if START_MODEL in text:
        text = text.split(START_MODEL, 1)[1]
    return text.split(END_TURN, 1)[0].strip()


def parse_recipe_output(text: str) -> ParsedRecipe:
    """Parse ``Title / Ingredients / Steps / Flavor notes`` from free-form generated text."""
    text = strip_chat(text)
    marks = []
    for name, pat in _SECTIONS.items():
        m = pat.search(text)
        if m:
            marks.append((m.start(), m.end(), name))
    marks.sort()
    out = ParsedRecipe()
    for idx, (_start, end, name) in enumerate(marks):
        stop = marks[idx + 1][0] if idx + 1 < len(marks) else len(text)
        body = text[end:stop].strip()
        if name == "title":
            out.title = body.splitlines()[0].strip(" *#") if body else ""
        elif name == "notes":
            out.notes = " ".join(body.split())
        else:
            items = [m.group(1).strip(" *") for m in map(_ITEM.match, body.splitlines()) if m]
            if not items:  # un-bulleted lines
                items = [ln.strip(" *") for ln in body.splitlines() if ln.strip(" *")]
            setattr(out, name, items)
    if not out.title:  # first markdown heading / bold line as a fallback title
        m = re.search(r"^\s*(?:#+\s*|\*\*)(.+?)(?:\*\*)?\s*$", text, re.M)
        if m and not any(p.match(m.group(0)) for p in _SECTIONS.values()):
            out.title = m.group(1).strip(" *#")
    return out


_PAIR_LINE = re.compile(
    r"^\s*(?:\d+[.)]|[-*•])\s*(?:\*\*)?([^:\-–—*(\n]+?)(?:\*\*)?\s*(?:[:\-–—(]|$)"
)


def parse_pairing_output(text: str, k: int = 5) -> list[str]:
    """Extract up to ``k`` suggested ingredient names (raw strings) from a pairing answer."""
    names = []
    for line in strip_chat(text).splitlines():
        m = _PAIR_LINE.match(line)
        if m:
            name = m.group(1).strip(" *")
            if name and len(name) <= 40:
                names.append(name)
        if len(names) >= k:
            break
    return names
