"""Recipe parsing, cleaning, filtering and de-duplication."""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t ]+")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")

# Abbreviation -> normalized unit. Applied with word boundaries, case-insensitive.
UNIT_MAP: list[tuple[str, str]] = [
    (r"\btbsp?s?\.?(?=\s|$)", "tbsp"),
    (r"\btablespoons?\b", "tbsp"),
    (r"\btsps?\.?(?=\s|$)", "tsp"),
    (r"\bteaspoons?\b", "tsp"),
    (r"\bc\.(?=\s|$)", "cup"),
    (r"\bcups?\b", "cup"),
    (r"\blbs?\.?(?=\s|$)", "lb"),
    (r"\bpounds?\b", "lb"),
    (r"\boz\.?(?=\s|$|\))", "oz"),
    (r"\bounces?\b", "oz"),
    (r"\bpkgs?\.?(?=\s|$)", "package"),
    (r"\bqts?\.?(?=\s|$)", "quart"),
    (r"\bpts?\.(?=\s|$)", "pint"),
    (r"\bgals?\.?(?=\s|$)", "gallon"),
]
_UNIT_RES = [(re.compile(p, re.IGNORECASE), r) for p, r in UNIT_MAP]


@dataclass
class Recipe:
    """A parsed recipe."""

    title: str
    ingredients: list[str]
    steps: list[str]
    canon: list[str] = field(default_factory=list)  # canonical ingredient names
    cuisine: str = "any"
    uid: str = ""

    def to_dict(self) -> dict:
        """Plain-dict view."""
        return asdict(self)


def clean_text(text: str) -> str:
    """Unescape HTML entities, drop tags, normalize whitespace."""
    text = html.unescape(text or "")
    text = _TAG.sub(" ", text)
    text = _WS.sub(" ", text)
    return text.strip()


def normalize_units(line: str) -> str:
    """Normalize common unit abbreviations: ``"1 c. milk"`` -> ``"1 cup milk"``."""
    for pat, rep in _UNIT_RES:
        line = pat.sub(rep, line)
    return _WS.sub(" ", line).strip()


def merge_step_fragments(steps: list[str]) -> list[str]:
    """Join direction fragments that were split mid-sentence (a known corpus artifact).

    A fragment is appended to the previous one until the accumulated text ends with terminal
    punctuation (``. ! ? )``).
    """
    merged: list[str] = []
    buf = ""
    for s in steps:
        s = s.strip()
        if not s:
            continue
        buf = f"{buf} {s}".strip() if buf else s
        if buf[-1] in ".!?)":
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return merged


def parse_recipe_text(text: str) -> Recipe | None:
    """Parse the ``Title / Ingredients: / Directions:`` text format of the corpus."""
    if not text:
        return None
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    title = ""
    section = None
    ingredients: list[str] = []
    steps: list[str] = []
    for raw in lines:
        line = clean_text(raw)
        if not line:
            continue
        low = line.lower().rstrip(":")
        if low in {"ingredients"}:
            section = "ing"
            continue
        if low in {"directions", "instructions", "steps", "method"}:
            section = "dir"
            continue
        if section is None:
            if not title:
                title = line
            continue
        item = _BULLET.sub("", line).strip()
        if not item:
            continue
        (ingredients if section == "ing" else steps).append(item)
    if not title:
        return None
    return Recipe(title=title, ingredients=ingredients, steps=steps)


def clean_recipe(r: Recipe) -> Recipe:
    """Clean title/ingredients/steps text and normalize units."""
    title = clean_text(r.title).strip(" .")
    title = title[:1].upper() + title[1:] if title else title
    ingredients = [normalize_units(clean_text(i)) for i in r.ingredients]
    steps = merge_step_fragments([clean_text(s) for s in r.steps])
    return Recipe(
        title=title,
        ingredients=[i for i in ingredients if i],
        steps=[s for s in steps if s],
        canon=list(r.canon),
        cuisine=r.cuisine,
        uid=r.uid,
    )


def is_valid(
    r: Recipe, min_ingredients: int = 3, min_steps: int = 2, max_chars: int = 2500
) -> bool:
    """Keep recipes with enough ingredients/steps and bounded length."""
    if len(r.ingredients) < min_ingredients or len(r.steps) < min_steps:
        return False
    total = len(r.title) + sum(map(len, r.ingredients)) + sum(map(len, r.steps))
    return total <= max_chars


def normalized_title(title: str) -> str:
    """Lowercase alphanumerics only (used for near-duplicate detection)."""
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def content_hash(r: Recipe) -> str:
    """SHA-1 of the whitespace/case-normalized full recipe."""
    blob = "\n".join([r.title, *r.ingredients, *r.steps]).lower()
    blob = re.sub(r"\s+", " ", blob)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def dedupe(recipes: Iterable[Recipe]) -> Iterator[Recipe]:
    """Drop exact duplicates (content hash) and normalized-title duplicates; keep first seen."""
    seen_hash: set[str] = set()
    seen_title: set[str] = set()
    for r in recipes:
        h = content_hash(r)
        t = normalized_title(r.title)
        if h in seen_hash or t in seen_title:
            continue
        seen_hash.add(h)
        seen_title.add(t)
        yield r
