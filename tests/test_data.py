from recipe_gemma.data.build import assign_split, flavor_notes, stable_bucket
from recipe_gemma.data.clean import (
    Recipe,
    clean_recipe,
    clean_text,
    dedupe,
    is_valid,
    merge_step_fragments,
    normalize_units,
    parse_recipe_text,
)
from recipe_gemma.data.prompts import (
    END_TURN,
    START_MODEL,
    START_USER,
    chat_prompt,
    format_pairing_target,
    format_recipe_target,
    pairing_user_message,
    parse_pairing_output,
    parse_recipe_output,
    recipe_user_message,
)
from recipe_gemma.flavor.graph import FlavorGraph

RAW = """No-Bake Nut Cookies

Ingredients:
- 1 c. firmly packed brown sugar
- 1/2 c. evaporated milk
- 2 Tbsp. butter or margarine

Directions:
- In a heavy 2-quart saucepan, mix brown sugar
- and milk.
- Stir over medium heat until mixture bubbles all over top.
"""


def test_clean_text_strips_html_and_entities() -> None:
    assert clean_text("<b>Mac &amp; cheese</b>\t  now") == "Mac & cheese now"


def test_normalize_units() -> None:
    assert normalize_units("1 c. milk") == "1 cup milk"
    assert normalize_units("2 Tbsp. butter") == "2 tbsp butter"
    assert normalize_units("1/2 tsp. salt") == "1/2 tsp salt"
    assert normalize_units("1 (8 oz.) pkg. cream cheese") == "1 (8 oz) package cream cheese"
    assert normalize_units("3 teaspoons sugar") == "3 tsp sugar"


def test_parse_and_clean_recipe_merges_fragments() -> None:
    r = clean_recipe(parse_recipe_text(RAW))
    assert r.title == "No-Bake Nut Cookies"
    assert r.ingredients[0] == "1 cup firmly packed brown sugar"
    assert len(r.ingredients) == 3
    assert r.steps == [
        "In a heavy 2-quart saucepan, mix brown sugar and milk.",
        "Stir over medium heat until mixture bubbles all over top.",
    ]


def test_merge_step_fragments_keeps_trailing() -> None:
    assert merge_step_fragments(["Mix", "well.", "Serve"]) == ["Mix well.", "Serve"]


def test_parse_returns_none_on_empty() -> None:
    assert parse_recipe_text("") is None


def test_is_valid_thresholds() -> None:
    ok = Recipe("T", ["a", "b", "c"], ["s1.", "s2."])
    assert is_valid(ok)
    assert not is_valid(Recipe("T", ["a", "b"], ["s1.", "s2."]))
    assert not is_valid(Recipe("T", ["a", "b", "c"], ["s1."]))
    assert not is_valid(ok, max_chars=3)


def test_dedupe_exact_and_title() -> None:
    a = Recipe("Apple Pie", ["a", "b", "c"], ["x.", "y."])
    a_copy = Recipe("Apple Pie", ["a", "b", "c"], ["x.", "y."])
    a_retitled = Recipe("apple  pie!", ["d", "e", "f"], ["z.", "w."])
    b = Recipe("Banana Bread", ["a", "b", "c"], ["x.", "y."])
    out = list(dedupe([a, a_copy, a_retitled, b]))
    assert [r.title for r in out] == ["Apple Pie", "Banana Bread"]


def test_split_is_deterministic_and_proportional() -> None:
    fr = {"train": 0.9, "val": 0.05, "test": 0.05}
    keys = [f"r{i}" for i in range(4000)]
    s1 = [assign_split(k, 7, fr) for k in keys]
    assert s1 == [assign_split(k, 7, fr) for k in keys]
    assert 0.87 < s1.count("train") / len(keys) < 0.93
    assert 0.0 <= stable_bucket("x", 1) < 1.0


def test_chat_prompt_format() -> None:
    p = chat_prompt("hi")
    assert p == f"{START_USER}hi{END_TURN}\n{START_MODEL}"


def test_recipe_and_pairing_messages() -> None:
    msg = recipe_user_message(["olive_oil", "tomato"], "Southern European")
    assert msg.startswith("Create a recipe using: olive oil, tomato. Cuisine: Southern European")
    assert pairing_user_message("bell_pepper") == (
        "Suggest 5 ingredients that pair well with bell pepper, with a one-line reason each."
    )


def test_recipe_target_roundtrips_through_parser() -> None:
    text = format_recipe_target(
        "Toast", ["1 slice bread", "1 tbsp butter"], ["Toast.", "Butter."], "x"
    )
    parsed = parse_recipe_output(START_MODEL + text + END_TURN)
    assert parsed.title == "Toast"
    assert parsed.ingredients == ["1 slice bread", "1 tbsp butter"]
    assert parsed.steps == ["Toast.", "Butter."]
    assert parsed.notes == "x"
    assert parsed.well_formed


def test_parser_handles_markdown_variant() -> None:
    text = (
        "## Spicy Tomato Soup\n\n**Ingredients:**\n* 2 tomatoes\n* 1 onion\n\n"
        "**Instructions:**\n1. Chop.\n2. Simmer.\n"
    )
    parsed = parse_recipe_output(text)
    assert parsed.title == "Spicy Tomato Soup"
    assert parsed.ingredients == ["2 tomatoes", "1 onion"]
    assert parsed.steps == ["Chop.", "Simmer."]


def test_parser_not_well_formed() -> None:
    assert not parse_recipe_output("I cannot help with that.").well_formed


def test_pairing_target_and_parser() -> None:
    t = format_pairing_target("basil", [("tomato", "classic."), ("olive_oil", "rich.")])
    assert parse_pairing_output(t) == ["Tomato", "Olive oil"]
    md = "1. **Garlic**: aromatic\n2. Lemon – bright\n- Parmesan cheese (umami)"
    assert parse_pairing_output(md) == ["Garlic", "Lemon", "Parmesan cheese"]


def test_flavor_notes_prefers_non_staples() -> None:
    g = FlavorGraph([("tomato", "basil", 10), ("butter", "milk", 80)])
    notes = flavor_notes(
        ["butter", "milk", "tomato", "basil"], g, "any", frozenset({"butter", "milk"})
    )
    assert notes.startswith("A dish built on compatible flavors: basil and tomato share 10")
