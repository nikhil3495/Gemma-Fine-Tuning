import math
from pathlib import Path

from recipe_gemma.flavor.cuisine import CuisineTagger
from recipe_gemma.flavor.graph import FlavorGraph
from recipe_gemma.flavor.normalize import IngredientNormalizer, clean_ingredient_line, singularize
from recipe_gemma.flavor.pmi import CooccurrenceStats
from recipe_gemma.flavor.scorer import PairingScorer

VOCAB = {"tomato", "basil", "garlic", "cream_cheese", "olive_oil", "chicken", "bell_pepper", "egg"}


def test_singularize() -> None:
    assert singularize("tomatoes") == "tomato"
    assert singularize("berries") == "berry"
    assert singularize("leaves") == "leaf"
    assert singularize("molasses") == "molasses"
    assert singularize("glass") == "glass"


def test_clean_ingredient_line() -> None:
    assert clean_ingredient_line("2 cups chopped fresh tomatoes, seeded") == "tomato"
    assert clean_ingredient_line("1 (8 oz.) package cream cheese, softened") == "cream cheese"


def test_normalizer_aliases_and_ngrams() -> None:
    n = IngredientNormalizer(VOCAB)
    assert n("3 boneless skinless chicken breasts") == "chicken"
    assert n("1 (8 oz) package cream cheese, softened") == "cream_cheese"
    assert n("2 Tbsp extra virgin olive oil") == "olive_oil"
    assert n("1 green pepper, diced") == "bell_pepper"
    assert n("2 large eggs") == "egg"
    assert n("1 tsp. garlic powder") == "garlic"
    assert n("1 cup zzz mystery thing") == "zzz_mystery_thing"  # out-of-vocab passthrough
    assert n.canonical("Tomatoes") == "tomato"


def test_graph_roundtrip(tmp_path: Path) -> None:
    g = FlavorGraph([("tomato", "basil", 37), ("garlic", "basil", 5)])
    g.add_node("salt")
    assert g.shared_compounds("basil", "tomato") == 37
    assert g.shared_compounds("tomato", "garlic") == 0
    assert g.covered("salt", "tomato") and not g.covered("unicorn")
    assert g.neighbors("basil") == [("tomato", 37), ("garlic", 5)]
    p = tmp_path / "g.json.gz"
    g.to_file(p)
    g2 = FlavorGraph.from_file(p)
    assert g2.nodes == g.nodes and g2.num_edges == 2


def test_graph_parses_ahn_csv(tmp_path: Path) -> None:
    p = tmp_path / "s2.csv"
    p.write_text("#\n# header\nbasil,tomato,37\nsoy_sauce,ginger,4\n", encoding="utf-8")
    g = FlavorGraph.from_ahn_csv(p)
    assert g.shared_compounds("tomato", "basil") == 37 and len(g) == 4


def test_pmi_math() -> None:
    # 4 recipes: a&b always together (2 docs), c alone twice.
    s = CooccurrenceStats.from_recipes([["a", "b"], ["a", "b"], ["c"], ["c", "c"]])
    assert s.n_docs == 4 and s.count("a", "b") == 2 and s.count("b", "a") == 2
    # p(a)=p(b)=p(ab)=0.5 -> PMI = log(0.5/(0.25)) = log 2 ; NPMI = log2 / -log0.5 = 1
    assert math.isclose(s.pmi("a", "b"), math.log(2))
    assert math.isclose(s.npmi("a", "b"), 1.0)
    assert s.pmi("a", "c") == -math.inf and s.npmi("a", "c") == -1.0
    assert math.isnan(s.pmi("a", "zzz"))


def test_pmi_serialization(tmp_path: Path) -> None:
    s = CooccurrenceStats.from_recipes([["a", "b"], ["a", "b", "c"]])
    s.to_file(tmp_path / "s.json.gz", min_pair_count=2)
    s2 = CooccurrenceStats.from_file(tmp_path / "s.json.gz")
    assert s2.count("a", "b") == 2 and s2.count("a", "c") == 0 and s2.single["c"] == 1


def test_scorer_rank_and_flags() -> None:
    recipes = [["tomato", "basil", "garlic"]] * 6 + [["chicken", "egg"]] * 6 + [["tomato"]] * 2
    stats = CooccurrenceStats.from_recipes(recipes)
    graph = FlavorGraph([("tomato", "basil", 37)])
    sc = PairingScorer(stats, graph)
    ranked = sc.rank("tomato", k=5, min_count=3)
    assert [r.partner for r in ranked] == ["basil", "garlic"]  # compound bonus breaks the tie
    s = sc.score("tomato", "basil")
    assert s.pmi_positive and s.shares_compound and s.in_graph
    bad = sc.score("tomato", "chicken")
    assert not bad.pmi_positive and not bad.shares_compound


def test_cuisine_tagger() -> None:
    rows = [("EastAsian", ["soy_sauce", "ginger", "scallion"])] * 20 + [
        ("SouthernEuropean", ["olive_oil", "tomato", "basil"])
    ] * 20
    t = CuisineTagger().fit(rows)
    assert t.predict(["soy_sauce", "ginger", "scallion"]) == "East Asian"
    assert t.predict(["olive_oil", "tomato", "basil"]) == "Southern European"
    assert t.predict(["unknown"]) == "any"  # too little evidence
