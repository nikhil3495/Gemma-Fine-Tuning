import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recipe_gemma.eval import runner
from recipe_gemma.eval.gemini import GeminiClient
from recipe_gemma.eval.judge import combine, judge_item, judge_orders, parse_verdict
from recipe_gemma.eval.metrics import pairing_item_metrics, recipe_item_metrics
from recipe_gemma.eval.stats import bootstrap_ci, cohen_kappa, fleiss_kappa, win_rates
from recipe_gemma.flavor.graph import FlavorGraph
from recipe_gemma.flavor.normalize import IngredientNormalizer
from recipe_gemma.flavor.pmi import CooccurrenceStats
from recipe_gemma.flavor.scorer import PairingScorer


@pytest.fixture
def scorer_and_norm() -> tuple[PairingScorer, IngredientNormalizer]:
    recipes = [["tomato", "basil", "garlic"]] * 5 + [["chicken", "rice"]] * 5
    graph = FlavorGraph([("tomato", "basil", 37), ("tomato", "garlic", 4)])
    graph.add_node("chicken")
    norm = IngredientNormalizer(graph.nodes | {"rice"})
    return PairingScorer(CooccurrenceStats.from_recipes(recipes), graph), norm


# --------------------------------------------------------------------------- stats
def test_bootstrap_ci_contains_mean_and_handles_nan() -> None:
    ci = bootstrap_ci([0, 1] * 50 + [math.nan], n_boot=500)
    assert ci["n"] == 100 and ci["mean"] == 0.5 and ci["lo"] < 0.5 < ci["hi"]
    assert math.isnan(bootstrap_ci([])["mean"])
    const = bootstrap_ci([0.3] * 10)
    assert const["lo"] == pytest.approx(0.3) and const["hi"] == pytest.approx(0.3)


def test_win_rates_sum_to_one() -> None:
    wr = win_rates(["win", "win", "tie", "loss"])
    assert wr["win"]["mean"] == 0.5 and wr["tie"]["mean"] == 0.25 and wr["loss"]["mean"] == 0.25


def test_kappas() -> None:
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    assert fleiss_kappa([[2, 0], [0, 2], [2, 0]]) == pytest.approx(1.0)


def test_perplexity_on_toy_distribution() -> None:
    # Uniform over 4 tokens: NLL per token = log 4 -> PPL = 4 exactly.
    per_item = np.array([[3 * math.log(4), 3], [5 * math.log(4), 5]])
    s = runner.perplexity_summary(per_item, n_boot=200, seed=0)
    assert s["ppl"] == pytest.approx(4.0) and s["lo"] == pytest.approx(4.0)
    assert s["tokens"] == 8 and s["n"] == 2


# --------------------------------------------------------------------------- metrics
def test_recipe_metrics(scorer_and_norm) -> None:
    scorer, norm = scorer_and_norm
    text = (
        "Title: Bruschetta\n\nIngredients:\n- 2 tomatoes\n- 4 basil leaves\n- 1 clove garlic\n"
        "- 1 cup unicorn dust\n\nSteps:\n1. Chop.\n2. Serve.\n\nFlavor notes: fresh"
    )
    m = recipe_item_metrics(text, ["tomato", "basil", "rice"], norm, scorer)
    assert m["well_formed"] == 1.0 and m["all_requested"] == 0.0
    assert m["requested_recall"] == pytest.approx(2 / 3)
    assert m["unrecognized_rate"] == pytest.approx(1 / 4)  # unicorn dust
    # distinct pairs among {basil, garlic, tomato, unicorn_dust}: 6 ; PMI>0 for 3 herb pairs
    assert m["n_pairs"] == 6 and m["pmi_positive"] == pytest.approx(3 / 6)
    assert m["shares_compound"] == pytest.approx(2 / 6)
    assert m["graph_covered"] == pytest.approx(3 / 6)


def test_pairing_metrics(scorer_and_norm) -> None:
    scorer, norm = scorer_and_norm
    out = (
        "1. Basil - classic\n2. Garlic - aromatic\n3. Chicken - protein\n"
        "4. Tomato - again\n5. Rice - x"
    )
    m = pairing_item_metrics(out, "tomato", norm, scorer)
    assert m["format_ok"] == 1.0 and m["anchor_repeated"] == 1.0
    assert m["n_pairs"] == 4  # anchor itself excluded
    assert m["pmi_positive"] == pytest.approx(2 / 4)
    assert m["shares_compound"] == pytest.approx(2 / 4)
    bad = pairing_item_metrics("I don't know.", "tomato", norm, scorer)
    assert bad["format_ok"] == 0.0 and math.isnan(bad["pmi_positive"])


# --------------------------------------------------------------------------- judge
def test_judge_orders_randomized_but_deterministic() -> None:
    firsts = {judge_orders(f"item{i}", ("ft", "base"), 0)[0] for i in range(40)}
    assert firsts == {("ft", "base"), ("base", "ft")}  # both orders occur first
    assert judge_orders("x", ("ft", "base"), 0) == judge_orders("x", ("ft", "base"), 0)
    o = judge_orders("x", ("ft", "base"), 0)
    assert set(o) == {("ft", "base"), ("base", "ft")}  # both orders always judged


def test_parse_verdict_and_combine() -> None:
    v = parse_verdict('blah {"overall": "a", "coherence": "B", "feasibility": "x"} end')
    assert v["overall"] == "A" and v["coherence"] == "B" and v["feasibility"] == "tie"
    assert parse_verdict("no json")["overall"] == "tie"
    assert combine("win", "win") == "win" and combine("win", "loss") == "tie"


def test_judge_item_cancels_position_bias() -> None:
    always_a = lambda prompt: '{"overall": "A"}'  # noqa: E731 - pure position bias
    r = judge_item(always_a, "q", {"ft": "x", "base": "y"}, "i1", ("ft", "base"), 0)
    assert r["overall"] == "tie"

    def prefers_ft(prompt: str) -> str:
        a = prompt.split("### Response A\n")[1].split("\n\n### Response B")[0]
        return '{"overall": "A"}' if a == "GOOD" else '{"overall": "B"}'

    r = judge_item(prefers_ft, "q", {"ft": "GOOD", "base": "BAD"}, "i2", ("ft", "base"), 0)
    assert r["overall"] == "win"


# --------------------------------------------------------------------------- gemini mock
class _FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_content(self, model: str, contents: str, config) -> SimpleNamespace:
        self.calls.append({"model": model, "contents": contents})
        if len(self.calls) == 1:
            raise RuntimeError("transient 503")
        return SimpleNamespace(text=f"  echo: {contents}  ")


def test_gemini_client_with_mock_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake = SimpleNamespace(models=_FakeModels())
    client = GeminiClient("gemini-test", sdk_client=fake, seed=1)
    assert client("hello") == "echo: hello"
    assert len(fake.models.calls) == 2 and fake.models.calls[-1]["model"] == "gemini-test"


def test_gemini_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        GeminiClient("gemini-test")


# --------------------------------------------------------------------------- runner e2e
def test_runner_end_to_end_tiny(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("keras")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    root = Path(__file__).resolve().parents[1]
    prompts = [
        json.loads(line) for line in (root / "tests/fixtures/val.jsonl").open(encoding="utf-8")
    ]
    for i, p in enumerate(prompts):
        p["eval_rank"] = i
    (tmp_path / "prompts.jsonl").write_text(
        "\n".join(json.dumps(p) for p in prompts), encoding="utf-8"
    )
    graph = FlavorGraph([("tomato", "basil", 3)])
    graph.to_file(tmp_path / "g.json.gz")
    CooccurrenceStats.from_recipes([["tomato", "basil"]]).to_file(tmp_path / "c.json.gz")
    cfg = {
        "seed": 0,
        "n_per_task": 5,
        "eval_prompts": str(tmp_path / "prompts.jsonl"),
        "graph": str(tmp_path / "g.json.gz"),
        "cooccur": str(tmp_path / "c.json.gz"),
        "model": {"preset": "tiny-random", "adapter": str(tmp_path / "missing.lora.npz")},
        "models": ["base", "finetuned", "gemini"],
        "decoding": {
            "sampler": "greedy", "temperature": 1.0, "k": 1, "p": 1.0, "seed": 0,
            "batch_size": 2, "max_length": {"recipe": 260, "pairing": 140},
            "max_output_tokens": {"recipe": 10, "pairing": 10},
        },
        "gemini": {"model": "gemini-x"},
        "judge": {"enabled": True, "model": "gemini-x", "pairs": [["finetuned", "base"]]},
        "perplexity": {"seq_len": 448, "batch_size": 2},
        "bootstrap": {"n": 50, "seed": 0},
        "out_dir": str(tmp_path / "res"),
        "generations_dir": str(tmp_path / "res" / "gen"),
    }  # fmt: skip
    out = runner.run(cfg, {"generate", "perplexity", "metrics", "judge"})
    assert out["status"]["base"] == "ran"
    assert out["status"]["finetuned"].startswith("not run (adapter missing")
    assert out["status"]["gemini"] == runner.NO_KEY
    assert out["judge"]["status"] == runner.NO_KEY
    assert "proxy" in out["judge"]["label"]
    ppl = json.loads((tmp_path / "res" / "perplexity.json").read_text(encoding="utf-8"))
    assert ppl["by_model"]["base"]["all"]["ppl"] > 1
    assert "not comparable" in ppl["by_model"]["gemini"]["status"]
    assert "base" in out["metrics"]["metrics"]
    # cached: a second run does not regenerate
    gen_file = tmp_path / "res" / "gen" / "base.jsonl"
    before = gen_file.read_text(encoding="utf-8")
    runner.run(cfg, {"generate"})
    assert gen_file.read_text(encoding="utf-8") == before
