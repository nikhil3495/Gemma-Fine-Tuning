"""Ingredient-name normalization: free-text ingredient line -> canonical ingredient name.

Pipeline: lowercase -> drop parentheticals/quantities/units/descriptors -> singularize ->
alias map -> longest n-gram match against a known vocabulary (the flavor-graph nodes plus
corpus ingredients). Pure Python, deterministic, no network.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

UNITS = {
    "c", "cup", "cups", "tsp", "teaspoon", "teaspoons", "tbsp", "tablespoon", "tablespoons",
    "tbs", "t", "oz", "ounce", "ounces", "lb", "lbs", "pound", "pounds", "g", "gram", "grams",
    "kg", "ml", "l", "liter", "liters", "litre", "qt", "quart", "quarts", "pt", "pint", "pints",
    "gal", "gallon", "pkg", "package", "packages", "can", "cans", "jar", "jars", "bottle",
    "box", "boxes", "bag", "carton", "envelope", "stick", "sticks", "slice", "slices", "clove",
    "cloves", "head", "heads", "bunch", "bunches", "sprig", "sprigs", "pinch", "dash", "piece",
    "pieces", "whole", "large", "medium", "small", "lg", "med", "sm", "container", "packet",
    "handful", "inch", "in", "x", "squares", "square", "loaf", "pkt", "env", "ctn", "jigger",
}  # fmt: skip
DESCRIPTORS = {
    "fresh", "freshly", "chopped", "diced", "minced", "sliced", "grated", "shredded", "crushed",
    "ground", "dried", "dry", "frozen", "canned", "cooked", "uncooked", "raw", "softened",
    "melted", "beaten", "peeled", "seeded", "cubed", "halved", "quartered", "finely", "coarsely",
    "thinly", "roughly", "lightly", "packed", "firmly", "heaping", "level", "about", "approx",
    "to", "taste", "optional", "divided", "plus", "more", "for", "of", "and", "or", "the", "a",
    "an", "into", "cut", "drained", "rinsed", "trimmed", "boneless", "skinless", "room",
    "temperature", "cold", "warm", "hot", "lukewarm", "extra", "virgin", "light", "dark",
    "low", "fat", "reduced", "sodium", "unsalted", "salted", "sweetened", "unsweetened", "plain",
    "pitted", "toasted", "roasted", "lean", "boiling", "well", "very", "ripe", "firm",
    "torn", "mashed", "pureed", "prepared", "instant", "quick", "favorite", "your", "good",
    "quality", "organic", "homemade", "store", "bought", "each", "few", "several", "some",
    "with", "without", "in", "on", "at", "fine", "coarse", "bite", "size", "inch-thick",
    "thick", "thin", "big", "little", "jumbo", "baby", "young", "mature", "pkg.",
}  # fmt: skip

# Common spelling variants -> canonical names (canonical names use flavor-graph spelling).
ALIASES = {
    "scallions": "scallion", "green onion": "scallion", "spring onion": "scallion",
    "green onions": "scallion", "cilantro": "coriander", "coriander leaf": "coriander",
    "confectioners sugar": "sugar", "powdered sugar": "sugar", "brown sugar": "sugar",
    "granulated sugar": "sugar", "white sugar": "sugar", "all-purpose flour": "wheat",
    "all purpose flour": "wheat", "flour": "wheat", "self-rising flour": "wheat",
    "bread flour": "wheat", "margarine": "butter", "oleo": "butter", "egg yolk": "egg",
    "egg white": "egg", "eggs": "egg", "bell pepper": "bell_pepper", "green pepper": "bell_pepper",
    "red pepper": "bell_pepper", "sweet pepper": "bell_pepper", "chili pepper": "pepper",
    "black pepper": "black_pepper", "pepper": "black_pepper", "kosher salt": "salt",
    "sea salt": "salt", "garlic powder": "garlic", "onion powder": "onion",
    "chicken breast": "chicken", "chicken thigh": "chicken", "ground beef": "beef",
    "hamburger": "beef", "ground chuck": "beef", "heavy cream": "cream",
    "whipping cream": "cream", "sour cream": "sour_cream", "half and half": "cream",
    "evaporated milk": "milk", "buttermilk": "milk", "condensed milk": "milk",
    "parmesan": "parmesan_cheese", "parmesan cheese": "parmesan_cheese",
    "cheddar": "cheddar_cheese", "cheddar cheese": "cheddar_cheese",
    "mozzarella": "mozzarella_cheese", "mozzarella cheese": "mozzarella_cheese",
    "cream cheese": "cream_cheese", "olive oil": "olive_oil", "vegetable oil": "vegetable_oil",
    "canola oil": "vegetable_oil", "salad oil": "vegetable_oil", "oil": "vegetable_oil",
    "soy sauce": "soy_sauce", "lemon juice": "lemon_juice", "lime juice": "lime_juice",
    "vanilla extract": "vanilla", "baking soda": "baking_soda",
    "baking powder": "baking_powder", "cocoa powder": "cocoa",
    "chocolate chip": "cocoa", "semisweet chocolate": "cocoa", "chocolate": "cocoa",
    "molasses": "cane_molasses", "sausage": "pork_sausage", "cornstarch": "starch",
    "corn starch": "starch", "cinnamon stick": "cinnamon", "smoked sausage": "smoked_sausage",
    "maple syrup": "maple_syrup", "corn syrup": "corn_syrup", "rolled oat": "oat",
    "oatmeal": "oat", "tomato paste": "tomato", "tomato sauce": "tomato",
    "crushed tomato": "tomato", "garlic clove": "garlic", "ginger root": "ginger",
    "gingerroot": "ginger", "chile": "pepper", "chili": "pepper", "jalapeno": "pepper",
    "jalapeno pepper": "pepper", "courgette": "zucchini",
    "aubergine": "eggplant", "prawn": "shrimp",
    "peanut butter": "peanut_butter",
    "white wine": "white_wine", "red wine": "red_wine", "rice vinegar": "rice_vinegar",
    "wine vinegar": "wine_vinegar", "cider vinegar": "cider", "apple cider vinegar": "cider",
    "sesame oil": "sesame_oil", "fish sauce": "fish", "basil leaf": "basil",
    "bay leaf": "bay", "parsley flake": "parsley", "mint leaf": "mint",
}  # fmt: skip

_PAREN = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_NUM = re.compile(r"\b\d+([./-]\d+)?(\s*\d+/\d+)?\b|[¼½¾⅓⅔⅛]")
_NONWORD = re.compile(r"[^a-z\s\-']")
_WS = re.compile(r"\s+")


def singularize(word: str) -> str:
    """Very small English singularizer adequate for ingredient nouns."""
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    irregular = {"leaves": "leaf", "loaves": "loaf", "halves": "half", "knives": "knife"}
    if word in {"molasses", "hummus", "couscous", "swiss", "grits", "oats", "asparagus"}:
        return {"oats": "oat"}.get(word, word)
    if word in irregular:
        return irregular[word]
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("oes", "ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def clean_ingredient_line(line: str) -> str:
    """Strip quantities, units, parentheticals and descriptors; return a short name phrase."""
    text = line.lower().replace("&", " and ")
    text = _PAREN.sub(" ", text)
    text = text.split(",")[0].split(";")[0]
    text = _NUM.sub(" ", text)
    text = text.replace(".", " ")
    text = _NONWORD.sub(" ", text)
    tokens = [t.strip("-'") for t in _WS.split(text) if t.strip("-'")]
    kept = [t for t in tokens if t not in UNITS and t not in DESCRIPTORS]
    return " ".join(singularize(t) for t in kept).strip()


class IngredientNormalizer:
    """Map ingredient lines / names onto a canonical vocabulary.

    ``vocab`` holds canonical names (flavor-graph spelling, e.g. ``"olive_oil"``). Matching tries
    aliases, then the longest contiguous n-gram (<=3 words) present in the vocabulary, then the
    last word (the head noun), and otherwise returns the cleaned phrase joined by underscores.
    """

    def __init__(self, vocab: Iterable[str] = (), aliases: dict[str, str] | None = None) -> None:
        self.vocab: set[str] = {v.lower() for v in vocab}
        self.aliases = {k: v for k, v in (aliases or ALIASES).items()}

    def canonical(self, phrase: str) -> str:
        """Canonicalize an already-cleaned phrase (or a bare ingredient name)."""
        phrase = phrase.lower().replace("_", " ").strip()
        phrase = " ".join(singularize(w) for w in phrase.split())
        if not phrase:
            return ""
        if phrase in self.aliases:
            return self.aliases[phrase]
        words = phrase.split()
        for n in range(min(3, len(words)), 0, -1):  # longest n-gram first, rightmost first
            for i in range(len(words) - n, -1, -1):
                gram = " ".join(words[i : i + n])
                if gram in self.aliases:
                    return self.aliases[gram]
                key = gram.replace(" ", "_")
                if key in self.vocab:
                    return key
        return "_".join(words)

    def __call__(self, line: str) -> str:
        """Normalize a raw ingredient line such as ``"2 c. chopped fresh tomatoes"``."""
        return self.canonical(clean_ingredient_line(line))

    def in_vocab(self, name: str) -> bool:
        """Whether a canonical name is in the known vocabulary."""
        return name in self.vocab


def display_name(canonical: str) -> str:
    """``olive_oil`` -> ``olive oil``."""
    return canonical.replace("_", " ")
