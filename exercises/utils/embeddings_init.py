import gensim.downloader
model = gensim.downloader.load("glove-wiki-gigaword-50") # replace with "word2vec-google-news-300" for Google News Word2Vec (Warning: much bigger size!)
print("Vocabulary size:", len(model.key_to_index))

categories = {
    "months": [
        "january", "february", "march", "april", "june", "july",
        "august", "september", "october", "november", "december"
    ],
    "seasons": [
        "spring", "summer", "fall", "autumn", "winter"
    ],
    "professions_and_people": [
        "man", "woman", "nurse", "doctor", "uncle", "aunt"
    ],
    "animals": [
        "dog", "cat", "horse", "lion", "cow", "elephant", "tiger", "wolf",
        "squid", "dolphin", "whale", "eagle", "snake", "frog", "monkey",
        "deer", "rabbit"
    ],
    "vehicles": [
        "car", "bicycle", "motorcycle", "bus", "truck", "van", "train",
        "tram", "subway", "airplane", "helicopter", "boat", "ship",
        "submarine", "rocket", "scooter", "tractor", "tank", "skateboard"
    ],
    "vegetables": [
        "carrot", "broccoli", "spinach", "potato", "tomato",
        "corn", "cucumber", "banana", "pumpkin"
    ],
    "colors": [
        "red", "green", "blue", "yellow", "white", "black", "orange", "purple", "brown"
    ],
    "nature": [
        "sky", "grass", "plant", "tree", "sun", "ocean", "leaf", "fire", "blood"
    ],
    "food": [
        "apple", "banana", "orange", "strawberry", "grape", "fruit",
        "pastry", "bread", "cake", "pie", "cookie", "rose"
    ],
    "education": [
        "school", "book", "pencil", "desk", "teacher", "backpack", "science"
    ],
    "sports": [
        "sport", "ball", "team", "goal", "run", "coach"
    ],
    "art": [
        "art", "paint", "brush", "color", "canvas", "draw"
    ]
}

"""
Interactive Similarity Explorer (Gensim-based)
----------------------------------------------
Uses a preloaded Gensim model, e.g.:
    model = gensim.downloader.load("glove-wiki-gigaword-50")

Lets you:
- Search for a token and find its most similar words.
- Restrict the search to selected semantic categories.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
from difflib import SequenceMatcher

# ---------- HELPERS ----------
def levenshtein_like(a, b):
    """Approximate text similarity ratio (for spelling suggestions)."""
    return SequenceMatcher(None, a, b).ratio()

# Build category lookup (optional, for filtering)
token_to_cat = {t: cat for cat, toks in categories.items() for t in toks if t in model.key_to_index}

# ---------- INTERACTIVE EXPLORER ----------
def create_interactive_explorer(model):
    search_box = widgets.Text(
        placeholder="Type a token...",
        description="Token:",
        style={"description_width": "80px"},
        layout=widgets.Layout(width="350px")
    )

    category_checkboxes = {
        cat: widgets.Checkbox(
            value=True,
            description=cat.replace("_", " ").title(),
            indent=False,
            layout=widgets.Layout(width="200px")
        )
        for cat in categories
    }

    checkbox_grid = widgets.GridBox(
        list(category_checkboxes.values()),
        layout=widgets.Layout(grid_template_columns="repeat(3, 200px)")
    )

    output = widgets.Output()

    def on_submit(change):
        token = change["new"].strip()

        with output:
            clear_output()

            # Validate token
            if token not in model.key_to_index:
                print(f"⚠️ Token '{token}' not found in model vocabulary.")
                return

            # Collect allowed tokens from selected categories
            active_cats = [c for c, cb in category_checkboxes.items() if cb.value]
            allowed_tokens = {t for c in active_cats for t in categories[c] if t in model.key_to_index and t != token}

            if not allowed_tokens:
                print("⚠️ No valid tokens found in selected categories.")
                return

            # Compute cosine similarity for all allowed tokens
            results = []
            for t in allowed_tokens:
                try:
                    sim = model.similarity(token, t)
                    results.append((t, sim))
                except KeyError:
                    continue

            # Sort by descending similarity
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:5]  # top 5

            if not results:
                print("⚠️ No results in selected categories.")
                return

            # Display
            print(f"Nearest neighbors for '{token}' (filtered by categories):\n")
            for t, sim in results:
                cat = token_to_cat.get(t, "unknown").replace("_", " ").title()
                print(f"  {t:15s}  similarity = {sim:.3f}   [{cat}]")

    # Observe user input
    search_box.observe(on_submit, names="value")

    # Update when checkboxes change
    def update_output(*_):
        if search_box.value.strip():
            on_submit({"new": search_box.value})
    for cb in category_checkboxes.values():
        cb.observe(update_output, names="value")

    display(widgets.VBox([
        search_box,
        widgets.Label("Select categories to include in the search:"),
        checkbox_grid,
        output
    ]))
