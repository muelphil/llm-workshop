# --- Inititialization ---

# --- Core GPT-2 BPE tokenizer (JS port), self-contained using encoder.json + vocab.bpe.txt ---

import json, hashlib, sys, html
from pathlib import Path

# We need Unicode character properties in the regex (like \p{L}, \p{N}).
# The 'regex' module supports these; fall back to 're' with a simpler pattern if unavailable.
try:
    import regex as re
    _HAS_REGEX = True
except ImportError:
    import re
    _HAS_REGEX = False
    print(
        "Note: The exact GPT-2 tokenization pattern uses the 'regex' package "
        "for \\p{L}/\\p{N}. Falling back to a simplified pattern.\n"
        "Install with: pip install regex",
        file=sys.stderr
    )

# --- Load resources (must be present in working dir) ---
ENCODER_PATH = Path("encoder.json")
BPE_MERGES_PATH = Path("vocab.bpe.txt")

with ENCODER_PATH.open("r", encoding="utf-8") as f:
    encoder = json.load(f)  # { token_string: token_id }

# Build decoder: { token_id: token_string }
decoder = {v: k for k, v in encoder.items()}

with BPE_MERGES_PATH.open("r", encoding="utf-8") as f:
    lines = f.read().splitlines()

# Skip first line; last line may be empty
bpe_merges = [tuple(line.split()) for line in lines[1:] if line.strip()]
bpe_ranks = {merge: i for i, merge in enumerate(bpe_merges)}

# --- Bytes<->Unicode mapping (same scheme as JS/official GPT-2) ---
def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + \
         list(range(ord('¡'), ord('¬') + 1)) + \
         list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}

# --- Regex pattern (same as JS notebook) ---
if _HAS_REGEX:
    pat = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
else:
    # Simpler fallback (less accurate than GPT-2’s exact pattern)
    pat = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|\S+|\s+")

# --- Helpers for BPE ---
def get_pairs(word_tuple):
    """Return set of symbol pairs in a word (word is a tuple of symbols)."""
    pairs = set()
    prev_char = word_tuple[0]
    for ch in word_tuple[1:]:
        pairs.add((prev_char, ch))
        prev_char = ch
    return pairs

_cache = {}

def bpe(token):
    """Byte-Pair Encoding merge loop."""
    if token in _cache:
        return _cache[token]

    word = tuple(token)
    pairs = get_pairs(word)
    if not pairs:
        return token

    while True:
        # find lowest-ranked bigram
        rank_pairs = {bpe_ranks.get(pair, float('inf')): pair for pair in pairs}
        bigram = rank_pairs[min(rank_pairs.keys())]
        if bigram not in bpe_ranks:
            break

        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
            except ValueError:
                new_word.extend(word[i:])
                break
            new_word.extend(word[i:j])
            i = j
            if i < len(word)-1 and word[i] == first and word[i+1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        word = tuple(new_word)
        if len(word) == 1:
            break
        pairs = get_pairs(word)

    out = " ".join(word)
    _cache[token] = out
    return out

# --- Encode & decode ---
def _encode_str(s: str):
    """Return the 'byte->unicode' mapped string used by GPT-2 BPE."""
    return "".join(byte_encoder[b] for b in s.encode("utf-8"))

def _decode_str(us: str):
    """Inverse of _encode_str."""
    return bytes(byte_decoder[c] for c in us).decode("utf-8", errors="strict")

def encode(text: str):
    """Encode text -> list[int] (token ids)."""
    bpe_tokens = []
    for match in pat.findall(text):
        token = _encode_str(match)
        bpe_out = bpe(token)
        bpe_piece_strings = bpe_out.split(" ")
        bpe_tokens.extend(encoder[p] for p in bpe_piece_strings)
    return bpe_tokens

def decode(tokens):
    """Decode list[int] -> str."""
    s = "".join(decoder[t] for t in tokens)
    return _decode_str(s)

# --- Bundle tokens for per-token visualization (handles multi-token chars like Japanese) ---
def bundle_tokens(tokens):
    """Faithful port of the JS bundleTokens."""
    bundles = []
    copied = list(tokens)
    current = []
    while copied:
        current.append(copied.pop(0))
        # If first code point is U+FFFD (replacement char), keep merging
        try:
            chunk = decode(current)
        except:
            continue
        bundles.append({"text": chunk, "tokens": current})
        current = []
    return bundles

# --- Color helper (similar to stringToPastelColor in JS) ---
def string_to_pastel_color(s: str) -> str:
    h = hashlib.sha256(s.encode("utf-8")).digest()[:3]
    # Adjust toward lighter range
    r, g, b = [(x // 2) + 128 for x in h]
    return f"#{r:02x}{g:02x}{b:02x}"


# --- Build HTML spans for a text (replicates the Observable styling closely) ---
def generate_spans_html(text: str) -> str:
    """Faithful port of the JS generateSpans using bundleTokens + string_to_pastel_color."""
    tokens = encode(text)
    bundles = bundle_tokens(tokens)
    parts = []
    for t in bundles:
        color = string_to_pastel_color(t["text"])
        token_ids = "<br>".join(str(x) for x in t["tokens"])
        # Escape HTML, then mimic JS: replace "\n" -> <br>, and only the FIRST " " -> &nbsp;
        rendered_text = html.escape(t["text"]).replace("\n", "<br>").replace(" ", "&nbsp;", 1)
        parts.append(
            f'''<span title="{html.escape(str(t["tokens"]))}" style="
        padding: 3px;
        border-right: 3px solid white;
        line-height: 3em;
        font-family: courier;
        background-color: {color};
        position: relative;
      "><span style="position:absolute; top: 100%; line-height:1em; left:-0.5px; font-size:0.7em">{token_ids}</span>{rendered_text}</span>'''
        )
    return "<div>" + "".join(parts) + "</div>"