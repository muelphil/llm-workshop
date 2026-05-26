# --- Visualization of OpenAI Responses ---
import math
import numpy as np
from IPython.display import HTML, display
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
import os
from IPython.display import JSON, display
import json

# --- color maps (probability: red→white→green, perplexity: white→red) ---
_red = (252/255, 121/255, 125/255, 1.0)
_white = (1.0, 1.0, 1.0, 1.0)
_green = (166/255, 234/255, 173/255, 1.0)

_cm_prob = LinearSegmentedColormap.from_list("prob_cmap", [_red, _white, _green], N=100)
_cm_pplx = LinearSegmentedColormap.from_list("pplx_cmap", [_white, _red], N=100)

def _safe_exp(x: float) -> float:
    """exp(x) with handling for -inf."""
    if x is None:
        return 0.0
    if x == float("-inf"):
        return 0.0
    try:
        return float(math.exp(x))
    except Exception:
        return 0.0

def _norm01(arr):
    """Normalize a 1D array to [0,1] (returns zeros if constant/empty)."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)

def _to_text(token: str) -> str:
    """
    Decode token display the same way as your prior code:
    - 'Ġ' -> space
    - 'Ċ' -> '\n'
    - replace spaces with '░' to visualize spacing
    """
    return token.replace("Ġ", " ").replace("Ċ", "\\n").replace(" ", "░")

def _get_attr_or_item(obj, key, default=None):
    """Try attribute first, then mapping access, else default."""
    if obj is None:
        return default
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default

def _normalize_top_logprobs(top) -> dict:
    """
    Normalize various SDK shapes of top_logprobs into {token: logprob} dict.
    Accepts:
      - list of objects (with .token and .logprob)
      - list of dicts (with ['token'], ['logprob'])
      - already a dict {token: logprob}
    """
    if top is None:
        return {}
    if isinstance(top, dict):
        return {str(k): float(v) for k, v in top.items()}
    if isinstance(top, (list, tuple)):
        out = {}
        for alt in top:
            tok = _get_attr_or_item(alt, "token", None)
            lp = _get_attr_or_item(alt, "logprob", None)
            if tok is not None and lp is not None:
                out[str(tok)] = float(lp)
        return out
    return {}

def _token_entropy_from_dict(top_dict: dict) -> float:
    """
    Shannon entropy (in nats) given {token: logprob}.
    Assumes values are log-probabilities that exponentiate to probabilities.
    """
    if not top_dict:
        return 0.0
    log_probs = np.array(list(top_dict.values()), dtype=float)
    probs = np.exp(log_probs)
    # guard against numerical issues
    probs = probs / (probs.sum() + 1e-12)
    return float(-np.sum(probs * np.log(probs + 1e-12)))

def _top_logprobs_to_html(top_dict: dict) -> str:
    """Build the hover table HTML with probability bar visualization."""
    if not top_dict:
        return '<table class="top-logprobs"><tr><td colspan="2">no alternatives</td></tr></table>'

    # Sort by probability (descending)
    items = sorted(top_dict.items(), key=lambda kv: _safe_exp(kv[1]), reverse=True)

    rows = []
    for tok, lp in items:
        prob = _safe_exp(float(lp))
        formatted = f"{prob:.3f}" if prob >= 0.001 else "<0.001"
        prob_pct = prob * 100  # convert to percent for CSS

        # Background gradient: blue (alpha=0.6) for first `prob_pct`, transparent rest
        bg_style = (
            f"background: linear-gradient(to right, "
            f"rgba(0, 120, 255, 0.6) {prob_pct:.1f}%, "
            f"rgba(0, 120, 255, 0.05) {prob_pct:.1f}%);"
        )

        rows.append(
            "<tr>"
            f"<td>{_to_text(str(tok))}</td>"
            f"<td style='{bg_style} text-align:right; padding:0 0.5em;'>{formatted}</td>"
            "</tr>"
        )

    return '<table class="top-logprobs" style="line-height: 1.4em;">' + "".join(rows) + "</table>"

def get_html_visualization(response, highlight: str = "probability"):
    """
    Build and display an HTML visualization of tokens with hoverable alternatives.
    - response: OpenAI API response object with logprobs (attribute-style access).
      Expected shape (OpenAI Chat API with logprobs=True):
        response.choices[0].logprobs.content -> list of entries
        each entry has: .token (str), .logprob (float), .top_logprobs (list/dict)
    - highlight: 'probability' or 'perplexity'
    Returns:
      tokens_list, html_string
      where tokens_list = [[decoded_token_str, probability], ...]
    """
    # Extract the first choice
    choices = _get_attr_or_item(response, "choices", None)
    if not choices:
        html = "<i>No choices in response.</i>"
        display(HTML(html))
        return [], html

    choice0 = choices[0]
    logprobs = _get_attr_or_item(choice0, "logprobs", None)
    content = _get_attr_or_item(logprobs, "content", None)
    if not content:
        html = "<i>No token-level logprobs found (did you set logprobs=True?).</i>"
        display(HTML(html))
        return [], html

    # Normalize entries
    tokens = []
    lps = []
    tops = []
    for entry in content:
        tok = _get_attr_or_item(entry, "token", "")
        lp = _get_attr_or_item(entry, "logprob", float("-inf"))
        top = _get_attr_or_item(entry, "top_logprobs", None)

        tok = str(tok)
        lp = float(lp)
        top_dict = _normalize_top_logprobs(top)

        tokens.append(tok)
        lps.append(lp)
        tops.append(top_dict)

    # Build the required list [[decoded_token, probability], ...]
    token_list = [[_to_text(t), _safe_exp(lp)] for t, lp in zip(tokens, lps)]

    # Perplexity (per-token) derived from top distribution: exp(entropy)
    entropies = np.array([_token_entropy_from_dict(d) for d in tops], dtype=float)
    perplexities = np.exp(entropies)  # >= 1
    # Normalize perplexity to [0,1] for color mapping
    pplx_norm = _norm01(perplexities)

    # HTML assembly
    spans = []
    for i, (tok, lp, top_dict) in enumerate(zip(tokens, lps, tops)):
        prob = _safe_exp(lp)
        if highlight.lower() == "perplexity":
            bg = rgb2hex(_cm_pplx(float(pplx_norm[i])))
        else:  # default to probability
            # prob is already in [0,1]; map to red→white→green
            bg = rgb2hex(_cm_prob(float(prob)))

        span = (
            f'<span class="token" style="background-color: {bg};">'
            f'{_to_text(tok)}'
            f'{_top_logprobs_to_html(top_dict)}'
            f'</span>'
        )
        spans.append(span)

    html = '<div style="border:1px solid grey; padding:4px; margin-right:100px;">' + "\n".join(spans) + "</div>"
    #display(HTML(html))
    return token_list, html

import ipywidgets as widgets
from IPython.display import display, clear_output

import ipywidgets as widgets
from IPython.display import display, HTML

def visualize(response, initial="Probability"):
    """
    Displays a static HTML block and radio buttons to toggle between
    'Probability' and 'Perplexity' visualizations without re-rendering.
    """
    options = ['Probability', 'Perplexity']
    assert initial in options, "radio_value parameter must be one of the options " + str(options)
    # Pre-generate both visualizations
    _, html_prob = get_html_visualization(response, highlight="probability")
    _, html_pplx = get_html_visualization(response, highlight="perplexity")

    # Create label and radio buttons
    label = widgets.HTML("<b>Highlight:</b>")
    toggle = widgets.RadioButtons(
        options=options,
        value=initial,
        layout=widgets.Layout(display='flex', flex_flow='row', align_items='center'),
    )

    # Create HTML output
    html_widget = widgets.HTML(value=html_prob if initial == 'Probability' else html_pplx)

    # Define what happens when user switches the toggle
    def on_toggle_change(change):
        if change['new'] == 'Probability':
            html_widget.value = html_prob
        else:
            html_widget.value = html_pplx

    toggle.observe(on_toggle_change, names='value')

    # Display both elements
    display(widgets.VBox([toggle, html_widget]))


# --- Visualization of Text Diff ---
from IPython.display import display, HTML
from typing import List, Tuple

def levenshtein_with_backtrace(a: str, b: str) -> Tuple[int, List[Tuple[str, int, int]]]:
    """
    Compute Levenshtein distance DP table and backtrace to produce alignment.
    Returns (distance, alignment)
    alignment is a list of tuples describing the operations in reverse-backtrace order:
      ('match'|'sub'|'ins'|'del', i_index_in_a_or_-1, j_index_in_b_or_-1)
    The indices point to the character in a (expected) and b (actual) for that operation, or -1
    when not applicable.
    """
    n, m = len(a), len(b)
    # dp costs
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    # backpointers: store which operation led to dp[i][j]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
        if i > 0:
            back[i][0] = ('del', i-1, -1)  # delete from a
    for j in range(m + 1):
        dp[0][j] = j
        if j > 0:
            back[0][j] = ('ins', -1, j-1)  # insert into b (extra char in b)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_sub = 0 if a[i-1] == b[j-1] else 1
            # candidates: deletion, insertion, substitution/match
            del_cost = dp[i-1][j] + 1
            ins_cost = dp[i][j-1] + 1
            sub_cost = dp[i-1][j-1] + cost_sub
            best = min(del_cost, ins_cost, sub_cost)
            dp[i][j] = best
            if best == sub_cost:
                back[i][j] = ('match' if cost_sub == 0 else 'sub', i-1, j-1)
            elif best == del_cost:
                back[i][j] = ('del', i-1, j-1)  # j-1 is kept for easier alignment (placeholder)
            else:
                back[i][j] = ('ins', i-1, j-1)

    # Backtrace from dp[n][m]
    i, j = n, m
    ops_reversed = []
    while i > 0 or j > 0:
        op = back[i][j]
        if op is None:
            break
        tag, ai, bj = op
        ops_reversed.append((tag, ai, bj))
        if tag in ('match', 'sub'):
            i -= 1
            j -= 1
        elif tag == 'del':
            i -= 1
        elif tag == 'ins':
            j -= 1

    ops = ops_reversed[::-1]  # correct order from start to end
    return dp[n][m], ops

def alignment_details(expected: str, actual: str):
    """
    Produce per-character info for the actual (second) string indicating whether each
    character was:
      - 'match' : exactly matches an expected char
      - 'sub'   : substituted for an expected char (wrong char)
      - 'ins'   : insertion (extra char in actual)
    Also returns info about deletions (chars in expected with no corresponding actual char).

    Returns:
      - distance (int)
      - actual_info: list of dicts for each char in actual with fields:
          {'char': c, 'status': 'match'|'sub'|'ins', 'expected_char': e or None, 'expected_index': idx or None}
      - deletions: list of dicts for each deletion in expected:
          {'char': c, 'index': idx}
    """
    dist, ops = levenshtein_with_backtrace(expected, actual)
    actual_info = []
    deletions = []

    # We'll walk through ops and build the mapping
    # Keep pointers for expected and actual indices as we progress
    exp_idx = 0
    act_idx = 0
    for tag, ai, bj in ops:
        if tag == 'match':
            # expected[exp_idx] == actual[act_idx]
            actual_info.append({'char': actual[act_idx], 'status': 'match',
                                'expected_char': expected[exp_idx], 'expected_index': exp_idx})
            exp_idx += 1
            act_idx += 1
        elif tag == 'sub':
            actual_info.append({'char': actual[act_idx], 'status': 'sub',
                                'expected_char': expected[exp_idx], 'expected_index': exp_idx})
            exp_idx += 1
            act_idx += 1
        elif tag == 'ins':
            # extra char in actual
            actual_info.append({'char': actual[act_idx], 'status': 'ins',
                                'expected_char': None, 'expected_index': None})
            act_idx += 1
        elif tag == 'del':
            # missing char from actual; record deletion for expected char at exp_idx
            deletions.append({'char': expected[exp_idx], 'index': exp_idx})
            exp_idx += 1
        else:
            raise RuntimeError("Unknown tag")

    # There might be leftover characters (shouldn't happen if ops cover all), handle defensively
    while act_idx < len(actual):
        actual_info.append({'char': actual[act_idx], 'status': 'ins',
                            'expected_char': None, 'expected_index': None})
        act_idx += 1
    while exp_idx < len(expected):
        deletions.append({'char': expected[exp_idx], 'index': exp_idx})
        exp_idx += 1

    return dist, actual_info, deletions

def visualize_diff(expected: str, actual: str):
    dist, actual_info, deletions = alignment_details(expected, actual)

    # Build HTML rows: expected (with deletions shown), actual with highlights
    expected_row = ""
    actual_row = ""
    guide_row = ""  # small visual markers

    # We need to render aligned columns. Reconstruct alignment by walking actual_info and deletions in order.
    # We'll iterate through ops again to get a faithful column alignment.
    _, ops = levenshtein_with_backtrace(expected, actual)
    exp_ptr = 0
    act_ptr = 0
    for tag, ai, bj in ops:
        if tag == 'match':
            expected_row += expected[exp_ptr]
            actual_row += actual[act_ptr]
            guide_row += " "
            exp_ptr += 1
            act_ptr += 1
        elif tag == 'sub':
            expected_row += f"<span style='background-color:#ff8282;text-decoration:line-through;font-style:italic;'>{expected[exp_ptr]}</span>"
            actual_row += f"<span style='background-color:#ff8282;'>{actual[act_ptr]}</span>"
            guide_row += "^"
            exp_ptr += 1
            act_ptr += 1
        elif tag == 'ins':
            expected_row += "<span style='color:#999;'>▯</span>"
            actual_row += f"<span style='background-color:#ff8282;'>{actual[act_ptr]}</span>"
            guide_row += "^"
            act_ptr += 1
        elif tag == 'del':
            expected_row += f"<span style='background-color:#ff8282;text-decoration:line-through;font-style:italic;'>{expected[exp_ptr]}</span>"
            actual_row += "<span style='color:#999;'>▯</span>"
            guide_row += "v"
            exp_ptr += 1

    # If any leftovers
    while exp_ptr < len(expected):
        expected_row += f"<span style='background-color:#ff8282;text-decoration:line-through;font-style:italic;'>{expected[exp_ptr]}</span>"
        actual_row += "<span style='color:#999;'>▯</span>"
        guide_row += "v"
        exp_ptr += 1
    while act_ptr < len(actual):
        expected_row += "<span style='color:#999;'>▯</span>"
        actual_row += f"<span style='background-color:#ff8282;'>{actual[act_ptr]}</span>"
        guide_row += "^"
        act_ptr += 1

    html = f"""
    <div style="font-family:system-ui, monospace; line-height:1.6;">
      <div><strong>Expected:</strong></div>
      <div style='white-space:pre;font-family:monospace;'>{expected}</div>
      <div style='margin-top:8px;'><strong>Model Response:</strong></div>
      <div style='white-space:pre;font-family:monospace;'>{actual_row}</div>
      </div>
    """
    #<div style='margin-top:10px;'><strong>Levenshtein distance:</strong> {dist}</div>

    display(HTML(html))

    # Also return the per-character info for programmatic use
    #return {
    #    'distance': dist,
    #    'actual_info': actual_info,
    #    'deletions': deletions
    #}

def apply_style():
    display(HTML("""<style>
    
    .token{
        position: relative;
        border-radius: 2px;
        padding: 1px;
        margin: 0 1px;
        cursor: default;
        border: 1px solid transparent;
    }
    
    .token:hover{
    border-color: red;
    }
    
    .top-logprobs {
        background-color: white;
        position: absolute;
        display:none;
        user-select:none;
        pointer-events: none;
        top:100%;
        left: 0;
        transform: translateX(-25%);
        z-index: 200;
        border: 1px solid black !important;
        background: initial !important;
        background-color: rgb(0,0,0) !important;
        color: white !important;
        border-radius: 2px;
    }
    
    .jp-RenderedHTMLCommon tbody tr:nth-child(odd), .jp-RenderedHTMLCommon tbody tr:nth-child(even){
    background: initial;
    }
    
    .lm-Widget.lm-Panel{
    overflow: visible;
    }
    
    .token:hover .top-logprobs{
    display:initial;
    }
    
    .top-logprobs tr td {
        padding: 0px 0.5em 2px !important;
    }
    
    :not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon td, :not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon th, :not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon tr {
        text-align: left;
    }
    
    .jp-OutputArea {
        overflow-y: initial;
    }
    .widget-radio-box{
        display: flex;
        flex-direction: row;
        gap: 10px;
        }
    </style>"""))