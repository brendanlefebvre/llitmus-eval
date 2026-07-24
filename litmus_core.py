"""Backend-agnostic core for Litmus.

Hosts every algorithm shared across backends and CLIs: reference-text loading,
reasoning-preamble stripping, the distinct-trigram decode-stability metric,
perplexity windowing + aggregation, the Run record, report rendering, and the
per-command perf drivers. Imports neither mlx nor torch — the model runtime is
reached only through a Backend (see litmus_common.get_backend).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

REFERENCE_TEXT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "reference.txt"
)

# Matches common reasoning-preamble openers seen in Bonsai 8B output.
PREAMBLE_RE = re.compile(
    r"^(Okay|Alright|Let me|So,|First,|I need to|The user|Looking at)"
)
# Matches an optional parenthetical prefix that sometimes precedes the
# preamble opener, e.g. "(150-200 words) Okay, so I need to..."
PARENTHETICAL_PREFIX_RE = re.compile(r"^\([^)]*\)\s*")


def _load_reference_text(path: str = REFERENCE_TEXT_PATH) -> str:
    """Load and strip a Project Gutenberg plain-text file.

    The script never embeds prose — the user downloads Gatsby once. See the
    module docstring for the curl command.
    """
    if not os.path.exists(path):
        raise SystemExit(
            f"reference text not found at {path}.\n"
            "Download The Great Gatsby from Project Gutenberg once:\n"
            "  curl -L https://www.gutenberg.org/cache/epub/64317/pg64317.txt "
            f"-o {path}"
        )
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # Strip Gutenberg header/footer if present.
    start = text.find("*** START OF")
    if start != -1:
        nl = text.find("\n", start)
        if nl != -1:
            text = text[nl + 1:]
    end = text.find("*** END OF")
    if end != -1:
        text = text[:end]
    return text.strip()


def _strip_thinking(text: str, tokenizer) -> tuple[str, int]:
    """Remove a reasoning preamble from the front of `text`.

    Returns (useful_text, scratchpad_token_count). Two detection paths:

    1. Explicit reasoning block delimited by a literal ``</think>`` tag
       (LFM2.5, DeepSeek-R1, QwQ, …). Everything up to and including the
       closing tag is scratchpad; the remainder is the answer. An opened
       ``<think>`` with no closing tag means the model ran out of budget
       mid-reasoning — the whole generation is treated as scratchpad.
    2. Fallback heuristic for untagged models: a prose preamble separated
       from the answer by the first ``\\n\\n`` blank-line break.

    If nothing is detected, returns (text, 0). Token counts are a heuristic
    — re-encoding a segment may not match the streaming count exactly, but
    it's close enough for a rough "useful t/s".
    """
    stripped = text.lstrip()

    # Path 1: explicit <think>…</think> tag — preferred when present.
    close_idx = text.find("</think>")
    if close_idx != -1:
        end = close_idx + len("</think>")
        return text[end:].lstrip(), len(tokenizer.encode(text[:end]))
    if stripped.startswith("<think>"):
        # Opened a think block but never closed it — whole output is scratchpad.
        return "", len(tokenizer.encode(text))

    # Path 2: untagged prose-preamble heuristic.
    after_paren = PARENTHETICAL_PREFIX_RE.sub("", stripped, count=1)
    if not PREAMBLE_RE.match(after_paren):
        return text, 0
    parts = text.split("\n\n", 1)
    if len(parts) < 2:
        # Preamble never ended — whole output is scratchpad.
        return "", len(tokenizer.encode(text))
    head, tail = parts
    scratchpad_tokens = len(tokenizer.encode(head + "\n\n"))
    return tail, scratchpad_tokens


def distinct_trigram_ratio(tokens: list[int]) -> float:
    """Fraction of distinct token trigrams (1.0 = no repetition).

    Returns nan for sequences shorter than 3 tokens. Extracted verbatim from
    the decode-stability loop so both backends and the golden tests share it.
    """
    if len(tokens) < 3:
        return float("nan")
    trigrams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
    return len(set(trigrams)) / len(trigrams)


@dataclass
class Run:
    label: str
    prompt: str
    prompt_tokens: int
    gen_tokens: int
    prefill_tps: float
    decode_tps: float
    ttft_ms: float
    peak_mem_mb: float
    sample: str
    useful_gen_tokens: Optional[int] = None
    useful_decode_tps: Optional[float] = None


def print_table(all_runs: list[Run]) -> None:
    has_useful = any(r.useful_decode_tps is not None for r in all_runs)
    width = 122 if has_useful else 110

    print("\n" + "=" * width)
    header = (
        f"{'size':<14} {'prompt':<40} {'p_tok':>6} {'g_tok':>6} "
        f"{'prefill t/s':>12} {'decode t/s':>12} {'TTFT ms':>10} {'peak MB':>10}"
    )
    if has_useful:
        header += f" {'useful t/s':>12}"
    print(header)
    print("-" * width)
    for r in all_runs:
        line = (
            f"{r.label:<14} {r.prompt:<40} {r.prompt_tokens:>6} {r.gen_tokens:>6} "
            f"{r.prefill_tps:>12.1f} {r.decode_tps:>12.1f} "
            f"{r.ttft_ms:>10.1f} {r.peak_mem_mb:>10.0f}"
        )
        if has_useful:
            u = r.useful_decode_tps
            line += f" {u:>12.1f}" if u is not None else f" {'-':>12}"
        print(line)

    print("\n--- Per-size summary ---")
    header = (
        f"{'size':<14} {'avg decode t/s':>16} {'avg prefill t/s':>18} "
        f"{'avg TTFT ms':>14} {'peak MB':>10}"
    )
    if has_useful:
        header += f" {'avg useful t/s':>16}"
    print(header)
    by_label: dict[str, list[Run]] = {}
    for r in all_runs:
        by_label.setdefault(r.label, []).append(r)
    for label, runs in by_label.items():
        avg_dec = sum(r.decode_tps for r in runs) / len(runs)
        avg_pre = sum(r.prefill_tps for r in runs) / len(runs)
        avg_ttft = sum(r.ttft_ms for r in runs) / len(runs)
        peak = max(r.peak_mem_mb for r in runs)
        line = (
            f"{label:<14} {avg_dec:>16.1f} {avg_pre:>18.1f} "
            f"{avg_ttft:>14.1f} {peak:>10.0f}"
        )
        if has_useful:
            useful_vals = [r.useful_decode_tps for r in runs if r.useful_decode_tps is not None]
            if useful_vals:
                avg_useful = sum(useful_vals) / len(useful_vals)
                line += f" {avg_useful:>16.1f}"
            else:
                line += f" {'-':>16}"
        print(line)

    print("\n--- Sample outputs (first 80 chars) ---")
    for r in all_runs:
        print(f"  [{r.label}] {r.prompt[:30]:<30} -> {r.sample}")
