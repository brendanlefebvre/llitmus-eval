"""Byte-strict golden tests for litmus_core's pure functions.

Captured before the orchestration is rewritten against the Backend protocol.
These pin the deterministic core's behavior — the guarantee that protects the
published results-bonsai-1bit.md numbers. No MLX, no model.
"""
import math

import litmus_core as core


class StubTokenizer:
    """Whitespace tokenizer: deterministic token counts with no model."""
    def encode(self, text: str) -> list[int]:
        return [len(w) for w in text.split()]


def test_strip_thinking_closed_tag():
    tok = StubTokenizer()
    text = "<think>reasoning here</think>The answer is 42."
    useful, scratch = core._strip_thinking(text, tok)
    assert useful == "The answer is 42."
    assert scratch == len(tok.encode("<think>reasoning here</think>"))


def test_strip_thinking_unclosed_tag_is_all_scratchpad():
    tok = StubTokenizer()
    text = "<think>never closed"
    useful, scratch = core._strip_thinking(text, tok)
    assert useful == ""
    assert scratch == len(tok.encode(text))


def test_strip_thinking_untagged_preamble_heuristic():
    tok = StubTokenizer()
    text = "Okay, let me think.\n\nThe final answer."
    useful, scratch = core._strip_thinking(text, tok)
    assert useful == "The final answer."
    assert scratch == len(tok.encode("Okay, let me think.\n\n"))


def test_strip_thinking_parenthetical_prefix():
    tok = StubTokenizer()
    text = "(150-200 words) Okay, so.\n\nAnswer body."
    useful, scratch = core._strip_thinking(text, tok)
    assert useful == "Answer body."


def test_strip_thinking_no_preamble_returns_unchanged():
    tok = StubTokenizer()
    text = "A direct answer with no reasoning."
    useful, scratch = core._strip_thinking(text, tok)
    assert useful == text
    assert scratch == 0


def test_distinct_trigram_ratio_all_distinct():
    assert core.distinct_trigram_ratio([1, 2, 3, 4, 5]) == 1.0


def test_distinct_trigram_ratio_all_repeated():
    # tokens [7,7,7,7] -> trigrams (7,7,7),(7,7,7): 1 distinct / 2 = 0.5
    assert core.distinct_trigram_ratio([7, 7, 7, 7]) == 0.5


def test_distinct_trigram_ratio_too_short_is_nan():
    assert math.isnan(core.distinct_trigram_ratio([1, 2]))


def test_print_table_golden(capsys):
    runs = [
        core.Run(
            label="1.7B", prompt="Explain quantum computing", prompt_tokens=5,
            gen_tokens=64, prefill_tps=120.0, decode_tps=45.0, ttft_ms=42.0,
            peak_mem_mb=512.0, sample="Quantum computing uses qubits",
        ),
    ]
    core.print_table(runs)
    out = capsys.readouterr().out
    # Pin the structural invariants of the rendered table.
    assert "prefill t/s" in out
    assert "1.7B" in out
    assert "--- Per-size summary ---" in out
    assert "45.0" in out  # decode t/s rendered at one decimal


def test_load_reference_text_strips_gutenberg_markers(tmp_path):
    p = tmp_path / "ref.txt"
    p.write_text(
        "header junk\n*** START OF THE BOOK ***\nReal body text.\n"
        "*** END OF THE BOOK ***\nfooter junk",
        encoding="utf-8",
    )
    assert core._load_reference_text(str(p)) == "Real body text."
