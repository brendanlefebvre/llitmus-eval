"""Shared core for Litmus: model tables, loading, and MLX memory helpers.

Imported by litmus.py (perf benchmarks) and litmus_spec.py (spec-check harness)
so the two share one loading/targeting path instead of drifting copies.
"""
from __future__ import annotations

import time
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*mx\.metal\.(clear_cache|get_peak_memory|reset_peak_memory).*deprecated.*",
)

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate

MODELS = {
    "1.7B": "prism-ml/Bonsai-1.7B-mlx-1bit",
    "4B": "prism-ml/Bonsai-4B-mlx-1bit",
    "8B": "prism-ml/Bonsai-8B-mlx-1bit",
}

# Stock 4-bit baselines (regular mlx_lm, no fork needed)
BASELINE_MODELS = {
    "llama3.2-1B-4bit": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "llama3.2-3B-4bit": "mlx-community/Llama-3.2-3B-Instruct-4bit",
}

PROMPTS = [
    "Explain quantum computing in two sentences.",
    "Write a haiku about a Mac Mini.",
    "What is the difference between a hash table and a B-tree?",
    "Summarize the plot of Hamlet in one paragraph.",
    "def fibonacci(n):",
]

WARMUP_PROMPT = "Hello."


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _resp_text(resp) -> str:
    """stream_generate yields GenerationResponse in newer mlx_lm, str in older."""
    return resp.text if hasattr(resp, "text") else str(resp)


def _peak_memory_mb() -> float:
    # mx.metal.get_peak_memory is deprecated; prefer top-level mx.get_peak_memory.
    if hasattr(mx, "get_peak_memory"):
        return mx.get_peak_memory() / (1024 * 1024)
    if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
        return mx.metal.get_peak_memory() / (1024 * 1024)
    return 0.0


def _reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
        return
    if hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()


def _clear_cache() -> None:
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
        return
    if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()


def _parse_sizes(sizes_str: str) -> list[str]:
    selected = [s.strip() for s in sizes_str.split(",") if s.strip()]
    unknown = [s for s in selected if s not in MODELS]
    if unknown:
        raise SystemExit(f"unknown size(s) {unknown}; pick from {list(MODELS)}")
    return selected


def _targets_for(args) -> list[tuple[str, str]]:
    """Return [(label, repo)] pairs for the current invocation.

    If --repo is passed, returns that single repo (overriding --sizes), with
    the user-provided --label or a generated one. Otherwise expands --sizes
    against the built-in Bonsai MODELS dict.
    """
    if args.repo:
        label = args.label or args.repo.split("/")[-1]
        return [(label, args.repo)]
    return [(size, MODELS[size]) for size in _parse_sizes(args.sizes)]


def _load_timed(repo: str) -> tuple[object, object, float]:
    t0 = time.perf_counter()
    model, tokenizer = load(repo)
    return model, tokenizer, time.perf_counter() - t0
