"""Shared core for Litmus: model tables, size parsing, target resolution,
and the Backend protocol.

Imported by litmus.py (perf benchmarks) and litmus_spec.py (spec-check harness)
so the two share one loading/targeting path instead of drifting copies. MLX
helpers live in litmus_mlx.py; torch helpers in litmus_torch.py. Backends are
resolved through get_backend(), keeping this module dependency-free at import.
"""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

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


@runtime_checkable
class Backend(Protocol):
    """The single model-runtime seam for Litmus.

    The core owns every algorithm; a backend owns only the library-specific
    parts: loading, token generation, one forward pass for log-probs, and
    memory telemetry.
    """
    name: str

    def load(self, repo: str, **opts) -> tuple[object, object, float]:
        """Return (model, tokenizer, load_seconds)."""

    def stream(self, model, tokenizer, prompt: str,
               max_tokens: int) -> Iterator[str]:
        """Yield one generated token's text per step."""

    def token_logprobs(self, model, tokenizer, ids: list[int]) -> list[float]:
        """Per-token log-probs as floats: logprob of ids[i] given ids[:i],
        for i in 1..len(ids)-1. Returns len(ids)-1 values."""

    def peak_memory_mb(self) -> float: ...
    def reset_peak_memory(self) -> None: ...
    def clear_cache(self) -> None: ...


def get_backend(name: str = "auto") -> "Backend":
    """Resolve a backend by name: 'mlx' | 'cuda' | 'auto'.

    'auto' picks MLX if importable, else torch. Imports only the selected
    backend's module, so importing litmus_common/litmus_core pulls in neither
    mlx nor torch.
    """
    if name == "auto":
        try:
            import mlx.core  # noqa: F401
            name = "mlx"
        except ImportError:
            name = "cuda"
    if name == "mlx":
        from litmus_mlx import MLXBackend
        return MLXBackend()
    if name == "cuda":
        from litmus_torch import TorchBackend
        return TorchBackend()
    raise SystemExit(f"unknown backend {name!r}; pick from mlx | cuda | auto")


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


def resolve_context_length(repo: str) -> int | None:
    """Native context length for a model repo, from config.json in the HF cache.

    Precedence: top-level ``max_position_embeddings``, then
    ``text_config.max_position_embeddings`` (Qwen3.6 / gemma-4 / Qwen3-VL
    nest it). Returns None when the config is not cached, unparseable, or
    carries neither key — the caller decides the fallback. A silent default
    here is what produced the 2026-07-28 over-context scoring bug.
    """
    import json
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo_id=repo, filename="config.json")
    if not isinstance(path, str):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict):
        # A top-level array/string/number parses cleanly, then .get() below
        # raises AttributeError -- which the clause above does NOT catch, so it
        # escapes resolve_context_length() and aborts the run. The contract is
        # None for anything unusable. (loxo's mirror of this function,
        # _read_hf_cache_config, guards the same shape.)
        return None
    for scope in (cfg, cfg.get("text_config") or {}):
        if not isinstance(scope, dict):
            continue
        v = scope.get("max_position_embeddings")
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return None

