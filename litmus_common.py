"""Shared core for Litmus: model tables, loading, and MLX memory helpers.

Imported by litmus.py (perf benchmarks) and litmus_spec.py (spec-check harness)
so the two share one loading/targeting path instead of drifting copies.
"""
from __future__ import annotations

import time
import warnings
from typing import Iterator, Protocol, runtime_checkable

warnings.filterwarnings(
    "ignore",
    message=r".*mx\.metal\.(clear_cache|get_peak_memory|reset_peak_memory).*deprecated.*",
)

# MLX is imported lazily (see _mlx() and __getattr__ below) so litmus_common's
# pure helpers — model tables, size parsing, target resolution — import on any
# platform, not just Apple Silicon. Only the model-running re-exports (mx, nn,
# load, stream_generate) and the memory helpers touch MLX; they resolve it on
# first use and let ImportError propagate to callers that genuinely need it.

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


# ---------------------------------------------------------------------------
# lazy MLX access
# ---------------------------------------------------------------------------

_MLX: dict = {}  # cache: name -> imported MLX object


def _mlx() -> dict:
    """Import MLX on first use and cache the names we re-export.

    ImportError is left to propagate: any caller reaching here is on a
    model-running path and genuinely needs MLX installed.
    """
    if not _MLX:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load, stream_generate
        _MLX.update(mx=mx, nn=nn, load=load, stream_generate=stream_generate)
    return _MLX


_LAZY_NAMES = ("mx", "nn", "load", "stream_generate")


def __getattr__(name: str):
    # PEP 562: resolve the MLX-backed re-exports lazily so importers can still
    # do `from litmus_common import stream_generate` without eager-loading MLX.
    if name in _LAZY_NAMES:
        return _mlx()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _resp_text(resp) -> str:
    """stream_generate yields GenerationResponse in newer mlx_lm, str in older."""
    return resp.text if hasattr(resp, "text") else str(resp)


def _peak_memory_mb() -> float:
    # mx.metal.get_peak_memory is deprecated; prefer top-level mx.get_peak_memory.
    mx = _mlx()["mx"]
    if hasattr(mx, "get_peak_memory"):
        return mx.get_peak_memory() / (1024 * 1024)
    if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
        return mx.metal.get_peak_memory() / (1024 * 1024)
    return 0.0


def _reset_peak_memory() -> None:
    mx = _mlx()["mx"]
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
        return
    if hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
        mx.metal.reset_peak_memory()


def _clear_cache() -> None:
    mx = _mlx()["mx"]
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
    load = _mlx()["load"]
    t0 = time.perf_counter()
    model, tokenizer = load(repo)
    return model, tokenizer, time.perf_counter() - t0
