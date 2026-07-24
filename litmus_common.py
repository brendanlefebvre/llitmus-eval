"""Shared core for Litmus: model tables, size parsing, target resolution,
and the Backend protocol.

Imported by litmus.py (perf benchmarks) and litmus_spec.py (spec-check harness)
so the two share one loading/targeting path instead of drifting copies. MLX
helpers live in litmus_mlx.py; the lazy shims below exist only until Phase D
retires litmus_spec.py's direct MLX imports.
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


# ---------------------------------------------------------------------------
# temporary lazy shims for litmus_spec.py (removed in Phase D / Task 10)
# ---------------------------------------------------------------------------

def stream_generate(*args, **kwargs):
    from mlx_lm import stream_generate as _impl
    return _impl(*args, **kwargs)


def _resp_text(resp) -> str:
    from litmus_mlx import _resp_text as _impl
    return _impl(resp)


def _clear_cache() -> None:
    from litmus_mlx import MLXBackend
    MLXBackend().clear_cache()


def _load_timed(repo: str) -> tuple[object, object, float]:
    from litmus_mlx import MLXBackend
    return MLXBackend().load(repo)
