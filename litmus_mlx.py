"""MLXBackend: the Apple-Silicon adapter for the Litmus Backend protocol.

mlx.core / mlx.nn / mlx_lm are imported lazily inside method bodies so that
importing this module (e.g. via get_backend) does not require MLX until a
model op actually runs.
"""
from __future__ import annotations

import time
import warnings
from typing import Iterator

warnings.filterwarnings(
    "ignore",
    message=r".*mx\.metal\.(clear_cache|get_peak_memory|reset_peak_memory).*deprecated.*",
)


def _resp_text(resp) -> str:
    """stream_generate yields GenerationResponse in newer mlx_lm, str in older."""
    return resp.text if hasattr(resp, "text") else str(resp)


class MLXBackend:
    name = "mlx"

    def load(self, repo: str, **opts) -> tuple[object, object, float]:
        from mlx_lm import load
        t0 = time.perf_counter()
        model, tokenizer = load(repo)
        return model, tokenizer, time.perf_counter() - t0

    def stream(self, model, tokenizer, prompt: str,
               max_tokens: int) -> Iterator[str]:
        from mlx_lm import stream_generate
        for resp in stream_generate(model, tokenizer, prompt,
                                    max_tokens=max_tokens):
            yield _resp_text(resp)

    def token_logprobs(self, model, tokenizer, ids: list[int]) -> list[float]:
        import mlx.core as mx
        import mlx.nn as nn
        x = mx.array(ids)[None, :]                       # (1, T)
        logits = model(x)                                # (1, T, V)
        log_probs = nn.log_softmax(logits[:, :-1, :], axis=-1)
        targets = x[:, 1:, None]                         # (1, T-1, 1)
        gathered = mx.take_along_axis(log_probs, targets, axis=-1).squeeze(-1)
        return [float(v) for v in gathered[0].tolist()]

    def peak_memory_mb(self) -> float:
        import mlx.core as mx
        if hasattr(mx, "get_peak_memory"):
            return mx.get_peak_memory() / (1024 * 1024)
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
            return mx.metal.get_peak_memory() / (1024 * 1024)
        return 0.0

    def reset_peak_memory(self) -> None:
        import mlx.core as mx
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
            mx.metal.reset_peak_memory()

    def clear_cache(self) -> None:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
