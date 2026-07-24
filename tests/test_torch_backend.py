import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for TorchBackend smoke tests",
                allow_module_level=True)

from litmus_torch import TorchBackend

REPO = "sshleifer/tiny-gpt2"  # tiny, CPU/GPU-loadable stand-in


def test_stream_yields_nonempty_text():
    be = TorchBackend()
    model, tok, secs = be.load(REPO, quant="bf16")
    assert secs >= 0.0
    chunks = list(be.stream(model, tok, "Hello", max_tokens=5))
    assert chunks and all(isinstance(c, str) for c in chunks)


def test_token_logprobs_shape_and_finiteness():
    be = TorchBackend()
    model, tok, _ = be.load(REPO, quant="bf16")
    ids = tok("Hello world", return_tensors="pt").input_ids[0].tolist()
    lps = be.token_logprobs(model, tok, ids)
    assert len(lps) == len(ids) - 1
    assert all(math.isfinite(v) for v in lps)
