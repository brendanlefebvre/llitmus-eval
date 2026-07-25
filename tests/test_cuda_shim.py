import types

import litmus_cuda


def test_shim_exposes_historical_commands():
    # The four historical commands must still be dispatchable names.
    assert set(litmus_cuda.CUDA_COMMANDS) >= {
        "throughput", "perplexity", "decode-stability", "assisted",
    }


def test_shim_quant_backend_threads_quant(monkeypatch):
    captured = {}

    class Spy:
        name = "cuda"
        def load(self, repo, **opts):
            captured["quant"] = opts.get("quant")
            return object(), object(), 0.0

    be = litmus_cuda._QuantBackend(Spy(), quant="nf4")
    be.load("some/repo")
    assert captured["quant"] == "nf4"
