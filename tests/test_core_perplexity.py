import math

import litmus_core as core


def test_compute_perplexity_matches_exp_mean_nll(fake_backend):
    # token_logprobs returns -1.5 per position -> mean NLL = 1.5 -> ppl = e^1.5
    class Tok:
        def encode(self, text):
            return list(range(10))
    ppl = core.compute_perplexity(fake_backend, object(), Tok(),
                                  "any text", window=8)
    assert math.isclose(ppl, math.exp(1.5), rel_tol=1e-12)


def test_compute_perplexity_short_input_is_nan(fake_backend):
    class Tok:
        def encode(self, text):
            return [1]
    ppl = core.compute_perplexity(fake_backend, object(), Tok(),
                                  "x", window=8)
    assert math.isnan(ppl)
