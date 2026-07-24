"""Shared deterministic test doubles for backend-driven core tests."""
import math

import pytest


class FakeBackend:
    """A deterministic in-memory Backend: no model, no GPU.

    - stream(): yields the words of a canned string, space-suffixed, one 'token'
      at a time (so "".join reconstructs the canned text).
    - token_logprobs(): a fixed value per position, so perplexity is exactly
      exp(-value).
    - memory: always zero.
    """
    name = "fake"

    def __init__(self, canned_text="hello world answer", per_token_logprob=-1.5,
                 encode=None):
        self.canned_text = canned_text
        self.per_token_logprob = per_token_logprob
        self._encode = encode or (lambda s: list(range(len(s.split()))))

    def load(self, repo, **opts):
        return object(), _StubTokenizer(self._encode), 0.0

    def stream(self, model, tokenizer, prompt, max_tokens):
        words = self.canned_text.split()
        for w in words[:max_tokens]:
            yield w + " "

    def token_logprobs(self, model, tokenizer, ids):
        return [self.per_token_logprob] * (len(ids) - 1)

    def peak_memory_mb(self):
        return 0.0

    def reset_peak_memory(self):
        pass

    def clear_cache(self):
        pass


class _StubTokenizer:
    def __init__(self, encode):
        self._encode = encode

    def encode(self, text):
        return self._encode(text)

    def decode(self, ids, **kw):
        return " ".join(str(i) for i in ids)


@pytest.fixture
def fake_backend():
    return FakeBackend()
