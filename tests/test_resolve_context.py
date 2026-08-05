"""resolve_context_length: native context from cached config.json, or None.

The resolver deliberately returns None instead of a default — a silent
default is what let 51k–85k-token cases score on a 40,960-context model
(spec 2026-07-29).
"""
import json

import pytest

from litmus_common import resolve_context_length

# Module-scope `import huggingface_hub` made this file a COLLECTION error on
# any machine without the dep, and a collection error aborts the whole pytest
# run -- taking every unrelated test with it, including the divisor guard in
# test_router_divisor_property.py. Skip this module instead; the tests here
# genuinely need the real API surface to monkeypatch.
huggingface_hub = pytest.importorskip("huggingface_hub")


def _fake_cache(monkeypatch, tmp_path, config: dict | None):
    """Point try_to_load_from_cache at a temp config.json (or a miss)."""
    if config is None:
        # Cache miss: the real API returns None or a sentinel object,
        # never a str. Use None.
        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                            lambda repo_id, filename: None)
        return
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo_id, filename: str(p))


def test_top_level_max_position_embeddings(monkeypatch, tmp_path):
    _fake_cache(monkeypatch, tmp_path, {"max_position_embeddings": 40960})
    assert resolve_context_length("org/model") == 40960


def test_nested_text_config(monkeypatch, tmp_path):
    # Qwen3.6 / gemma-4 / Qwen3-VL nest it under text_config.
    _fake_cache(monkeypatch, tmp_path,
                {"text_config": {"max_position_embeddings": 262144}})
    assert resolve_context_length("org/model") == 262144


def test_top_level_wins_over_nested(monkeypatch, tmp_path):
    _fake_cache(monkeypatch, tmp_path,
                {"max_position_embeddings": 40960,
                 "text_config": {"max_position_embeddings": 262144}})
    assert resolve_context_length("org/model") == 40960


def test_neither_key_returns_none(monkeypatch, tmp_path):
    _fake_cache(monkeypatch, tmp_path, {"architectures": ["X"]})
    assert resolve_context_length("org/model") is None


def test_not_cached_returns_none(monkeypatch, tmp_path):
    _fake_cache(monkeypatch, tmp_path, None)
    assert resolve_context_length("org/model") is None


def test_unparseable_config_returns_none(monkeypatch, tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo_id, filename: str(p))
    assert resolve_context_length("org/model") is None


def test_bool_true_is_not_a_context_length(monkeypatch, tmp_path):
    # bool is an int subclass; a config with True must not resolve to 1.
    _fake_cache(monkeypatch, tmp_path, {"max_position_embeddings": True})
    assert resolve_context_length("org/model") is None
