def test_common_exposes_shared_helpers():
    import litmus_common as lc
    for name in ["MODELS", "BASELINE_MODELS", "_parse_sizes",
                 "_targets_for", "_load_timed", "_clear_cache", "_resp_text"]:
        assert hasattr(lc, name), name


def test_parse_sizes_roundtrip():
    import litmus_common as lc
    assert lc._parse_sizes("1.7B,4B") == ["1.7B", "4B"]


def test_litmus_still_imports_and_reexports():
    import litmus
    assert litmus.MODELS == __import__("litmus_common").MODELS
