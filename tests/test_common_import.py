def test_common_exposes_shared_helpers():
    import litmus_common as lc
    for name in ["MODELS", "BASELINE_MODELS", "_parse_sizes",
                 "_targets_for", "_load_timed", "_clear_cache", "_resp_text"]:
        assert hasattr(lc, name), name


def test_parse_sizes_roundtrip():
    import litmus_common as lc
    assert lc._parse_sizes("1.7B,4B") == ["1.7B", "4B"]


def test_litmus_still_imports_and_reexports():
    # litmus.py is the MLX perf CLI: it re-exports mx/nn/load/stream_generate as
    # module globals and uses them throughout, so importing it requires MLX.
    # Skip (rather than fail) when MLX is absent so the pure-logic suite still
    # runs under `pip install .[dev]` alone.
    import pytest
    pytest.importorskip("mlx_lm")
    import litmus
    assert litmus.MODELS == __import__("litmus_common").MODELS


def test_importing_litmus_common_does_not_load_mlx():
    # litmus_common's pure helpers (model tables, target resolution) must be
    # usable without MLX so the suite is not Apple-only at import time; MLX is
    # resolved lazily on first access to a model-running name. Fresh interpreter
    # so other tests' imports don't pollute sys.modules.
    import subprocess
    import sys
    code = (
        "import litmus_common, sys; "
        "bad = [m for m in sys.modules if m == 'mlx' or m.startswith('mlx.') "
        "or m == 'mlx_lm' or m.startswith('mlx_lm.')]; "
        "assert not bad, bad"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
