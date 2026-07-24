def test_common_exposes_shared_helpers():
    import litmus_common as lc
    for name in ["MODELS", "BASELINE_MODELS", "_parse_sizes",
                 "_targets_for", "get_backend"]:
        assert hasattr(lc, name), name


def test_parse_sizes_roundtrip():
    import litmus_common as lc
    assert lc._parse_sizes("1.7B,4B") == ["1.7B", "4B"]


def test_litmus_is_thin_cli_without_mlx():
    # litmus.py is now a backend-agnostic thin CLI: it imports neither mlx nor
    # torch, resolves a backend via get_backend, and dispatches to the core
    # COMMANDS. Importing it must not require MLX. Fresh interpreter so other
    # tests' imports don't pollute sys.modules.
    import subprocess
    import sys
    code = (
        "import litmus, sys; "
        "bad = [m for m in sys.modules if m == 'mlx' or m.startswith('mlx.') "
        "or m == 'mlx_lm' or m.startswith('mlx_lm.') or m == 'torch' "
        "or m.startswith('torch.')]; "
        "assert not bad, bad"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import litmus
    import litmus_core
    assert callable(litmus.main)
    assert "throughput" in litmus_core.COMMANDS


def test_importing_litmus_common_does_not_load_mlx_or_torch():
    # litmus_common's pure helpers (model tables, target resolution, the
    # Backend protocol) must be usable without MLX or torch so the suite is
    # not platform-locked at import time. Fresh interpreter so other tests'
    # imports don't pollute sys.modules.
    import subprocess
    import sys
    code = (
        "import litmus_common, sys; "
        "assert hasattr(litmus_common, 'get_backend'); "
        "bad = [m for m in sys.modules if m == 'mlx' or m.startswith('mlx.') "
        "or m == 'mlx_lm' or m.startswith('mlx_lm.') "
        "or m == 'torch' or m.startswith('torch.')]; "
        "assert not bad, bad"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
