import subprocess
import sys


def test_importing_litmus_spec_does_not_load_mlx():
    # litmus_spec must be usable as a pure-logic library without MLX; MLX is
    # only imported function-locally in the CLI/real-model path. Run in a fresh
    # interpreter so other tests' imports don't pollute sys.modules.
    code = (
        "import litmus_spec, sys; "
        "bad = [m for m in sys.modules if m == 'mlx' or m.startswith('mlx.') or m == 'litmus_common']; "
        "assert not bad, bad"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
