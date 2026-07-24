import importlib
import sys

import pytest


def test_get_backend_unknown_name_raises():
    import litmus_common
    with pytest.raises((ValueError, SystemExit)):
        litmus_common.get_backend("nonsense")


def test_importing_common_pulls_in_neither_mlx_nor_torch():
    # Fresh import must not eagerly import the heavy libs.
    for mod in ("mlx", "mlx.core", "torch", "litmus_common"):
        sys.modules.pop(mod, None)
    importlib.import_module("litmus_common")
    assert "mlx" not in sys.modules
    assert "torch" not in sys.modules


def test_get_backend_mlx_is_lazy(monkeypatch):
    # Selecting mlx imports litmus_mlx but the call itself must not fail on a
    # non-Apple box until a model op runs; construction alone is cheap.
    import litmus_common
    be = litmus_common.get_backend("mlx")
    assert be.name == "mlx"
