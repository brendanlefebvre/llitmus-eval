import tomllib
from pathlib import Path


def test_pyproject_lists_new_modules_and_extras():
    data = tomllib.loads(Path("pyproject.toml").read_text())
    modules = set(data["tool"]["setuptools"]["py-modules"])
    assert {"litmus_core", "litmus_mlx", "litmus_torch"} <= modules
    extras = data["project"]["optional-dependencies"]
    assert "mlx" in extras and "cuda" in extras and "dev" in extras
