import json
from litmus_spec import format_tool_table, format_constraint_table, write_sidecar


def test_tool_table_shows_prompted_and_native_and_gap():
    result = {"aggregate": {
        "prompted": {"well_formed": 0.9, "right_tool": 0.8, "args_ok": 0.7, "abstained_ok": 0.6},
        "native": {"well_formed": 1.0, "right_tool": 0.9, "args_ok": 0.8, "abstained_ok": 0.7}},
        "cases": [], "errored": []}
    table = format_tool_table("Qwen", result)
    assert "prompted" in table and "native" in table
    assert "Qwen" in table
    assert "gap(right)" in table


def test_tool_table_handles_no_native():
    result = {"aggregate": {
        "prompted": {"well_formed": 0.7, "right_tool": 0.5, "args_ok": 0.4, "abstained_ok": 0.4},
        "native": None}, "cases": [], "errored": []}
    table = format_tool_table("Bonsai", result)
    assert "no native" in table.lower()


def test_constraint_table_shows_strict_loose():
    result = {"aggregate": {"strict": 0.8, "loose": 0.9, "by_kind": {"all_lowercase": 1.0},
                            "n_cases": 5}, "cases": [], "errored": []}
    table = format_constraint_table("M", result)
    assert "strict" in table.lower() and "0.8" in table


def test_write_sidecar_roundtrips(tmp_path):
    result = {"aggregate": {"strict": 0.5, "loose": 0.5, "by_kind": {}, "n_cases": 2},
              "cases": [{"id": "c1"}], "errored": []}
    path = tmp_path / "out.json"
    write_sidecar(str(path), "constraints", "org/model", "model",
                  {"prompted": True, "native": False}, result)
    data = json.loads(path.read_text())
    assert data["profile"] == "constraints"
    assert data["convention_support"]["native"] is False
    assert data["n_cases"] == 2
    assert data["cases"][0]["id"] == "c1"
