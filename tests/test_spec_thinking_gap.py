from litmus_spec import format_thinking_gap


def _constraints(strict):
    return {"aggregate": {"strict": strict, "loose": strict, "by_kind": {},
                          "n_cases": 2}, "cases": [], "errored": []}


def _tools(right):
    return {"aggregate": {"prompted": {"well_formed": 1.0, "right_tool": right,
                                       "args_ok": right, "abstained_ok": right},
                          "native": None}, "cases": [], "errored": []}


def test_gap_reports_constraints_delta_and_token_cost():
    per_mode = {"no-think": _constraints(0.50), "think": _constraints(0.80)}
    line = format_thinking_gap("M", per_mode, "constraints",
                               {"no-think": 20.0, "think": 300.0})
    assert "+0.30" in line
    assert "20" in line and "300" in line


def test_gap_reports_tool_calling_delta_on_right_tool():
    per_mode = {"no-think": _tools(0.75), "think": _tools(0.83)}
    line = format_thinking_gap("M", per_mode, "tool-calling",
                               {"no-think": 18.0, "think": 320.0})
    assert "+0.08" in line


def test_gap_is_negative_when_thinking_hurts():
    per_mode = {"no-think": _constraints(0.90), "think": _constraints(0.60)}
    line = format_thinking_gap("M", per_mode, "constraints",
                               {"no-think": 20.0, "think": 300.0})
    assert "-0.30" in line
