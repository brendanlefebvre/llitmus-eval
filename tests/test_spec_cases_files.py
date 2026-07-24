from litmus_spec import load_cases


def test_shipped_constraint_cases_load():
    cases = load_cases("cases/constraints.jsonl", "constraints")
    assert len(cases) >= 10
    kinds = {k for c in cases for k, _ in c.checks}
    # every registry kind is exercised by at least one shipped case
    from litmus_spec import CHECKS
    assert set(CHECKS) <= kinds


def test_shipped_tool_cases_load_with_abstentions():
    cases = load_cases("cases/tool_calling.jsonl", "tool-calling")
    assert len(cases) >= 8
    assert sum(1 for c in cases if c.expect["tool"] is None) >= 2
