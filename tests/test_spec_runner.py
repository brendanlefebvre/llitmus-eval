from litmus_spec import (
    ConstraintCase, ToolCase, run_constraints, run_tool_calling,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=False, tools=None):
        return messages[-1]["content"]


def test_run_constraints_scores_and_aggregates():
    cases = [
        ConstraintCase("c1", "p", [("exact_bullets", {"n": 2})]),
        ConstraintCase("c2", "p", [("all_lowercase", {})]),
    ]
    # generate_fn keyed by prompt content; c1 prompt renders to "p" for both,
    # so key on case via a canned mapping instead:
    outputs = iter(["- a\n- b", "HELLO"])
    result = run_constraints(cases, FakeTokenizer(), lambda p: next(outputs))
    assert result["aggregate"]["n_cases"] == 2
    assert result["aggregate"]["strict"] == 0.5   # c1 passes, c2 fails
    assert len(result["cases"]) == 2


def test_run_constraints_records_errors():
    cases = [ConstraintCase("c1", "p", [("all_lowercase", {})])]

    def boom(_):
        raise RuntimeError("model exploded")

    result = run_constraints(cases, FakeTokenizer(), boom)
    assert result["aggregate"]["n_cases"] == 0    # errored case not counted
    assert result["errored"][0]["id"] == "c1"
    assert "exploded" in result["errored"][0]["error"]


def test_run_tool_calling_prompted_only():
    cases = [ToolCase("t1", "weather?", [{"name": "get_weather"}],
                      {"tool": "get_weather", "arguments": {"location": "Paris"}})]
    out = '{"tool":"get_weather","arguments":{"location":"Paris"}}'
    result = run_tool_calling(cases, FakeTokenizer(), lambda p: out, native=False)
    assert result["aggregate"]["prompted"]["right_tool"] == 1.0
    assert result["aggregate"]["native"] is None
    assert result["cases"][0]["native"] is None


def test_run_tool_calling_with_native_column():
    cases = [ToolCase("t1", "weather?", [{"name": "get_weather"}],
                      {"tool": "get_weather", "arguments": {"location": "Paris"}})]
    # native parser reads name/arguments; prompted reads tool/arguments — both
    # here happen to parse from the same generic-JSON shape via name/tool keys.
    out = '{"name":"get_weather","tool":"get_weather","arguments":{"location":"Paris"}}'
    result = run_tool_calling(cases, FakeTokenizer(), lambda p: out, native=True)
    assert result["aggregate"]["native"]["right_tool"] == 1.0
    assert result["aggregate"]["native_parse_failed"] == 0


def test_run_tool_calling_counts_native_parse_failures():
    cases = [ToolCase("t1", "weather?", [{"name": "get_weather"}],
                      {"tool": "get_weather", "arguments": {"location": "Paris"}})]
    # 1st generate call = prompted (parseable), 2nd = native (malformed structure)
    calls = {"n": 0}
    def gen(prompt):
        calls["n"] += 1
        return ('{"tool":"get_weather","arguments":{"location":"Paris"}}'
                if calls["n"] == 1 else '<tool_call>{{"name":"get_weather"}}</tool_call>')
    result = run_tool_calling(cases, FakeTokenizer(), gen, native=True)
    assert result["aggregate"]["native_parse_failed"] == 1


def test_run_tool_calling_native_abstention_scored_correct():
    cases = [ToolCase("t1", "capital of France?", [{"name": "get_weather"}], {"tool": None})]
    # prompted emits explicit null; native emits prose (no call) -> correct abstention
    calls = {"n": 0}
    def gen(prompt):
        calls["n"] += 1
        return '{"tool": null, "arguments": {}}' if calls["n"] == 1 else "Paris is the capital."
    result = run_tool_calling(cases, FakeTokenizer(), gen, native=True)
    assert result["aggregate"]["native"]["abstained_ok"] == 1.0
    assert result["aggregate"]["native"]["right_tool"] == 1.0
    assert result["aggregate"]["native_parse_failed"] == 0


def test_run_tool_calling_native_abstention_with_broken_call_not_credited():
    cases = [ToolCase("t1", "write a haiku", [{"name": "get_weather"}], {"tool": None})]
    calls = {"n": 0}
    def gen(prompt):
        calls["n"] += 1
        return ('{"tool":null,"arguments":{}}' if calls["n"] == 1
                else '<tool_call>{{"name":"get_weather"}}</tool_call>')
    result = run_tool_calling(cases, FakeTokenizer(), gen, native=True)
    assert result["aggregate"]["native"]["abstained_ok"] == 0.0
    assert result["aggregate"]["native_parse_failed"] == 1


def test_run_tool_calling_records_raw_output():
    cases = [ToolCase("t1", "weather?", [{"name": "get_weather"}],
                      {"tool": "get_weather", "arguments": {"location": "Paris"}})]
    out = '{"tool":"get_weather","arguments":{"location":"Paris"}}'
    result = run_tool_calling(cases, FakeTokenizer(), lambda p: out, native=False)
    assert result["cases"][0]["prompted_output"].startswith('{"tool"')


def test_run_tool_calling_native_error_excludes_prompted_from_aggregate():
    cases = [ToolCase("t1", "weather?", [{"name": "get_weather"}],
                      {"tool": "get_weather", "arguments": {"location": "Paris"}})]
    calls = {"n": 0}
    def gen(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"tool":"get_weather","arguments":{"location":"Paris"}}'  # prompted OK
        raise RuntimeError("native call exploded")                            # native leg raises
    result = run_tool_calling(cases, FakeTokenizer(), gen, native=True)
    assert result["errored"][0]["id"] == "t1"
    assert result["cases"] == []
    # the phantom prompted pass must NOT survive into the aggregate
    assert result["aggregate"]["prompted"]["right_tool"] is None
