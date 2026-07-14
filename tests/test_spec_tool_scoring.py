from litmus_spec import ParsedCall, score_tool_call, aggregate_tool


def test_correct_call_all_pass():
    p = ParsedCall(True, "get_weather", {"location": "Paris"}, "ok")
    s = score_tool_call(p, {"tool": "get_weather", "arguments": {"location": "Paris"}})
    assert s == {"well_formed": True, "right_tool": True,
                 "args_ok": True, "abstained_ok": None}


def test_wrong_tool_makes_args_na():
    p = ParsedCall(True, "send_email", {"to": "x"}, "ok")
    s = score_tool_call(p, {"tool": "get_weather", "arguments": {"location": "Paris"}})
    assert s["right_tool"] is False and s["args_ok"] is None


def test_hallucinated_extra_arg_fails_args():
    p = ParsedCall(True, "get_weather", {"location": "Paris", "unit": "C"}, "ok")
    s = score_tool_call(p, {"tool": "get_weather", "arguments": {"location": "Paris"}})
    assert s["args_ok"] is False


def test_abstention_correct():
    p = ParsedCall(True, None, None, "abstain")
    s = score_tool_call(p, {"tool": None})
    assert s["abstained_ok"] is True and s["right_tool"] is True and s["args_ok"] is None


def test_abstention_violated_by_calling():
    p = ParsedCall(True, "get_weather", {"location": "X"}, "ok")
    s = score_tool_call(p, {"tool": None})
    assert s["abstained_ok"] is False and s["right_tool"] is False


def test_parse_failure_on_abstention_is_not_credited():
    # unparseable output also has tool=None, but it is NOT a real abstention
    p = ParsedCall(False, None, None, "no json found")
    s = score_tool_call(p, {"tool": None})
    assert s["well_formed"] is False
    assert s["abstained_ok"] is False and s["right_tool"] is False


def test_parse_failure_on_call_case_is_not_right():
    p = ParsedCall(False, None, None, "truncated")
    s = score_tool_call(p, {"tool": "get_weather", "arguments": {"location": "Paris"}})
    assert s["right_tool"] is False and s["args_ok"] is None


def test_native_abstention_no_call_is_correct():
    # native: a no-call (well_formed False, tool None) on an abstention case is correct
    p = ParsedCall(False, None, None, "no json found")
    s = score_tool_call(p, {"tool": None}, native=True)
    assert s["abstained_ok"] is True and s["right_tool"] is True and s["well_formed"] is True


def test_native_abstention_violated_by_calling():
    p = ParsedCall(True, "get_weather", {"location": "X"}, "ok")
    s = score_tool_call(p, {"tool": None}, native=True)
    assert s["abstained_ok"] is False and s["right_tool"] is False


def test_prompted_abstention_still_requires_wellformed():
    # prompted (native=False): garbage is NOT a valid abstention
    p = ParsedCall(False, None, None, "no json")
    s = score_tool_call(p, {"tool": None})
    assert s["abstained_ok"] is False and s["well_formed"] is False


def test_aggregate_excludes_none_from_denominator():
    per_case = [
        {"well_formed": True, "right_tool": True, "args_ok": True, "abstained_ok": None},
        {"well_formed": True, "right_tool": False, "args_ok": None, "abstained_ok": None},
        {"well_formed": True, "right_tool": True, "args_ok": None, "abstained_ok": True},
    ]
    agg = aggregate_tool(per_case)
    assert agg["well_formed"] == 1.0
    assert agg["right_tool"] == 2 / 3
    assert agg["args_ok"] == 1.0          # only one non-None, and it passed
    assert agg["abstained_ok"] == 1.0     # only one non-None, and it passed


def test_rate_all_none_returns_none():
    from litmus_spec import _rate
    assert _rate([None, None]) is None


def test_args_ok_missing_key_fails():
    p = ParsedCall(True, "get_weather", {}, "ok")  # missing required 'location'
    s = score_tool_call(p, {"tool": "get_weather", "arguments": {"location": "Paris"}})
    assert s["args_ok"] is False
