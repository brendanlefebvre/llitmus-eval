"""Tests for the main-replay profile: loader, tier-0 validators, runner, aggregation."""
import json

import pytest

from litmus_spec import (
    CaseError, ReplayCase, ParsedCall, load_cases,
    score_replay_call, aggregate_replay, run_main_replay,
    build_replay_prompt, format_replay_table, _check_args_schema,
    _capture_tool_names, _capture_tool_params,
)

# Build tool-call tag strings via chr() so the literal markers never appear
# in this source file (they would otherwise confuse the harness).
TC_OPEN = chr(60) + chr(60) + "tool_call" + chr(62)
TC_CLOSE = chr(60) + chr(47) + "tool_call" + chr(62)


# ---------------------------------------------------------------------------
# capture / case fixtures
# ---------------------------------------------------------------------------

READ_TOOL = {"type": "function", "function": {
    "name": "read",
    "parameters": {"type": "object",
                   "properties": {"filePath": {"type": "string"},
                                   "limit": {"type": "integer"}},
                   "required": ["filePath"]},
}}

GREP_TOOL = {"type": "function", "function": {
    "name": "grep",
    "parameters": {"type": "object",
                   "properties": {"pattern": {"type": "string"},
                                   "path": {"type": "string"},
                                   "include": {"type": "string"}},
                   "required": ["pattern"]},
}}


def _make_capture(tmp_path, name, *, messages=None, tools=None, max_tokens=32000,
                  model="loxo/auto"):
    body = {
        "model": model,
        "messages": messages or [{"role": "system", "content": "sys"},
                                  {"role": "user", "content": "do the thing"}],
        "tools": tools if tools is not None else [],
        "max_tokens": max_tokens,
        "tool_choice": "auto",
    }
    p = tmp_path / name
    p.write_text(json.dumps(body))
    return str(p)


def _replay_case(capture_path, *, id_="mr-001", depth_stratum="mid",
                 chain_id="chain-01", est_tokens=20000, acted=True,
                 tools=None, arguments=None):
    return ReplayCase(
        id=id_, capture_path=capture_path, chain_id=chain_id,
        depth_stratum=depth_stratum, est_tokens=est_tokens,
        reference={"acted": acted, "tools": tools or [], "arguments": arguments or []},
    )


def _replay_jsonl(tmp_path, name, *cases):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    return str(p)


def _valid_case_dict(capture_path):
    return {
        "id": "mr-001", "capture_path": capture_path, "chain_id": "chain-01",
        "depth_stratum": "mid", "est_tokens": 23515,
        "reference": {"acted": True, "tools": ["read"], "arguments": [{"filePath": "x"}]},
    }


# ===========================================================================
# loader
# ===========================================================================

class TestLoader:
    def test_load_valid_replay_case(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json", tools=[READ_TOOL])
        path = _replay_jsonl(tmp_path, "m.jsonl", _valid_case_dict(cap))
        cases = load_cases(path, "main-replay")
        assert len(cases) == 1
        c = cases[0]
        assert isinstance(c, ReplayCase)
        assert c.id == "mr-001"
        assert c.capture_path == cap
        assert c.depth_stratum == "mid"
        assert c.est_tokens == 23515
        assert c.reference["acted"] is True
        assert c.reference["tools"] == ["read"]

    def test_missing_id_raises(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json")
        d = _valid_case_dict(cap); del d["id"]
        path = _replay_jsonl(tmp_path, "m.jsonl", d)
        with pytest.raises(CaseError, match="id"):
            load_cases(path, "main-replay")

    def test_missing_capture_path_raises(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json")
        d = _valid_case_dict(cap); del d["capture_path"]
        path = _replay_jsonl(tmp_path, "m.jsonl", d)
        with pytest.raises(CaseError, match="capture_path"):
            load_cases(path, "main-replay")

    def test_nonexistent_capture_path_raises(self, tmp_path):
        d = _valid_case_dict("/no/such/file-xyz.json")
        path = _replay_jsonl(tmp_path, "m.jsonl", d)
        with pytest.raises(CaseError, match="does not exist"):
            load_cases(path, "main-replay")

    def test_bad_depth_stratum_raises(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json")
        d = _valid_case_dict(cap); d["depth_stratum"] = "bottomless"
        path = _replay_jsonl(tmp_path, "m.jsonl", d)
        with pytest.raises(CaseError, match="depth_stratum"):
            load_cases(path, "main-replay")

    def test_est_tokens_must_be_int(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json")
        d = _valid_case_dict(cap); d["est_tokens"] = "23515"
        path = _replay_jsonl(tmp_path, "m.jsonl", d)
        with pytest.raises(CaseError, match="est_tokens"):
            load_cases(path, "main-replay")

    def test_est_tokens_rejects_bool(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json")
        d = _valid_case_dict(cap); d["est_tokens"] = True
        path = _replay_jsonl(tmp_path, "m.jsonl", d)
        with pytest.raises(CaseError, match="est_tokens"):
            load_cases(path, "main-replay")

    def test_reference_acted_must_be_bool(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json")
        d = _valid_case_dict(cap); d["reference"]["acted"] = "yes"
        path = _replay_jsonl(tmp_path, "m.jsonl", d)
        with pytest.raises(CaseError, match="acted"):
            load_cases(path, "main-replay")

    def test_reference_tools_must_be_list(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json")
        d = _valid_case_dict(cap); d["reference"]["tools"] = "read"
        path = _replay_jsonl(tmp_path, "m.jsonl", d)
        with pytest.raises(CaseError, match="tools"):
            load_cases(path, "main-replay")

    def test_reference_arguments_must_be_list(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json")
        d = _valid_case_dict(cap); d["reference"]["arguments"] = {"filePath": "x"}
        path = _replay_jsonl(tmp_path, "m.jsonl", d)
        with pytest.raises(CaseError, match="arguments"):
            load_cases(path, "main-replay")

    def test_missing_reference_raises(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json")
        d = _valid_case_dict(cap); del d["reference"]
        path = _replay_jsonl(tmp_path, "m.jsonl", d)
        with pytest.raises(CaseError, match="reference"):
            load_cases(path, "main-replay")

    def test_line_number_in_error(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json")
        good = _valid_case_dict(cap)
        bad = _valid_case_dict(cap); bad["id"] = "mr-002"; del bad["chain_id"]
        path = _replay_jsonl(tmp_path, "m.jsonl", good, bad)
        with pytest.raises(CaseError, match="line 2"):
            load_cases(path, "main-replay")

    def test_all_three_strata_load(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json")
        cases = []
        for i, s in enumerate(("shallow", "mid", "deep"), 1):
            d = _valid_case_dict(cap)
            d["id"] = f"mr-{i:03d}"
            d["depth_stratum"] = s
            cases.append(d)
        path = _replay_jsonl(tmp_path, "m.jsonl", *cases)
        loaded = load_cases(path, "main-replay")
        assert [c.depth_stratum for c in loaded] == ["shallow", "mid", "deep"]


# ===========================================================================
# score_replay_call — tier-0 validators
# ===========================================================================

class TestActedOk:
    def test_ref_acted_candidate_acts_passes(self):
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(True, "read", {"filePath": "f"}, "ok", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["acted_ok"] is True

    def test_ref_acted_candidate_does_not_act_fails(self):
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(True, None, None, "abstain")
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["acted_ok"] is False

    def test_ref_prose_candidate_prose_passes(self):
        case = _replay_case("x", acted=False)
        parsed = ParsedCall(True, None, None, "abstain")
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["acted_ok"] is True

    def test_ref_prose_candidate_acts_fails(self):
        case = _replay_case("x", acted=False)
        parsed = ParsedCall(True, "read", {"filePath": "f"}, "ok", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["acted_ok"] is False


class TestWellFormed:
    def test_well_formed_passes(self):
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(True, "read", {"filePath": "f"}, "ok", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["well_formed"] is True

    def test_not_well_formed_fails(self):
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(False, None, None, "truncated", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["well_formed"] is False


class TestToolExists:
    def test_known_tool_passes(self):
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(True, "read", {"filePath": "f"}, "ok", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL, GREP_TOOL])
        assert s["tool_exists"] is True

    def test_unknown_tool_fails(self):
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(True, "send_email", {"to": "x"}, "ok", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["tool_exists"] is False

    def test_no_tool_call_is_not_applicable(self):
        case = _replay_case("x", acted=False)
        parsed = ParsedCall(True, None, None, "abstain")
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["tool_exists"] is None


class TestArgsSchema:
    def test_correct_args_pass(self):
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(True, "read",
                            {"filePath": "f", "limit": 50}, "ok", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["args_schema_ok"] is True
        assert s["action_valid"] is True

    def test_optional_only_args_pass(self):
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(True, "read", {"filePath": "f"}, "ok", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["args_schema_ok"] is True

    def test_missing_required_key_fails(self):
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(True, "read", {"limit": 50}, "ok", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["args_schema_ok"] is False

    def test_hallucinated_key_fails(self):
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(True, "read",
                            {"filePath": "f", "bogus": 1}, "ok", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["args_schema_ok"] is False

    def test_wrong_type_fails(self):
        case = _replay_case("x", acted=True)
        # filePath should be string, given int
        parsed = ParsedCall(True, "read", {"filePath": 42}, "ok", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["args_schema_ok"] is False

    def test_wrong_type_integer_rejects_bool(self):
        # python bool is a subclass of int; JSON integer must reject a bool
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(True, "read",
                            {"filePath": "f", "limit": True}, "ok", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["args_schema_ok"] is False

    def test_unknown_tool_args_schema_is_none(self):
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(True, "send_email", {"to": "x"}, "ok", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["args_schema_ok"] is None
        # action_valid is False because tool_exists is False
        assert s["action_valid"] is False

    def test_no_tool_call_args_schema_is_none(self):
        case = _replay_case("x", acted=False)
        parsed = ParsedCall(True, None, None, "abstain")
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["args_schema_ok"] is None
        assert s["action_valid"] is True  # acted_ok only (well_formed is N/A)


class TestClosedGuard:
    def test_unclosed_thinking_none_shape(self):
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(True, "read", {"filePath": "f"}, "ok", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL], closed=False)
        # acted_ok is False (it did not act); remaining dimensions are None
        # (nothing was produced to check) per the dimension-applicability rule.
        assert s == {"acted_ok": False, "well_formed": None,
                     "tool_exists": None, "args_schema_ok": None,
                     "action_valid": False}


class TestActionValidComposition:
    def test_all_pass(self):
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(True, "read", {"filePath": "f"}, "ok", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["action_valid"] is True

    def test_tool_missing_breaks_action_valid(self):
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(True, "nope", {}, "ok", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["action_valid"] is False

    def test_bad_args_breaks_action_valid(self):
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(True, "read",
                            {"filePath": "f", "bogus": 1}, "ok", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["action_valid"] is False

    def test_prose_when_prose_expected_passes(self):
        case = _replay_case("x", acted=False)
        parsed = ParsedCall(True, None, None, "abstain")
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["action_valid"] is True


class TestDimensionApplicability:
    """None, never False, for not-applicable dimensions (spec 2026-07-28)."""

    def test_malformed_attempted_call_vs_prose_reference(self):
        """ParsedCall(well_formed=False, attempted=True) vs prose reference.

        The model TRIED to make a call but failed. acted_ok=False (reference
        didn't act), well_formed=False (meaningful — it tried and failed),
        tool_exists/args_schema_ok=None (no well-formed call to check).
        """
        case = _replay_case("x", acted=False)
        parsed = ParsedCall(False, None, None, "garbled", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["acted_ok"] is False
        assert s["well_formed"] is False
        assert s["tool_exists"] is None
        assert s["args_schema_ok"] is None
        assert s["action_valid"] is False

    def test_prose_response_vs_acted_reference(self):
        """Prose response (not attempted) vs acted reference.

        acted_ok=False (should have acted), well_formed=None (not applicable —
        no action to parse), tool_exists/args_schema_ok=None.
        """
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(True, None, None, "abstain")  # attempted=False
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["acted_ok"] is False
        assert s["well_formed"] is None
        assert s["tool_exists"] is None
        assert s["args_schema_ok"] is None
        assert s["action_valid"] is False

    def test_correct_prose_abstention_all_dims_none(self):
        """Correct prose abstention: acted_ok=True, all others None."""
        case = _replay_case("x", acted=False)
        parsed = ParsedCall(True, None, None, "abstain")  # attempted=False
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["acted_ok"] is True
        assert s["well_formed"] is None
        assert s["tool_exists"] is None
        assert s["args_schema_ok"] is None
        assert s["action_valid"] is True

    def test_malformed_attempted_call_vs_acted_reference(self):
        """Attempted-but-failed call vs acted reference.

        acted_ok=True (model tried to act when it should), well_formed=False
        (meaningful — it tried and failed), tool_exists/args_schema_ok=None.
        """
        case = _replay_case("x", acted=True)
        parsed = ParsedCall(False, None, None, "garbled", attempted=True)
        s = score_replay_call(parsed, case, [READ_TOOL])
        assert s["acted_ok"] is True
        assert s["well_formed"] is False
        assert s["tool_exists"] is None
        assert s["args_schema_ok"] is None
        assert s["action_valid"] is False


# ===========================================================================
# schema validation helpers
# ===========================================================================

class TestSchemaHelpers:
    def test_capture_tool_names(self):
        names = _capture_tool_names([READ_TOOL, GREP_TOOL])
        assert names == {"read", "grep"}

    def test_capture_tool_names_skips_malformed(self):
        names = _capture_tool_names([READ_TOOL, "junk", {"no": "function"}])
        assert names == {"read"}

    def test_capture_tool_params_found(self):
        params = _capture_tool_params([READ_TOOL, GREP_TOOL], "read")
        assert params is not None
        assert "filePath" in params["properties"]
        assert params["required"] == ["filePath"]

    def test_capture_tool_params_missing_returns_none(self):
        assert _capture_tool_params([READ_TOOL], "grep") is None

    def test_check_args_schema_correct(self):
        props = {"filePath": {"type": "string"}, "limit": {"type": "integer"}}
        required = ["filePath"]
        assert _check_args_schema({"filePath": "f", "limit": 5}, props, required)

    def test_check_args_schema_no_required(self):
        props = {"filePath": {"type": "string"}}
        assert _check_args_schema({}, props, [])

    def test_check_args_schema_type_list_union(self):
        # JSON schema allows type: ["string", "null"]
        props = {"x": {"type": ["string", "null"]}}
        assert _check_args_schema({"x": "hi"}, props, [])
        assert _check_args_schema({"x": None}, props, [])
        assert not _check_args_schema({"x": 5}, props, [])

    def test_check_args_schema_unknown_type_skipped(self):
        props = {"x": {"type": "weird"}}
        # unknown type name is not checked (don't fail on what we can't check)
        assert _check_args_schema({"x": object()}, props, [])


# ===========================================================================
# aggregation
# ===========================================================================

class TestAggregateReplay:
    def test_aggregate_basic_rates(self):
        per_case = [
            {"acted_ok": True, "well_formed": True, "tool_exists": True,
             "args_schema_ok": True, "action_valid": True, "depth_stratum": "mid"},
            {"acted_ok": True, "well_formed": True, "tool_exists": False,
             "args_schema_ok": None, "action_valid": False, "depth_stratum": "mid"},
            {"acted_ok": False, "well_formed": True, "tool_exists": True,
             "args_schema_ok": True, "action_valid": False, "depth_stratum": "deep"},
        ]
        agg = aggregate_replay(per_case)
        assert agg["n_cases"] == 3
        # weighted: mid=0.5*0.383 + deep=0.0*0.542 = 0.1915
        assert agg["action_valid_weighted"] == pytest.approx(0.1915)
        assert agg["by_dimension"]["acted_ok"] == {"rate": 2 / 3, "n_applicable": 3}
        assert agg["by_dimension"]["well_formed"] == {"rate": 1.0, "n_applicable": 3}
        assert agg["by_dimension"]["tool_exists"] == {"rate": 2 / 3, "n_applicable": 3}
        # args_schema_ok: one None excluded, two True => 2/2 = 1.0
        assert agg["by_dimension"]["args_schema_ok"] == {"rate": 1.0, "n_applicable": 2}

    def test_aggregate_by_depth(self):
        per_case = [
            {"action_valid": True, "depth_stratum": "shallow"},
            {"action_valid": False, "depth_stratum": "shallow"},
            {"action_valid": True, "depth_stratum": "mid"},
            {"action_valid": True, "depth_stratum": "mid"},
            {"action_valid": False, "depth_stratum": "deep"},
        ]
        agg = aggregate_replay(per_case)
        assert agg["by_depth"]["shallow"] == {"action_valid": 0.5, "n": 2}
        assert agg["by_depth"]["mid"] == {"action_valid": 1.0, "n": 2}
        assert agg["by_depth"]["deep"] == {"action_valid": 0.0, "n": 1}

    def test_aggregate_by_chain(self):
        per_case = [
            {"action_valid": True, "depth_stratum": "mid", "chain_id": "chain-01"},
            {"action_valid": False, "depth_stratum": "mid", "chain_id": "chain-01"},
            {"action_valid": True, "depth_stratum": "deep", "chain_id": "chain-02"},
        ]
        agg = aggregate_replay(per_case)
        assert agg["by_chain"]["chain-01"] == {"action_valid": 0.5, "n": 2}
        assert agg["by_chain"]["chain-02"] == {"action_valid": 1.0, "n": 1}

    def test_aggregate_reference_model_default(self):
        per_case = [{"action_valid": True, "depth_stratum": "mid"}]
        agg = aggregate_replay(per_case)
        assert agg["reference_model"] == "z-ai/glm-5.2"

    def test_aggregate_reference_model_from_case(self):
        per_case = [{"action_valid": True, "depth_stratum": "mid",
                     "reference_model": "mlx-community/Qwen3-14B-4bit"}]
        agg = aggregate_replay(per_case)
        assert agg["reference_model"] == "mlx-community/Qwen3-14B-4bit"

    def test_aggregate_dimension_denominators(self):
        """Pin per-dimension n_applicable: None values excluded from denominators."""
        per_case = [
            # acted_ok: T, well_formed: None, tool_exists: None, args_schema: None
            {"acted_ok": True, "well_formed": None, "tool_exists": None,
             "args_schema_ok": None, "action_valid": True, "depth_stratum": "mid"},
            # acted_ok: T, well_formed: T, tool_exists: T, args_schema: T
            {"acted_ok": True, "well_formed": True, "tool_exists": True,
             "args_schema_ok": True, "action_valid": True, "depth_stratum": "mid"},
            # acted_ok: F, well_formed: F, tool_exists: None, args_schema: None
            {"acted_ok": False, "well_formed": False, "tool_exists": None,
             "args_schema_ok": None, "action_valid": False, "depth_stratum": "deep"},
        ]
        agg = aggregate_replay(per_case)
        assert agg["by_dimension"]["acted_ok"] == {"rate": 2 / 3, "n_applicable": 3}
        assert agg["by_dimension"]["well_formed"] == {"rate": 1 / 2, "n_applicable": 2}
        assert agg["by_dimension"]["tool_exists"] == {"rate": 1.0, "n_applicable": 1}
        assert agg["by_dimension"]["args_schema_ok"] == {"rate": 1.0, "n_applicable": 1}

    def test_aggregate_depth_weights_present(self):
        agg = aggregate_replay([])
        assert agg["depth_weights"] == {"shallow": 0.075, "mid": 0.383, "deep": 0.542}

    def test_aggregate_empty(self):
        agg = aggregate_replay([])
        assert agg["n_cases"] == 0
        assert agg["action_valid_weighted"] == 0.0
        for s in ("shallow", "mid", "deep"):
            assert agg["by_depth"][s] == {"action_valid": 0.0, "n": 0}

    def test_aggregate_args_schema_all_none(self):
        per_case = [
            {"acted_ok": True, "well_formed": True, "tool_exists": True,
             "args_schema_ok": None, "action_valid": True, "depth_stratum": "mid"},
        ]
        agg = aggregate_replay(per_case)
        assert agg["by_dimension"]["args_schema_ok"] == {"rate": None, "n_applicable": 0}


# ===========================================================================
# build_replay_prompt
# ===========================================================================

class ReplayFakeTokenizer:
    """Captures the messages/tools passed to apply_chat_template."""
    def __init__(self):
        self.last_messages = None
        self.last_tools = None
        self.last_enable_thinking = None

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=False, tools=None, **kw):
        self.last_messages = messages
        self.last_tools = tools
        self.last_enable_thinking = kw.get("enable_thinking")
        return "PROMPT:" + (messages[-1]["content"] if messages else "")


class TestBuildReplayPrompt:
    def test_native_forwards_captured_tools(self, tmp_path):
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "do it"}]
        cap = _make_capture(tmp_path, "req-1.json", messages=msgs,
                            tools=[READ_TOOL])
        case = _replay_case(cap)
        tok = ReplayFakeTokenizer()
        prompt = build_replay_prompt(tok, case, native=True)
        assert tok.last_tools == [READ_TOOL]
        assert tok.last_messages == msgs
        assert "do it" in prompt

    def test_prompted_omits_tools(self, tmp_path):
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "do it"}]
        cap = _make_capture(tmp_path, "req-1.json", messages=msgs,
                            tools=[READ_TOOL])
        case = _replay_case(cap)
        tok = ReplayFakeTokenizer()
        build_replay_prompt(tok, case, native=False)
        assert tok.last_tools is None
        assert tok.last_messages == msgs

    def test_enable_thinking_forwarded(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json", tools=[READ_TOOL])
        case = _replay_case(cap)
        tok = ReplayFakeTokenizer()
        build_replay_prompt(tok, case, native=True, enable_thinking=False)
        assert tok.last_enable_thinking is False


# ===========================================================================
# run_main_replay — end-to-end
# ===========================================================================

class TestRunMainReplay:
    def test_prompted_runner_correct_call(self, tmp_path):
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "read the file"}]
        cap = _make_capture(tmp_path, "req-1.json", messages=msgs,
                            tools=[READ_TOOL], max_tokens=32000)
        case = _replay_case(cap, acted=True, tools=["read"],
                            arguments=[{"filePath": "x"}])
        tok = ReplayFakeTokenizer()

        seen = {}
        def gen(prompt, max_tokens=0):
            seen["prompt"] = prompt
            seen["max_tokens"] = max_tokens
            return '{"tool": "read", "arguments": {"filePath": "f"}}'

        result = run_main_replay([case], tok, gen, native=False)
        assert result["errored"] == []
        assert len(result["cases"]) == 1
        rec = result["cases"][0]
        assert rec["id"] == "mr-001"
        assert rec["native"] is False
        assert rec["depth_stratum"] == "mid"
        assert rec["chain_id"] == "chain-01"
        score = rec["score"]
        assert score["acted_ok"] is True
        assert score["well_formed"] is True
        assert score["tool_exists"] is True
        assert score["args_schema_ok"] is True
        assert score["action_valid"] is True
        # max_tokens comes from the captured body, not the CLI
        assert seen["max_tokens"] == 32000
        assert result["aggregate"]["by_depth"]["mid"]["action_valid"] == 1.0

    def test_prompted_runner_prose_when_ref_prose(self, tmp_path):
        msgs = [{"role": "user", "content": "what is 2+2"}]
        cap = _make_capture(tmp_path, "req-1.json", messages=msgs,
                            tools=[READ_TOOL])
        case = _replay_case(cap, acted=False)
        tok = ReplayFakeTokenizer()
        # model responds in prose: no JSON object -> parse_prompted returns
        # well_formed=False, tool=None, attempted=False. A correct abstention:
        # acted_ok=True (ref didn't act, candidate didn't attempt). Dimensions
        # for the action itself are not applicable (None).
        result = run_main_replay([case], tok,
                                 lambda p, max_tokens=0: "The answer is 4.",
                                 native=False)
        score = result["cases"][0]["score"]
        assert score["acted_ok"] is True   # ref didn't act, candidate didn't act
        assert score["well_formed"] is None  # not applicable (no action to parse)
        assert score["action_valid"] is True

    def test_native_runner_correct_call(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json", tools=[READ_TOOL])
        case = _replay_case(cap, acted=True)
        tok = ReplayFakeTokenizer()
        out = TC_OPEN + '\n{"name": "read", "arguments": {"filePath": "f"}}\n' + TC_CLOSE
        result = run_main_replay([case], tok, lambda p, max_tokens=0: out,
                                 native=True)
        assert tok.last_tools == [READ_TOOL]
        score = result["cases"][0]["score"]
        assert score["well_formed"] is True
        assert score["tool_exists"] is True
        assert score["args_schema_ok"] is True
        assert score["action_valid"] is True

    def test_native_runner_unknown_tool(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json", tools=[READ_TOOL])
        case = _replay_case(cap, acted=True)
        tok = ReplayFakeTokenizer()
        out = TC_OPEN + '\n{"name": "send_email", "arguments": {"to": "x"}}\n' + TC_CLOSE
        result = run_main_replay([case], tok, lambda p, max_tokens=0: out,
                                 native=True)
        score = result["cases"][0]["score"]
        assert score["tool_exists"] is False
        assert score["action_valid"] is False

    def test_runner_passes_captured_max_tokens(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json", tools=[READ_TOOL],
                            max_tokens=9999)
        case = _replay_case(cap, acted=True)
        tok = ReplayFakeTokenizer()
        seen = {}
        def gen(p, max_tokens=0):
            seen["max_tokens"] = max_tokens
            return '{"tool": "read", "arguments": {"filePath": "f"}}'
        run_main_replay([case], tok, gen, native=False)
        assert seen["max_tokens"] == 9999

    def test_runner_thinking_unclosed_scores_all_false(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json", tools=[READ_TOOL])
        case = _replay_case(cap, acted=True)
        tok = ReplayFakeTokenizer()
        # unclosed thinking tag -> strip_thinking returns ("", False) ->
        # all-false. Build the thinking tag via chr() to avoid literal markers.
        think_open = chr(60) + "think" + chr(62)
        out = think_open + " still reasoning..."  # no closing tag
        result = run_main_replay([case], tok, lambda p, max_tokens=0: out,
                                 native=False)
        rec = result["cases"][0]
        assert rec["thinking_unclosed"] is True
        score = rec["score"]
        assert score["action_valid"] is False
        assert score["acted_ok"] is False
        assert score["well_formed"] is None  # nothing produced to check

    def test_runner_records_error(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json", tools=[READ_TOOL])
        case = _replay_case(cap, acted=True)
        tok = ReplayFakeTokenizer()

        def boom(p, max_tokens=0):
            raise RuntimeError("model exploded")

        result = run_main_replay([case], tok, boom, native=False)
        assert result["cases"] == []
        assert result["errored"][0]["id"] == "mr-001"
        assert "exploded" in result["errored"][0]["error"]
        # errored case is not counted in the aggregate
        assert result["aggregate"]["n_cases"] == 0

    def test_runner_aggregates_multiple_strata(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json", tools=[READ_TOOL])
        cases = [
            _replay_case(cap, id_="mr-001", depth_stratum="shallow",
                         chain_id="chain-01", acted=True),
            _replay_case(cap, id_="mr-002", depth_stratum="shallow",
                         chain_id="chain-02", acted=True),
            _replay_case(cap, id_="mr-003", depth_stratum="mid",
                         chain_id="chain-01", acted=True),
        ]
        tok = ReplayFakeTokenizer()
        # first two pass, third fails (wrong tool)
        outs = iter([
            '{"tool": "read", "arguments": {"filePath": "a"}}',
            '{"tool": "read", "arguments": {"filePath": "b"}}',
            '{"tool": "nope", "arguments": {}}',
        ])

        def gen(p, max_tokens=0):
            return next(outs)

        result = run_main_replay(cases, tok, gen, native=False)
        agg = result["aggregate"]
        assert agg["n_cases"] == 3
        # shallow: 2/2 pass, mid: 0/1 pass
        assert agg["by_depth"]["shallow"] == {"action_valid": 1.0, "n": 2}
        assert agg["by_depth"]["mid"] == {"action_valid": 0.0, "n": 1}
        # weighted: 1.0*0.075 + 0.0*0.383 = 0.075
        assert agg["action_valid_weighted"] == pytest.approx(0.075)


# ===========================================================================
# format_replay_table
# ===========================================================================

class TestFormatReplayTable:
    def test_table_shows_action_valid_and_dimensions(self):
        result = {"aggregate": {
            "action_valid_weighted": 0.19, "n_cases": 3,
            "by_dimension": {"acted_ok": {"rate": 1.0, "n_applicable": 3},
                             "well_formed": {"rate": 0.67, "n_applicable": 3},
                             "tool_exists": {"rate": 0.67, "n_applicable": 3},
                             "args_schema_ok": {"rate": 0.5, "n_applicable": 2}},
            "by_depth": {"shallow": {"action_valid": 1.0, "n": 1},
                         "mid": {"action_valid": 0.5, "n": 2},
                         "deep": {"action_valid": 0.0, "n": 0}},
            "by_chain": {"chain-01": {"action_valid": 0.5, "n": 2}},
            "reference_model": "z-ai/glm-5.2",
            "depth_weights": {"shallow": 0.075, "mid": 0.383, "deep": 0.542},
        }, "cases": [], "errored": []}
        table = format_replay_table("Qwen", result)
        assert "Qwen" in table
        assert "action_valid" in table
        assert "0.19" in table
        assert "by dimension" in table
        assert "by depth" in table
        assert "shallow" in table and "mid" in table and "deep" in table

    def test_table_handles_none_dimension(self):
        result = {"aggregate": {
            "action_valid_weighted": 0.0, "n_cases": 1,
            "by_dimension": {"acted_ok": {"rate": None, "n_applicable": 0},
                             "well_formed": {"rate": None, "n_applicable": 0},
                             "tool_exists": {"rate": None, "n_applicable": 0},
                             "args_schema_ok": {"rate": None, "n_applicable": 0}},
            "by_depth": {"shallow": {"action_valid": 0.0, "n": 0},
                         "mid": {"action_valid": 0.0, "n": 1},
                         "deep": {"action_valid": 0.0, "n": 0}},
            "by_chain": {},
            "reference_model": "z-ai/glm-5.2",
            "depth_weights": {"shallow": 0.075, "mid": 0.383, "deep": 0.542},
        }, "cases": [], "errored": [{"id": "x", "error": "boom"}]}
        table = format_replay_table("M", result)
        assert "errored" in table

    def test_table_with_empty_depth(self):
        result = {"aggregate": {
            "action_valid_weighted": 0.383, "n_cases": 1,
            "by_dimension": {"acted_ok": {"rate": 1.0, "n_applicable": 1},
                             "well_formed": {"rate": 1.0, "n_applicable": 1},
                             "tool_exists": {"rate": 1.0, "n_applicable": 1},
                             "args_schema_ok": {"rate": 1.0, "n_applicable": 1}},
            "by_depth": {},
            "by_chain": {},
            "reference_model": "z-ai/glm-5.2",
            "depth_weights": {"shallow": 0.075, "mid": 0.383, "deep": 0.542},
        }, "cases": [], "errored": []}
        table = format_replay_table("M", result)
        assert "action_valid" in table


# ===========================================================================
# integration: load -> run end-to-end via load_cases
# ===========================================================================

class TestLoadAndRun:
    def test_load_then_run(self, tmp_path):
        msgs = [{"role": "user", "content": "read it"}]
        cap = _make_capture(tmp_path, "req-1.json", messages=msgs,
                            tools=[READ_TOOL], max_tokens=32000)
        d = _valid_case_dict(cap)
        path = _replay_jsonl(tmp_path, "m.jsonl", d)
        cases = load_cases(path, "main-replay")
        tok = ReplayFakeTokenizer()
        gen = lambda p, max_tokens=0: '{"tool": "read", "arguments": {"filePath": "f"}}'
        result = run_main_replay(cases, tok, gen, native=False)
        assert result["aggregate"]["by_depth"]["mid"]["action_valid"] == 1.0
        assert result["aggregate"]["n_cases"] == 1
