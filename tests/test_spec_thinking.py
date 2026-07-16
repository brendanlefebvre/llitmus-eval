import pytest

from litmus_spec import (
    ConstraintCase, ToolCase, _chat, parse_native, run_constraints,
    run_tool_calling, score_constraint_case, score_tool_call, strip_thinking,
    supports_thinking,
)


class ThinkingTokenizer:
    """Template that honours enable_thinking, like Qwen3.5 / Ternary-Bonsai."""

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=False, tools=None, enable_thinking=None):
        base = messages[-1]["content"]
        if enable_thinking is False:
            return base + "<think>\n\n</think>\n\n"
        return base + "<think>\n"


class PlainTokenizer:
    """Template that accepts the kwarg but ignores it (no thinking mode)."""

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=False, tools=None, **kw):
        return messages[-1]["content"]


class StrictTokenizer:
    """Older tokenizer whose signature rejects the kwarg outright."""

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=False, tools=None):
        return messages[-1]["content"]


def test_supports_thinking_true_when_template_honours_flag():
    assert supports_thinking(ThinkingTokenizer()) is True


def test_supports_thinking_false_when_template_ignores_flag():
    assert supports_thinking(PlainTokenizer()) is False


def test_supports_thinking_false_when_tokenizer_rejects_kwarg():
    # Must not crash the run for models predating thinking modes.
    assert supports_thinking(StrictTokenizer()) is False


def test_chat_passes_enable_thinking_through_to_template():
    msgs = [{"role": "user", "content": "hi"}]
    assert _chat(ThinkingTokenizer(), msgs, enable_thinking=False).endswith(
        "<think>\n\n</think>\n\n")
    assert _chat(ThinkingTokenizer(), msgs).endswith("<think>\n")


def test_chat_omits_enable_thinking_when_not_requested():
    # A tokenizer that cannot accept the kwarg must still work when the caller
    # does not ask for a thinking mode.
    assert _chat(StrictTokenizer(), [{"role": "user", "content": "hi"}]) == "hi"


def test_strip_thinking_returns_answer_after_closed_block():
    answer, closed = strip_thinking('<think>\nlots of reasoning\n</think>\n\n{"tool": null}')
    assert answer == '{"tool": null}'
    assert closed is True


def test_strip_thinking_is_noop_without_a_think_block():
    answer, closed = strip_thinking('{"tool": "get_weather"}')
    assert answer == '{"tool": "get_weather"}'
    assert closed is True


def test_strip_thinking_unclosed_block_yields_empty_answer():
    # Ran out of budget mid-reasoning: the whole output is scratchpad, so there
    # is no answer to score. Mirrors litmus.py's _strip_thinking semantics.
    answer, closed = strip_thinking("<think>\nI will output the JSON.\nOne last check.")
    assert answer == ""
    assert closed is False


def test_strip_thinking_handles_empty_think_block():
    # enable_thinking=False renders '<think>\n\n</think>\n\n' before the answer.
    answer, closed = strip_thinking("<think>\n\n</think>\n\nParis")
    assert answer == "Paris"
    assert closed is True


def test_empty_answer_vacuously_passes_prohibition_checks():
    """Guards WHY unclosed thinking needs explicit failure.

    A prohibition-style constraint is trivially satisfied by an empty string,
    so a model that burned its whole budget reasoning and emitted no answer
    would be *rewarded* if we scored its empty answer normally.
    """
    case = ConstraintCase("c1", "p", [("all_lowercase", {})])
    assert score_constraint_case("", case)[0]["passed"] is True


def test_score_constraint_case_fails_every_check_when_thinking_unclosed():
    case = ConstraintCase("c1", "p", [("all_lowercase", {}), ("max_words", {"n": 5})])
    checks = score_constraint_case("", case, closed=False)
    assert [c["passed"] for c in checks] == [False, False]
    assert all("think" in c["detail"].lower() for c in checks)


def test_run_constraints_scores_unclosed_thinking_as_a_miss():
    """A runaway reasoner must not score strict=1.0 on a prohibition case."""
    case = ConstraintCase("c1", "p", [("all_lowercase", {})])
    runaway = "<think>\nI will output the answer.\nOne last check."
    result = run_constraints([case], ThinkingTokenizer(), lambda p: runaway)
    assert result["aggregate"]["strict"] == 0.0
    assert result["cases"][0]["thinking_unclosed"] is True


def test_run_constraints_scores_the_answer_not_the_reasoning():
    """Uppercase inside <think> must not fail an all_lowercase case."""
    case = ConstraintCase("c1", "p", [("all_lowercase", {})])
    out = "<think>\nI should use Lowercase Only here.\n</think>\n\nhello there"
    result = run_constraints([case], ThinkingTokenizer(), lambda p: out)
    assert result["aggregate"]["strict"] == 1.0
    assert result["cases"][0]["thinking_unclosed"] is False


def test_run_tool_calling_ignores_json_inside_the_think_block():
    """Reasoning models weigh candidate calls *as JSON* inside <think>.

    parse_prompted takes the first JSON object it finds, so without stripping
    it would score the model's discarded first guess instead of its answer.
    """
    case = ToolCase("t1", "weather?", [{"name": "get_weather"}],
                    {"tool": "get_weather", "arguments": {"location": "Paris"}})
    out = ('<think>\nMaybe {"tool": "send_email"}? No, that is wrong.\n</think>\n\n'
           '{"tool": "get_weather", "arguments": {"location": "Paris"}}')
    result = run_tool_calling([case], ThinkingTokenizer(), lambda p: out,
                              native=False)
    assert result["aggregate"]["prompted"]["right_tool"] == 1.0
    assert result["aggregate"]["prompted"]["args_ok"] == 1.0


def test_run_tool_calling_runaway_thinking_gets_no_abstention_credit():
    """A model that never answers must not be credited for 'correctly' abstaining."""
    case = ToolCase("t11", "write a haiku", [{"name": "send_email"}],
                    {"tool": None})
    runaway = '<think>\nShould I call a tool? One last check.'
    result = run_tool_calling([case], ThinkingTokenizer(), lambda p: runaway,
                              native=False)
    assert result["aggregate"]["prompted"]["abstained_ok"] == 0.0
    assert result["cases"][0]["thinking_unclosed"] is True


def test_native_abstention_vacuously_credits_an_empty_answer():
    """Guards WHY `closed` must reach score_tool_call.

    Native abstention is inferred from the ABSENCE of tool-call structure, so an
    empty answer looks identical to a deliberate no-call. This documents the
    trap that `closed=False` exists to close.
    """
    assert score_tool_call(parse_native(""), {"tool": None},
                           native=True)["abstained_ok"] is True


def test_score_tool_call_unclosed_thinking_is_not_a_native_abstention():
    score = score_tool_call(parse_native(""), {"tool": None},
                            native=True, closed=False)
    assert score["abstained_ok"] is False
    assert score["well_formed"] is False
    assert score["right_tool"] is False


def test_score_tool_call_unclosed_thinking_on_a_tool_case():
    # Non-abstention convention: abstained_ok stays None, args_ok unevaluable.
    score = score_tool_call(parse_native(""), {"tool": "get_weather"},
                            native=True, closed=False)
    assert score["well_formed"] is False
    assert score["right_tool"] is False
    assert score["abstained_ok"] is None
    assert score["args_ok"] is None
