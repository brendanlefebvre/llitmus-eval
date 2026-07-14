import pytest
from litmus_spec import (
    ToolCase, build_prompted_tool_prompt, supports_native_tools,
    build_native_tool_prompt, build_constraint_prompt, ConstraintCase,
)


class FakeTokenizer:
    """apply_chat_template records what it got; supports tools= by default."""
    def __init__(self, tools_supported=True):
        self.tools_supported = tools_supported
        self.last_tools = None

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=False, tools=None):
        if tools is not None and not self.tools_supported:
            raise ValueError("template has no tools support")
        self.last_tools = tools
        body = " | ".join(m["content"] for m in messages)
        suffix = f" [tools={len(tools)}]" if tools else ""
        return f"<s>{body}{suffix}"


def test_prompted_prompt_has_no_tools_kwarg_and_lists_schema():
    tok = FakeTokenizer()
    case = ToolCase("t1", "weather?", [{"name": "get_weather"}],
                    {"tool": "get_weather", "arguments": {}})
    out = build_prompted_tool_prompt(tok, case)
    assert tok.last_tools is None            # prompted path never uses tools=
    assert "get_weather" in out              # schema is in the preamble
    assert "json" in out.lower()


def test_supports_native_true_and_false():
    assert supports_native_tools(FakeTokenizer(tools_supported=True)) is True
    assert supports_native_tools(FakeTokenizer(tools_supported=False)) is False


def test_native_prompt_passes_tools_kwarg():
    tok = FakeTokenizer()
    case = ToolCase("t1", "weather?", [{"name": "get_weather"}],
                    {"tool": "get_weather", "arguments": {}})
    build_native_tool_prompt(tok, case)
    assert tok.last_tools == case.tools


def test_constraint_prompt_wraps_user_message():
    tok = FakeTokenizer()
    case = ConstraintCase("c1", "list 3 fruits", [("exact_bullets", {"n": 3})])
    out = build_constraint_prompt(tok, case)
    assert "list 3 fruits" in out
