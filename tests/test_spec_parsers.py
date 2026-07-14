from litmus_spec import parse_prompted, parse_native, ParsedCall


def test_prompted_plain_json():
    r = parse_prompted('{"tool": "get_weather", "arguments": {"location": "Paris"}}')
    assert r.well_formed and r.tool == "get_weather"
    assert r.arguments == {"location": "Paris"}


def test_prompted_fenced_with_preamble():
    text = 'Sure!\n```json\n{"tool": "send_email", "arguments": {"to": "a@b.c"}}\n```'
    r = parse_prompted(text)
    assert r.well_formed and r.tool == "send_email"


def test_prompted_explicit_null_is_wellformed_abstention():
    r = parse_prompted('{"tool": null, "arguments": {}}')
    assert r.well_formed is True and r.tool is None


def test_prompted_garbage_is_not_wellformed():
    r = parse_prompted("I cannot help with that.")
    assert r.well_formed is False and r.tool is None


def test_prompted_empty_is_truncated():
    r = parse_prompted("   ")
    assert r.well_formed is False and r.detail == "truncated"


def test_native_qwen_tool_call_tag():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'
    r = parse_native(text)
    assert r.well_formed and r.tool == "get_weather"
    assert r.arguments == {"location": "Paris"}


def test_native_llama_python_tag():
    text = '<|python_tag|>{"name": "get_weather", "arguments": {"location": "Rome"}}'
    r = parse_native(text)
    assert r.well_formed and r.tool == "get_weather" and r.arguments["location"] == "Rome"


def test_native_generic_json_fallback():
    text = '{"name": "search", "arguments": {"q": "cats"}}'
    r = parse_native(text)
    assert r.well_formed and r.tool == "search"


def test_native_unparseable():
    r = parse_native("no tool here")
    assert r.well_formed is False


def test_prompted_brace_inside_string_argument():
    text = '{"tool": "search", "arguments": {"query": "func() { return 1 }"}}'
    r = parse_prompted(text)
    assert r.well_formed and r.tool == "search"
    assert r.arguments == {"query": "func() { return 1 }"}


def test_native_brace_inside_string_argument():
    text = '<tool_call>{"name": "run", "arguments": {"code": "if (x) {y}"}}</tool_call>'
    r = parse_native(text)
    assert r.well_formed and r.tool == "run"
    assert r.arguments == {"code": "if (x) {y}"}


def test_prompted_escaped_quote_then_brace_in_string():
    text = r'{"tool": "echo", "arguments": {"msg": "she said \"hi\" }"}}'
    r = parse_prompted(text)
    assert r.well_formed and r.tool == "echo"
    assert r.arguments == {"msg": 'she said "hi" }'}


def test_native_broken_tag_is_attempted_not_wellformed():
    r = parse_native('<tool_call>{{"name":"get_weather"}}</tool_call>')
    assert r.well_formed is False and r.attempted is True


def test_native_prose_is_not_attempted():
    r = parse_native("The capital of France is Paris.")
    assert r.attempted is False and r.well_formed is False


def test_native_clean_call_is_attempted():
    r = parse_native('<tool_call>{"name":"get_weather","arguments":{"location":"Paris"}}</tool_call>')
    assert r.well_formed is True and r.attempted is True


def test_native_unclosed_tool_call_is_attempted():
    # real Qwen behavior: opening <tool_call> tag, no closing tag, malformed JSON
    text = ('Autumn leaves falling,\nNature\'s symphony ends.\n'
            '<tool_call>\n{{"name": "send_email", "arguments": {"to": "x"')
    r = parse_native(text)
    assert r.attempted is True and r.well_formed is False


def test_native_unclosed_tag_with_valid_json_is_parsed():
    # opening tag, no close, but valid single-brace JSON after -> generic fallback reads it
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Rome"}}'
    r = parse_native(text)
    assert r.attempted is True and r.well_formed is True and r.tool == "get_weather"
