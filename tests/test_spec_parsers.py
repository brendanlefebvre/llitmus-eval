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
