import pytest
from litmus_spec import load_cases, CaseError, ConstraintCase, ToolCase


def write(tmp_path, name, *lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_load_valid_constraint_cases(tmp_path):
    path = write(
        tmp_path, "c.jsonl",
        '{"id":"a","prompt":"p","checks":[{"kind":"exact_bullets","n":3}]}',
        '{"id":"b","prompt":"q","checks":[{"kind":"all_lowercase"}]}',
    )
    cases = load_cases(path, "constraints")
    assert len(cases) == 2
    assert isinstance(cases[0], ConstraintCase)
    assert cases[0].checks == [("exact_bullets", {"n": 3})]


def test_load_valid_tool_cases(tmp_path):
    path = write(
        tmp_path, "t.jsonl",
        '{"id":"t1","prompt":"weather?","tools":[{"name":"get_weather"}],'
        '"expect":{"tool":"get_weather","arguments":{"location":"Paris"}}}',
        '{"id":"t2","prompt":"hi","tools":[{"name":"get_weather"}],'
        '"expect":{"tool":null}}',
    )
    cases = load_cases(path, "tool-calling")
    assert isinstance(cases[0], ToolCase)
    assert cases[1].expect["tool"] is None


def test_unknown_kind_raises(tmp_path):
    path = write(tmp_path, "c.jsonl",
                 '{"id":"a","prompt":"p","checks":[{"kind":"no_such_check"}]}')
    with pytest.raises(CaseError, match="no_such_check"):
        load_cases(path, "constraints")


def test_missing_field_raises(tmp_path):
    path = write(tmp_path, "c.jsonl", '{"id":"a","checks":[]}')  # no prompt
    with pytest.raises(CaseError, match="prompt"):
        load_cases(path, "constraints")


def test_bad_param_type_raises(tmp_path):
    path = write(tmp_path, "c.jsonl",
                 '{"id":"a","prompt":"p","checks":[{"kind":"exact_bullets","n":"three"}]}')
    with pytest.raises(CaseError, match="n"):
        load_cases(path, "constraints")


def test_line_number_in_error(tmp_path):
    path = write(
        tmp_path, "c.jsonl",
        '{"id":"a","prompt":"p","checks":[{"kind":"all_lowercase"}]}',
        '{"id":"b","prompt":"p","checks":[{"kind":"bogus"}]}',
    )
    with pytest.raises(CaseError, match="line 2"):
        load_cases(path, "constraints")


def test_non_object_line_raises(tmp_path):
    path = write(tmp_path, "c.jsonl", "5")
    with pytest.raises(CaseError, match="line 1"):
        load_cases(path, "constraints")


def test_non_list_checks_raises(tmp_path):
    path = write(tmp_path, "c.jsonl",
                 '{"id":"a","prompt":"p","checks":{"kind":"all_lowercase"}}')
    with pytest.raises(CaseError, match="checks.*list"):
        load_cases(path, "constraints")


def test_non_dict_check_entry_raises(tmp_path):
    path = write(tmp_path, "c.jsonl",
                 '{"id":"a","prompt":"p","checks":["all_lowercase"]}')
    with pytest.raises(CaseError, match="check.*object"):
        load_cases(path, "constraints")


def test_non_dict_expect_raises(tmp_path):
    path = write(tmp_path, "t.jsonl",
                 '{"id":"t","prompt":"p","tools":[{"name":"x"}],"expect":5}')
    with pytest.raises(CaseError, match="expect.*object"):
        load_cases(path, "tool-calling")


def test_bool_for_int_param_raises(tmp_path):
    path = write(tmp_path, "c.jsonl",
                 '{"id":"a","prompt":"p","checks":[{"kind":"exact_bullets","n":true}]}')
    with pytest.raises(CaseError, match="n"):
        load_cases(path, "constraints")


def test_invalid_regex_pattern_raises_at_load(tmp_path):
    # A bad regex must fail fast during loading, not crash mid-run after the
    # model is loaded (score_constraint_case runs outside run_constraints' guard).
    path = write(tmp_path, "c.jsonl",
                 '{"id":"a","prompt":"p","checks":[{"kind":"regex_match","pattern":"([unclosed"}]}')
    with pytest.raises(CaseError, match="line 1.*regex_match.*pattern"):
        load_cases(path, "constraints")


def test_valid_regex_pattern_loads(tmp_path):
    path = write(tmp_path, "c.jsonl",
                 '{"id":"a","prompt":"p","checks":[{"kind":"regex_match","pattern":"^ok$"}]}')
    cases = load_cases(path, "constraints")
    assert cases[0].checks == [("regex_match", {"pattern": "^ok$"})]
