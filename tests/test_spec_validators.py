from litmus_spec import CHECKS, CheckResult


def run(kind, text, **params):
    return CHECKS[kind].fn(text, params)


def test_exact_bullets_pass_and_fail():
    three = "- a\n- b\n- c"
    assert run("exact_bullets", three, n=3).passed is True
    assert run("exact_bullets", "- a\n- b", n=3).passed is False
    assert run("exact_bullets", "- a\n- b\n- c\n- d", n=3).passed is False


def test_exact_bullets_counts_numbered_and_star():
    assert run("exact_bullets", "1. a\n2. b\n3. c", n=3).passed is True
    assert run("exact_bullets", "* a\n* b\n* c", n=3).passed is True


def test_min_and_max_words():
    assert run("min_words", "one two three", n=3).passed is True
    assert run("min_words", "one two", n=3).passed is False
    assert run("max_words", "one two", n=3).passed is True
    assert run("max_words", "one two three four", n=3).passed is False


def test_casing():
    assert run("all_lowercase", "hello there").passed is True
    assert run("all_lowercase", "Hello").passed is False
    assert run("all_uppercase", "HELLO 1!").passed is True
    assert run("all_uppercase", "Hello").passed is False


def test_forbidden_word_is_case_insensitive_word_boundary():
    assert run("forbidden_word", "I like cats", word="dog").passed is True
    assert run("forbidden_word", "A DOG appeared", word="dog").passed is False
    # substring that is not the whole word does not trip it
    assert run("forbidden_word", "dogma is fine", word="dog").passed is True


def test_required_phrase_and_ends_with():
    assert run("required_phrase", "the answer is 42", phrase="answer is").passed is True
    assert run("required_phrase", "nope", phrase="answer is").passed is False
    assert run("ends_with", "goodbye now  ", phrase="now").passed is True
    assert run("ends_with", "now then", phrase="now").passed is False


def test_valid_json():
    assert run("valid_json", '{"a": 1}').passed is True
    assert run("valid_json", 'not json').passed is False


def test_regex_match():
    assert run("regex_match", "abc123", pattern=r"\d{3}").passed is True
    assert run("regex_match", "abc", pattern=r"\d{3}").passed is False


def test_result_is_checkresult_with_detail():
    r = run("exact_bullets", "- a\n- b", n=3)
    assert isinstance(r, CheckResult)
    assert r.detail  # non-empty explanation
