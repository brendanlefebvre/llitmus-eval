import litmus_spec


def test_run_constraints_with_fake_generate():
    # run_constraints only needs a tokenizer + generate_fn; prove the seam is
    # generate_fn-shaped so any Backend.stream can drive it.
    cases = litmus_spec.load_cases(
        litmus_spec.DEFAULT_CASES["constraints"], "constraints")

    class Tok:
        def apply_chat_template(self, msgs, **kw):
            return msgs[-1]["content"]
        def encode(self, text):
            return text.split()

    def fake_generate(prompt):
        return "ok " * 20

    result = litmus_spec.run_constraints(cases[:2], Tok(), fake_generate,
                                         enable_thinking=None)
    assert isinstance(result, dict)
    assert set(result) >= {"aggregate", "cases", "errored"}
    assert len(result["cases"]) == 2         # one record per case fed in
    assert result["errored"] == []           # fake_generate never raises
