import types

import litmus_core as core
from conftest import FakeBackend


def test_run_one_counts_tokens_and_builds_run():
    be = FakeBackend(canned_text="one two three four")

    class Tok:
        def encode(self, text):
            return text.split()

    r = core.run_one(be, object(), Tok(), "prompt here", max_tokens=4,
                     label="fake")
    assert isinstance(r, core.Run)
    assert r.label == "fake"
    assert r.gen_tokens == 4              # four canned words yielded
    assert r.sample.startswith("one two three four")
    assert r.peak_mem_mb == 0.0


def test_cmd_perplexity_runs_end_to_end(capsys, monkeypatch, tmp_path):
    be = FakeBackend(per_token_logprob=-2.0)
    ref = tmp_path / "ref.txt"
    ref.write_text("word " * 50, encoding="utf-8")
    args = types.SimpleNamespace(
        reference_text=str(ref), ppl_window=16, repo="some/repo",
        label="fake", sizes="1.7B",
    )
    # _targets_for reads args.repo -> single (label, repo)
    core.cmd_perplexity(be, args)
    out = capsys.readouterr().out
    assert "perplexity" in out.lower()
