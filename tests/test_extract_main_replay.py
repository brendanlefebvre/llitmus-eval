"""Tests for scripts/extract_main_replay.py — chain grouping, skip rules,
depth strata, sampling, and case format.

Uses temp directories with synthetic capture files. Does not depend on the
real captures directory.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
import extract_main_replay as emr  # noqa: E402


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

MAIN_SYS = ("You are opencode, an interactive CLI tool that helps users "
            "with software engineering tasks.")
NONMAIN_SYS = "You are a title generator. Generate a short title."
TOOLS = [
    {"type": "function",
     "function": {"name": "read", "parameters": {"type": "object",
                    "properties": {"filePath": {"type": "string"}}}}},
]


def _body(messages, tools=TOOLS, max_tokens=32000):
    return {
        "model": "test/model",
        "max_tokens": max_tokens,
        "messages": messages,
        "tools": tools,
        "stream": True,
    }


def write_capture(dirpath, name, messages, tools=TOOLS, max_tokens=32000):
    p = dirpath / name
    p.write_text(json.dumps(_body(messages, tools, max_tokens)),
                 encoding="utf-8")
    return p


def sys_msg():
    return {"role": "system", "content": MAIN_SYS}


def user_msg(content="hello"):
    return {"role": "user", "content": content}


def assistant_tool_msg(name="read", args=None):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call-0", "type": "function",
             "function": {"name": name,
                          "arguments": json.dumps(args or {"filePath": "f.py"})}},
        ],
    }


def assistant_prose_msg(text="Here is the answer."):
    return {"role": "assistant", "content": text}


def assistant_empty_msg():
    return {"role": "assistant", "content": None}


def tool_msg(content="result"):
    return {"role": "tool", "content": content, "tool_call_id": "call-0"}


def make_chain(dirpath, names_and_msgs):
    """Write a sequence of captures that form a prefix chain.

    names_and_msgs: list of (filename, messages_list)
    """
    for name, msgs in names_and_msgs:
        write_capture(dirpath, name, msgs)


# ---------------------------------------------------------------------------
# 1. Chain grouping
# ---------------------------------------------------------------------------

class TestChainGrouping:
    def test_two_captures_grow_into_one_chain(self, tmp_path):
        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(),
                  assistant_tool_msg(), tool_msg()]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)

        assert len(chains) == 1
        assert len(chains[0]) == 2

    def test_divergent_messages_start_new_chain(self, tmp_path):
        msgs_a = [sys_msg(), user_msg("question A")]
        msgs_b = [sys_msg(), user_msg("question B")]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)

        assert len(chains) == 2
        assert len(chains[0]) == 1
        assert len(chains[1]) == 1

    def test_shorter_capture_starts_new_chain(self, tmp_path):
        msgs_a = [sys_msg(), user_msg(), assistant_tool_msg(), tool_msg()]
        msgs_b = [sys_msg(), user_msg()]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)

        assert len(chains) == 2

    def test_same_length_stays_in_chain(self, tmp_path):
        msgs = [sys_msg(), user_msg()]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)

        assert len(chains) == 1
        assert len(chains[0]) == 2

    def test_three_captures_growing_chain(self, tmp_path):
        m1 = [sys_msg(), user_msg()]
        m2 = [sys_msg(), user_msg(), assistant_tool_msg(), tool_msg()]
        m3 = [sys_msg(), user_msg(), assistant_tool_msg(), tool_msg(),
              user_msg("next"), assistant_tool_msg(name="edit")]
        for i, msgs in enumerate((m1, m2, m3)):
            write_capture(tmp_path,
                          f"req-20260101T00000{i}.000000-000{i}.json", msgs)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)

        assert len(chains) == 1
        assert len(chains[0]) == 3

    def test_py_files_skipped(self, tmp_path):
        (tmp_path / "helper.py").write_text("# not a capture")
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json",
                      [sys_msg(), user_msg()])
        rows = emr.load_captures(tmp_path)
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# 2. Skip rules
# ---------------------------------------------------------------------------

class TestSkipRules:
    def test_skip_non_main_class(self, tmp_path, capsys):
        msgs_a = [{"role": "system", "content": NONMAIN_SYS},
                  user_msg()]
        msgs_b = list(msgs_a) + [assistant_tool_msg(), tool_msg()]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json",
                      msgs_a, tools=[])
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json",
                      msgs_b, tools=[])

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, [])

        assert len(usable) == 0
        captured = capsys.readouterr()
        assert "not main" in captured.err

    def test_skip_appended_starts_with_user(self, tmp_path, capsys):
        # Capture N: [sys, user, assistant, tool]
        # Capture N+1: [sys, user, assistant, tool, user, assistant]
        # Appended = [user, assistant] — starts with user, not assistant
        msgs_a = [sys_msg(), user_msg(), assistant_tool_msg(), tool_msg()]
        msgs_b = list(msgs_a) + [user_msg("more"), assistant_prose_msg()]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, [])

        assert len(usable) == 0
        captured = capsys.readouterr()
        assert "not assistant" in captured.err

    def test_skip_pathological_reference(self, tmp_path, capsys):
        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), assistant_empty_msg()]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, [])

        assert len(usable) == 0
        captured = capsys.readouterr()
        assert "pathological reference" in captured.err

    def test_skip_curl_probe(self, tmp_path, capsys):
        curl_name = "req-20260727T121220.315928-0000.json"
        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), assistant_tool_msg(), tool_msg()]
        write_capture(tmp_path, curl_name, msgs_a)
        write_capture(tmp_path, "req-20260727T121221.000000-0001.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, [])

        assert len(usable) == 0
        captured = capsys.readouterr()
        assert "curl probe" in captured.err

    def test_skip_qwen_authored(self, tmp_path, capsys):
        # Create a capture whose filename timestamp matches a ledger entry
        # within ±2 seconds.
        cap_name = "req-20260727T122717.000000-0003.json"
        ledger_ts = "2026-07-27T12:27:16.500000+00:00"  # ~0.5s difference

        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), assistant_tool_msg(), tool_msg()]
        write_capture(tmp_path, cap_name, msgs_a)
        write_capture(tmp_path, "req-20260727T122718.000000-0004.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, [ledger_ts])

        assert len(usable) == 0
        captured = capsys.readouterr()
        assert "Qwen3-14B" in captured.err
        assert ledger_ts in captured.err

    def test_skip_over_limit(self, tmp_path, capsys):
        # Need est_tokens > 60000. estimate_prompt_tokens counts chars // 4
        # for content + tool schema JSON chars // 4.
        # A 240001-char user message → 60000 tokens (content alone).
        # Need >60000, so 240005 chars → 60001 tokens.
        big_content = "x" * 240005
        msgs_a = [sys_msg(), user_msg(big_content)]
        msgs_b = [sys_msg(), user_msg(big_content),
                  assistant_tool_msg(), tool_msg()]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, [])

        assert len(usable) == 0
        captured = capsys.readouterr()
        assert "over limit" in captured.err

    def test_usable_pair_passes_all_rules(self, tmp_path, capsys):
        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), assistant_tool_msg(), tool_msg()]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, [])

        assert len(usable) == 1
        pair = usable[0]
        assert pair["chain_id"] == "chain-01"
        assert pair["depth_stratum"] == "shallow"
        assert pair["reference"]["acted"] is True
        assert pair["reference"]["tools"] == ["read"]
        assert pair["reference"]["arguments"] == [{"filePath": "f.py"}]

    def test_skip_no_appended_messages(self, tmp_path, capsys):
        msgs = [sys_msg(), user_msg()]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, [])

        assert len(usable) == 0
        captured = capsys.readouterr()
        assert "no appended messages" in captured.err


# ---------------------------------------------------------------------------
# 3. Depth stratum assignment
# ---------------------------------------------------------------------------

class TestStratumAssignment:
    def test_shallow_boundary(self):
        assert emr.assign_stratum(0) == "shallow"
        assert emr.assign_stratum(15999) == "shallow"

    def test_mid_boundary(self):
        assert emr.assign_stratum(16000) == "mid"
        assert emr.assign_stratum(39999) == "mid"

    def test_deep_boundary(self):
        assert emr.assign_stratum(40000) == "deep"
        assert emr.assign_stratum(60000) == "deep"

    def test_over_limit_excluded(self):
        assert emr.assign_stratum(60001) is None
        assert emr.assign_stratum(100000) is None


# ---------------------------------------------------------------------------
# 4. Sampling
# ---------------------------------------------------------------------------

class TestSampling:
    def _make_pairs(self, n, chain_id, stratum):
        return [
            {
                "capture_path": f"/fake/{chain_id}-{i}.json",
                "chain_id": chain_id,
                "depth_stratum": stratum,
                "est_tokens": 1000,
                "reference": {"acted": True, "tools": ["read"],
                              "arguments": [{}]},
            }
            for i in range(n)
        ]

    def test_five_per_stratum_single_chain(self):
        usable = []
        for s in ("shallow", "mid", "deep"):
            usable.extend(self._make_pairs(10, "chain-01", s))
        sample, coverage = emr.sample_cases(usable, 5)

        by_stratum = {"shallow": 0, "mid": 0, "deep": 0}
        for c in sample:
            by_stratum[c["depth_stratum"]] += 1
        assert by_stratum == {"shallow": 5, "mid": 5, "deep": 5}
        assert len(sample) == 15

    def test_multi_chain_per_stratum(self):
        usable = []
        for s in ("shallow", "mid", "deep"):
            usable.extend(self._make_pairs(10, "chain-01", s))
            usable.extend(self._make_pairs(5, "chain-02", s))
        sample, coverage = emr.sample_cases(usable, 5)

        for s in ("shallow", "mid", "deep"):
            chains_in_stratum = set(
                c["chain_id"] for c in sample
                if c["depth_stratum"] == s
            )
            assert len(chains_in_stratum) >= 2, (
                f"{s} should have 2+ chains, got {chains_in_stratum}")

    def test_shortfall_when_fewer_than_target(self):
        usable = self._make_pairs(3, "chain-01", "shallow")
        usable.extend(self._make_pairs(10, "chain-01", "mid"))
        usable.extend(self._make_pairs(10, "chain-01", "deep"))
        sample, coverage = emr.sample_cases(usable, 5)

        by_stratum = {"shallow": 0, "mid": 0, "deep": 0}
        for c in sample:
            by_stratum[c["depth_stratum"]] += 1
        assert by_stratum["shallow"] == 3
        assert by_stratum["mid"] == 5
        assert by_stratum["deep"] == 5

    def test_even_sampling_within_chain(self):
        usable = self._make_pairs(20, "chain-01", "mid")
        sample, _ = emr.sample_cases(usable, 5)
        # Should not be the first 5 — indices should be spread out
        paths = [c["capture_path"] for c in sample]
        first_five = [f"/fake/chain-01-{i}.json" for i in range(5)]
        assert paths != first_five

    def test_empty_usable(self):
        sample, coverage = emr.sample_cases([], 5)
        assert sample == []
        for s in ("shallow", "mid", "deep"):
            assert coverage[s]["n"] == 0

    def test_coverage_reports_single_chain(self):
        usable = []
        for s in ("shallow", "mid", "deep"):
            usable.extend(self._make_pairs(10, "chain-01", s))
        _, coverage = emr.sample_cases(usable, 5)
        for s in ("shallow", "mid", "deep"):
            assert coverage[s]["chains"] == ["chain-01"]
            assert coverage[s]["n"] == 5


# ---------------------------------------------------------------------------
# 5. Case format
# ---------------------------------------------------------------------------

class TestCaseFormat:
    def test_output_jsonl_fields(self, tmp_path):
        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), assistant_tool_msg(), tool_msg()]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, [])
        sample, _ = emr.sample_cases(usable, 5)

        case = {
            "id": "mr-001",
            "capture_path": sample[0]["capture_path"],
            "chain_id": sample[0]["chain_id"],
            "depth_stratum": sample[0]["depth_stratum"],
            "est_tokens": sample[0]["est_tokens"],
            "reference": sample[0]["reference"],
        }

        assert case["id"] == "mr-001"
        assert case["capture_path"].startswith("/")
        assert case["chain_id"] == "chain-01"
        assert case["depth_stratum"] == "shallow"
        assert isinstance(case["est_tokens"], int)
        assert case["reference"]["acted"] is True
        assert case["reference"]["tools"] == ["read"]
        assert case["reference"]["arguments"] == [{"filePath": "f.py"}]

    def test_reference_prose_turn(self, tmp_path):
        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), assistant_prose_msg("The answer.")]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, [])

        assert len(usable) == 1
        ref = usable[0]["reference"]
        assert ref["acted"] is False
        assert ref["tools"] == []
        assert ref["arguments"] == []

    def test_parallel_tool_calls(self, tmp_path):
        assistant_parallel = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c0", "type": "function",
                 "function": {"name": "read",
                              "arguments": '{"filePath": "a.py"}'}},
                {"id": "c1", "type": "function",
                 "function": {"name": "read",
                              "arguments": '{"filePath": "b.py"}'}},
            ],
        }
        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), assistant_parallel]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, [])

        assert len(usable) == 1
        ref = usable[0]["reference"]
        assert ref["acted"] is True
        assert ref["tools"] == ["read", "read"]
        assert ref["arguments"] == [
            {"filePath": "a.py"}, {"filePath": "b.py"}]

    def test_jsonl_one_object_per_line(self, tmp_path):
        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), assistant_tool_msg(), tool_msg()]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs_b)

        out_path = tmp_path / "out.jsonl"
        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, [])
        sample, _ = emr.sample_cases(usable, 5)

        lines = []
        for i, c in enumerate(sample, 1):
            case = {
                "id": f"mr-{i:03d}",
                "capture_path": c["capture_path"],
                "chain_id": c["chain_id"],
                "depth_stratum": c["depth_stratum"],
                "est_tokens": c["est_tokens"],
                "reference": c["reference"],
            }
            lines.append(json.dumps(case))

        text = "\n".join(lines) + "\n"
        for line in text.strip().splitlines():
            obj = json.loads(line)
            assert "id" in obj
            assert "capture_path" in obj
            assert "chain_id" in obj
            assert "depth_stratum" in obj
            assert "est_tokens" in obj
            assert "reference" in obj
