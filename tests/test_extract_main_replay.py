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

    def test_chain_id_stable_from_first_capture_ts(self, tmp_path):
        # chain_id derives from the first capture's timestamp stem, not the
        # chain's position — stable across extractions regardless of corpus
        # growth or filtering.
        msgs_a = [sys_msg(), user_msg("question A")]
        msgs_b = [sys_msg(), user_msg("question B")]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260315T091500.123456-0007.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)

        assert emr.chain_stable_id(chains[0]) == "chain-20260101T000000"
        assert emr.chain_stable_id(chains[1]) == "chain-20260315T091500"
        # Dropping the first chain leaves the second chain's id unchanged.
        assert emr.chain_stable_id(chains[1:][0]) == "chain-20260315T091500"


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
        # Create a capture whose filename timestamp matches a ledger entry's
        # arrival time (ledger ts - latency_ms) within ±2 seconds.
        cap_name = "req-20260727T122717.000000-0003.json"
        # Ledger ts is stamped at completion; subtract latency_ms to recover
        # the arrival time that should match the capture filename.
        ledger_ts = "2026-07-27T12:27:16.500000+00:00"  # ~0.5s after capture
        latency_ms = 0
        qwen_entries = [{"ts": ledger_ts, "latency_ms": latency_ms}]

        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), assistant_tool_msg(), tool_msg()]
        write_capture(tmp_path, cap_name, msgs_a)
        write_capture(tmp_path, "req-20260727T122718.000000-0004.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, qwen_entries)

        assert len(usable) == 0
        captured = capsys.readouterr()
        assert "Qwen3-14B" in captured.err
        assert ledger_ts in captured.err

    def test_qwen_latency_ms_subtraction(self, tmp_path, capsys):
        # The ledger ts is stamped at response completion (arrival + latency).
        # Without subtracting latency_ms, a capture whose arrival is well
        # before completion would NOT match; with the subtraction it should.
        # Capture arrival: 12:27:17.000. Ledger completion: 12:27:19.500.
        # latency_ms = 2500 → arrival = 12:27:17.000 → matches exactly.
        cap_name = "req-20260727T122717.000000-0003.json"
        ledger_ts = "2026-07-27T12:27:19.500000+00:00"
        latency_ms = 2500
        qwen_entries = [{"ts": ledger_ts, "latency_ms": latency_ms}]

        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), assistant_tool_msg(), tool_msg()]
        write_capture(tmp_path, cap_name, msgs_a)
        write_capture(tmp_path, "req-20260727T122718.000000-0004.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, qwen_entries)

        assert len(usable) == 0
        captured = capsys.readouterr()
        assert "Qwen3-14B" in captured.err

    def test_qwen_null_latency_conservative_skip(self, tmp_path, capsys):
        # Same ledger completion ts as above, but latency_ms omitted (None).
        # The arrival time cannot be recovered, so a WIDE ±600s window
        # against the completion ts applies: any plausibly-related capture
        # pair is skipped rather than kept, with a distinct reason.
        cap_name = "req-20260727T122717.000000-0003.json"
        ledger_ts = "2026-07-27T12:27:19.500000+00:00"
        qwen_entries = [{"ts": ledger_ts, "latency_ms": None}]

        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), assistant_tool_msg(), tool_msg()]
        write_capture(tmp_path, cap_name, msgs_a)
        write_capture(tmp_path, "req-20260727T122718.000000-0004.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, qwen_entries)

        assert len(usable) == 0  # conservatively skipped
        captured = capsys.readouterr()
        assert "Qwen3-14B" in captured.err
        assert "null latency_ms" in captured.err
        assert "±600s" in captured.err

    def test_qwen_null_latency_far_outside_window_not_skipped(
            self, tmp_path, capsys):
        # Null latency, but the completion ts is ~13 minutes (>600s) after
        # the capture arrival — even the wide conservative window does not
        # reach it, so the pair is kept.
        cap_name = "req-20260727T122717.000000-0003.json"
        ledger_ts = "2026-07-27T12:40:30.000000+00:00"
        qwen_entries = [{"ts": ledger_ts, "latency_ms": None}]

        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), assistant_tool_msg(), tool_msg()]
        write_capture(tmp_path, cap_name, msgs_a)
        write_capture(tmp_path, "req-20260727T122718.000000-0004.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, qwen_entries)

        assert len(usable) == 1  # NOT skipped — outside even ±600s

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
        assert pair["chain_id"] == "chain-20260101T000000"
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

    def test_singleton_chain_drop_logged(self, tmp_path, capsys):
        # A single-capture chain (length 1) must be logged, not silently
        # dropped. Divergent first messages produce two length-1 chains.
        msgs_a = [sys_msg(), user_msg("question A")]
        msgs_b = [sys_msg(), user_msg("question B")]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        emr.process_pairs(chains, [])

        captured = capsys.readouterr()
        assert "no pairs possible" in captured.err
        assert "chain-20260101T000000" in captured.err

    def test_malformed_args_skips_pair(self, tmp_path, capsys):
        # A tool_call whose arguments are not valid JSON must skip the pair
        # with a logged reason, not silently substitute empty args.
        bad_assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c0", "type": "function",
                 "function": {"name": "read",
                              "arguments": "{not valid json"}},
            ],
        }
        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), bad_assistant]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, [])

        assert len(usable) == 0
        captured = capsys.readouterr()
        assert "malformed tool_call arguments" in captured.err
        assert "JSONDecodeError" in captured.err

    def test_unexpected_args_type_skips_pair_with_type_label(
            self, tmp_path, capsys):
        # A tool_call whose arguments are neither null/""/str/dict must skip
        # the pair with a reason naming the unexpected type — not the
        # JSONDecodeError label, which would misname the cause.
        weird_assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c0", "type": "function",
                 "function": {"name": "read",
                              "arguments": 12345}},
            ],
        }
        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), weird_assistant]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, [])

        assert len(usable) == 0
        captured = capsys.readouterr()
        assert "unexpected tool_call arguments type (int)" in captured.err
        assert "JSONDecodeError" not in captured.err

    def test_null_args_pair_kept(self, tmp_path, capsys):
        # Explicit null arguments must NOT shrink the pair population.
        null_assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c0", "type": "function",
                 "function": {"name": "list_files",
                              "arguments": None}},
            ],
        }
        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), null_assistant]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, [])

        assert len(usable) == 1
        assert usable[0]["reference"]["arguments"] == [{}]


# ---------------------------------------------------------------------------
# 2b. Class-filter before chain grouping
# ---------------------------------------------------------------------------

class TestClassFilterBeforeGrouping:
    def test_non_main_filtered_before_grouping(self, tmp_path, capsys):
        # Interleaved non-main capture between two main captures would
        # fragment the chain if not filtered first. With class-filtering
        # applied before group_chains, the two main captures form one chain.
        main_msgs_a = [sys_msg(), user_msg()]
        main_msgs_b = [sys_msg(), user_msg(), assistant_tool_msg(), tool_msg()]
        nonmain_msgs = [{"role": "system", "content": NONMAIN_SYS},
                        user_msg()]

        write_capture(tmp_path, "req-20260101T000000.000000-0000.json",
                      main_msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json",
                      nonmain_msgs, tools=[])
        write_capture(tmp_path, "req-20260101T000002.000000-0002.json",
                      main_msgs_b)

        rows = emr.load_captures(tmp_path)
        # Mirror the class-filter logic from main().
        main_rows = [r for r in rows
                     if emr.classify(r["body"]).cls == "main"]
        chains = emr.group_chains(main_rows)

        assert len(chains) == 1
        assert len(chains[0]) == 2

    def test_build_reference_returns_none_on_malformed_json(self):
        bad_assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c0", "type": "function",
                 "function": {"name": "read",
                              "arguments": "{not valid json"}},
            ],
        }
        assert emr.build_reference(bad_assistant) is None

    def test_build_reference_returns_none_on_non_string_non_dict_args(self):
        weird_assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c0", "type": "function",
                 "function": {"name": "read",
                              "arguments": 12345}},
            ],
        }
        assert emr.build_reference(weird_assistant) is None

    def test_build_reference_null_args_kept_as_empty(self):
        # Explicit JSON null arguments are legitimate (zero-parameter tools
        # on some stacks) and map to {}, not a skip.
        null_assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c0", "type": "function",
                 "function": {"name": "list_files",
                              "arguments": None}},
            ],
        }
        ref = emr.build_reference(null_assistant)
        assert ref == {"acted": True, "tools": ["list_files"],
                       "arguments": [{}]}

    def test_build_reference_empty_string_args_kept_as_empty(self):
        empty_assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c0", "type": "function",
                 "function": {"name": "list_files",
                              "arguments": ""}},
            ],
        }
        ref = emr.build_reference(empty_assistant)
        assert ref == {"acted": True, "tools": ["list_files"],
                       "arguments": [{}]}


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
    def test_reference_model_constant(self):
        assert emr.REFERENCE_MODEL == "z-ai/glm-5.2"

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
        assert case["chain_id"] == "chain-20260101T000000"
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


# ---------------------------------------------------------------------------
# 6. Ledger accounting audit
# ---------------------------------------------------------------------------

class TestLedgerAudit:
    def _corpus_and_sample(self, tmp_path):
        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), assistant_tool_msg(), tool_msg()]
        write_capture(tmp_path, "req-20260727T122717.000000-0003.json", msgs_a)
        write_capture(tmp_path, "req-20260727T122718.000000-0004.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        # Empty qwen_entries: the skip rule missed the row (e.g. a
        # served_model spelling load_qwen_timestamps does not recognize),
        # so the pair lands in the sample — the audit's failure mode.
        usable = emr.process_pairs(chains, [])
        sample, _ = emr.sample_cases(usable, 5)
        return rows, set(c["capture_path"] for c in sample)

    def test_audit_trips_on_sampled_local_served_capture(
            self, tmp_path, capsys):
        rows, sampled_paths = self._corpus_and_sample(tmp_path)
        assert sampled_paths  # the leaked pair really was sampled

        ledger_rows = [{"ts": "2026-07-27T12:27:17.500000+00:00",
                        "latency_ms": 500,
                        "served_model": "Qwen/Qwen3-14B-AWQ"}]
        with pytest.raises(SystemExit) as exc_info:
            emr.audit_ledger(rows, ledger_rows, sampled_paths)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "MATCHED capture req-20260727T122717.000000-0003.json" \
            in captured.out
        assert "AUDIT FAILURE" in captured.out

    def test_audit_matches_nearest_capture_not_first_in_window(
            self, tmp_path, capsys):
        # Two captures 0.55s apart, both inside the ±2s window of the
        # reconstructed arrival (12:27:17.000). The earlier file (-0002) is a
        # decoy 0.45s off; the later (-0003) is 6ms off and its successor
        # pair IS sampled. First-in-window matching would name the decoy and
        # falsely pass; nearest-match must name -0003 and trip.
        msgs_decoy = [sys_msg(), user_msg("decoy")]
        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), assistant_tool_msg(), tool_msg()]
        write_capture(tmp_path, "req-20260727T122716.550000-0002.json",
                      msgs_decoy)
        write_capture(tmp_path, "req-20260727T122717.006000-0003.json",
                      msgs_a)
        write_capture(tmp_path, "req-20260727T122718.000000-0004.json",
                      msgs_b)
        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, [])
        sample, _ = emr.sample_cases(usable, 5)
        sampled_paths = set(c["capture_path"] for c in sample)
        assert sampled_paths

        ledger_rows = [{"ts": "2026-07-27T12:27:17.500000+00:00",
                        "latency_ms": 500,
                        "served_model": "Qwen/Qwen3-14B-AWQ"}]
        with pytest.raises(SystemExit) as exc_info:
            emr.audit_ledger(rows, ledger_rows, sampled_paths)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "MATCHED capture req-20260727T122717.006000-0003.json" \
            in captured.out

    def test_audit_passes_when_row_absent_from_corpus(
            self, tmp_path, capsys):
        rows, sampled_paths = self._corpus_and_sample(tmp_path)

        # Ledger ts months away from any capture — no match possible even
        # with the wide null-latency window.
        ledger_rows = [{"ts": "2026-01-01T00:00:00.000000+00:00",
                        "latency_ms": None,
                        "served_model": "Qwen/Qwen3-14B-AWQ"}]
        emr.audit_ledger(rows, ledger_rows, sampled_paths)  # must not exit
        captured = capsys.readouterr()
        assert ("ledger row 2026-01-01T00:00:00.000000+00:00 "
                "served_model=Qwen/Qwen3-14B-AWQ: "
                "no matching capture in corpus") in captured.out
        assert ("Qwen audit: 1 local-served main rows, 0 matched "
                "(all excluded), 1 absent") in captured.out

    def test_audit_matched_but_excluded_passes(self, tmp_path, capsys):
        # A matched capture whose successor pair was NOT sampled is fine.
        rows, _ = self._corpus_and_sample(tmp_path)

        ledger_rows = [{"ts": "2026-07-27T12:27:17.500000+00:00",
                        "latency_ms": 500,
                        "served_model": "Qwen/Qwen3-14B-AWQ"}]
        emr.audit_ledger(rows, ledger_rows, set())  # must not exit
        captured = capsys.readouterr()
        assert ("Qwen audit: 1 local-served main rows, 1 matched "
                "(all excluded), 0 absent") in captured.out

    def test_load_local_main_rows_filters_class_and_route(self, tmp_path):
        ledger = tmp_path / "adequacy.jsonl"
        entries = [
            {"class": "main", "route": "local",
             "ts": "2026-07-27T12:00:00+00:00", "latency_ms": 1200,
             "served_model": "Qwen/Qwen3-14B"},
            {"class": "main", "route": "remote",
             "ts": "2026-07-27T12:01:00+00:00", "latency_ms": 900,
             "served_model": "z-ai/glm-5.2"},
            {"class": "chore", "route": "local",
             "ts": "2026-07-27T12:02:00+00:00", "latency_ms": 300,
             "served_model": "Qwen/Qwen3-14B"},
            {"class": "main", "route": "local",
             "ts": "2026-07-27T12:03:00+00:00", "latency_ms": None,
             "served_model": None},
        ]
        ledger.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n",
            encoding="utf-8")

        rows = emr.load_local_main_rows(ledger)

        assert len(rows) == 2
        assert rows[0]["latency_ms"] == 1200
        assert rows[1]["latency_ms"] is None
        assert rows[1]["served_model"] == "(unknown)"


# ---------------------------------------------------------------------------
# 7. Corpus snapshot metadata
# ---------------------------------------------------------------------------

class TestCorpusMeta:
    def test_meta_file_written_with_snapshot_fields(self, tmp_path):
        msgs_a = [sys_msg(), user_msg()]
        msgs_b = [sys_msg(), user_msg(), assistant_tool_msg(), tool_msg()]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs_b)

        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        usable = emr.process_pairs(chains, [])
        sample, coverage = emr.sample_cases(usable, 5)

        output_path = tmp_path / "cases" / "main_replay.jsonl"
        meta_path = emr.write_meta(
            output_path, len(rows), rows[-1]["name"], sample, coverage)

        assert meta_path == tmp_path / "cases" / "main_replay.meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["corpus_files"] == 2
        assert meta["newest_capture"] == \
            "req-20260101T000001.000000-0001.json"
        assert meta["n_cases"] == 1
        assert meta["strata"] == {"shallow": 1, "mid": 0, "deep": 0}
        assert meta["chains_represented"] == ["chain-20260101T000000"]
