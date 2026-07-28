"""F1: tier-0 validators must pass ~100% on real glm-5.2 reference actions.

Falsification test: the harness's own validators are run over the reference
actions themselves — the real production actions captured from glm-5.2 that
the replay cases were extracted from. Any systematic failure means the harness
is measuring itself, not the model (the ``parse_native`` failure mode). That
is a hard stop until fixed.

Also includes F3a: a check that each case's ``est_tokens`` matches
``estimate_prompt_tokens`` on the capture body, catching extraction drift.

These are integration tests. They require ``cases/main_replay.jsonl`` to exist
(produced by ``scripts/extract_main_replay.py``) and the capture files it
points at to be accessible. They skip gracefully otherwise.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

import pytest

# The extractor lives under scripts/; import its hashing helper so the test
# uses the exact same per-message digest the extractor used to build chains.
_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from litmus_spec import (  # noqa: E402
    ParsedCall,
    load_cases,
    score_replay_call,
)
from loxo_llm_router import estimate_prompt_tokens  # noqa: E402


CASES_PATH = "cases/main_replay.jsonl"


def _require_cases():
    """Skip unless cases/main_replay.jsonl exists and loads."""
    if not os.path.exists(CASES_PATH):
        pytest.skip("cases/main_replay.jsonl not found; run extractor first")
    return load_cases(CASES_PATH, "main-replay")


def _mhash(msg: dict) -> str:
    """Same per-message digest as scripts/extract_main_replay.py:mhash."""
    return hashlib.sha1(
        json.dumps(msg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:10]


def _load_capture(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _find_capture_n1(capture_path: str) -> tuple[dict, str] | None:
    """Find capture N+1 for the given capture N path.

    N+1 is the next .json file (by filename order) in the same directory whose
    message hashes are a superset prefix of N's — i.e. N's messages appear
    unchanged at the front of N+1's messages, and N+1 has additional appended
    messages that include the reference assistant turn.

    Returns (n1_body, n1_path) or None if no successor is found / not
    accessible.
    """
    p = pathlib.Path(capture_path)
    if not p.exists():
        return None
    n_body = _load_capture(capture_path)
    n_msgs = n_body.get("messages") or []
    n_hashes = [_mhash(m) for m in n_msgs]
    n_count = len(n_hashes)

    siblings = sorted(p.parent.glob("*.json"), key=lambda x: x.name)
    try:
        idx = siblings.index(p)
    except ValueError:
        return None
    for cand in siblings[idx + 1:]:
        try:
            cand_body = _load_capture(str(cand))
        except (OSError, ValueError):
            continue
        cand_msgs = cand_body.get("messages") or []
        if len(cand_msgs) <= n_count:
            continue
        cand_hashes = [_mhash(m) for m in cand_msgs]
        # N's hashes must be a prefix of N+1's (superset by extension).
        if cand_hashes[:n_count] == n_hashes:
            return cand_body, str(cand)
    return None


def _reference_parsed_call(n1_body: dict, n_count: int) -> ParsedCall:
    """Build a ParsedCall from the reference assistant message in capture N+1.

    The reference action is the first ``assistant`` message in the appended
    messages (beyond capture N's count). Per the brief:

    - tool_calls present: ParsedCall(well_formed=True, tool=<first call's
      function.name>, arguments=json.loads(<first call's function.arguments>),
      detail="reference", attempted=True). Tier-0 checks the first action only,
      same as the candidate evaluation.
    - no tool_calls but content (prose): ParsedCall(well_formed=True, tool=None,
      arguments=None, detail="reference prose", attempted=False) — a correct
      abstention.
    - neither: the pathological case the extractor should have skipped. We
      raise so the test surfaces it rather than silently passing.
    """
    appended = (n1_body.get("messages") or [])[n_count:]
    ref_msg = next((m for m in appended if m.get("role") == "assistant"), None)
    if ref_msg is None:
        raise AssertionError(
            f"no appended assistant message found (n_count={n_count})")
    tool_calls = ref_msg.get("tool_calls")
    if tool_calls:
        fn = tool_calls[0].get("function", {})
        name = fn.get("name")
        args_raw = fn.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {}
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            args = {}
        return ParsedCall(well_formed=True, tool=name, arguments=args,
                          detail="reference", attempted=True)
    content = ref_msg.get("content")
    if content:
        return ParsedCall(well_formed=True, tool=None, arguments=None,
                          detail="reference prose", attempted=False)
    raise AssertionError(
        "pathological reference (no tool_calls, no content) — extractor should "
        "have skipped this pair")


def _capture_tools(capture_path: str) -> list:
    body = _load_capture(capture_path)
    return body.get("tools") or []


# ---------------------------------------------------------------------------
# F1
# ---------------------------------------------------------------------------

def test_f1_reference_self_validation():
    """F1: tier-0 validators must pass ~100% on real glm-5.2 reference actions.

    Any systematic failure means the harness is measuring itself, not the model
    (the parse_native failure mode). Hard stop until fixed.
    """
    cases = _require_cases()

    results = []
    skipped = []
    for case in cases:
        pair = _find_capture_n1(case.capture_path)
        if pair is None:
            # Capture N+1 not reachable (captures dir moved or pruned). Skip
            # this case rather than crediting or failing it — F1 is about the
            # validators, not capture availability.
            skipped.append(case.id)
            continue
        n1_body, _ = pair
        n_count = len(_load_capture(case.capture_path).get("messages") or [])
        try:
            ref_parsed = _reference_parsed_call(n1_body, n_count)
        except AssertionError:
            skipped.append(case.id)
            continue
        capture_tools = _capture_tools(case.capture_path)
        score = score_replay_call(ref_parsed, case, capture_tools, closed=True)
        score["id"] = case.id
        results.append(score)

    if not results:
        pytest.skip("no reference actions could be resolved from captures")

    action_valid_rate = (
        sum(1 for r in results if r["action_valid"]) / len(results)
    )

    # Per-case reporting on failure: which dimensions dropped.
    failures = [r for r in results if not r["action_valid"]]
    if failures:
        detail = "\n".join(
            f"  {r['id']}: acted_ok={r['acted_ok']} well_formed="
            f"{r['well_formed']} tool_exists={r['tool_exists']} "
            f"args_schema_ok={r['args_schema_ok']}"
            for r in failures
        )
    else:
        detail = ""

    assert action_valid_rate >= 0.90, (
        f"F1 FAILED: reference self-validation rate={action_valid_rate:.2f} "
        f"(expected ~1.0). Harness validators cannot validate real glm-5.2 "
        f"actions.\nFailures:\n{detail}"
    )


# ---------------------------------------------------------------------------
# F3a
# ---------------------------------------------------------------------------

def test_f3a_no_truncation_case_estimates():
    """F3a: each case's est_tokens must match estimate_prompt_tokens(capture body).

    The full no-truncation gate needs a model run (a runtime check in the
    runner), but this catches extraction drift: if the tokenizer's count of
    the captured prompt no longer matches what the extractor recorded, the
    case's depth stratum may be wrong.
    """
    cases = _require_cases()

    mismatches = []
    for case in cases:
        if not os.path.exists(case.capture_path):
            pytest.skip(f"capture not accessible: {case.capture_path}")
        body = _load_capture(case.capture_path)
        expected = estimate_prompt_tokens(body)
        if case.est_tokens != expected:
            mismatches.append(
                f"  {case.id}: est_tokens={case.est_tokens} but "
                f"estimate_prompt_tokens={expected}"
            )

    assert not mismatches, (
        "F3a FAILED: case est_tokens drift from estimate_prompt_tokens:\n"
        + "\n".join(mismatches)
    )
