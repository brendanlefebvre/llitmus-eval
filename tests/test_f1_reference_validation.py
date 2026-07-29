"""F1: tier-0 validators must pass ~100% on real glm-5.2 reference actions.

Falsification test: the harness's own validators are run over the reference
actions themselves — the real production actions captured from glm-5.2 that
the replay cases were extracted from. Any systematic failure means the harness
is measuring itself, not the model (the ``parse_native`` failure mode). That
is a hard stop until fixed.

Also includes a ref_tokens drift check: each case's ``ref_tokens`` must match
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
    ReplayCase,
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


def _reference_parsed_calls(n1_body: dict, n_count: int) -> list[ParsedCall]:
    """Build one ParsedCall per action in the reference assistant message.

    The reference turn is the first ``assistant`` message in the appended
    messages (beyond capture N's count). EVERY call in a parallel-call turn is
    returned — the caller scores each as its own row, so a reference turn
    passes a dimension only if all its calls pass.

    - tool_calls present: one ParsedCall(well_formed=True, tool=<the call's
      function.name>, arguments=<parsed function.arguments>,
      detail="reference", attempted=True) per call, in order.
    - no tool_calls but content (prose): a single ParsedCall(well_formed=True,
      tool=None, arguments=None, detail="reference prose", attempted=False) —
      a correct abstention.
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
        calls = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name")
            args_raw = fn.get("arguments", "{}")
            if isinstance(args_raw, str):
                args = json.loads(args_raw)
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}
            calls.append(ParsedCall(well_formed=True, tool=name,
                                    arguments=args, detail="reference",
                                    attempted=True))
        return calls
    content = ref_msg.get("content")
    if content:
        return [ParsedCall(well_formed=True, tool=None, arguments=None,
                           detail="reference prose", attempted=False)]
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

    Gated per dimension (acted_ok, well_formed, tool_exists, args_schema_ok),
    each at ≈1.0 over its applicable-row denominator — every call in a
    parallel-call reference turn is its own row (ids suffixed "#2", "#3", …).
    Uncheckable dimensions (None) are excluded from denominators. The
    diagnostic value of F1 is *which* dimension fails, so a pooled rate is
    never used.
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
            ref_calls = _reference_parsed_calls(n1_body, n_count)
        except AssertionError:
            skipped.append(case.id)
            continue
        capture_tools = _capture_tools(case.capture_path)
        # One row per reference call: a parallel-call turn passes a dimension
        # only if every call passes. Rows past the first get an id suffix.
        for i, ref_parsed in enumerate(ref_calls):
            score = score_replay_call(ref_parsed, case, capture_tools,
                                      closed=True)
            score["id"] = case.id if i == 0 else f"{case.id}#{i + 1}"
            results.append(score)

    if not results:
        pytest.skip("no reference actions could be resolved from captures")

    # Per-dimension rates over applicable cases (None excluded).
    dims = ["acted_ok", "well_formed", "tool_exists", "args_schema_ok"]
    dim_stats = {}
    for d in dims:
        vals = [r[d] for r in results]
        present = [v for v in vals if v is not None]
        rate = (sum(1 for v in present if v) / len(present)) if present else None
        dim_stats[d] = {"rate": rate, "n_applicable": len(present)}

    # Always print all four rates + denominators.
    print("\nF1 per-dimension reference self-validation:")
    for d in dims:
        s = dim_stats[d]
        rate_str = f"{s['rate']:.3f}" if s['rate'] is not None else "N/A"
        print(f"  {d}: rate={rate_str} n_applicable={s['n_applicable']}")
    if skipped:
        print(f"  (skipped {len(skipped)} case(s): {', '.join(skipped)})")

    # Gate each dimension separately at ≈1.0.
    failures = []
    for d in dims:
        s = dim_stats[d]
        if s["rate"] is not None and s["rate"] < 0.99:
            # Collect per-case detail for the failing dimension.
            failing_ids = [r["id"] for r in results if r[d] is False]
            failures.append(
                f"{d}: rate={s['rate']:.3f} n_applicable={s['n_applicable']} "
                f"(failing cases: {', '.join(failing_ids)})"
            )

    assert not failures, (
        "F1 FAILED: per-dimension reference self-validation below ~1.0.\n"
        "Harness validators cannot validate real glm-5.2 actions.\n"
        + "\n".join(f"  {f}" for f in failures)
    )


def test_reference_parsed_calls_validates_all_parallel_calls():
    """Every call in a parallel-call reference turn is built and scorable.

    Unit test (no captures needed): a two-call reference where the second
    call names a tool absent from the capture's tools — the second row must
    fail tool_exists, so the turn cannot pass on the strength of its first
    call alone.
    """
    n1_body = {"messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"function": {"name": "read", "arguments": '{"filePath": "x"}'}},
            {"function": {"name": "bogus_tool", "arguments": "{}"}},
        ]},
    ]}
    calls = _reference_parsed_calls(n1_body, n_count=1)
    assert len(calls) == 2
    assert [c.tool for c in calls] == ["read", "bogus_tool"]

    case = ReplayCase(
        id="t-parallel", capture_path="unused", chain_id="c-1",
        depth_stratum="mid", ref_tokens=1,
        reference={"acted": True, "tools": ["read"], "arguments": []})
    capture_tools = [{"type": "function", "function": {
        "name": "read",
        "parameters": {"type": "object",
                       "properties": {"filePath": {"type": "string"}},
                       "required": ["filePath"]},
    }}]
    scores = [score_replay_call(c, case, capture_tools, closed=True)
              for c in calls]
    assert scores[0]["tool_exists"] is True
    assert scores[0]["action_valid"] is True
    assert scores[1]["tool_exists"] is False
    assert scores[1]["action_valid"] is False


# ---------------------------------------------------------------------------
# ref_tokens drift check (formerly mislabelled "F3a")
# ---------------------------------------------------------------------------

def test_ref_tokens_drift_check():
    """ref_tokens drift check: each case's ref_tokens must match
    estimate_prompt_tokens(capture body).

    This is NOT the F3a truncation gate — what exists of that lives in
    run_main_replay: per-case ``prompt_tokens_fed`` recording (the tokenizer's
    count of the *rendered* prompt), a ``tokens_fed_error`` note when the
    tokenizer fails, and an overflow gate that errors any case whose rendered
    prompt exceeds a sane tokenizer ``model_max_length``. Comparison against
    what the backend *actually consumed* remains future work. This test
    instead catches extraction drift: if the estimator's count of the captured
    prompt no longer matches what the extractor recorded, the case's depth
    stratum may be wrong. Estimator-vs-itself can't detect truncation, but it
    can detect a stale case file.

    Skip is per-case (collected into a list) rather than via pytest.skip,
    which aborts the whole loop on the first missing capture.
    """
    cases = _require_cases()

    mismatches = []
    skipped = []
    for case in cases:
        if not os.path.exists(case.capture_path):
            skipped.append(case.id)
            continue
        body = _load_capture(case.capture_path)
        expected = estimate_prompt_tokens(body)
        if case.ref_tokens != expected:
            mismatches.append(
                f"  {case.id}: ref_tokens={case.ref_tokens} but "
                f"estimate_prompt_tokens={expected}"
            )

    if skipped:
        print(f"  (skipped {len(skipped)} case(s) with missing captures: "
              f"{', '.join(skipped)})")

    assert not mismatches, (
        "ref_tokens drift: case ref_tokens no longer match "
        "estimate_prompt_tokens(capture body):\n"
        + "\n".join(mismatches)
    )
