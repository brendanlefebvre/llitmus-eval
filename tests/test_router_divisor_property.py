"""Property: the router's estimate_prompt_tokens never underestimates the
reference-tokenizer count on the calibration corpus. Pins the divisor to the
corpus it was calibrated on; new captures may require recalibration
(scripts/calibrate_router_divisor.py).

Two paths verify the same property from different evidence:

* **live** — recompute ref_tokens from the corpus with the real tokenizer.
  Authoritative, but needs the capture files, transformers, and a cached
  tokenizer, so it only runs on the machine that built the corpus.
* **recorded** — replay the committed (chars, ref_tokens) table written by
  ``calibrate_router_divisor.py --emit-ref-table``. Pure arithmetic, runs
  anywhere. Cannot notice the corpus changing, but does catch the divisor
  being edited without recalibration — the failure that actually happens.

``test_divisor_guard_actually_ran`` fails if NEITHER path executed. That is
the point of this file: before it existed, all three preconditions for the
live path were absent on every machine except one, each absence produced a
skip, and a skipped guard reports green. Set LITMUS_ALLOW_UNGUARDED_DIVISOR=1
to downgrade that to a skip when you knowingly have neither.
"""
import json
import os
import pathlib

import pytest

from litmus_spec import count_ref_tokens, load_cases
from loxo_llm_router import ESTIMATE_CHARS_PER_TOKEN, estimate_prompt_tokens

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
# Absolute, not cwd-relative: a relative path silently skipped the whole guard
# whenever pytest ran from anywhere but the repo root.
CASES = REPO_ROOT / "cases" / "main_replay.jsonl"
REF_TABLE = REPO_ROOT / "cases" / "main_replay.divisor_ref.json"

# Recorded for diagnostics only -- test_divisor_guard_actually_ran does NOT
# read this, precisely so it survives xdist, random ordering, and -k selection.
_ran: set = set()


def _evidence_available() -> bool:
    """Could either evidence path run in THIS environment?

    Mirrors the skip conditions of the two checks above. Deliberately
    independent of whether they actually executed in this process.
    """
    if REF_TABLE.exists():
        return True
    if not CASES.exists():
        return False
    try:                      # the live path additionally needs transformers
        import transformers   # noqa: F401
    except Exception:         # noqa: BLE001
        return False
    return True


def _assert_no_undercounts(pairs, source: str) -> None:
    """pairs: iterable of (case_id, estimate, reference_tokens)."""
    under = [f"  {cid}: est={est} < ref={ref}" for cid, est, ref in pairs
             if est < ref]
    assert not under, (
        f"estimator underestimates on the {source} evidence — an undercount "
        f"routes an over-long prompt to a local model. Recalibrate: "
        f"scripts/calibrate_router_divisor.py (then repin "
        f"ESTIMATE_CHARS_PER_TOKEN in loxo and refresh {REF_TABLE.name})\n"
        + "\n".join(under))


# --- recorded path: runs anywhere ------------------------------------------

def test_pinned_divisor_holds_on_recorded_table():
    if not REF_TABLE.exists():
        pytest.skip(
            f"{REF_TABLE.name} missing — regenerate on the machine holding "
            f"the corpus: python scripts/calibrate_router_divisor.py "
            f"--emit-ref-table")
    table = json.loads(REF_TABLE.read_text(encoding="utf-8"))
    cases = table.get("cases") or []
    assert cases, f"{REF_TABLE.name} records no cases — regenerate it"
    assert len(cases) == table.get("n_cases"), (
        f"{REF_TABLE.name} is truncated: n_cases={table.get('n_cases')} but "
        f"{len(cases)} rows present")

    # Recomputed, not read back: the table stores the estimator's *input*
    # (chars), so the pinned divisor is re-applied here. Storing the estimate
    # itself would make this a tautology.
    pairs = [(c["id"], int(c["chars"] / ESTIMATE_CHARS_PER_TOKEN),
              c["ref_tokens"]) for c in cases]
    _assert_no_undercounts(pairs, "recorded")
    _ran.add("recorded")


def test_recorded_table_matches_the_pinned_tokenizer():
    """A table built against a different tokenizer cannot vouch for this
    divisor — cross-family segmentation variance dwarfs the corpus variance
    the divisor was fitted to."""
    if not REF_TABLE.exists():
        pytest.skip(f"{REF_TABLE.name} missing")
    from loxo_llm_router import ESTIMATE_DIVISOR_REF_TOKENIZER
    recorded = json.loads(REF_TABLE.read_text(encoding="utf-8")).get("tokenizer")
    assert recorded == ESTIMATE_DIVISOR_REF_TOKENIZER, (
        f"{REF_TABLE.name} was built against {recorded!r} but loxo pins "
        f"{ESTIMATE_DIVISOR_REF_TOKENIZER!r} — recalibrate before trusting "
        f"either")


# --- live path: authoritative, corpus-machine only -------------------------

@pytest.fixture(scope="module")
def ref_tokenizer():
    meta_path = REPO_ROOT / "cases" / "main_replay.meta.json"
    repo = "mlx-community/Qwen3-14B-4bit"
    if meta_path.exists():
        repo = json.loads(meta_path.read_text()).get("tokenizer", repo)
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            repo, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"reference tokenizer {repo} not in local HF cache "
                    f"({type(e).__name__})")


def test_estimator_never_underestimates_on_corpus(ref_tokenizer):
    if not CASES.exists():
        pytest.skip(f"{CASES} missing")
    # load_cases raises CaseError on a missing capture_path, so a partially
    # materialized corpus errors here rather than quietly checking a subset.
    cases = load_cases(str(CASES), "main-replay")
    assert cases, f"{CASES} loaded zero cases"

    pairs = []
    for case in cases:
        with open(case.capture_path, encoding="utf-8") as f:
            body = json.load(f)
        pairs.append((case.id, estimate_prompt_tokens(body),
                      count_ref_tokens(ref_tokenizer, body)))
    assert len(pairs) == len(cases)
    _assert_no_undercounts(pairs, "live corpus")
    _ran.add("live")


# --- the anti-silent-skip check --------------------------------------------

def test_divisor_guard_actually_ran():
    """A guard that skips is not a guard: a run where no evidence path could
    execute proved nothing about ESTIMATE_CHARS_PER_TOKEN and must not read
    as green.

    Preconditions are RECOMPUTED here rather than read from what earlier tests
    recorded. Deriving this from shared module state coupled it to in-process,
    file-order execution, so three ordinary setups produced a false failure
    with a misleading message: pytest-xdist puts the writers in other workers,
    a random-order plugin can run this first, and -k / a node id skips them
    entirely. The question "was any evidence available" is answerable on its
    own, so it should be asked directly.
    """
    if _evidence_available():
        return
    msg = (
        "the divisor guard did not execute — neither the recorded table nor "
        f"the live corpus was available, so nothing verified "
        f"ESTIMATE_CHARS_PER_TOKEN={ESTIMATE_CHARS_PER_TOKEN}.\n"
        f"Fix: on the machine holding the corpus, run\n"
        f"  python scripts/calibrate_router_divisor.py --emit-ref-table\n"
        f"and commit {REF_TABLE.name} (integers only, no captured content).\n"
        f"Override with LITMUS_ALLOW_UNGUARDED_DIVISOR=1 if you accept an "
        f"unguarded run.")
    if os.environ.get("LITMUS_ALLOW_UNGUARDED_DIVISOR") == "1":
        pytest.skip(msg)
    pytest.fail(msg)
