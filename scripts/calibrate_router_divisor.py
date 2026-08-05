#!/usr/bin/env python3
"""Calibrate the router's ESTIMATE_CHARS_PER_TOKEN divisor.

For every corpus case: chars = loxo_llm_router._count_prompt_chars(body),
real = count_ref_tokens(reference tokenizer, body). min(chars/real) over the
corpus is the tightest divisor that does not underestimate ON THIS CORPUS —
but it is an empirical minimum, not a bound, so pinning it leaves the router
sitting at the edge of the observed data. SAFETY pulls the divisor below that
minimum deliberately, which raises every estimate and buys headroom against
traffic denser than anything the corpus has seen.

The margin lives here rather than in the pinned constant on purpose: hand-
editing loxo's value would be silently undone the next time someone runs this
script and pastes the output.

Prints a per-case table, the raw and safety-adjusted divisors, and worst-case
under/over margins at the value it recommends. Paste that value into
loxo_llm_router/__init__.py, and repin it in loxo's test_routing.py
(test_estimate_divisor_is_pinned) — that pin is what makes a divisor edit
fail in the repo that runs CI.

Run with --emit-ref-table whenever you repin. It writes the per-case
(chars, ref_tokens) integers to cases/main_replay.divisor_ref.json; committing
that file is what lets tests/test_router_divisor_property.py verify the pinned
divisor on machines without the corpus, the captures, or transformers — which
is every machine but this one.
"""
import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from litmus_spec import count_ref_tokens, load_cases  # noqa: E402
from loxo_llm_router import _count_prompt_chars       # noqa: E402

# Fraction of the observed minimum ratio to actually recommend. 0.95 gives
# ~5% headroom, chosen against the shape of the observed spread rather than
# picked round: the corpus ratios span roughly the raw minimum to +22% above
# it, so 5% sits well inside the lower tail without pushing meaningful traffic
# to cloud unnecessarily. Raise it toward 1.0 only once the corpus is large
# enough that its minimum approximates a real bound; lower it if a case ever
# lands near the edge.
SAFETY = 0.95


# Committed alongside the case file: the two integers per case that the
# calibration actually turns on. See _write_ref_table for why this exists.
REF_TABLE = "cases/main_replay.divisor_ref.json"


def _write_ref_table(path: str, tokenizer: str, cases_path: str,
                     rows: list, divisor: float) -> None:
    """Record (case_id, chars, ref_tokens) so the guard can run off-corpus.

    The corpus itself cannot be committed -- it points at capture files
    containing real conversation content, and computing ref_tokens needs
    transformers plus a locally cached tokenizer. The consequence, before this
    existed, was that tests/test_router_divisor_property.py skipped on every
    machine but the one that generated the corpus, and a skip reports green:
    the divisor's only guard was silently absent wherever it was most likely
    to be edited.

    These two integers per case are the entire input to the calibration --
    everything downstream is arithmetic on them. Committing them makes the
    pinned divisor checkable anywhere, with no content leaving the machine
    (ids, counts, and hashes only, same discipline as extract_main_replay).

    A snapshot cannot notice the corpus changing -- regenerate it whenever you
    regenerate the corpus. It catches the failure that actually happens: the
    divisor edited without recalibration.
    """
    payload = {
        "tokenizer": tokenizer,
        "cases_path": cases_path,
        "safety": SAFETY,
        "recommended_divisor": divisor,
        "n_cases": len(rows),
        "cases": [{"id": cid, "chars": chars, "ref_tokens": real}
                  for cid, chars, real, _ in sorted(rows, key=lambda r: r[0])],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\nreference table written: {path} ({len(rows)} cases) — commit it, "
          f"it is what lets the property test run off this machine")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="cases/main_replay.jsonl")
    ap.add_argument("--tokenizer",
                    default="mlx-community/Qwen3-14B-4bit")
    ap.add_argument("--emit-ref-table", nargs="?", const=REF_TABLE, default=None,
                    metavar="PATH",
                    help=f"write the per-case (chars, ref_tokens) table to PATH "
                         f"(default {REF_TABLE}) so the property test can verify "
                         f"the pinned divisor without the corpus or a tokenizer")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                            local_files_only=True)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"reference tokenizer {args.tokenizer!r} not cached "
                 f"({type(e).__name__}: {e}); run: hf download {args.tokenizer}")

    rows = []
    for case in load_cases(args.cases, "main-replay"):
        with open(case.capture_path, encoding="utf-8") as f:
            body = json.load(f)
        chars = _count_prompt_chars(body)
        real = count_ref_tokens(tok, body)
        rows.append((case.id, chars, real, chars / real))

    if not rows:
        sys.exit(f"no cases loaded from {args.cases!r} -- the corpus is empty "
                 f"or all lines are blank. Regenerate it with "
                 f"scripts/extract_main_replay.py before calibrating.")

    rows.sort(key=lambda r: r[3])
    print(f"{'case':8} {'chars':>10} {'ref_tokens':>10} {'chars/tok':>10}")
    for cid, chars, real, ratio in rows:
        print(f"{cid:8} {chars:>10} {real:>10} {ratio:>10.3f}")

    raw = math.floor(rows[0][3] * 100) / 100
    divisor = math.floor(rows[0][3] * SAFETY * 100) / 100
    print(f"\nraw min(chars/tok), floored:      {raw}   "
          f"(edge of observed data — do not pin)")
    print(f"recommended ESTIMATE_CHARS_PER_TOKEN = {divisor}   "
          f"(raw x SAFETY={SAFETY})")
    for label, d in (("raw", raw), ("recommended", divisor)):
        margins = [(int(chars / d) - real) / real
                   for _, chars, real, _ in rows]
        print(f"  margins at {label:12} {d}: worst under {min(margins):+.1%}, "
              f"worst over {max(margins):+.1%}")
    margins = [(int(chars / divisor) - real) / real
               for _, chars, real, _ in rows]
    if min(margins) < 0:
        sys.exit("floor produced an undercount — investigate before pinning")

    if args.emit_ref_table:
        _write_ref_table(args.emit_ref_table, args.tokenizer, args.cases,
                         rows, divisor)


if __name__ == "__main__":
    main()
