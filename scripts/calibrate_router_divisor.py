#!/usr/bin/env python3
"""Calibrate the router's ESTIMATE_CHARS_PER_TOKEN divisor.

For every corpus case: chars = loxo_llm_router._count_prompt_chars(body),
real = count_ref_tokens(reference tokenizer, body). The safe divisor is
min(chars/real) over the corpus, floored to 2 decimals — then
chars/divisor >= real everywhere (never underestimates). Prints a per-case
table, the recommended divisor, and worst-case under/over margins at that
divisor. Paste the value into loxo_llm_router/__init__.py.
"""
import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from litmus_spec import count_ref_tokens, load_cases  # noqa: E402
from loxo_llm_router import _count_prompt_chars       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="cases/main_replay.jsonl")
    ap.add_argument("--tokenizer",
                    default="mlx-community/Qwen3-14B-4bit")
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

    rows.sort(key=lambda r: r[3])
    print(f"{'case':8} {'chars':>10} {'ref_tokens':>10} {'chars/tok':>10}")
    for cid, chars, real, ratio in rows:
        print(f"{cid:8} {chars:>10} {real:>10} {ratio:>10.3f}")

    divisor = math.floor(rows[0][3] * 100) / 100
    print(f"\nrecommended ESTIMATE_CHARS_PER_TOKEN = {divisor}")
    margins = [(int(chars / divisor) - real) / real
               for _, chars, real, _ in rows]
    print(f"margins at that divisor: worst under {min(margins):+.1%}, "
          f"worst over {max(margins):+.1%}")
    if min(margins) < 0:
        sys.exit("floor produced an undercount — investigate before pinning")


if __name__ == "__main__":
    main()
