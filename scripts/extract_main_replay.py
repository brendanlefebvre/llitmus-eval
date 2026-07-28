#!/usr/bin/env python3
"""Extract main-replay eval cases from Loxo capture files.

Walks the captures directory, groups files into prefix chains by per-message
hashing, applies skip rules, assigns depth strata, samples 15 cases (5 per
stratum from at least two chains), and emits JSONL to cases/main_replay.jsonl.

Captures are successive snapshots of growing conversations — consecutive
captures in a prefix chain represent (state, known-good-action) pairs. The
appended assistant message in capture N+1 is the reference action.

No message content is printed to stdout/stderr — only counts, roles, hashes,
filenames, and skip reasons. Safe to paste.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
from datetime import datetime

from loxo_llm_router import (
    classify,
    estimate_prompt_tokens,
    LOCAL_CONTEXT_LIMIT,
)

CURL_PROBES = frozenset({
    "req-20260727T121220.315928-0000.json",
    "req-20260727T121227.015255-0001.json",
})

STRATA = ("shallow", "mid", "deep")

ADEQUACY_LEDGER = pathlib.Path(
    os.path.expanduser("~/.local/state/loxo-llm-router/adequacy.jsonl")
)

FILENAME_TS_RE = re.compile(r"req-(\d{8}T\d{6}\.\d+)-\d+\.json")


# ---------------------------------------------------------------------------
# Message hashing and chain grouping
# ---------------------------------------------------------------------------

def mhash(msg: dict) -> str:
    return hashlib.sha1(
        json.dumps(msg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:10]


def load_captures(capture_dir: pathlib.Path) -> list[dict]:
    rows = []
    for f in sorted(capture_dir.glob("*.json")):
        try:
            body = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"SKIP {f.name}: {e}", file=sys.stderr)
            continue
        msgs = body.get("messages") or []
        rows.append({
            "name": f.name,
            "path": f,
            "body": body,
            "hashes": [mhash(m) for m in msgs],
            "roles": [m.get("role") for m in msgs],
            "n": len(msgs),
        })
    return rows


def group_chains(rows: list[dict]) -> list[list[dict]]:
    chains: list[list[dict]] = []
    cur: list[dict] = []
    prev: dict | None = None
    for r in rows:
        if prev is None:
            cur = [r]
        else:
            common = min(prev["n"], r["n"])
            div = next(
                (i for i in range(common)
                 if prev["hashes"][i] != r["hashes"][i]),
                None,
            )
            if div is not None:
                chains.append(cur)
                cur = [r]
            elif r["n"] < prev["n"]:
                chains.append(cur)
                cur = [r]
            else:
                cur.append(r)
        prev = r
    if cur:
        chains.append(cur)
    return chains


# ---------------------------------------------------------------------------
# Timestamp correlation for Qwen3-14B skip
# ---------------------------------------------------------------------------

def parse_capture_ts(name: str) -> datetime | None:
    m = FILENAME_TS_RE.match(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S.%f")
    except ValueError:
        return None


def load_qwen_timestamps(ledger_path: pathlib.Path) -> list[str]:
    if not ledger_path.exists():
        return []
    out = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("class") != "main":
            continue
        if "Qwen3-14B" not in (entry.get("served_model") or ""):
            continue
        ts = entry.get("ts")
        if ts:
            out.append(ts)
    return out


def _ts_within_2s(capture_ts: datetime, ledger_ts_str: str) -> bool:
    try:
        ledger_ts = datetime.fromisoformat(ledger_ts_str)
        if ledger_ts.tzinfo is not None:
            ledger_ts = ledger_ts.replace(tzinfo=None)
        return abs((capture_ts - ledger_ts).total_seconds()) <= 2
    except (ValueError, TypeError):
        return False


def is_qwen_authored(capture_name: str,
                     qwen_timestamps: list[str]) -> str | None:
    capture_ts = parse_capture_ts(capture_name)
    if capture_ts is None:
        return None
    for ts in qwen_timestamps:
        if _ts_within_2s(capture_ts, ts):
            return ts
    return None


# ---------------------------------------------------------------------------
# Stratum assignment
# ---------------------------------------------------------------------------

def assign_stratum(tokens: int, limit: int = LOCAL_CONTEXT_LIMIT) -> str | None:
    if tokens < 16000:
        return "shallow"
    if tokens < 40000:
        return "mid"
    if tokens <= limit:
        return "deep"
    return None


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------

def build_reference(assistant_msg: dict) -> dict | None:
    tool_calls = assistant_msg.get("tool_calls")
    if tool_calls:
        tools = []
        arguments = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            tools.append(fn.get("name"))
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
            arguments.append(args)
        return {"acted": True, "tools": tools, "arguments": arguments}
    content = assistant_msg.get("content")
    if content:
        return {"acted": False, "tools": [], "arguments": []}
    return None


# ---------------------------------------------------------------------------
# Pair extraction and skip rules
# ---------------------------------------------------------------------------

def process_pairs(chains: list[list[dict]],
                  qwen_timestamps: list[str]) -> list[dict]:
    usable = []
    for chain_idx, chain in enumerate(chains, 1):
        chain_id = f"chain-{chain_idx:02d}"
        if len(chain) < 2:
            continue
        for i in range(len(chain) - 1):
            cap_n = chain[i]
            cap_n1 = chain[i + 1]
            pair_label = f"{cap_n['name']} -> {cap_n1['name']}"

            # Skip rule 5 (checked first so the explicit message fires
            # before the class filter catches them): curl probes.
            if cap_n["name"] in CURL_PROBES:
                print(f"SKIP {cap_n['name']}: curl probe (named explicitly)",
                      file=sys.stderr)
                continue

            # Skip rule 1: non-main captures
            classification = classify(cap_n["body"])
            if classification.cls != "main":
                print(f"SKIP {cap_n['name']}: class={classification.cls} "
                      f"(not main)", file=sys.stderr)
                continue

            # Skip rule 2: non-assistant-start pairs
            appended_roles = cap_n1["roles"][cap_n["n"]:]
            if not appended_roles:
                print(f"SKIP {pair_label}: no appended messages",
                      file=sys.stderr)
                continue
            if appended_roles[0] != "assistant":
                print(f"SKIP {pair_label}: appended roles start with "
                      f"{appended_roles[0]}, not assistant", file=sys.stderr)
                continue

            ref_msg = cap_n1["body"]["messages"][cap_n["n"]]

            # Skip rule 3: pathological reference turns
            ref = build_reference(ref_msg)
            if ref is None:
                print(f"SKIP {pair_label}: pathological reference "
                      f"(no tool_calls, no content)", file=sys.stderr)
                continue

            # Skip rule 4: Qwen3-14B-authored turns
            qwen_ts = is_qwen_authored(cap_n["name"], qwen_timestamps)
            if qwen_ts is not None:
                print(f"SKIP {pair_label}: reference authored by Qwen3-14B "
                      f"(ledger ts={qwen_ts})", file=sys.stderr)
                continue

            # Depth stratum (over-limit is excluded)
            tokens = estimate_prompt_tokens(cap_n["body"])
            stratum = assign_stratum(tokens)
            if stratum is None:
                print(f"SKIP {pair_label}: over limit ({tokens} > "
                      f"{LOCAL_CONTEXT_LIMIT})", file=sys.stderr)
                continue

            usable.append({
                "capture_path": str(cap_n["path"].resolve()),
                "chain_id": chain_id,
                "depth_stratum": stratum,
                "est_tokens": tokens,
                "reference": ref,
            })
    return usable


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_evenly(items: list, n: int) -> list:
    if n <= 0:
        return []
    if n >= len(items):
        return list(items)
    stride = len(items) / n
    return [items[int(i * stride)] for i in range(n)]


def sample_cases(usable: list[dict],
                 n_per_stratum: int = 5) -> tuple[list[dict], dict]:
    by_stratum: dict[str, list[dict]] = {s: [] for s in STRATA}
    for pair in usable:
        by_stratum[pair["depth_stratum"]].append(pair)

    sample: list[dict] = []
    coverage: dict[str, dict] = {}

    for stratum in STRATA:
        pairs = by_stratum[stratum]
        if not pairs:
            coverage[stratum] = {"chains": [], "n": 0, "population": 0}
            continue

        by_chain: dict[str, list[dict]] = {}
        for p in pairs:
            by_chain.setdefault(p["chain_id"], []).append(p)

        chain_ids = sorted(by_chain.keys())
        n = min(n_per_stratum, len(pairs))

        if len(chain_ids) == 1:
            selected = sample_evenly(pairs, n)
        else:
            sorted_chains = sorted(
                chain_ids,
                key=lambda c: (-len(by_chain[c]), c),
            )
            selected: list[dict] = []
            remaining = n
            for cid in sorted_chains[1:]:
                if remaining <= 1:
                    break
                selected.extend(sample_evenly(by_chain[cid], 1))
                remaining -= 1
            dominant = sorted_chains[0]
            selected.extend(sample_evenly(by_chain[dominant], remaining))

        sample.extend(selected)
        coverage[stratum] = {
            "chains": sorted(set(p["chain_id"] for p in selected)),
            "n": len(selected),
            "population": len(pairs),
        }

    return sample, coverage


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--capture-dir", default=None,
                    help="captures directory (default: $LOXO_CAPTURE_DIR "
                         "or ~/.local/state/loxo-llm-router/captures)")
    ap.add_argument("--output", default="cases/main_replay.jsonl",
                    help="output JSONL path")
    ap.add_argument("--sample-per-stratum", type=int, default=5,
                    help="cases to sample per depth stratum (default: 5)")
    args = ap.parse_args()

    capture_dir = pathlib.Path(
        args.capture_dir
        or os.environ.get("LOXO_CAPTURE_DIR")
        or os.path.expanduser("~/.local/state/loxo-llm-router/captures")
    )
    output_path = pathlib.Path(args.output)
    n_per = args.sample_per_stratum

    rows = load_captures(capture_dir)
    chains = group_chains(rows)

    qwen_ts = load_qwen_timestamps(ADEQUACY_LEDGER)
    usable = process_pairs(chains, qwen_ts)

    pop_by_stratum = {s: 0 for s in STRATA}
    for p in usable:
        pop_by_stratum[p["depth_stratum"]] += 1

    sample, coverage = sample_cases(usable, n_per)

    final_cases = []
    for i, case in enumerate(sample, 1):
        final_cases.append({
            "id": f"mr-{i:03d}",
            "capture_path": case["capture_path"],
            "chain_id": case["chain_id"],
            "depth_stratum": case["depth_stratum"],
            "est_tokens": case["est_tokens"],
            "reference": case["reference"],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for case in final_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    # --- Summary (stdout) ---
    print(f"Captures scanned: {len(rows)}")
    print(f"Chains found: {len(chains)}")
    for i, c in enumerate(chains, 1):
        print(f"  chain-{i:02d}: {len(c)} captures "
              f"({c[0]['name']} .. {c[-1]['name']})")
    print(f"Total usable pairs: {len(usable)}")
    print("Pairs per stratum (before sampling):")
    for s in STRATA:
        print(f"  {s}: {pop_by_stratum[s]}")
    print(f"Sampled cases: {len(final_cases)}")
    print("Sampled cases per stratum:")
    for s in STRATA:
        c = coverage.get(s, {})
        n = c.get("n", 0)
        if n < n_per:
            print(f"  {s}: {n}" +
                  (f" (shortfall: {n_per - n})" if n else " (no cases)"))
        else:
            print(f"  {s}: {n}")
    sampled_chains = sorted(set(c["chain_id"] for c in final_cases))
    print(f"Chains represented in sample: {', '.join(sampled_chains)}")
    print("Per-stratum chain coverage:")
    for s in STRATA:
        c = coverage.get(s, {})
        chains_list = c.get("chains", [])
        n = c.get("n", 0)
        if n == 0:
            print(f"  {s}: no cases")
        elif len(chains_list) == 1:
            print(f"  {s}: {chains_list[0]} (SINGLE-CHAIN)")
        else:
            print(f"  {s}: {', '.join(chains_list)}")
    single = [
        s for s in STRATA
        if coverage.get(s, {}).get("n", 0) > 0
        and len(coverage.get(s, {}).get("chains", [])) == 1
    ]
    if single:
        multi = [
            s for s in STRATA
            if s not in single
            and coverage.get(s, {}).get("n", 0) > 0
        ]
        if multi:
            print(f"  NOTE: two-chain coverage only in: {', '.join(multi)}")
        print(f"  SINGLE-CHAIN strata: {', '.join(single)}")
    print(f"\nWrote {len(final_cases)} cases to {output_path}")


if __name__ == "__main__":
    main()
