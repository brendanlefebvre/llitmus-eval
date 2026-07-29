#!/usr/bin/env python3
"""Extract main-replay eval cases from Loxo capture files.

Walks the captures directory, filters to ``main``-class captures, groups files
into prefix chains by per-message hashing, applies skip rules, assigns depth
strata, samples up to 15 cases (5 per stratum), drawn from multiple chains when
available, and emits JSONL to cases/main_replay.jsonl.

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
from datetime import datetime, timedelta

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

# The model that authored the captured reference turns. Sourced here (not
# hardcoded in litmus_spec) so the runner/sidecar can read it from the case
# file. Matches the increment-1 reference per the adequacy design spec.
REFERENCE_MODEL = "z-ai/glm-5.2"

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


def chain_stable_id(chain: list[dict]) -> str:
    """Stable chain id derived from the first capture's timestamp stem.

    "req-20260728T115606.817785-0209.json" -> "chain-20260728T115606".
    Unlike a positional chain-NN index, this is stable across extractions
    regardless of corpus growth or filtering, so regenerated case files'
    by_chain groupings are comparable across runs.
    """
    name = chain[0]["name"]
    m = FILENAME_TS_RE.match(name)
    if m:
        return f"chain-{m.group(1).split('.')[0]}"
    return f"chain-{pathlib.Path(name).stem}"


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


def load_qwen_timestamps(ledger_path: pathlib.Path) -> list[dict]:
    """Return Qwen3-14B-served ``main`` ledger entries as dicts.

    Each entry has ``ts`` (ISO string, stamped at response *completion*) and
    ``latency_ms`` (int or None). Callers must subtract ``latency_ms`` from
    ``ts`` to recover the request *arrival* time before correlating against
    capture filename timestamps.
    """
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
            latency = entry.get("latency_ms")
            out.append({
                "ts": ts,
                "latency_ms": int(latency) if latency is not None else None,
            })
    return out


NULL_LATENCY_WINDOW_S = 600


def _ts_within_window(capture_ts: datetime, ledger_ts_str: str,
                      latency_ms: int | None) -> bool:
    """Compare a capture arrival ts to a ledger completion ts.

    The ledger ``ts`` is stamped at response completion (arrival + latency).
    Subtract ``latency_ms`` to recover the true arrival time, then compare
    within ±2s. When ``latency_ms`` is None the arrival time cannot be
    recovered, so fall back to a WIDE conservative window (±600s) against the
    completion ts: any plausibly-related capture counts as a match, and
    callers skip matches rather than keep them.
    """
    delta = _ts_delta_seconds(capture_ts, ledger_ts_str, latency_ms)
    if delta is None:
        return False
    window = NULL_LATENCY_WINDOW_S if latency_ms is None else 2
    return delta <= window


def _ts_delta_seconds(capture_ts: datetime, ledger_ts_str: str,
                      latency_ms: int | None) -> float | None:
    """Absolute seconds between a capture arrival and a ledger row's
    reconstructed arrival (completion ts when latency is unknown), or None
    when the ledger ts cannot be parsed. Split out so audit_ledger can rank
    near-simultaneous captures by proximity instead of file order.
    """
    try:
        ledger_ts = datetime.fromisoformat(ledger_ts_str)
        if ledger_ts.tzinfo is not None:
            ledger_ts = ledger_ts.replace(tzinfo=None)
        if latency_ms is not None:
            ledger_ts = ledger_ts - timedelta(milliseconds=latency_ms)
        return abs((capture_ts - ledger_ts).total_seconds())
    except (ValueError, TypeError):
        return None


def is_qwen_authored(capture_name: str,
                     qwen_entries: list[dict]) -> dict | None:
    """Return the matching ledger entry if capture N was Qwen3-14B-served.

    ``qwen_entries`` are the dicts from :func:`load_qwen_timestamps`. The
    returned entry carries the ledger's completion ``ts`` (preserved for log
    clarity) and ``latency_ms``; correlation uses arrival = ts - latency_ms,
    or the wide null-latency window when latency_ms is None.
    """
    capture_ts = parse_capture_ts(capture_name)
    if capture_ts is None:
        return None
    for entry in qwen_entries:
        if _ts_within_window(capture_ts, entry["ts"],
                             entry.get("latency_ms")):
            return entry
    return None


# ---------------------------------------------------------------------------
# Ledger accounting audit
# ---------------------------------------------------------------------------

def load_local_main_rows(ledger_path: pathlib.Path) -> list[dict]:
    """Return all ``main``-class, locally-routed ledger entries.

    Unlike :func:`load_qwen_timestamps` this does not filter on served_model:
    the audit must account for EVERY locally-served main request, whatever
    model name the ledger recorded.
    """
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
        if entry.get("route") != "local":
            continue
        ts = entry.get("ts")
        if ts:
            latency = entry.get("latency_ms")
            out.append({
                "ts": ts,
                "latency_ms": int(latency) if latency is not None else None,
                "served_model": entry.get("served_model") or "(unknown)",
            })
    return out


def audit_ledger(capture_rows: list[dict], ledger_rows: list[dict],
                 sampled_paths: set[str]) -> None:
    """Account for every locally-served main ledger row against the corpus.

    For each row, reconstruct the serving-capture arrival (ts - latency_ms,
    or the wide null-latency window) and match it against the loaded capture
    corpus. A matched capture whose successor pair landed in the sampled case
    file is a hard failure — a locally-served reference must never be
    replayed as ground truth — so exit(1). Unmatched rows are logged ABSENT.
    """
    matched = 0
    absent = 0
    for entry in ledger_rows:
        # Nearest in-window capture, not first: adjacent requests can arrive
        # within the same ±2s window (e.g. a chore fired 0.4s before a main
        # turn), and auditing the wrong neighbor's successor would let the
        # real pair through.
        match = None
        best_delta = None
        for r in capture_rows:
            cap_ts = parse_capture_ts(r["name"])
            if cap_ts is None:
                continue
            if _ts_within_window(cap_ts, entry["ts"],
                                 entry.get("latency_ms")):
                delta = _ts_delta_seconds(cap_ts, entry["ts"],
                                          entry.get("latency_ms"))
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    match = r
        if match is None:
            absent += 1
            print(f"ledger row {entry['ts']} "
                  f"served_model={entry['served_model']}: "
                  f"no matching capture in corpus")
            continue
        matched += 1
        print(f"ledger row {entry['ts']} "
              f"served_model={entry['served_model']}: "
              f"MATCHED capture {match['name']}")
        if str(match["path"].resolve()) in sampled_paths:
            print(f"AUDIT FAILURE: capture {match['name']} was locally "
                  f"served ({entry['served_model']}) but its successor pair "
                  f"was emitted into the sampled case file")
            sys.exit(1)
    print(f"Qwen audit: {len(ledger_rows)} local-served main rows, "
          f"{matched} matched (all excluded), {absent} absent")


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
    """Build the reference action from an assistant turn.

    Returns ``None`` if the turn is malformed (no tool_calls and no content),
    if any tool_call's string ``arguments`` fail to parse as JSON, or if
    ``arguments`` is an unexpected type. An explicit JSON null or ``""`` is
    NOT malformed — some stacks emit those for zero-parameter tools — and
    maps to ``{}``. Genuine parse failures signal "skip the pair" rather than
    silently substituting empty args — tier-1 will consume these arguments in
    increment 2, so masking malformed JSON would hide a real failure mode.
    """
    tool_calls = assistant_msg.get("tool_calls")
    if tool_calls:
        tools = []
        arguments = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            tools.append(fn.get("name"))
            args_raw = fn.get("arguments", "{}")
            if args_raw is None or args_raw == "":
                args = {}  # legitimate empty (zero-parameter tool)
            elif isinstance(args_raw, dict):
                args = args_raw
            elif isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    return None  # signal skip in process_pairs
            else:
                return None  # signal skip in process_pairs
            arguments.append(args)
        return {"acted": True, "tools": tools, "arguments": arguments}
    content = assistant_msg.get("content")
    if content:
        return {"acted": False, "tools": [], "arguments": []}
    return None


def _bad_args_reason(tool_calls: list) -> str:
    """Name the cause of a build_reference args rejection for the skip log.

    Mirrors the acceptance logic in :func:`build_reference` so the logged
    reason distinguishes a genuine JSON decode error from an unexpected
    arguments type.
    """
    for tc in tool_calls:
        args_raw = tc.get("function", {}).get("arguments", "{}")
        if args_raw is None or args_raw == "" or isinstance(args_raw, dict):
            continue
        if isinstance(args_raw, str):
            try:
                json.loads(args_raw)
            except json.JSONDecodeError:
                return "malformed tool_call arguments (JSONDecodeError)"
            continue
        return (f"unexpected tool_call arguments type "
                f"({type(args_raw).__name__})")
    return "malformed tool_call arguments (JSONDecodeError)"


# ---------------------------------------------------------------------------
# Pair extraction and skip rules
# ---------------------------------------------------------------------------

def process_pairs(chains: list[list[dict]],
                  qwen_entries: list[dict]) -> list[dict]:
    usable = []
    for chain in chains:
        chain_id = chain_stable_id(chain)
        if len(chain) < 2:
            print(f"SKIP chain {chain_id}: {len(chain)} capture(s) "
                  f"— no pairs possible", file=sys.stderr)
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

            # Skip rule 1: non-main captures (belt-and-suspenders; the
            # corpus is already class-filtered before grouping in main()).
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

            # Determine whether the reference has bad tool_call arguments
            # (decode error or unexpected type) vs is pathological (no
            # tool_calls, no content). build_reference returns None for both;
            # disambiguate by inspecting ref_msg so the skip reason is
            # accurate.
            ref = build_reference(ref_msg)
            if ref is None:
                tcs = ref_msg.get("tool_calls")
                if tcs:
                    print(f"SKIP {pair_label}: {_bad_args_reason(tcs)}",
                          file=sys.stderr)
                else:
                    print(f"SKIP {pair_label}: pathological reference "
                          f"(no tool_calls, no content)", file=sys.stderr)
                continue

            # Skip rule 4: Qwen3-14B-authored turns
            qwen_entry = is_qwen_authored(cap_n["name"], qwen_entries)
            if qwen_entry is not None:
                if qwen_entry.get("latency_ms") is None:
                    print(f"SKIP {pair_label}: reference plausibly authored "
                          f"by Qwen3-14B (ledger ts={qwen_entry['ts']}, "
                          f"null latency_ms — conservative "
                          f"±{NULL_LATENCY_WINDOW_S}s window)",
                          file=sys.stderr)
                else:
                    print(f"SKIP {pair_label}: reference authored by "
                          f"Qwen3-14B (ledger ts={qwen_entry['ts']})",
                          file=sys.stderr)
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
                "ref_tokens": tokens,
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
# Corpus snapshot metadata
# ---------------------------------------------------------------------------

def depth_weights_from_population(pop_by_stratum: dict) -> dict | None:
    """In-scope depth weights observed on THIS corpus walk.

    weight_s = usable pairs in stratum s / total usable in-scope pairs.
    This is what aggregate_replay should weight by — the hardcoded constant
    in litmus_spec is only a fallback from the 2026-07-27 measurement, and
    the corpus has grown since. None when there are no usable pairs.
    """
    total = sum(pop_by_stratum.get(s, 0) for s in STRATA)
    if not total:
        return None
    return {s: pop_by_stratum.get(s, 0) / total for s in STRATA}


def write_meta(output_path: pathlib.Path, corpus_count: int,
               newest_name: str, final_cases: list[dict],
               coverage: dict, pop_by_stratum: dict) -> pathlib.Path:
    """Persist the corpus snapshot as a sibling ``<stem>.meta.json``.

    The captures dir grows over time; the stdout snapshot line alone leaves
    no durable record of which population an extraction saw.
    """
    meta_path = output_path.with_name(output_path.stem + ".meta.json")
    meta = {
        "corpus_files": corpus_count,
        "newest_capture": newest_name,
        "n_cases": len(final_cases),
        "strata": {s: coverage.get(s, {}).get("n", 0) for s in STRATA},
        "population_by_stratum": {s: pop_by_stratum.get(s, 0)
                                  for s in STRATA},
        "depth_weights": depth_weights_from_population(pop_by_stratum),
        "chains_represented": sorted(
            set(c["chain_id"] for c in final_cases)),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta_path


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
    # Class-filter before chain grouping: interleaved non-main captures
    # (chore/compaction) fragment chains because group_chains compares only
    # consecutive files. The per-pair classify() in process_pairs stays as a
    # belt-and-suspenders check.
    main_rows: list[dict] = []
    for r in rows:
        cls = classify(r["body"]).cls
        if cls != "main":
            print(f"DROP {r['name']}: class={cls} (filtered before grouping)",
                  file=sys.stderr)
            continue
        main_rows.append(r)
    chains = group_chains(main_rows)

    # Corpus snapshot: record which population this extraction saw. The
    # captures dir grows over time and nothing else records the file count or
    # newest ts an extraction was run against.
    corpus_count = len(rows)
    newest_name = rows[-1]["name"] if rows else "(none)"

    qwen_entries = load_qwen_timestamps(ADEQUACY_LEDGER)
    usable = process_pairs(chains, qwen_entries)

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
            "ref_tokens": case["ref_tokens"],
            "reference_model": REFERENCE_MODEL,
            "reference": case["reference"],
        })

    # Ledger accounting audit: every locally-served main row must map to a
    # capture whose successor pair was excluded from the sample. Runs before
    # the case file is written so a failed audit never persists a poisoned
    # case set.
    local_rows = load_local_main_rows(ADEQUACY_LEDGER)
    sampled_paths = set(c["capture_path"] for c in final_cases)
    audit_ledger(rows, local_rows, sampled_paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for case in final_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    # --- Summary (stdout) ---
    print(f"Corpus snapshot: {corpus_count} captures, newest: {newest_name}")
    print(f"Captures scanned: {len(rows)}")
    print(f"Main-class captures: {len(main_rows)} "
          f"({len(rows) - len(main_rows)} non-main dropped)")
    print(f"Chains found: {len(chains)}")
    for c in chains:
        print(f"  {chain_stable_id(c)}: {len(c)} captures "
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
    meta_path = write_meta(output_path, corpus_count, newest_name,
                           final_cases, coverage, pop_by_stratum)
    weights = depth_weights_from_population(pop_by_stratum)
    if weights:
        print("Observed depth weights (usable in-scope pairs): "
              + ", ".join(f"{s}={weights[s]:.3f}" for s in STRATA))
    print(f"\nWrote {len(final_cases)} cases to {output_path}")
    print(f"Wrote corpus snapshot metadata to {meta_path}")


if __name__ == "__main__":
    main()
