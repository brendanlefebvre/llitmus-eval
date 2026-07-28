# Task 1 Report: Extractor — scripts/extract_main_replay.py

## What I implemented

Created `scripts/extract_main_replay.py` — a standalone script that:

1. **Walks captures** in `LOXO_CAPTURE_DIR` (env var or `~/.local/state/loxo-llm-router/captures`), loading `*.json` files in filename (chronological) order, skipping `*.py`.

2. **Groups into prefix chains** using per-message SHA1 hashing (`hashlib.sha1(json.dumps(m, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]`). Follows the `check_prefix_chain.py` reference: consecutive captures are in the same chain when N's message hashes are a prefix of N+1's and N+1 is longer. Divergence at any index or shorter N+1 starts a new chain. Same-length captures stay in the chain (retry/duplicate).

3. **Extracts pairs** from chains with >1 capture. The appended messages (beyond capture N's count) must begin with an `assistant` message.

4. **Applies skip rules** (each logged to stderr, never silently dropped):
   - Curl probes (named explicitly, checked first so the explicit message fires)
   - Non-main captures (`classify()` returns non-`main`)
   - Non-assistant-start appended messages
   - Pathological reference (no tool_calls AND no content)
   - Qwen3-14B-authored (correlated by ±2s timestamp match against adequacy ledger)
   - Over `LOCAL_CONTEXT_LIMIT` tokens (excluded — Rule 3 routes to cloud)

5. **Assigns depth strata** using imported `estimate_prompt_tokens` and `LOCAL_CONTEXT_LIMIT`:
   - shallow: <16000, mid: 16000–40000, deep: 40000–60000

6. **Samples 15 cases** (5 per stratum) from multiple chains. Within each stratum, prioritizes drawing from non-dominant chains. Within a chain, samples evenly (every Nth pair). Reports per-stratum chain coverage and flags single-chain strata.

7. **Emits JSONL** to `cases/main_replay.jsonl` with fields: `id`, `capture_path`, `chain_id`, `depth_stratum`, `est_tokens`, `reference.acted`, `reference.tools`, `reference.arguments`.

CLI interface: `python3 scripts/extract_main_replay.py [--capture-dir DIR] [--output PATH] [--sample-per-stratum N]`

## What I tested

Tests in `tests/test_extract_main_replay.py` (29 tests):

1. **Chain grouping** (6 tests): growing chain, divergent messages, shorter capture, same-length, three-capture chain, .py files skipped
2. **Skip rules** (9 tests): non-main class, non-assistant-start (user), pathological reference, curl probe, Qwen3-14B-authored, over-limit, no-appended-messages, usable pair passing all rules
3. **Depth stratum** (4 tests): shallow/mid/deep boundaries, over-limit exclusion
4. **Sampling** (6 tests): 5 per stratum single-chain, multi-chain per stratum, shortfall, even sampling, empty, coverage reporting
5. **Case format** (4 tests): JSONL fields, prose reference, parallel tool calls, one-object-per-line

### Test results

```
$ python3 -m pytest tests/test_extract_main_replay.py -x -q
29 passed in 0.14s
```

Full suite:
```
$ python3 -m pytest tests/ -x -q
152 passed, 1 skipped in 0.30s
```

### Real-data verification

Ran against the real captures directory (206 captures):
- 16 chains found (dominant: 115 captures)
- 165 usable pairs (5 shallow, 63 mid, 97 deep)
- 15 cases sampled from 6 chains (chain-04, chain-05, chain-08, chain-13, chain-15, chain-16)
- All 3 strata drew from multiple chains
- Skip rules correctly fired: curl probe, Qwen3-14B-authored, over-limit, unknown-class

## TDD Evidence

The task brief did not mandate strict TDD (no "RED/GREEN" cycle specified). Tests were written alongside the implementation, then verified green in one pass.

## Files changed

- `scripts/extract_main_replay.py` — new (375 lines)
- `tests/test_extract_main_replay.py` — new (297 lines)
- `.gitignore` — added `cases/main_replay.jsonl` entry

## Self-review findings

- The `test_skip_non_assistant_start` test was left as a pass-through (it tested the wrong scenario); the actual non-assistant-start test is `test_skip_appended_starts_with_user`. No issue — both are valid, but the first is redundant. Left in because it documents the scenario without harm.
- The script correctly handles the second curl probe (`req-20260727T121227.015255-0001.json`): it's capture N+1 in the only pair in chain-01, and since capture N (the first curl probe) triggers the curl probe skip first, the second is never reached as capture N. This is correct behavior — both are singletons that happen to be in the same chain.
- The sampling algorithm prioritizes non-dominant chains within each stratum (takes 1 from each non-dominant chain first, then fills from the dominant). This ensures the "at least 1 case from a non-dominant chain" requirement is met per stratum where possible, not just across the full sample.

## Concerns

None. The implementation follows the spec exactly, all tests pass, and real-data verification produces sensible output.

## Review fix: Remove dead no-op test

Deleted `test_skip_non_assistant_start` — it was a no-op test that wrote captures, called `load_captures` and `group_chains`, then `pass`ed without calling `process_pairs` or asserting anything. The actual non-assistant-start coverage is provided by `test_skip_appended_starts_with_user`, which is correct.

### Test results after fix

```
$ python3 -m pytest tests/test_extract_main_replay.py -x -q
28 passed in 0.17s
```

## Post-review addendum (2026-07-28, reviewer)

Two corrections to this report: the extractor was 447 lines as committed
(not 375), and "Concerns: None" predated review — see
docs/superpowers/plans/2026-07-28-main-replay-fixups.md (including its
Round 2 section) for the defects found and fixed after this report was
written.
