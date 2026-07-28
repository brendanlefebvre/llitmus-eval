# main-replay fixups: post-review corrections before increment 1 can run

**Date:** 2026-07-28
**Context:** Code review of the six increment-1 commits (`1787776`..`30a249f`),
four parallel reviewers + per-finding verification pass. Every item below was
independently verified against the code, and where noted, against the real
ledger/captures. Confidence scores use a 0–100 rubric; ≥80 means directly
confirmed with evidence.
**Authority:** `docs/superpowers/specs/2026-07-27-main-adequacy-design.md`
(revised 2026-07-28 post-review — the F1 and prompted-convention bullets there
supersede the copies embedded in `2026-07-28-main-replay-increment-1.md`).
**Definition of done:** all must-fix items landed, `cases/main_replay.jsonl`
regenerated with the fixed extractor, F1 green per-dimension, then re-run the
three-model matrix.

## Must-fix

### 1. Prompted-mode convention: show the model the format it is graded on (verified 100)

`build_replay_prompt` (`litmus_spec.py:767-784`) passes `tools=None` and
injects nothing for non-native models, while `run_main_replay`
(`litmus_spec.py:899`) grades with `parse_prompted` (`:402-411`), which only
recognizes the `_PROMPTED_SYSTEM` JSON shape — a shape nothing in the captured
prompt describes. All three increment-1 models register `native: false` in
their chore sidecars, so this path is the ONLY one the run exercises: as
shipped, the eval measures format mismatch, not capability.

**Decision (made 2026-07-28, recorded in the spec):** for non-native models,
append `_PROMPTED_SYSTEM` **plus the request's own `tools` schemas** as an
additional system message. Sole deviation from body-verbatim; everything else
stays untouched. Update `build_replay_prompt`, and add tests: (a) non-native
prompt contains the convention text and the 11 schemas; (b) native prompt is
byte-verbatim.

### 2. Sidecar contract: weighted headline, reference_model, by_chain (verified 100)

`aggregate_replay` (`litmus_spec.py:639-670`) emits the spec-banned unweighted
pooled `action_valid` (consumed by `_headline` `:981` and
`format_thinking_gap` `:994`), never applies `_DEPTH_WEIGHTS` (`:639`, only
echoed at `:668`), and lacks `reference_model` and `by_chain` entirely.

Fix per spec "The number the router reads": compute `action_valid_weighted` =
Σ stratum_rate × depth_weight over in-scope strata (0.075/0.383/0.542); delete
the pooled `action_valid` key; point `_headline`/`format_replay_table`/gap
formatting at the weighted number; add `reference_model` ("z-ai/glm-5.2",
sourced from the extractor — put it in the case file or a run header, not a
hardcode in litmus_spec) and `by_chain` {chain_id: {action_valid, n}}.

### 3. Qwen-authored skip: off-by-one timestamp correlation (verified 90, corpus contaminated)

Ledger `ts` is stamped at response completion (`ledger.py:225` via
`record()` at `__init__.py:420`), i.e. arrival + latency. `is_qwen_authored`
(`scripts/extract_main_replay.py:140-160`) matches capture N's *arrival*
against it with ±2s, so it actually fires on capture N+1's arrival (agentic
loops re-request within seconds) and skips the WRONG pair. Verified against
the real ledger: `ts − latency_ms` matches the true capture arrival within
5–63 ms on all three 2026-07-27 Qwen turns.

**Live consequence:** `cases/main_replay.jsonl` line 2 (shallow) has the
134-second Qwen3-14B response as its reference — a candidate model authored
one of its own "known-good" references, violating the F1 premise.

Fix: correlate `ledger_ts − latency_ms` against capture N's filename ts (keep
±2s; it's now aligned to the right event), assert every local-served `main`
ledger row is accounted for (matched or explicitly absent from captures), and
**regenerate the case file**.

### 4. F3a: implement the truncation gate for real (verified 80)

The shipped `test_f3a_no_truncation_case_estimates`
(`tests/test_f1_reference_validation.py:213-238`) re-runs
`estimate_prompt_tokens` against the extractor's own recording —
estimator-vs-itself; it can never detect truncation. `run_main_replay` has no
runtime check either, and deep cases at 40–60k tokens are exactly where a
smaller effective context window truncates silently.

Fix: in the runner, after building the prompt, compare the tokenizer's token
count of the rendered prompt against what the backend actually consumed (or at
minimum assert rendered-token-count < model context limit and record it
per-case in the sidecar as `prompt_tokens_fed`); fail the case file's run —
not just a test — on mismatch. Keep the existing drift test but rename it so
it stops claiming to be F3a; also fix its `pytest.skip`-aborts-the-loop bug
(skip per-case via a collected list, like F1 does).

### 5. Abstention scoring: honor `parsed.attempted` (verified 75, proven by direct execution)

`score_replay_call` (`litmus_spec.py:599`) uses `candidate_acted =
parsed.tool is not None`; line `:610` then forces `well_formed=True` when
`acted_ok`. A malformed attempted call (`attempted=True, well_formed=False,
tool=None`) against a prose reference scores `action_valid=True`. Mirror
`score_tool_call:462`: treat `parsed.attempted` as acting — `candidate_acted
= parsed.attempted or parsed.tool is not None` — and add the missing test
(`ParsedCall(False, None, None, attempted=True)` vs `reference.acted=False`
must fail `well_formed`).

### 6. F1 per-dimension gates (owner directive + verified 75)

`test_f1_reference_self_validation` (`tests/test_f1_reference_validation.py:186-206`)
asserts one pooled rate ≥ 0.90. Replace with per-dimension rates —
`acted_ok`, `well_formed`, `tool_exists`, `args_schema_ok` — each asserted
≈1.0, computed with `_rate()`-style None-exclusion, and always printed (not
only on failure). Delete the pooled assert. Spec bullet already updated.

### 7. closed=False dimension semantics (verified 75; prerequisite for 6)

On unclosed thinking, `score_replay_call` (`litmus_spec.py:592-595`) returns
`args_schema_ok=False` / `tool_exists=False` where `score_tool_call`
(`:447-454`) returns `None` so `_rate` excludes uncheckable dimensions.
Align replay with the `None` convention (keep `action_valid=False`), update
`TestClosedGuard` (`tests/test_spec_replay.py:312-319`), and fix the
"matching the existing profiles' treatment" comment, which is currently
inaccurate.

## Should-fix

- **Class-filter before chain grouping** (verified 75, real data):
  `load_captures` globs every class and `group_chains`
  (`scripts/extract_main_replay.py:76-101`) compares only consecutive files,
  so interleaved non-main captures fragment chains — measured on the current
  291-file corpus: 25 chains where 14 exist, 2 pairs destroyed, 1 fully
  usable mid-stratum case silently lost. Classify and drop non-main files
  BEFORE grouping (the per-pair `classify()` at `:229` stays as a belt).
- **Log singleton-chain drops** (verified 75): `:214-215` drops chains of
  length < 2 with a bare `continue`, violating the spec's "each skip logged,
  never silently dropped." The named curl probes reach the explicit log only
  because their bodies happen to be identical.
- **Remove the extractor's malformed-args fallback** (verified 75):
  `build_reference` (`scripts/extract_main_replay.py:187-196`) still has the
  `except JSONDecodeError: args = {}` masking that commit `528459d` removed
  from the F1 helper — skip the pair with a logged reason instead. Tier-1
  will consume these arguments in increment 2.

## Minor (batch opportunistically)

- F1 validates only `tool_calls[0]` of parallel-call references
  (`tests/test_f1_reference_validation.py:124`) — validate all.
- Extractor docstring overstates "15 cases from at least two chains"
  (best-effort in code); either enforce with non-zero exit or soften the
  docstring.
- Record the capture-corpus snapshot (file count + newest filename ts) in the
  case file header and sidecar — the dir has grown 137 → 291 across two days
  and nothing records which population an extraction saw.
- `parse_native` still only parses JSON-in-`<tool_call>`, not the
  `<function=><parameter=>` XML family (2026-07-16 learning). Moot for
  increment 1 (nothing runs native after fix 1's decision), but becomes
  load-bearing for any native model — leave a TODO where fix 1 lands.
- `.superpowers/sdd/task-1-report.md` says "375 lines" (extractor is 447) and
  "Concerns: None" — amend or annotate when closing these fixups.

## Verification checklist for the fixup PR/commit

1. `pytest tests/test_spec_replay.py tests/test_extract_main_replay.py
   tests/test_f1_reference_validation.py` green.
2. Extractor re-run: skip log shows the Qwen pair skipped by the corrected
   correlation (and names it), singleton drops logged, chain count matches
   the class-filtered grouping (14 on the current corpus).
3. Regenerated `cases/main_replay.jsonl` contains NO reference served by a
   candidate model (cross-check every case against the ledger).
4. F1 prints four dimension rates, all ≈1.0, and fails loudly per-dimension.
5. Sidecar contains `action_valid_weighted`, `reference_model`, `by_chain`,
   `depth_weights` — and no pooled `action_valid`.
