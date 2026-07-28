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

### 5. Dimension applicability rule: None, never False, for not-applicable (unifies former items 5, 6, 7 — owner rule, 2026-07-28)

**THE RULE:** a scored dimension that is NOT APPLICABLE to a case must be
`None`, never `False`. `False` means "checked and failed." `None` means "not
a meaningful question here." The code already does this correctly for
`args_schema_ok` when the named tool isn't in the capture's `tools` array —
apply the same treatment consistently.

**Why it matters:** the sidecar's `by_dimension` breakdown exists for
per-dimension fault localization. Any dimension returning `False` on an
inapplicable case double-counts a single underlying failure across several
dimensions and makes the breakdown useless for locating the actual fault.

**Fix site 1 — the abstention branch** (`litmus_spec.py:599`, `:610`; the
verified-75, execution-proven bug). Use `parsed.attempted` as the
discriminator, mirroring `score_tool_call:462`:
- attempted a call that didn't parse → `acted_ok` per reference,
  `well_formed=False` (meaningful — it tried and failed);
- chose prose, never attempted → `well_formed=None` (not applicable), and
  `tool_exists`/`args_schema_ok` likewise `None`.
This simultaneously kills both verified defects in the branch: the malformed
attempted call scored as a perfect abstention (`candidate_acted` must treat
`parsed.attempted` as acting), and the reference-acted/candidate-prose case
failing `acted_ok` AND `well_formed` for one underlying failure.

**Fix site 2 — the `closed=False` branch** (`litmus_spec.py:592-595`; former
item 7). A thinking-budget overrun currently returns all-`False` across every
dimension, registering one overrun as five distinct failures. `acted_ok=False`
is defensible — it did not act. `well_formed`/`tool_exists`/`args_schema_ok`
→ `None`; nothing was produced to check. Fix the "matching the existing
profiles' treatment" comment, which is currently inaccurate.

**Aggregation** (`aggregate_replay`): skip `None` rather than coercing to
`False` when computing `by_dimension` rates, and report each dimension's
denominator — how many cases it was actually applicable to — alongside its
rate (e.g. `{"rate": ..., "n_applicable": ...}`). A rate over an unstated
denominator is the thing that made `loose` useless.

**F1 consumes the same semantics** (former item 6; owner directive):
`test_f1_reference_self_validation`
(`tests/test_f1_reference_validation.py:186-206`) drops its pooled ≥ 0.90
assert entirely and gates each dimension separately at ≈1.0 over its
applicable denominator, printing all four rates + denominators always, not
only on failure. Spec bullet already updated.

**Unchanged by design:** `action_valid` semantics — a case with `None`
dimensions still fails `action_valid` if any applicable dimension failed.

**Scope guard:** these two branches plus aggregation and the F1 test.
Nothing else.

**Tests:** `ParsedCall(False, None, None, attempted=True)` vs prose reference
→ `well_formed=False`, `action_valid=False`; prose response vs acted
reference → `acted_ok=False`, `well_formed=None`; update `TestClosedGuard`
(`tests/test_spec_replay.py:312-319`) to the `None` shape; an aggregation
test pinning per-dimension denominators.

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
4. F1 prints four dimension rates with applicable-case denominators, all
   ≈1.0, and fails loudly per-dimension; no pooled assert remains.
5. Sidecar contains `action_valid_weighted`, `reference_model`, `by_chain`,
   `depth_weights` — and no pooled `action_valid`.

## Round 2 (2026-07-28, applied post-review of the fixup commits)

The fixup commits resolved items 1, 2, 5 and all should-fixes but left two
silent shortfalls and introduced one HIGH defect. All closed in this round:

1. **Weighted headline renormalization** — absent strata no longer drop their
   weight silently; the rate renormalizes over present strata and the
   aggregate publishes `depth_weight_coverage` (Σ present weights), with a
   visible `coverage=… — missing strata` marker in the table when < 1.0. The
   two tests that pinned the shrinking headline were re-pinned to the
   renormalized values. Spec updated (renormalization rule).
2. **F3a minimum gate** — tokenizer failures now surface as
   `tokens_fed_error` on the case record instead of silent None, and a
   prompt exceeding a sane `model_max_length` lands the case in `errored`
   ("prompt exceeds model context: N > M") rather than scoring a silently
   truncated generation. The drift test's overclaiming docstring was
   rewritten to describe what exists; comparison against actual backend
   consumption remains future work, stated as such.
3. **Ledger accounting audit** — `audit_ledger` accounts for every
   local-served `main` ledger row: nearest-in-window match (not
   first-in-window — adjacent requests can arrive inside the same ±2s
   window; verified live where the first-match variant named a chore decoy
   450 ms off instead of the true serving capture 6 ms off), hard exit(1)
   before the case file is written if a matched capture's successor pair was
   sampled, ABSENT logged otherwise. Live run: 7 rows, 3 matched (all
   excluded), 4 absent (pre-corpus).
4. **Null-latency conservatism** — rows without `latency_ms` use a ±600s
   window against completion ts and skip on match; the test that cemented
   the leak was inverted.
5. **Legitimate empty args** — `null`/`""` tool-call arguments are kept as
   `{}`; only genuine decode failures and unexpected types skip, each with
   an accurate log label.
6. **Stable chain ids** — `chain-<first-capture-timestamp-stem>` instead of
   positional numbering, so `by_chain` is comparable across extractions.
7. **Minors closed** — F1 validates all parallel calls (per-call rows,
   `id#N`); corpus snapshot persisted to `cases/main_replay.meta.json`;
   `reference_model` sourced from case records (hardcoded fallbacks
   removed); task report annotated.

Remaining known gaps, deliberate: F3a backend-consumption comparison;
strict-template (single-system-message) guard for the prompted injection —
both noted in code/docstrings, neither blocks the increment-1 model set.

8. **Live depth weights (owner directive, post-round-2)** — `_DEPTH_WEIGHTS`
   was stale hardcoded data. The extractor now computes the observed
   in-scope usable-pair distribution on every walk
   (`depth_weights_from_population`), persists it (plus
   `population_by_stratum`) in `main_replay.meta.json`, and the runner
   consumes it via `load_replay_meta`; the 2026-07-27 constant is removed
   entirely: missing weights yield `action_valid_weighted: null` with
   `depth_weights_source: "missing"` (owner directive — the applicability
   rule applied to the headline; formatters render "n/a (weights missing)"). Measured drift that motivated
   this: 0.075/0.383/0.542 (07-27, 120 pairs) vs 0.129/0.397/0.473 (07-28,
   224 pairs).
