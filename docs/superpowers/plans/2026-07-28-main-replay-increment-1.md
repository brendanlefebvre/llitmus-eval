# main-replay increment 1: extractor, tier-0 profile, falsification run

**Date:** 2026-07-28
**Source:** extracted verbatim from
`docs/superpowers/specs/2026-07-27-main-adequacy-design.md` (rev. 2026-07-28,
commit `82ad227`) — the sections "First increment (one day), and what would
kill it" and "Falsification tests." The spec is canonical; if this file and the
spec diverge, the spec wins.
**Status:** Not started. Design approved for planning; no code written.

## First increment (one day), and what would kill it

1. **Extractor** — `scripts/extract_main_replay.py`: walk the captures dir,
   group files into prefix chains by per-message hashing, and for each selected
   turn emit a case: `{id, capture_path, chain_id, depth_stratum, reference:
   {acted, tools: [...], arguments: {...}}}` to `cases/main_replay.jsonl`.
   Stratum assignment uses `from loxo_llm_router import estimate_prompt_tokens`
   and the deployment's configured `LOCAL_CONTEXT_LIMIT` — imported, never
   reimplemented. The prompt is **the captured body verbatim by reference**
   (path, not embedded copy): fixture fidelity is structural —
   `max_tokens: 32000`, `tool_choice`, the full tool array, everything, because
   the case *is* the request. This avoids the half-real-fixture bug by
   construction, and keeps 460 KB bodies (and operator content) out of the
   repo; the runner reads `LOXO_CAPTURE_DIR`.

   **Skip rules** (each logged with a reason, never silently dropped):
   - non-`main` captures (class recomputed via `classify()`);
   - pairs whose appended messages do not begin with an `assistant` message
     (not a decision point);
   - pathological reference turns (reasoning-only, no content or tool call);
   - **the 3 turns served locally by Qwen3-14B** (per the ledger,
     correlated by timestamp) — Qwen3-14B is a candidate in this increment and
     must not be scored against references it authored;
   - the zero/near-zero-token curl probes
     (`req-20260727T121220.315928-0000.json` and
     `req-20260727T121227.015255-0001.json` estimate exactly 0; two more
     probes estimate 5–6). These classify as `unknown`, so the class filter
     already excludes them, but the skip list names them explicitly rather
     than relying on that coincidence.

   **Sample: 15 cases, 5 per stratum, drawn from at least two chains** (the
   dominant 115-capture chain plus the 8-capture chain, which is a different
   task). Fifteen cases from two tasks is materially better evidence than
   fifteen from one at identical runtime. Stated plainly: the shallow stratum's
   population is only 9, so its sample of 5 covers most of what exists; and
   equal-N is a curve-measuring choice, corrected at reporting time by the
   depth weights above.
2. **Profile** — `main-replay` in `litmus_spec.py`: loader + runner reusing
   the native/prompted machinery; tier-0 checks only.
3. **Run** Llama-3.2-1B, Qwen3-4B (think + no-think), Qwen3-14B (think +
   no-think). Sidecars out.
4. **Execute F1, F2, F3a; report F3b; report raw outputs alongside scores**
   (the standing rule: read the outputs, not just the number).

Kill criteria for the day: F1 fails (validators can't validate real glm-5.2
actions), F2 fails (no separation), or F3a fails (prompts truncated — fix the
harness before believing anything). Any of these falsifies the approach before
any router integration exists — which is the point of building this increment
first.

Deliberate parameter decision: generation runs with the captured
`max_tokens=32000`, not a tight action budget. A tight cap silently zeroes out
thinking models (2026-07-16 lesson); a runaway thinker under the honest cap is
*itself signal*, captured in the cost axis exactly as the 125-second chore
title was.

Deferred to increment 2+: tier 1 agreement + F4; tier 2 judge calibration;
prompt-cache reuse across same-chain cases (cases from one chain share long
byte-identical prefixes — the natural lever if prefill cost grows); the
`adequacy_scores.json` exporter; sharing the scorer with Loxo A4 shadow
evaluation.

## Falsification tests (referenced above; copied for standalone use)

- **F1 — reference self-validation.** Run the tier-0 validators over the
  *reference actions themselves*. They are real production actions from
  glm-5.2; they must pass ≈100%. Any systematic failure means the harness is
  measuring itself, not the model (the `parse_native` failure mode). Hard stop
  until fixed.
- **F2 — known-good/known-bad separation.** Llama-3.2-1B vs Qwen3-14B on
  `action_valid`. The chore eval showed the 1B matching the 14B on compliance;
  if that happens *here* — the 1B holding a high valid-action rate on 11-tool,
  15k+-token contexts — then tier 0 has no discriminating power for `main` and
  the gate is falsified. Expected result: the 1B collapses; if it does not,
  this design is wrong and the honest conclusion is that mechanical validity
  does not separate models on this class.
- **F3a — no truncation (hard gate, checked directly).** For every case,
  compare the tokenizer's count of the prompt actually fed to the model against
  the case's expected length. Any mismatch is a harness bug (truncation,
  template overflow) — the tight-token-budget lesson in new clothes — and is
  caught by measurement, not inferred from a score curve.
- **F3b — depth-curve direction (observation, not pass/fail).** Report the
  `action_valid` curve across shallow→mid→deep per model and account for its
  direction. A non-increasing curve for small models is the expected shape, but
  an inversion is a *finding to explain*, not an automatic failure: deep turns
  in this corpus may genuinely be easier (late-session work is often
  follow-through — read this file, apply that edit — while early turns are
  open-ended), and with 114/126 pairs from one session, depth and
  session-progress are nearly the same variable. Note also that at n=5 per
  stratum a single case moves a rate by 0.20, so no strict monotonicity
  assertion is falsifiable at this sample size; F3b is reported, discussed, and
  used to direct the next increment's sampling — nothing more.
- **F4 — agreement noise floor** (second increment). The strongest local
  model's `tool_agreement` bounds what agreement can mean on this case set. If
  the strongest model agrees with glm-5.2 on, say, <40% of turns while holding
  high `action_valid`, then tier 1 is too noisy at current case counts to rank
  with, and must be recalibrated (coarser classes, more cases, or tier-2
  calibration) before the router reads it.

## Context the plan references but does not restate

Defined in the spec: the tier-0 check definitions (`acted_ok`, `well_formed`,
`tool_exists`, `args_schema_ok`, thinking-budget guard), the depth strata and
measured distribution (9/46/65/17 via the authoritative estimator), the depth
weights (0.075/0.383/0.542), and the sidecar shape including
`reference_model` and `action_valid_weighted`.
