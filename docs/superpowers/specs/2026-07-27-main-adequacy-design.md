# What "adequacy" means for the `main` class: next-action replay against captured trajectories

**Status:** Proposed — design only, no implementation. A recommendation to argue with.
**Date:** 2026-07-27
**Inputs:** 137 captures (129 `main`) in `~/.local/state/loxo-llm-router/captures/`, the Litmus tool-calling profile, the Loxo adequacy ledger, learnings entries of 2026-07-27.

## The question

`main` is 90% of routed traffic and all of the routing economics, and it has no
checkable output shape. `chore` reduced to regex validators; `main` is an
open-ended stream of tool calls and edits where "did it do a good job" does not
obviously reduce to anything mechanical. Loxo's ROADMAP explicitly defers this
("main-turn quality-judging... an explicitly deferred open question"). This doc
proposes a definition of per-model `main` adequacy that a router can look up.

## The empirical fact the whole design rests on

The captures are not 129 independent request bodies. They are **successive
snapshots of the same growing conversations**: the dominant group is one
OpenCode session of 126 requests whose message arrays grow monotonically from 12
to 274 messages, and each capture's messages are a byte-identical prefix of the
next (verified by per-message SHA1 comparison during exploration).

Consequence: **capture N ends exactly where a model had to act, and capture N+1
contains what the incumbent production model actually did** — the appended
assistant message, with its real tool calls and arguments. The corpus is
therefore ~125 (state, known-good-action) pairs from real work, at context
depths from 12k to 90k real prompt tokens, with the client's verbatim
parameters and all 11 tool schemas. That is a replay evaluation set that no
synthetic benchmark can reproduce, and it is the asset the brief says is
irreplaceable.

## The proposed definition

> **A model is adequate for `main` to the extent that, when dropped verbatim
> into real mid-session states from captured traffic, it (a) produces a
> mechanically valid next action, and (b) chooses the same kind of action the
> incumbent model actually took at that same point — with both rates reported
> per context-depth stratum, alongside latency and peak memory.**

The unit of evaluation is the **turn**, not the session. This is partly forced
(a session-level eval needs an execution environment; see rejected alternative
1) but it is also *correct for the consumer*: the router never sees a session.
It prices and routes one request body at a time. Per-turn adequacy is measured
at exactly the granularity at which the routing decision is made.

Scoring has two load-bearing tiers plus one bounded calibration tier.

### Tier 0 — mechanical validity (the gate)

Pure validators in the existing `litmus_spec.py` style, no judgment:

- `acted_ok` — the model acted when the reference acted, or responded in prose
  when the reference responded in prose. (Failing to call any tool when the
  session obviously demands one — emitting commentary instead of acting — is a
  primary small-model failure mode and is checkable without judgment.)
- `well_formed` — the action parses, via the existing `parse_native` /
  `parse_prompted` machinery, under the same fixed-convention rule as the
  tool-calling profile (native `tools=` where supported, prompted JSON
  otherwise; report which).
- `tool_exists` — named tool is one of the 11 in the request's own `tools`
  array.
- `args_schema_ok` — required keys present, types correct, **no hallucinated
  keys**. Schema-validation, deliberately *not* the tool-calling profile's
  exact-value equality: at a real mid-session decision point there is no single
  correct argument value, but there is exactly one schema.
- Thinking-budget guard: an unclosed `<think>` scores all-false, as in the
  existing profiles.

`action_valid` = all of the above per case, aggregated as a rate. This is the
hard gate: a model failing tier 0 at rate X will burn approximately X% of
production turns, because these are the same signals the adequacy ledger
already records live (`had_tool_calls`, `tool_calls_valid_json`,
`finish_reason`). **Tier 0 offline is the prior; the ledger is the posterior of
the identical statistic.** That symmetry is deliberate: when a model is ever
promoted, the bench number and the production number are directly comparable
with no translation.

### Tier 1 — reference agreement (the ranking signal, not a gate)

- `tool_match` — candidate's tool is among the reference turn's tool(s)
  (reference turns with parallel calls count as a set).
- `action_class_match` — agreement at a coarse partition: *gather*
  (read/grep/glob/webfetch) | *mutate* (edit/write/bash) | *orchestrate*
  (todowrite/task/skill/question) | *respond* (no tool).

This is where the multiple-valid-actions problem lives: `read foo.py` and
`grep -n symbol` can both be right, so per-case disagreement is not per-case
error. Usage rule, stated as policy: **tier 1 ranks models against each other;
it never passes or fails a case.** The methodology precedent is BFCL's
AST-matching-against-reference as a scalable proxy for execution
([Patil et al., ICML 2025](https://proceedings.mlr.press/v267/patil25a.html);
[gorilla.cs.berkeley.edu](https://gorilla.cs.berkeley.edu/leaderboard.html) —
verified 2026-07-27); the difference is that our references come from captured
real trajectories rather than authored cases, which buys distribution fidelity
at the price of reference non-uniqueness — hence ranking-only.

### Tier 2 — bounded LLM judge (calibration only, not in the routing score)

A frontier judge is used once, on a **sample of tier-1 disagreement cases
only**, answering a single rubric question at temperature 0: "given this
context, is the candidate's action a defensible next step — yes/no." Its output
calibrates the noise floor of tier 1 (what fraction of disagreement is benign),
and is cached, never re-run per routing decision.

Why this bounded role survives the standard objections
([Zheng et al., NeurIPS 2023](https://arxiv.org/pdf/2306.05685) — position,
verbosity, and self-enhancement bias; verified 2026-07-27):

- No pairwise comparison → position bias is structurally absent.
- Binary defensibility of a *tool call* → verbosity bias has little surface.
- The judge never scores its own outputs → self-enhancement bias is absent.
- Judge-stronger-than-judged is acceptable *here* because the judge is not
  defining quality; the captured incumbent already did. The judge only
  estimates how often "different from the incumbent" means "wrong."
- Non-determinism and cost are bounded because the judge output is a
  calibration constant, not a per-model score. If the judge is dropped
  entirely, tiers 0–1 still function; the routing number never depends on it.

### Depth stratification

Every rate above is reported per prompt-size stratum: **shallow** (<16k
estimated tokens), **mid** (16–40k), **deep** (40k up to the local context
limit). Beyond the limit, Rule 3 already routes to cloud, so it is out of
scope. Rationale: for local candidates the routing-relevant failure mode is
precisely "at what depth does this model stop being able to act." A single
pooled number would average away the cliff; the stratified curve *is* the
signal that separates "can carry a coding session" from "cannot," because
carrying a session means still acting correctly at turn 200, not just turn 2.

## The number the router reads

Sidecar `results_main-replay_<label>.json`, same envelope as existing sidecars
(per-mode detail, `median_latency_ms`, `peak_memory_mb`, `mean_tokens_per_case`),
with aggregate:

```json
{
  "action_valid": 0.93,
  "by_dimension": {"acted_ok": ..., "well_formed": ..., "tool_exists": ..., "args_schema_ok": ...},
  "tool_agreement": 0.61,
  "action_class_agreement": 0.78,
  "by_depth": {
    "shallow": {"action_valid": ..., "tool_agreement": ..., "n": ...},
    "mid":     {...},
    "deep":    {...}
  },
  "n_cases": 15
}
```

(Values illustrative.) The eventual `adequacy_scores.json` entry for
`class=main` is `{model, action_valid, tool_agreement, by_depth}` plus the cost
axes — two-axis adequacy, exactly as the chore work concluded (a model that
passes every check at 125 s is still the wrong route). Promotion stays human
per the ROADMAP non-goal; this number narrows candidates, it does not flip
switches.

## Rejected alternatives

**1. Outcome-based agentic benchmark (SWE-bench-style, or a bespoke task
environment).** Highest fidelity to "carried the session," and the only method
that measures multi-step recovery. Rejected because: it grades the
model+scaffold *system*, crossing the model-characterizer boundary Litmus
deliberately set on 2026-07-13; it requires an agent loop plus execution
environment that does not exist here; each task is a full agentic session, so a
model evaluation is hours-to-days on a 24 GB M4 Pro shared with everything else
(fails the re-run constraint); and it evaluates a public distribution
(GitHub issues in SWE-bench's case — Jimenez et al., ICLR 2024,
arXiv:2310.06770, cited from training knowledge, not re-checked today) rather
than this operator's OpenCode traffic with these 11 tool schemas. Decisively:
it cannot consume the captures at all — it discards the one irreplaceable
asset.

**2. LLM judge as the primary scorer** (judge grades every candidate turn).
Produces a number, but the number inherits the judge's taste for how *it* would
act; it costs cases × models × reruns judge calls forever; and it is
non-deterministic in exactly the place a routing table wants stability. The
sharpest argument: with captures in hand, "would a strong model approve this
action" is a noisy, expensive approximation of **checking against what a strong
model actually did at that exact point** — which is tier 1, computed by string
comparison for free. The judge earns only the bounded calibration role of
tier 2.

**3. Production-signals only** (skip the benchmark; read
`finish_reason`/`tool_calls_valid_json` from the ledger). Rejected as the sole
answer for the chicken-and-egg reason: the ledger only ever scores models that
already serve traffic, and the router needs a prior *before* risking a real
request on a candidate. It also cannot answer counterfactuals ("would the 4B
have been fine on this turn?"). It is half the answer — the posterior half, per
the 2026-07-27 latency-swing learning (Litmus = prior, ledger = posterior) —
and this proposal is designed so both halves measure the same statistic. Note
that ROADMAP A4 shadow evaluation *is* this proposal run on live traffic
instead of captures: building the tier-0/tier-1 scorer once yields both the
offline bench and the shadow scorer.

**4. Perplexity of the reference actions under the candidate** (no generation;
seconds per model — Litmus already has `compute_perplexity`). Rejected
precisely because it is the easiest to compute (warning #1): token-level
likelihood of a cloud model's serialization under a local model's template
conflates format mismatch with incompetence (the `parse_native` lesson — two
"native" conventions behind one tag), rewards distributional mimicry rather
than decoding-time ability to emit a valid call, and cannot detect over-calling
or failure-to-act at all.

## Falsification tests (built into the first increment)

- **F1 — reference self-validation.** Run the tier-0 validators over the
  *reference actions themselves*. They are real production actions from the
  incumbent; they must pass ≈100%. Any systematic failure means the harness is
  measuring itself, not the model (the `parse_native` failure mode). Hard stop
  until fixed.
- **F2 — known-good/known-bad separation.** Llama-3.2-1B vs Qwen3-14B on
  `action_valid`. The chore eval showed the 1B matching the 14B on compliance;
  if that happens *here* — the 1B holding a high valid-action rate on 11-tool,
  15k+-token contexts — then tier 0 has no discriminating power for `main` and
  the gate is falsified. Expected result: the 1B collapses; if it does not,
  this design is wrong and the honest conclusion is that mechanical validity
  does not separate models on this class.
- **F3 — depth-curve direction.** For small models, `action_valid` must be
  non-increasing across shallow→mid→deep. An inversion indicates a harness
  artifact (truncation, parsing, template overflow), not capability — the
  tight-token-budget lesson in new clothes.
- **F4 — agreement noise floor** (second increment). The strongest local
  model's `tool_agreement` bounds what agreement can mean on this case set. If
  the strongest model agrees with the reference on, say, <40% of turns while
  holding high `action_valid`, then tier 1 is too noisy at current case counts
  to rank with, and must be recalibrated (coarser classes, more cases, or tier-2
  calibration) before the router reads it.

## First increment (one day), and what would kill it

1. **Extractor** — `scripts/extract_main_replay.py`: walk the captures dir,
   group files into sessions by message-prefix matching, and for each selected
   turn emit a case: `{id, capture_path, depth_stratum, reference: {acted,
   tools: [...], arguments: {...}}}` to `cases/main_replay.jsonl`. The prompt
   is **the captured body verbatim by reference** (path, not embedded copy):
   fixture fidelity is structural — `max_tokens: 32000`, `tool_choice`,
   the full tool array, everything, because the case *is* the request. This
   avoids yesterday's half-real-fixture bug by construction, and keeps 460 KB
   bodies (and operator content) out of the repo; the runner reads
   `LOXO_CAPTURE_DIR`. Skip pathological reference turns (reasoning-only, no
   content or tool call). Sample: 5 shallow / 5 mid / 5 deep = 15 cases.
2. **Profile** — `main-replay` in `litmus_spec.py`: loader + runner reusing
   the native/prompted machinery; tier-0 checks only.
3. **Run** Llama-3.2-1B, Qwen3-4B (think + no-think), Qwen3-14B (think +
   no-think). Sidecars out.
4. **Execute F1–F3 and report raw outputs alongside scores** (yesterday's rule:
   read the outputs, not just the number).

Kill criteria for the day: F1 fails (validators can't validate real incumbent
actions) or F2 fails (no separation). Either result falsifies the approach
before any router integration exists — which is the point of building this
increment first.

Deliberate parameter decision: generation runs with the captured
`max_tokens=32000`, not a tight action budget. A tight cap silently zeroes out
thinking models (2026-07-16 lesson); a runaway thinker under the honest cap is
*itself signal*, captured in the cost axis exactly as the 125-second chore
title was.

Deferred to increment 2+: tier 1 agreement + F4; tier 2 judge calibration;
prompt-cache reuse across same-session cases (cases from one session share long
byte-identical prefixes — the natural lever if prefill cost grows); the
`adequacy_scores.json` exporter; sharing the scorer with Loxo A4 shadow
evaluation.

## Cost estimate (honest)

Prefill dominates: deep cases are 40–90k real prompt tokens. Per-case decode is
small (an action is tens of tokens, plus thinking where enabled). Prefill
throughput for these exact shapes is **unmeasured** — `litmus.py
prefill-scaling` exists and should produce the number before anyone quotes one.
Order-of-magnitude expectation: minutes per deep case on the 14B, so a full
15-case × 2-mode run plausibly lands in the tens-of-minutes-to-low-hours range
per large model; the 1B is trivial. If that proves too slow to re-run per model
release, the prompt-cache-reuse lever above is the mitigation, and the honest
fallback is fewer deep cases (with the coverage loss stated in the sidecar, not
silently).

## What this does NOT measure

- **Compounding error and self-recovery — the structural gap.** Replay scores
  every candidate on the *incumbent's* state distribution. A candidate's own
  mistakes would take a real session somewhere these captures never go, and
  nothing here measures whether it recovers. This is the covariate-shift
  critique of behavioral cloning (Ross, Gordon & Bagnell, AISTATS 2011 —
  DAgger; arXiv:1011.0686, cited from training knowledge, not re-checked
  today). A model can be turn-wise valid and session-wise doomed. No offline
  method short of rejected alternative 1 closes this; the sufficiency proof is
  and remains production shadow evaluation and earned promotion (A4/A6/A7).
  **This eval is the filter, not the verdict.**
- **Edit and code quality.** `args_schema_ok` does not mean the edit compiles,
  the bash command is safe, or the code is right. Nothing executes.
- **Prose-turn quality.** When the right move is to answer the user in text,
  only the act-vs-respond decision is scored, not the answer.
- **Goal achievement.** Whether the session accomplished what the user wanted
  is invisible at turn granularity.
- **Tomorrow's distribution.** Cases are one day of one operator's OpenCode
  traffic; an OpenCode release can change the system prompt and tool schemas
  (the classifier's fingerprints are versioned for the same reason).
  Re-extraction per harness version is part of the method, not a footnote.

If the position is "a metric that cannot see compounding error is not adequacy
at all" — that is the strongest counterargument to this design, and the answer
it must defend: per-turn validity is *necessary* and measurable now; session
sufficiency is only ever provable on real traffic, which is what the
shadow/promotion pipeline is for. The alternative readings are an environment
benchmark this project has already scoped out, or waiting — and `main` traffic
stays 100% cloud while waiting.
