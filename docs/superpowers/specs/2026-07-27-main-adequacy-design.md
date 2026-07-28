# What "adequacy" means for the `main` class: next-action replay against captured trajectories

**Status:** Proposed — design only, no implementation. Revised 2026-07-28 per
measurement review (chain structure, authoritative depth profile, reference-set
identity, weighting, F3 split, confound). Revised again 2026-07-28 post
code-review: prompted-mode convention resolved, F1 made per-dimension. A
recommendation to argue with.
**Date:** 2026-07-27, revised 2026-07-28
**Inputs:** 137 captures (129 `main`) in `~/.local/state/loxo-llm-router/captures/`,
the Litmus tool-calling profile, the Loxo adequacy ledger, learnings entries of
2026-07-27 (including the authoritative depth-profile entry).

## The question

`main` is 90% of routed traffic and all of the routing economics, and it has no
checkable output shape. `chore` reduced to regex validators; `main` is an
open-ended stream of tool calls and edits where "did it do a good job" does not
obviously reduce to anything mechanical. Loxo's ROADMAP explicitly defers this
("main-turn quality-judging... an explicitly deferred open question"). This doc
proposes a definition of per-model `main` adequacy that a router can look up.

## The empirical fact the whole design rests on

The captures are not 129 independent request bodies. They are **successive
snapshots of growing conversations** — verified by hashing each message and
comparing overlaps. Measured structure (2026-07-28): **eleven prefix chains**,
not one. The dominant chain is **115 captures** (13:39→15:22) yielding **114 of
the 126 usable pairs**; the rest are one chain of 8, two of 3, one of 2, and
six singletons (curl probes, session-start title generations, a replay).

Consequence: **capture N ends exactly where a model had to act, and capture N+1
contains what the serving model actually did** — the appended assistant
message, with its real tool calls and arguments. A pair is usable only when the
appended messages begin with an `assistant` message; appends that are
`tool`/`user` only are not decision points and are skipped. The corpus is
therefore 126 (state, known-good-action) pairs from real work — fewer after
the skip rules in the first-increment section — with the
client's verbatim parameters and all 11 tool schemas — a replay evaluation set
no synthetic benchmark can reproduce, and the asset the brief calls
irreplaceable.

Two honesty notes on this corpus, expanded in the limits section: **114 of the
126 pairs come from one session about one task** (Litmus chore-profile work, in
Python, in this repo), and the reference actions are overwhelmingly one
model's: the ledger attributes **126 `main` turns to `z-ai/glm-5.2` and 3 to
`mlx-community/Qwen3-14B-4bit`**. The reference set is glm-5.2, and the doc
says "glm-5.2" rather than "the incumbent" wherever the distinction pays rent.

## The proposed definition

> **A model is adequate for `main` to the extent that, when dropped verbatim
> into real mid-session states from captured traffic, it (a) produces a
> mechanically valid next action, and (b) chooses the same kind of action
> glm-5.2 — the model that actually served these turns — took at that same
> point, with both rates reported per context-depth stratum, alongside latency
> and peak memory.**

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
  otherwise; report which). **For non-native models the prompted convention
  must be shown to the model:** `_PROMPTED_SYSTEM` plus the request's own tool
  schemas are appended as an additional system message. This is the sole
  deliberate deviation from body-verbatim, and it exists because the captured
  system prompt teaches the serving harness's native protocol, not the litmus
  prompted shape — grading a model on a format it was never shown measures the
  harness, not the model (2026-07-28 review, verified: all three increment-1
  candidates register non-native, so an uninstructed prompted path would have
  been the *only* path exercised). Everything else in the body stays verbatim.
- `tool_exists` — named tool is one of the 11 in the request's own `tools`
  array.
- `args_schema_ok` — required keys present, types correct, **no hallucinated
  keys**. Schema-validation, deliberately *not* the tool-calling profile's
  exact-value equality: at a real mid-session decision point there is no single
  correct argument value, but there is exactly one schema.
- Thinking-budget guard: an unclosed `<think>` scores all-false, as in the
  existing profiles.

`action_valid` = all of the above per case. This is the hard gate: a model
failing tier 0 at rate X will burn approximately X% of production turns,
because these are the same signals the adequacy ledger already records live
(`had_tool_calls`, `tool_calls_valid_json`, `finish_reason`). **Tier 0 offline
is the prior; the ledger is the posterior of the identical statistic.** That
symmetry is deliberate: when a model is ever promoted, the bench number and the
production number are directly comparable with no translation.

### Tier 1 — reference agreement (the ranking signal, not a gate)

- `tool_match` — candidate's tool is among the reference turn's tool(s)
  (reference turns with parallel calls count as a set).
- `action_class_match` — agreement at a coarse partition: *gather*
  (read/grep/glob/webfetch) | *mutate* (edit/write/bash) | *orchestrate*
  (todowrite/task/skill/question) | *respond* (no tool).

**"Agreement" here means agreement with glm-5.2 specifically** — the model that
served 126 of the 129 captured `main` turns — not with an abstract authority.
The sidecar records `reference_model` so the number can never be read as more
than that.

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
  defining quality; the captured glm-5.2 actions already did. The judge only
  estimates how often "different from glm-5.2" means "wrong."
- Non-determinism and cost are bounded because the judge output is a
  calibration constant, not a per-model score. If the judge is dropped
  entirely, tiers 0–1 still function; the routing number never depends on it.

### Depth stratification

Every rate above is reported per prompt-size stratum. Strata and their measured
population — computed with **Loxo's own `estimate_prompt_tokens()` imported
from the package and the deployment's real `LOCAL_CONTEXT_LIMIT = 60000`**,
never a reimplementation or a char-count proxy (a naive proxy inverted the
strategic conclusion before being caught; see the 2026-07-27 depth-profile
learning) — over the 137 captures:

| stratum | est. tokens | n | share |
|---|---|---|---|
| shallow | <16k | 9 | 6.6% |
| mid | 16k–40k | 46 | 33.6% |
| deep (in scope) | 40k–60k | 65 | 47.4% |
| over limit (Rule 3 → cloud) | >60k | 17 | 12.4% |

p50 43,515 · p90 60,610 · max 64,644. The ~5.6k-token tool-schema floor on
every request means genuinely shallow `main` traffic barely exists — **shallow
is the binding stratum at n=9**, and any equal-N sample takes most of its
population. Beyond the limit, Rule 3 already routes to cloud, so >60k is out of
scope; only 12.4% of traffic exits that way, leaving ~88% of `main` genuinely
eligible for an adequacy decision — the eval's addressable share is large. One
marginal note: the distribution is pressed against the limit (p90 60,610 vs a
60,000 ceiling), so `local_context_limit` is the binding constraint at the
margin and longer sessions cross it routinely — worth a config experiment,
not a reframe.

Cases are stratified **by chain as well as depth**: with 114 of 126 pairs
coming from one session, a depth-only sample is silently also a
one-task sample. The first increment draws from at least two chains (the
8-capture chain is a different task) and records chain identity per case.

Rationale for the depth axis is unchanged: for local candidates the
routing-relevant failure mode is "at what depth does this model stop being able
to act." A single pooled number would average away the cliff; the stratified
curve *is* the signal that separates "can carry a coding session" from
"cannot" — subject to the depth/session-progress confound stated in the limits
section.

## The number the router reads

Sidecar `results_main-replay_<label>.json`, same envelope as existing sidecars
(per-mode detail, `median_latency_ms`, `peak_memory_mb`, `mean_tokens_per_case`),
with aggregate:

```json
{
  "reference_model": "z-ai/glm-5.2",
  "action_valid_weighted": 0.91,
  "by_dimension": {"acted_ok": ..., "well_formed": ..., "tool_exists": ..., "args_schema_ok": ...},
  "tool_agreement": 0.61,
  "action_class_agreement": 0.78,
  "by_depth": {
    "shallow": {"action_valid": ..., "tool_agreement": ..., "n": ...},
    "mid":     {...},
    "deep":    {...}
  },
  "by_chain": {"chain-01": {"action_valid": ..., "n": ...}, "chain-10": {...}},
  "depth_weights": {"shallow": 0.075, "mid": 0.383, "deep": 0.542},
  "n_cases": 15
}
```

(Values illustrative; weights are the measured in-scope traffic shares
9/46/65 over 120.) Equal-N sampling per stratum is correct for measuring the
depth *curve* — equal precision per point — but a pooled rate over an equal-N
sample would describe a traffic mix that does not exist. So the sidecar reports
**`action_valid_weighted`** — per-stratum rates weighted by the observed
in-scope depth distribution — and **no unweighted pooled number exists in the
sidecar at all**.

The eventual `adequacy_scores.json` entry for `class=main` is `{model,
reference_model, action_valid_weighted, tool_agreement, by_depth}` plus the
cost axes — two-axis adequacy, exactly as the chore work concluded (a model
that passes every check at 125 s is still the wrong route). Promotion stays
human per the ROADMAP non-goal; this number narrows candidates, it does not
flip switches.

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
  *reference actions themselves*. They are real production actions from
  glm-5.2; they must pass ≈100% — **reported and gated per dimension**
  (`acted_ok`, `well_formed`, `tool_exists`, `args_schema_ok` each ≈1.0, with
  uncheckable dimensions excluded from denominators per `_rate()` semantics),
  never as one pooled number. The diagnostic value of F1 is *which* dimension
  fails: `args_schema_ok` failures indict the schema validator, `well_formed`
  the parser (the `parse_native` failure mode), `acted_ok` the reference
  extraction. A pooled rate hides exactly that signal. Any systematic failure
  means the harness is measuring itself, not the model. Hard stop until fixed.
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

## Cost estimate (honest)

Prefill dominates, and the measured distribution says the *typical* case is
big: p50 ≈ 43.5k estimated prompt tokens, with the deep stratum (47% of
in-scope traffic) running 40–60k. Per-case decode is small (an action is tens
of tokens, plus thinking where enabled). Prefill throughput for these exact
shapes is **unmeasured** — `litmus.py prefill-scaling` exists and should
produce the number before anyone quotes one. Order-of-magnitude expectation:
minutes per deep case on the 14B, so a full 15-case × 2-mode run plausibly
lands in the tens-of-minutes-to-low-hours range per large model; the 1B is
trivial. Because the corpus is deep-heavy, same-chain prompt-cache reuse is
**load-bearing for re-runnability, not a nice-to-have** — cases from one chain
share byte-identical prefixes by construction. Deep cases are the last thing to
cut: they are where the routing decision actually lives. If the runtime still
proves too slow per model release, the honest fallback is fewer *shallow* cases
(a stratum with population 9 anyway), with the coverage change stated in the
sidecar, not silently.

## What this does NOT measure

- **Compounding error and self-recovery — the structural gap.** Replay scores
  every candidate on the *reference model's* state distribution. A candidate's
  own mistakes would take a real session somewhere these captures never go, and
  nothing here measures whether it recovers. This is the covariate-shift
  critique of behavioral cloning (Ross, Gordon & Bagnell, AISTATS 2011 —
  DAgger; arXiv:1011.0686, cited from training knowledge, not re-checked
  today). A model can be turn-wise valid and session-wise doomed. No offline
  method short of rejected alternative 1 closes this; the sufficiency proof is
  and remains production shadow evaluation and earned promotion (A4/A6/A7).
  **This eval is the filter, not the verdict.**
- **One session, one task, one codebase.** 114 of 126 usable pairs come from a
  single session doing a single task (Litmus chore-profile work — Python, in an
  eval harness, in this repo). "One day of one operator's traffic" would
  understate it. A model could score well by being good at Python eval-harness
  work specifically; two-chain sampling mitigates at the margin, and only
  accumulating more capture days fixes it.
- **Depth vs session-progress is confounded — a property of this data, not a
  fixable flaw in the method.** Because nearly all pairs are one session,
  "deeper context" and "later in one task" are nearly the same variable. The
  depth curve cannot cleanly attribute degradation to context length until the
  corpus contains deep turns from multiple sessions. F3b treats curve shape as
  an observation partly for this reason.
- **glm-5.2 is the reference, not the truth.** Tier 1 measures
  glm-5.2-likeness. Where glm-5.2 chose a suboptimal action, agreeing with it
  scores as agreement all the same.
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
