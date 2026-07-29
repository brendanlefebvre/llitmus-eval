# Token accounting and context gate — design

**Status:** reviewed; open questions resolved 2026-07-29 (reference tokenizer =
`Qwen3-14B-4bit`, `ref_tokens` rename kept). The tokenizer choice was first
resolved to `Qwen3.6-35B-A3B-4bit` as "the router's actual local target", then
re-resolved same day after checking the running deployment: the backend
actually serves `Qwen3-14B-4bit` (the 35B's ~18.7 GiB resident footprint can't
run always-on within 24 GiB; see `~/bin/mlx-vlm-serve.sh`).
**Date:** 2026-07-29
**Repos affected:** `llitmus-eval`, `loxo-llm-router`

## Background

A Qwen3-14B main-replay run kernel-panicked the Mac at 17:11:48 on 2026-07-28
(`watchdog timeout: no checkins from watchdogd in 93 seconds`). Sizing showed
case mr-013 required 21.55 GiB on a 24 GiB machine. The memory blowup was the
symptom; two upstream defects made it inevitable:

1. `estimate_prompt_tokens` (chars//4, text content only) understated real
   tokenized length by up to 1.95x — mr-012 was recorded as 40,027 tokens and
   is really 77,908.
2. The runner's over-context guard checked `tokenizer.model_max_length`.
   `Qwen3-14B-4bit` ships that as 131,072 — a real value (the guard's 10^9
   sentinel check was armed and comparing), but it is the *tokenizer's*
   YaRN-extended maximum, not the model's native 40,960 context from
   `config.json`. Every deep case (51k–85k tokens) passed a 131,072 gate and
   was scored as if the results meant something.

Verified context lengths:

| model | real context | role |
|---|---|---|
| `Qwen3.6-35B-A3B-4bit` | **262,144** (nested in `text_config`) | configured in loxo.toml, but NOT served — manual-start only (memory) |
| `Llama-3.2-1B-Instruct-4bit` | 131,072 | eval candidate |
| `Qwen3-4B-4bit` | 40,960 | eval candidate |
| `Qwen3-14B-4bit` | 40,960 | eval candidate; what the 7979 backend actually serves (`LOCAL_MODELS` env, verified 2026-07-29) |

`LOCAL_CONTEXT_LIMIT = 60000` was arbitrary and doing three unrelated jobs:
routing threshold, advertised `context_length` in `/v1/models`, and the eval's
corpus in-scope gate.

## Section 1 — Shared context resolver (llitmus-eval)

New in `litmus_common.py`:

```python
resolve_context_length(repo: str) -> int | None
    1. config.json from the HF cache (lazy import, keeps litmus_common
       dependency-free at import time, per its module docstring)
    2. max_position_embeddings
    3. text_config.max_position_embeddings      # Qwen3.6, gemma-4, Qwen3-VL
    4. None  -> caller decides the fallback
```

Returning `None` rather than a default is deliberate: a silent default is what
produced this bug. Callers must handle absence explicitly.

## Section 2 — Exact counting in the extractor

In `scripts/extract_main_replay.py`, `est_tokens` (chars//4) is replaced by an
exact count of the *rendered* prompt.

The subtlety: the runner's render varies by model and mode (native tools vs.
prompted, thinking on/off), but a stratum is a property of the **corpus**, not
of a candidate. So the extractor uses a fixed **reference tokenizer** (new
`--tokenizer`, default `mlx-community/Qwen3-14B-4bit`) and a canonical render.
The reference is the model that actually serves the router's local traffic
(verified against the deployment env 2026-07-29) — strata are meant to
describe the routing population, and that population is served by the 14B.
`Qwen3.6-35B-A3B-4bit` was considered (it's the loxo.toml-configured target)
and rejected: the deployment doesn't serve it, and can't always-on within
24 GiB. Per-model reality is already captured separately by
`prompt_tokens_fed`. If the served local target ever changes, the reference
tokenizer (and the router divisor calibrated against it) should be revisited.

The canonical render must live as an importable shared function (not inline in
the extractor), because the drift check below has to reproduce it exactly.

Field renamed `est_tokens` → `ref_tokens`. This is a schema break:

- `cases/main_replay.jsonl` must be regenerated; `litmus_spec.py:305` and its
  loader tests updated.
- `tests/test_f1_reference_validation.py:278` (`test_est_tokens_drift_check`)
  currently recomputes `estimate_prompt_tokens(body)` per case and compares.
  Reproducing `ref_tokens` instead requires the canonical render plus a loaded
  reference tokenizer — the test becomes HF-cache-dependent and must skip
  loudly (per-case, as it does for missing captures) when the tokenizer isn't
  cached. This is the churniest single test change in the plan.

The rename is worth it (`est_` would now be a lie), but the drift-check churn
is the real cost.

**Cross-repo dependency end-state:** the extractor imports `classify`,
`estimate_prompt_tokens`, and `LOCAL_CONTEXT_LIMIT` from `loxo_llm_router`
(`scripts/extract_main_replay.py:28-31`). The estimator and limit imports go
away; `classify` stays — llitmus-eval remains coupled to the router package
for capture classification. That coupling is intended, not an oversight.

Corpus in-scope gate becomes **fleet max** via `resolve_context_length` over the
candidate set (131,072 today), replacing `LOCAL_CONTEXT_LIMIT` — including the
`limit=LOCAL_CONTEXT_LIMIT` default on `assign_stratum`
(`scripts/extract_main_replay.py:320`). It doesn't bind today — nothing exceeds
131,072 — but it's now principled rather than arbitrary. `meta.json` gains
`tokenizer`, `fleet_max_context`, and `over_limit` (a flat count — anything
over the fleet max is past the deep edge by definition, so a per-stratum
breakdown would be vacuous) so a snapshot records how it was counted.

Strata edges stay 16,000 / 40,000. They're now exact-token edges instead of
chars/4 edges, so the population shifts — `depth_weights` already recompute from
each walk (commit `5ab12bf`), so that self-corrects.

## Section 3 — Per-model gate in the runner

`litmus_spec.py:1110` currently checks `tokenizer.model_max_length`, which for
Qwen3 is the tokenizer's 131,072 YaRN maximum rather than the model's native
40,960 — so the gate compares against the wrong ceiling. Replace with
`resolve_context_length(repo)`, falling back to `model_max_length` when `None`.
Gate behaviour is unchanged — over-context cases land in `errored` — and the
existing tests at `tests/test_spec_replay.py:967-999` already cover that path.
The sidecar records which context length was used.

For Qwen3-14B this errors all 5 deep cases as over-context rather than scoring
RoPE-extrapolated garbage, and the headline reports reduced coverage instead of
a fake number.

## Section 4 — Router (loxo-llm-router)

**Estimator fix.** Count every message field, not just `content` text —
`tool_calls`, `reasoning_content`, `tool_call_id` were 126,183 chars (87% of
counted) on mr-012 alone. Divisor lowered from 4 so it **never underestimates**
across the 15-case corpus. The value 3.5 was calibrated against
`Qwen3-14B-4bit` counts (worst under +0.2%, worst over +22.1%) on
2026-07-29 by `scripts/calibrate_router_divisor.py` — the floor of
`min(chars/ref_tokens)` over the corpus, so `chars/divisor >= ref_tokens`
everywhere. (The provisional 3.6 underestimated mr-008 and mr-013, where the
corpus's chars/token ratio dipped to 3.51; the standing property test in
`tests/test_router_divisor_property.py` fails loudly if the pinned value ever
underestimates again.) Deliberately conservative, because the failure modes
are asymmetric: undercount → an over-long prompt hits a local model → garbage
or a crash; overcount → goes to cloud → costs money but works.

**Derived threshold.** Probe `{local_base_url}/v1/models` once at startup,
cached, for the target's `context_length`/`max_model_len`. `local_context_limit`
becomes an explicit override *and* the fallback when the probe fails or the
field is absent. No new dependency (httpx is already there), and it stays
backend-agnostic. A down local server must not block requests — fall back and
move on.

**`/v1/models` truthfulness.** Local-pinned tiers advertise
`effective_local_context()` (explicit > probe > legacy default) instead of the
raw constant. Deployment reality check (2026-07-29): the env file already pins
`LOCAL_CONTEXT_LIMIT=32768` explicitly (below the served 14B's native 40,960,
per the serve script's guidance), so the running instance advertises a
truthful-conservative 32,768 today. The fix matters for default-config
deployments, where the advertised number was the arbitrary 60,000 that commit
`7e4842c` called "rate-card-truthful".

## Section 5 — Testing & risks

New tests: resolver (top-level / nested / missing); extractor count matches the
runner's `prompt_tokens_fed` on a fixture; runner gate over/at/None-context;
router estimator regression on a `tool_calls`+`reasoning_content` capture;
divisor-never-underestimates property over the corpus; probe success /
missing-field / unreachable.

Existing-test migrations (router side, not just additions):

- `test_routing.py:103` hard-asserts chars//4 arithmetic
  (`"a" * 40` → 10 tokens); the 3.6 divisor breaks it, as it does any test
  doing /4 math (e.g. the image-part test at `test_routing.py:118`).
- `LOCAL_CONTEXT_LIMIT` is monkeypatched in ~8 places across
  `test_routing.py` and `test_metadata.py`. Once the threshold is
  probe-derived, patching the constant no longer controls routing unless the
  probe cache is also patched/disabled — otherwise those tests silently stop
  exercising what they think they exercise. Each site needs an explicit
  decision: patch the derived value, or force the fallback path.

Risks: the reference tokenizer must be cached (fail loudly, don't silently fall
back); the `est_tokens` rename breaks existing case files. The probe question
is resolved (2026-07-29, server verified): mlx_lm's `/v1/models` reports
neither `context_length` nor `max_model_len`, so the probe derives nothing on
the current backend and the explicit 32,768 governs. The probe stays in scope
anyway — it is cheap and self-activates if the backend is ever swapped for one
that reports context (e.g. vLLM's `max_model_len`).

## Resolved decisions (2026-07-29)

- `est_tokens` → `ref_tokens` rename: **keep**. The drift-check churn
  (Section 2) is accepted; keeping `est_` on an exact count would be
  misleading.
- Reference tokenizer: **`Qwen3-14B-4bit`** — re-resolved 2026-07-29 after
  verifying the deployment. First resolution picked `Qwen3.6-35B-A3B-4bit` as
  "the router's actual local target"; checking the running stack showed that
  premise false (the backend serves the 14B; the 35B is manual-start only on
  this 24 GiB machine). Strata describe the routing population, and the
  routing population is served by the 14B. Consequence: the provisional 3.6
  divisor was revised to 3.5 by calibration against this tokenizer — the corpus's
  chars/token ratio dipped to 3.51 on mr-008 and mr-013, so the
  `min(chars/ref_tokens)` floor landed at 3.5 to keep the estimator from
  underestimating those cases.
- Context probe: **kept in scope** despite deriving nothing from the current
  mlx_lm backend (verified: its `/v1/models` carries no context field) —
  explicit config wins today; the probe self-activates on a future backend
  that reports context.
