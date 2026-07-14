# Spec-check harness — design

**Date:** 2026-07-13
**Branch:** `feat/spec-check-harness`
**Status:** approved design, pre-implementation

## Purpose

Add two model-level capability evals to Litmus — **tool-calling** and
**instruction/constraint-following** — built on a single unified **spec-check
harness**. Both are intrinsic, fixed-input, and objectively checkable (no LLM
judge, no reference model), and both produce per-task-type capability numbers
that feed Loxo's router.

A case = `{prompt, inputs, validators}`. The harness runs the model, runs the
validators against the output, and aggregates pass rates. Tool-calling and
constraint-following are two case files + two validator families feeding the
same runner — one mechanism, two capability profiles.

**Explicitly out of scope (deferred):** agentic, multi-turn tool *use* (call a
tool, read the result, decide the next action, complete a task). That grades the
model+loop system, not the model, and needs an agent harness + task environment
— a separate, larger tool. This harness characterizes the model only.

## Coherence check every metric must pass

Intrinsic, fixed-input, per-task-type model capability, AND it gives Loxo's
router something to route on. Both profiles pass: a profile like "model X: valid
tool calls 95%, correct abstention 70%, constraints 92% strict" is exactly a
per-task-type routing input.

## Architecture & module layout

```
litmus_common.py   # extracted shared core (new)
  - _load_timed, _targets_for, _parse_sizes, _resp_text
  - _peak_memory_mb / _reset_peak_memory / _clear_cache
  - MODELS / BASELINE_MODELS constants
litmus.py          # unchanged behavior; imports from litmus_common
litmus_spec.py     # the spec-check harness (new)
cases/
  tool_calling.jsonl     # Litmus-authored tool cases
  constraints.jsonl      # Litmus-authored constraint cases
  # ifeval_subset.jsonl  # vendored later, same loader
tests/                   # first test suite in the repo (pytest)
```

**One harness, two profiles.** `litmus_spec.py` exposes a single case-runner:
load model → for each case, elicit output → run the case's validators →
aggregate. Invoked as `python litmus_spec.py --profile tool-calling` and
`--profile constraints`. Shares `--repo` / `--sizes` / `--label` / `--max-tokens`
with litmus.py's conventions.

The only change to existing code is a mechanical extraction of shared helpers
into `litmus_common.py`; `litmus.py` keeps working identically by importing from
it.

## Case & validator data model

**Case file format: JSONL** (one case per line). Chosen over YAML to keep the
repo zero-runtime-dependency (no PyYAML) and because JSONL is IFEval's native
format — the vendored IFEval subset drops in later with no format translation.

**A constraint case:**
```json
{"id": "constr-007", "prompt": "List exactly three primary colors, all lowercase.",
 "checks": [{"kind": "exact_bullets", "n": 3}, {"kind": "all_lowercase"}]}
```

**A tool-calling case** carries tool schemas + the expected outcome (`tools` holds
one JSON-Schema object per available tool, elided here for brevity):
```jsonc
{"id": "tool-012", "prompt": "What's the weather in Paris?",
 "tools": [ /* get_weather schema */, /* send_email schema */ ],
 "expect": {"tool": "get_weather", "arguments": {"location": "Paris"}}}
```
Abstention case: `"expect": {"tool": null}`.

**Validator registry.** A module-level dict `CHECKS = {"exact_bullets": fn, ...}`.
Each validator has signature `(output_text, params) -> CheckResult(passed: bool,
detail: str)`. A case's `checks` list is resolved against the registry at load
time. This string-keyed registry is what lets the vendored IFEval
`instruction_id_list` map onto the same machinery later.

**Loader** validates every case on load: required fields present, every `kind`
known, params type-checked against each validator's declared param spec. An
unknown `kind` or malformed case is a **hard load error** — the run stops before
any model loads. No silent skips.

## Profile 1 — tool-calling

Both conventions are measured for every model (the "both columns" decision).

### Elicitation — two paths per case

1. **Prompted-JSON (every model).** A fixed system preamble lists the tool
   schemas and instructs: *emit exactly one JSON object
   `{"tool": <name|null>, "arguments": {...}}` and nothing else.* Wrapped in the
   model's chat template with **no** `tools=`. This is the apples-to-apples
   baseline and the only honest path for models without native tool support.
2. **Native `tools=` (only if supported).** Support is detected by probing
   `tokenizer.apply_chat_template(..., tools=[...])` once per model at load and
   caching the result. If it raises or the template ignores `tools`, native is
   marked unavailable for that model. When available, the same schemas are passed
   via `tools=`.

### Measurement philosophy (why this stays a model eval)

A convention is not a thumb on the scale — it is the measurement apparatus, the
same way perplexity fixes Gatsby and decode-stability fixes the chat template.
The threat to significance is not "flattering weak models" (prompted-JSON is the
*only* real deployment path for a no-native model, so measuring it measures
reality). The real threat is the opposite: forcing a *natively*-trained model
through a prompted-only path can measure it below its ceiling and even invert a
ranking. The mitigation is to **not collapse** to one number: report both
columns, labeled by convention, plus the gap. Numbers are always reported as
"capability under a named convention," never as convention-free.

### Parsing (known-risk surface)

There is no universal native tool-call output parser (Qwen emits
`<tool_call>…</tool_call>`, Hermes/Llama differ). This is an accepted risk,
handled explicitly rather than glossed:

- **Prompted:** extract the first balanced `{...}` JSON object from the output
  (tolerate markdown fences / leading prose), `json.loads` it. Failure →
  `well_formed = False`.
- **Native:** best-effort multi-format parser trying, in order: Qwen/Hermes
  `<tool_call>…</tool_call>`, Llama `<|python_tag|>`, then a generic
  balanced-JSON fallback. Native function-calling has no explicit "no-call"
  token, so `ParsedCall.attempted` distinguishes "a tool-call structure was
  present" (a `<tool_call>`/`<|python_tag|>` tag matched, or the generic
  fallback actually parsed an object) from a genuine no-call — prose with no
  structure at all. A native abstention is credited only when nothing was
  attempted; a structure that matched but whose inner JSON didn't parse (e.g.
  a broken/doubled-brace `<tool_call>`) is scored as an attempted, failed
  call, not a free abstention. `native_parse_failed` counts these
  attempted-but-unparseable native calls — distinct from `well_formed=False`
  and from `abstained` — so the native column's parser reliability is
  auditable independent of expect.

### Four dimensions, scored independently per case

| dimension | definition |
|---|---|
| `well_formed` | output parses to a valid call (or a clean "no call") |
| `right_tool` | selected tool == `expect.tool` (including correctly selecting *no* tool) |
| `args_ok` | required args present, values match `expect.arguments`, no hallucinated keys, right types |
| `abstained_ok` | on `expect.tool == null` cases, the model correctly made no call |

Aggregate = pass rate per dimension, computed separately for the prompted and
native columns, plus the **gap** (e.g. `prompted.right_tool − native.right_tool`).
`args_ok` is only evaluated when `right_tool` passed (wrong tool ⇒ args moot;
reported N/A, not counted against the args rate).

## Profile 2 — instruction/constraint-following

**Elicitation:** one path — prompt wrapped in the chat template, generate, run
validators on the output. No convention fork (constraints are about the prose,
not a call format).

### v0 validator vocabulary (all pure regex/parser, no LLM judge)

Drawn from IFEval's verifiable-constraint categories:

| `kind` | params | check |
|---|---|---|
| `exact_bullets` | `n` | exactly n markdown/`-`/numbered list items |
| `min_words` / `max_words` | `n` | word-count bound |
| `all_lowercase` / `all_uppercase` | — | casing |
| `forbidden_word` | `word` | word absent (case-insensitive) |
| `required_phrase` | `phrase` | phrase present |
| `ends_with` | `phrase` | trimmed output ends with phrase |
| `valid_json` | — | whole output parses as JSON |
| `regex_match` | `pattern` | catch-all for one-off constraints |

### Scoring (IFEval's two numbers)

- **strict** — a case passes only if *all* its checks pass; reported as the
  fraction of cases fully satisfied.
- **loose** — per-check pass rate across all checks in the run.

Both reported side by side, as IFEval does.

## Case sourcing

- **v0:** Litmus-authored cases in `cases/tool_calling.jsonl` and
  `cases/constraints.jsonl` — fast, proves the harness, objective but not
  IFEval-comparable.
- **Later:** vendor a curated subset of real IFEval prompts (Apache-2.0) as
  `cases/ifeval_subset.jsonl`, consumed by the *same* loader and validator
  registry, giving an externally-comparable anchor. The registry and check
  semantics are proven against our own cases first.

## Results output

**Human table** (mirrors litmus.py's `print_table` style).

Tool-calling — one row per model per convention:
```
model              conv       well  right  args  abst    gap(right)
Qwen2.5-3B         prompted   0.95  0.88   0.81  0.70
                   native     0.98  0.94   0.90  0.75    -0.06
Bonsai-4B-1bit     prompted   0.70  0.55   0.40  0.45    (no native)
```

Constraints — one row per model with `strict` / `loose`, plus a per-`kind`
breakdown table.

**Machine-readable sidecar** — `results_<profile>_<label>.json`, the Loxo-facing
artifact. Retains full per-case detail, not just aggregates, so a failure can be
inspected without re-running (same principle as decode-stability dumping full
output to a temp file):
```json
{"profile": "tool-calling", "model": "prism-ml/Bonsai-4B-mlx-1bit",
 "label": "Bonsai-4B-1bit",
 "convention_support": {"prompted": true, "native": false},
 "aggregate": {"prompted": {"well_formed": 0.70, "right_tool": 0.55,
                            "args_ok": 0.40, "abstained_ok": 0.45}},
 "n_cases": 24,
 "cases": [ {"id": "tool-012", "prompted": {"...per-dimension..."}, "native": null} ]}
```

The two profiles do *not* share a per-case record shape: constraint records are
`{id, checks, output_sample}` and tool-calling records are `{id, prompted,
native, prompted_output, native_output}`. What's shared is the output seam —
`write_sidecar` takes each profile's `result["cases"]` opaquely and writes it
generically, so the sidecar and table formatters don't need a common record
shape to stay one code path at that boundary.

## Error handling & edge cases

- **Malformed case file** → hard error at load, before any model loads.
- **Generation exception** on one case → record the case as `errored` with the
  message, continue the run; errored cases are excluded from rate denominators
  and counted separately (a run that errored 5/24 cases says so loudly rather
  than reporting a silently-deflated rate).
- **Native parse failure** → its own `native_parse_failed` tally, distinct from
  `well_formed=False` and from `abstained`.
- **Empty / truncated output** (model hit max-tokens mid-call) → `well_formed =
  False` with detail `"truncated"`. `--max-tokens` is tunable per profile with
  sane defaults (tool calls short, constraint prose longer).
- **`tools=` probe** done once per model at load, cached — not per case.

## Testing

The whole value proposition is objective, trustworthy checks, so the harness
ships with the repo's first test suite (`pytest`, tests under `tests/`).

- **Validators** are pure functions `(text, params) -> CheckResult` → unit-tested
  directly with crafted strings, no model needed. Each gets positive + negative +
  boundary cases (e.g. `exact_bullets(3)` passes on 3, fails on 2 and 4, handles
  numbered vs dash vs `*`).
- **Parsers** (prompted-JSON extractor, native multi-format) → unit-tested against
  captured example outputs including adversarial ones: fenced JSON,
  JSON-with-preamble, `<tool_call>`-wrapped, malformed.
- **Loader** → tests for unknown `kind`, missing fields, bad param types all
  raising at load.
- **Runner** → tested with a tiny fake model stub (returns canned strings) so
  scoring/aggregation is verified without loading MLX weights.

## Open items intentionally deferred

- Vendoring the real IFEval subset (follow-up, same loader).
- Native `tools=` output-format coverage beyond Qwen/Hermes/Llama + generic JSON
  fallback (extend the multi-format parser as new families are tested).
- Agentic multi-turn tool use (out of scope entirely; separate tool).
