# Main-replay matrix — 2026-07-30

First matrix on the post-rethink corpus. Every candidate processed every case
inside its own context window, and no stratum is weighted heavily enough for a
single case to move the headline. Results and corpus are gitignored derived
artifacts, so this note is the durable record.

**Provenance**

| | |
|---|---|
| corpus | `cases/main_replay.jsonl` @ `fd044fc` — 15 cases, 5/5/5, max `ref_tokens` 37,574 |
| strata edges | 16,000 / 32,000 |
| depth weights | shallow 0.370, mid 0.397, deep 0.233 |
| gate | `context_length` 40,960, `context_length_source: config` on all four |
| generation budget | per-case from the capture (32,000), unchanged from prior runs |
| reference tokenizer | `mlx-community/Qwen3-14B-4bit` |
| wall clock | 13:18 → 20:00 (6h42m) |

All four: **15 scored, 0 errored**, `max prompt_tokens_fed` 37,574.

## Headline

| model | no-think | think | gap | tok/case | peak | leg |
|---|---|---|---|---|---|---|
| Qwen3-0.6B-4bit | 0.08 | 0.13 | +0.05 | 6525 → 7098 (1.1×) | 8.5 GB | 1h57m |
| Qwen3-1.7B-4bit | **0.70** | 0.61 | **−0.09** | 2379 → 6979 (2.9×) | 9.0 GB | 1h32m |
| Qwen3-4B-4bit | 0.50 | 0.75 | +0.25 | 387 → 1930 (5.0×) | 8.9 GB | 56m |
| Qwen3-14B-4bit | 0.83 | **0.91** | +0.08 | 1787 → **1525** (0.9×) | 18.3 GB | 2h17m |

Best-mode scores order monotonically with size — 0.13, 0.70, 0.75, 0.91 — which
is the first time this eval has produced a defensible capability gradient. The
2026-07-29 run's apparent "4B beats 14B" was an n=1 artifact; see
`docs/superpowers/plans/2026-07-29-matrix-rerun.md`.

## By depth (`action_valid`, n=5 per cell)

| model | mode | shallow | mid | deep |
|---|---|---|---|---|
| 0.6B | no-think | 0.0 | 0.2 | 0.0 |
| 0.6B | think | 0.0 | 0.2 | 0.2 |
| 1.7B | no-think | 1.0 | 0.6 | **0.4** |
| 1.7B | think | 1.0 | 0.6 | **0.0** |
| 4B | no-think | 0.8 | 0.4 | 0.2 |
| 4B | think | 1.0 | 0.6 | 0.6 |
| 14B | no-think | 1.0 | 0.8 | 0.6 |
| 14B | think | 1.0 | 1.0 | 0.6 |

Deep is where every model runs out of road: nothing exceeds 0.6, including the
14B with thinking. Given that ~84% of real traffic is deeper than any of these
models can even accept, that ceiling is the operationally important number.

## Findings

**1. Qwen3-1.7B's thinking regression is entirely a deep-stratum collapse.**
Shallow and mid are identical between modes (1.0/1.0 and 0.6/0.6); deep goes
0.4 → 0.0. Thinking does not make this model worse in general — it fails
specifically on long context, while costing 2.9× the tokens. Pin it to
no-think, or don't ship it.

**2. Qwen3-14B is the only model where thinking is free.** It gained accuracy
while spending *fewer* tokens (1787 → 1525, 0.9×). Every smaller model paid
1.1–5.0× more for its gains. This is the strongest single argument for the 14B
as the local tier: the usual accuracy-for-tokens tradeoff does not apply.

**3. Qwen3-0.6B is disqualified on latency before accuracy.** It scored 0.0 on
shallow — the easiest stratum — and burned 6,525 tokens/case with thinking
*off*. Several cases ran ~18 minutes; `mr-013` ended in `unclosed-think`
(thinking budget exhausted). Its leg took 1h57m, more than twice the 4B's 56m
despite being 7× smaller. A model that occasionally consumes its entire 32,000
token budget on an ordinary request has a tail-latency problem no mean score
shows.

**4. Model size does not predict runtime.** Ordering by leg duration is 14B
(2h17m) > 0.6B (1h57m) > 1.7B (1h32m) > 4B (56m). Large models are
prefill-bound; small models are degeneration-bound. Do not schedule a matrix
assuming small-first is fast.

## Caveats

- **Peak memory on the 14B was 18,349 MB, above the ~18 GB abort line** in the
  runbook. The run stayed safe — kernel pressure never left normal — but the
  criterion was crossed without the watchdog firing, because the watchdog was
  keyed to pressure level and a 19 GB wired threshold rather than the runner's
  reported peak. Fix before the next run.
- Not comparable to 2026-07-29 on the corpus axis: different cases, different
  strata edges, different weights. The harness axis (runner, gate, scoring,
  generation budget, reference tokenizer) is unchanged, so the two runs share a
  method but not a measurement.
- `action_valid` at n=5 per depth cell still means one case is worth 0.2 within
  a stratum. The weights no longer amplify that to a headline-sized swing, but
  do not read two-model differences under ~0.15 as signal.
