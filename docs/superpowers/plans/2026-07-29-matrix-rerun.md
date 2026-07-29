# Main-Replay Matrix Re-Run (post token-accounting fix)

> **For the implementor:** This is an operational runbook, not a coding plan — no code changes are expected. If anything deviates from an "Expected" line below, STOP and report; do not patch code or thresholds to make the run complete. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Produce honest matrix numbers for the three candidates on the regenerated exact-count corpus, replacing the pre-fix sidecars whose deep-case scores were RoPE-extrapolated garbage (and whose last run kernel-panicked the machine on 2026-07-28).

**Why it's safe now:** the runner gates every case at the model's resolved native context (`context_length_source: "config"`), so the 14B never processes a >40,960-token prompt again. Worst-case expected peak is ~15 GiB (14B: ~7.8 GiB weights + ~6.5 GiB KV at the 40,960 ceiling) on a 24 GiB machine. The prior panic path (85k-token prompt on the 14B, 21.5+ GiB) is structurally unreachable.

**Where:** `/Users/brendanl/src/llitmus-eval`, branch `token-accounting-context-gate` (or `main` after merge — verify the branch contains commit `9f45112`).

**Duration estimate:** tens of minutes total. Llama-1B is the slow one now (it legitimately processes prompts up to 106k tokens); the Qwens got faster because their four biggest cases error out in milliseconds at the gate.

## Expected outcomes (verify against these, computed 2026-07-29 from the regenerated corpus)

| model | context (source: config) | cases scored | cases errored (over-context) |
|---|---|---|---|
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | 131,072 | 15 | none |
| `mlx-community/Qwen3-4B-4bit` | 40,960 | 11 | exactly `mr-011, mr-012, mr-013, mr-014` |
| `mlx-community/Qwen3-14B-4bit` | 40,960 | 11 | exactly `mr-011, mr-012, mr-013, mr-014` |

Corpus: 15 cases, 5/5/5 by stratum; one deep case (≤40,960 ref tokens) survives the Qwen gate, so deep coverage on the Qwens is reduced (n=1), not vacated. Depth weights come from `cases/main_replay.meta.json` (deep ≈ 0.77) — the headline `action_valid_weighted` will lean heavily on whatever deep evidence exists. For the Qwens that is 1 case; treat their headline accordingly (real, but thin).

## Pre-flight

- [ ] **1. Right branch, clean tree, suite green**

```bash
cd /Users/brendanl/src/llitmus-eval
git status --short            # expect: clean (untracked scratch is OK, tracked files unmodified)
git log --oneline -1          # expect: 9f45112 or a descendant
.venv/bin/python -m pytest tests/ -q   # expect: all pass; drift + divisor-property tests PASS (not skip)
```

If the drift check or `tests/test_router_divisor_property.py` SKIPS, the reference tokenizer isn't cached — stop and fix that first (`hf download mlx-community/Qwen3-14B-4bit`); do not run the matrix on an unverified corpus.

- [ ] **2. Free the memory the run needs**

The eval loads models in-process via mlx — the local servers are not involved, but a resident mlx server model would compete for RAM. Check and stop:

```bash
pgrep -fl "uvicorn|mlx_lm|mlx-lm" || echo "nothing running"
# If the 7979 backend or the 9090 router is up, stop them for the duration:
pkill -f "mlx-lm-launch.py" 2>/dev/null; pkill -f "uvicorn.*loxo_llm_router" 2>/dev/null
```

Close other heavy apps (browsers with many tabs, IDEs you aren't using). Restart the servers afterwards with `~/bin/mlx-vlm-serve.sh` and `/Users/brendanl/src/loxo-llm-router/llm-router-serve.sh` if needed.

- [ ] **3. Precompute the expected errored set (guards against silent corpus drift)**

```bash
.venv/bin/python -c "
import json
cases = [json.loads(l) for l in open('cases/main_replay.jsonl')]
print(sorted(c['id'] for c in cases if c['ref_tokens'] > 40960))"
```

Expected: `['mr-011', 'mr-012', 'mr-013', 'mr-014']`. If it prints anything else, the corpus changed since this runbook was written — update the Expected table above from the new output before proceeding (that is the one sanctioned edit to this document).

## The runs — one process per model, smallest first

Logs go to `/tmp`, NOT the repo root (the last matrix left debris that a reviewer had to flag). Sidecars overwrite the tracked `results_main-replay_<label>.json` files in the repo root — that is intended; the pre-fix numbers stay in git history.

- [ ] **4. Llama-3.2-1B** (the long one — up to 106k-token prompts, expect minutes per deep case)

```bash
.venv/bin/python litmus_spec.py --profile main-replay \
  --repo mlx-community/Llama-3.2-1B-Instruct-4bit --label Llama-3.2-1B \
  2>&1 | tee /tmp/matrix-llama1b.log
```

- [ ] **5. Qwen3-4B**

```bash
.venv/bin/python litmus_spec.py --profile main-replay \
  --repo mlx-community/Qwen3-4B-4bit --label Qwen3-4B \
  2>&1 | tee /tmp/matrix-qwen4b.log
```

- [ ] **6. Qwen3-14B** (the one that panicked the machine pre-fix — watch it)

Keep Activity Monitor (or `memory_pressure` in a second terminal) visible. Abort criteria: if memory pressure goes red or the peak reported per case climbs past ~18 GiB, Ctrl-C the run and report — do not ride it out.

```bash
.venv/bin/python litmus_spec.py --profile main-replay \
  --repo mlx-community/Qwen3-14B-4bit --label Qwen3-14B \
  2>&1 | tee /tmp/matrix-qwen14b.log
```

The per-case progress line (`[ n/15] mr-0xx ...`) shows the four over-context cases erroring instantly with `prompt exceeds model context: <n> > 40960 (config)` — that is the fix working, not a failure.

## Verification (all three sidecars)

- [ ] **7. Structural check**

```bash
.venv/bin/python -c "
import json
expected_err = {'mr-011', 'mr-012', 'mr-013', 'mr-014'}
for label, ctx, err in [('Llama-3.2-1B', 131072, set()),
                        ('Qwen3-4B', 40960, expected_err),
                        ('Qwen3-14B', 40960, expected_err)]:
    d = json.load(open(f'results_main-replay_{label}.json'))
    got_err = {e['id'] for e in d['errored']}
    modes = d.get('modes') or {}
    for m, r in modes.items():
        got_err |= {e['id'] for e in r.get('errored', [])}
    assert d['context_length'] == ctx, (label, d['context_length'])
    assert d['context_length_source'] == 'config', (label, d['context_length_source'])
    assert got_err == err, (label, sorted(got_err))
    fed = [c['prompt_tokens_fed'] for c in d['cases'] if c.get('prompt_tokens_fed')]
    assert all(f <= ctx for f in fed), (label, max(fed))
    print(f'{label}: OK  scored={len(d[\"cases\"])}  errored={sorted(got_err)}  peak={d.get(\"peak_memory_mb\")}')"
```

Expected: three `OK` lines matching the table in this document. Every assertion here is a hard requirement — an `AssertionError` means stop and report, with the sidecar and the log file for that model.

- [ ] **8. Sanity-read the headlines**

For each model note `action_valid_weighted`, the by-depth breakdown, and errored count from the tee'd logs. Two things that would be *wrong* and worth flagging even though step 7 passed: a Qwen model scoring deep cases at n>1 (gate not applied per-mode?), or Llama-1B with any errored case.

- [ ] **9. Commit the sidecars**

```bash
git add results_main-replay_Llama-3.2-1B.json results_main-replay_Qwen3-4B.json results_main-replay_Qwen3-14B.json
git commit -m "results: matrix re-run on exact-count corpus with native-context gate

Llama-1B scores all 15 cases (131,072 context); Qwen3-4B/14B score 11
with mr-011..mr-014 errored as over-context (40,960) instead of being
scored beyond their window. First matrix where the deep-stratum numbers
are real.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **10. Restart the servers if you stopped them in step 2**, and delete the `/tmp/matrix-*.log` files or leave them (they're outside the repo either way).

## Report back

Return: the three headline numbers with by-depth breakdowns, peak memory per model, wall-clock per model, confirmation that step 7 printed three OKs, and the commit SHA. If anything was aborted or deviated, the log path and the last 30 lines of that log.
