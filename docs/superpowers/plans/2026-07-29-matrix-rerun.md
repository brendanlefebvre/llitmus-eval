# Main-Replay Matrix Re-Run (post token-accounting fix)

> **For the implementor:** This is an operational runbook, not a coding plan — no code changes are expected. If anything deviates from an "Expected" line below, STOP and report; do not patch code or thresholds to make the run complete. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Produce matrix numbers for the four Qwen candidates on the post-rethink corpus (`fd044fc`), where every case is inside every candidate's context and no stratum is weighted heavily enough for one case to distort the headline. Supersedes the 2026-07-29 run, whose corpus no longer exists.

**Why it's safe now:** the runner gates every case at the model's resolved native context (`context_length_source: "config"`), so the 14B never processes a >40,960-token prompt again. Worst-case expected peak is ~15 GiB (14B: ~7.8 GiB weights + ~6.5 GiB KV at the 40,960 ceiling) on a 24 GiB machine. The prior panic path (85k-token prompt on the 14B, 21.5+ GiB) is structurally unreachable.

**Where:** `/Users/brendanl/src/llitmus-eval`, branch `token-accounting-context-gate` (or `main` after merge — verify the branch contains commit `fd044fc`).

> **Standing invariant — matrix runs never transit loxo.** Drive candidates directly (mlx_vlm on `:7979` locally, OpenRouter directly for cloud candidates), never through the router on `:9090`. Two reasons, and the second is the load-bearing one:
>
> 1. Router log spam corrupts timing measurements (the original reason for the direct-to-`:7979` convention).
> 2. **Anything transiting loxo is captured, and captures are the corpus.** A matrix run routed through the router writes replay traffic into `~/.local/state/loxo-llm-router/` captures, where the next extraction can pick it up as if it were real agent work — the eval quietly starts measuring its own output. The extractor's allowlist gate (`--reference-gate allowlist`, the default) drops anything the ledger doesn't attribute to `z-ai/glm-5.2`, which covers every candidate except one: a matrix run against **the reference model itself** produces captures the allowlist cannot distinguish from genuine references. That case is only preventable here, by not routing.

**Duration estimate:** ~2.5–3 hours. Every model now scores all 15 cases (nothing errors out early at the gate), and the deep stratum is 5 real cases instead of 1, so the two larger models take longer than on 2026-07-29. Rough shape: 0.6B ~10 min, 1.7B ~15 min, 4B ~40 min, 14B ~90 min.

## Expected outcomes

The corpus (`cases/main_replay.jsonl`, regenerated 2026-07-29) holds 15 cases,
5/5/5 by stratum, max `ref_tokens` **37,574** — inside every candidate's 40,960
window. So unlike the previous run, **no case should error as over-context**.

| model | context (source: config) | scored | errored |
|---|---|---|---|
| `mlx-community/Qwen3-0.6B-4bit` | 40,960 | 15 | none |
| `mlx-community/Qwen3-1.7B-4bit` | 40,960 | 15 | none |
| `mlx-community/Qwen3-4B-4bit` | 40,960 | 15 | none |
| `mlx-community/Qwen3-14B-4bit` | 40,960 | 15 | none |

`depth_weights` are now **0.37 / 0.40 / 0.23** (shallow/mid/deep) — deliberately
flat. On 2026-07-29 deep carried 0.77 with n=1, and a single coin-flip case
produced a 0.71 headline gap between two models whose shallow and mid scores
were identical. No stratum is now dominant enough to do that. Report per-stratum
n beside any weighted headline anyway.

**If a case DOES error as over-context, that is informative, not a failure.**
`ref_tokens` is the canonical render (messages + tools, native). The runner's
actual render can be larger — prompted mode appends the tool schemas as a
system message via `json.dumps(tools, indent=2)`, which can exceed the native
tools encoding. A case at 37,574 `ref_tokens` therefore has ~3,400 tokens of
headroom that template overhead could consume. If that happens, record which
case and its `prompt_tokens_fed`; it means the corpus gate needs a margin below
`fleet_max_context`, not that the run is broken.

**Llama-3.2-1B is no longer in the fleet** (dropped 2026-07-29, commit
`fd044fc`). Its chat template raises `This model only supports single
tool-calls at once!` on any message with parallel `tool_calls`, at render, in
both native and prompted mode — 9 of 15 cases and every deep one. Do not add it
back to compare; it cannot score this corpus.

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

- [ ] **3. Confirm the corpus matches this runbook (guards against silent drift)**

```bash
.venv/bin/python -c "
import json, collections
cs = [json.loads(l) for l in open('cases/main_replay.jsonl')]
m = json.load(open('cases/main_replay.meta.json'))
print('n:', len(cs), '| max ref_tokens:', max(c['ref_tokens'] for c in cs))
print('strata:', dict(collections.Counter(c['depth_stratum'] for c in cs)))
print('edges:', m['stratum_edges'], '| fleet_max:', m['fleet_max_context'])
print('weights:', {k: round(v, 3) for k, v in m['depth_weights'].items()})
print('beyond fleet:', m['routing_population']['beyond_fleet_max_pct'], '%')"
```

Expected: `n: 15`, max `ref_tokens` **37,574**, strata 5/5/5, edges
`[16000, 32000]`, fleet_max `40960`, weights ≈ `0.37/0.40/0.23`, beyond fleet
`83.6%`. Anything else means the corpus was regenerated against different
captures — re-derive the Expected table above before proceeding (the one
sanctioned edit to this document).

## The runs

Logs go to `/tmp`, NOT the repo root (the last matrix left debris that a reviewer had to flag). Sidecars overwrite the tracked `results_main-replay_<label>.json` files in the repo root — that is intended; the pre-fix numbers stay in git history.

- [ ] **4–7. Run each model, smallest first**

One process per model so each gets a clean allocator. Smallest first so a
memory problem surfaces cheaply.

```bash
for m in Qwen3-0.6B-4bit:Qwen3-0.6B Qwen3-1.7B-4bit:Qwen3-1.7B \
         Qwen3-4B-4bit:Qwen3-4B Qwen3-14B-4bit:Qwen3-14B; do
  repo="mlx-community/${m%%:*}"; label="${m#*:}"
  echo "=== $label start $(date '+%H:%M:%S') ==="
  .venv/bin/python litmus_spec.py --profile main-replay \
    --repo "$repo" --label "$label" 2>&1 | tee "/tmp/matrix-$label.log"
  echo "=== $label done $(date '+%H:%M:%S') ==="
done
```

Watch the 14B leg: it peaked at 15.5 GB on 2026-07-29 with 11 cases and now
runs 15, four of them in the 32k–40,960 band. Abort criteria unchanged — if
kernel memory pressure reaches critical (`sysctl -n
kern.memorystatus_vm_pressure_level` returns 4) or reported peak passes ~18 GB,
Ctrl-C and report. Note that `ps` RSS badly understates MLX usage, which lands
in *wired* memory; `vm_stat` wired is the number to watch.

## Verification (all four sidecars)

- [ ] **7. Structural check**

```bash
.venv/bin/python -c "
import json
for label in ('Qwen3-0.6B','Qwen3-1.7B','Qwen3-4B','Qwen3-14B'):
    d = json.load(open(f'results_main-replay_{label}.json'))
    errs = {e['id']: e['error'] for e in d['errored']}
    for m, r in (d.get('modes') or {}).items():
        for e in r.get('errored', []): errs.setdefault(e['id'], e['error'])
    fed = [c['prompt_tokens_fed'] for c in d['cases'] if c.get('prompt_tokens_fed')]
    assert d['context_length'] == 40960, (label, d['context_length'])
    assert d['context_length_source'] == 'config', (label, d['context_length_source'])
    assert all(f <= 40960 for f in fed), (label, max(fed))
    strata = {}
    for c in d['cases']: strata[c['depth_stratum']] = strata.get(c['depth_stratum'], 0) + 1
    print(f\"{label}: scored={len(d['cases'])} errored={sorted(errs)} \"
          f\"max_fed={max(fed) if fed else None} strata={strata} peak={d.get('peak_memory_mb'):.0f}MB\")
    if errs: print(f'   !! {label} errored: {errs}')
"
```

Expected: four lines, each `scored=15 errored=[]` with `strata={'shallow': 5, 'mid': 5, 'deep': 5}` and `max_fed` at or below 40,960. Every assertion here is a hard requirement — an `AssertionError` means stop and report, with the sidecar and the log file for that model.

- [ ] **8. Sanity-read the headlines**

For each model note `action_valid_weighted`, the by-depth breakdown, and errored count from the tee'd logs. Things worth flagging even though step 7 passed: any stratum scoring n<5 (a case errored silently), or a weighted headline whose spread traces to a single case despite the flattened weights — check the per-depth lines before quoting any comparison.

- [ ] **9. Do NOT commit the sidecars — record the findings instead**

`results_*.json` is gitignored (`.gitignore:11`), as are `cases/main_replay.jsonl`
and its `.meta.json` (lines 13–14). Results and corpus are derived artifacts,
regenerable from captures; captures are the source of truth. `git add` on them
fails, and `-f` would be wrong. (An earlier draft of this runbook told you to
commit them — that was incorrect.)

Because nothing lands in git, the run's findings have no durable home unless you
give them one. Write up anything that would otherwise be re-derived next time:

- Per-model headline, per-stratum n, peak memory, wall-clock.
- Any model that errored for a reason **other** than over-context — those are
  compatibility findings, not scores (see the Llama-3.2 note above).
- Whether any weighted headline rested on a stratum with n < 3.

Cross-project lessons belong in `~/src/learnings`
(`~/src/learnings/bin/capture-learning --title "..." --tags "..."`, body on
stdin). Repo-specific outcomes belong in a dated note under `docs/`.

- [ ] **10. Restart the servers if you stopped them in step 2**, and delete the `/tmp/matrix-*.log` files or leave them (they're outside the repo either way).

## Report back

Return: the three headline numbers with by-depth breakdowns, peak memory per model, wall-clock per model, confirmation that step 7 printed three OKs, and the commit SHA. If anything was aborted or deviated, the log path and the last 30 lines of that log.
