# Decoupling Litmus from MLX: a shared core + pluggable backends

**Status:** Design approved, pending spec review
**Branch:** `refactor/decouple-mlx-backend`
**Date:** 2026-07-24

## Motivation

Litmus is currently Apple-only at the seams that matter, and it carries a
maintenance hazard: `litmus.py` (MLX perf harness, 669 lines) and
`litmus_cuda.py` (torch/CUDA port, 577 lines) are hand-maintained *copies* of
the same six-command harness. They share prompts, the reference text, the
distinct-trigram decode-stability metric, the perplexity algorithm, the
128-token windowing, the report format, and the CLI shape — each reimplemented
per backend. The two drift independently.

Separately, `litmus_common.py` was just made MLX-lazy at import (PR #1), and
`litmus_spec.py` is already backend-agnostic except that its model-running path
imports MLX function-locally. So the runtime coupling to Apple Silicon is now
concentrated in a small, well-understood surface.

This refactor extracts the backend-agnostic core once and puts MLX and torch
behind a single `Backend` protocol that serves **both** the perf harness and
the capability-eval harness. It kills the `litmus.py`/`litmus_cuda.py`
duplication and makes the whole project runnable off Apple Silicon.

## Goals

- One backend-agnostic core hosting all shared logic (metrics, windowing,
  perplexity aggregation, report rendering, run orchestration).
- A thin `Backend` protocol as the single model-runtime seam for the project —
  used by `litmus.py` (perf) and `litmus_spec.py` (evals).
- MLX and torch as thin, lazily-imported adapters.
- No duplicated torch logic: `litmus_cuda.py` becomes a compatibility CLI shim
  over the unified core, not a parallel implementation.
- The deterministic core's behavior is an invariant, protected by byte-strict
  golden tests captured before the refactor.

## Non-goals

- **No new CI in this refactor.** A Linux CI workflow is the natural payoff but
  is explicitly deferred to a later change.
- **No full-model golden output.** Real MLX/torch inference is not
  bit-reproducible across library versions; we do not gate on it.
- **No cross-backend numerical parity guarantee.** Different kernels make exact
  agreement impossible; parity, if observed, is an informal manual sanity check,
  never a test gate.
- No change to which model *repos* work on which backend. The 1-bit Bonsai
  models still need the PrismML MLX fork; that stays the user's concern via
  `--repo`.

## Target architecture

Flat modules, matching the existing convention:

```
litmus_common.py   model tables, target resolution, Backend protocol + get_backend()
litmus_core.py     backend-agnostic perf logic: metrics, windowing, perplexity
                   aggregation, report rendering, per-command orchestration
litmus_mlx.py      MLXBackend    (mlx.core, mlx.nn, mlx_lm)     — lazy import
litmus_torch.py    TorchBackend  (torch, transformers)         — lazy import
litmus.py          modern CLI: --backend mlx|cuda -> litmus_core   (mlx default)
litmus_spec.py     capability evals; model-running path -> Backend (+ --backend)
litmus_cuda.py     compat SHIM: legacy --repo/--quant/--cmd CLI -> litmus_core
                   + TorchBackend. Preserves historical invocations.
```

`a6000_bootstrap.sh` and the results docs reference `python litmus_cuda.py ...`;
the shim keeps those working unchanged.

## The Backend protocol (the crux)

A thin protocol. The core owns every algorithm; the backend owns only what is
genuinely library-specific: loading, token generation, a single forward pass for
log-probs, and memory telemetry.

```python
class Backend(Protocol):
    name: str

    def load(self, repo: str, **opts) -> tuple[model, tokenizer, float]:
        """Return (model, tokenizer, load_seconds).
        MLX: mlx_lm.load(repo).
        Torch: transformers load + quant kwargs (bf16 / nf4 / prequant),
        migrated from litmus_cuda's --quant."""

    def stream(self, model, tokenizer, prompt: str, max_tokens: int) -> Iterator[str]:
        """Yield generated token *texts*. Drives throughput, TTFT,
        decode-stability, and (joined) the spec-check generate_fn.
        MLX: stream_generate + _resp_text. Torch: streamer / manual loop."""

    def token_logprobs(self, model, tokenizer, ids) -> list[float]:
        """THE NUMERIC SEAM. One forward pass over `ids`; return per-token
        log-probabilities as plain Python floats. The backend does the
        library-specific forward + log_softmax + gather; the core owns
        tokenization boundaries, the 128-token windowing, and the
        mean -> exp perplexity aggregation."""

    def peak_memory_mb(self) -> float: ...
    def reset_peak_memory(self) -> None: ...
    def clear_cache(self) -> None: ...
```

Selection lives in `litmus_common`:

```python
def get_backend(name: str = "auto") -> Backend:
    # "mlx" | "cuda" | "auto"; auto = MLX if importable else torch.
    # Imports only the selected backend's module, so importing litmus_common
    # or litmus_core still pulls in neither mlx nor torch.
```

### Why `token_logprobs` is the right seam

Today `compute_perplexity(model, tokenizer, text, window)` bundles two concerns:
the forward pass (backend-specific: `model(x)` -> logits, `log_softmax`,
`take_along_axis`) and the windowing + `mean`/`exp` aggregation (pure). We split
exactly along that line. `token_logprobs` returns floats — a library-neutral
value — so the core's perplexity math is identical for every backend and is
covered by the golden tests. Returning per-token log-probs (rather than a
finished perplexity number) keeps windowing and aggregation in the core where
the golden tests can pin them.

## Module-by-module changes

**`litmus_core.py` (new)** — hosts, moved verbatim from `litmus.py` where
possible to preserve behavior:
- `PROMPTS`, `WARMUP_PROMPT`, `REFERENCE_TEXT_PATH`, `_load_reference_text`
- the distinct-trigram decode-stability metric
- `compute_perplexity(backend, model, tokenizer, text, window)` — windowing +
  aggregation, calling `backend.token_logprobs` for the forward pass
- `run_one` / `bench_model` orchestration, rewritten against a `Backend`
- `print_table` / report rendering (pure)
- per-command drivers for all six commands: throughput, perplexity,
  prefill-scaling, decode-stability, baseline, cold-start. All are
  backend-agnostic once they ride on the protocol.
- `_strip_thinking` (shared with the spec harness's `strip_thinking`; unify or
  cross-import — decided in the plan)

**`litmus_mlx.py` (new)** — `MLXBackend` implementing the protocol. Absorbs the
MLX bodies of `_load_timed`, `stream_generate`+`_resp_text`, the
`compute_perplexity` forward pass, and the `_peak_memory_mb` /
`_reset_peak_memory` / `_clear_cache` helpers currently in `litmus_common`.

**`litmus_torch.py` (new)** — `TorchBackend`. Absorbs `litmus_cuda.py`'s torch
bodies: transformers load with `bf16`/`nf4`/`prequant`, the streaming/generate
loop, the `torch.log_softmax`+`torch.gather` forward, and
`torch.cuda.max_memory_allocated` / `reset_peak_memory_stats` / `empty_cache`.

**`litmus.py`** — shrinks to a thin CLI: parse args, `get_backend(args.backend)`
(default `mlx`), dispatch to `litmus_core`. Keeps its existing six subcommands.

**`litmus_spec.py`** — the model-running path swaps its function-local
`_mlx_generate` / `_targets_for` / `_load_timed` for the `Backend`:
`generate_fn = lambda p: "".join(backend.stream(model, tokenizer, p, budget))`,
loading via `backend.load`. Add `--backend`. The pure loader/parser/scorer code
is untouched.

**`litmus_cuda.py`** — reduced from a 577-line implementation to a thin shim:
keep its `--repo/--quant/--cmd/--max-tokens/--chat` argparse surface, translate
to `litmus_core` + `TorchBackend`. Historical invocations still run; no edits to
`a6000_bootstrap.sh` or results docs.

**`litmus_common.py`** — gains the `Backend` protocol and `get_backend`. The
MLX-specific memory helpers move to `litmus_mlx.py`; `litmus_common` keeps the
pure tables/targeting. (The `_mlx()` lazy shim from PR #1 is superseded by the
backend modules and can be retired once nothing imports the re-exports directly.)

**`pyproject.toml`** — extras: `mlx = ["mlx-lm"]`, `cuda = ["torch",
"transformers", "accelerate", "bitsandbytes", "hf_transfer"]`, `dev =
["pytest>=8.0"]`. Add `litmus_core`, `litmus_mlx`, `litmus_torch` to
`py-modules`.

## Testing strategy

Strict at the deterministic seam, loose at the stochastic one.

1. **Golden / characterization tests on `litmus_core` (byte-strict).** Before
   extraction, capture current outputs of the pure functions on fixed synthetic
   inputs: the distinct-trigram metric, perplexity windowing + aggregation fed
   synthetic per-token log-probs, and `print_table` rendering. After extraction,
   assert byte-identical. This is the guarantee that protects the published
   `results-bonsai-1bit.md` numbers. No model required; fully deterministic.

2. **`FakeBackend`.** A deterministic in-memory `Backend` (canned `stream`
   output, canned `token_logprobs`, zero memory). Lets the run orchestration and
   all six command drivers be unit-tested end to end with no real model or GPU.

3. **Backend primitives — smoke tests.** Per backend, `pytest.importorskip` the
   dependency, then assert shapes and plausible values (load returns a
   tokenizer, `stream` yields non-empty strings, `token_logprobs` returns the
   right count of finite floats). Not golden snapshots.

The existing spec-check suite continues to pass unchanged (its pure logic is
untouched); the `FakeBackend` also lets its runners be exercised backend-free.

## Phased execution (one spec, staged)

- **A. Extract deterministic core + golden tests.** Move shared logic into
  `litmus_core.py`; capture golden tests first. MLX-only, no behavior change,
  no protocol yet — `litmus.py` calls the core directly.
- **B. Introduce `Backend` + `MLXBackend`.** Define the protocol in
  `litmus_common`, implement `litmus_mlx.py`, route `litmus.py` through
  `get_backend`. Add `FakeBackend` and orchestration tests.
- **C. `TorchBackend` + shim.** Fold `litmus_cuda.py`'s torch logic into
  `litmus_torch.py`; reduce `litmus_cuda.py` to a compat shim. Torch smoke tests.
- **D. Route spec-check through `Backend`.** Swap `litmus_spec.py`'s
  model-running path; add `--backend` to both CLIs.
- **E. Packaging.** Extras and `py-modules` updates.

Each phase keeps the full suite green. CI is deferred (would be a later Phase F).

## Risks & open items

- **`_strip_thinking` duplication.** It exists in both `litmus.py` and
  `litmus_spec.py`. Unifying into the core is desirable but touches thinking-mode
  behavior that was carefully tuned in PR #1; the plan decides whether to unify
  now or leave both and only share going forward.
- **Torch `stream` fidelity.** `litmus_cuda.py` currently uses greedy decode to
  match `mlx_lm`'s default. The `TorchBackend.stream` must preserve that; smoke
  tests check shape, not exact tokens.
- **Backend-specific commands.** `baseline` references MLX-community 4-bit
  repos; the command logic is backend-agnostic but the default model list is
  MLX-flavored. The plan decides whether `baseline`'s defaults are
  backend-parameterized or left MLX-default with `--repo` override.
- **PR #1 interaction.** This branch is stacked on `feat/spec-check-harness`
  (PR #1, unmerged). The `_mlx()` lazy shim from PR #1 is superseded here; retire
  it cleanly to avoid two lazy-import mechanisms coexisting.
