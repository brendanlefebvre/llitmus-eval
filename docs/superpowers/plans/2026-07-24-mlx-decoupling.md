# Decoupling Litmus from MLX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract Litmus's backend-agnostic perf/eval logic into a shared core behind a thin `Backend` protocol, so MLX and torch become interchangeable lazily-imported adapters and the whole project runs off Apple Silicon.

**Architecture:** A new `litmus_core.py` owns every algorithm (metrics, windowing, perplexity aggregation, report rendering, the six perf-command drivers). A `Backend` protocol in `litmus_common.py` is the single model-runtime seam — `load`, `stream`, `token_logprobs`, memory telemetry — implemented by `litmus_mlx.py` (MLXBackend) and `litmus_torch.py` (TorchBackend). `litmus.py` and `litmus_cuda.py` shrink to thin CLIs that pick a backend and dispatch to the core; `litmus_spec.py`'s model-running path swaps its MLX-local helpers for the protocol.

**Tech Stack:** Python 3.11+, `mlx` / `mlx_lm` (Apple), `torch` / `transformers` (CUDA), `pytest>=8.0`. Standard-library `typing.Protocol` for the seam.

## Global Constraints

- **Core stays dependency-free at import.** After Phase B, `import litmus_core` and `import litmus_common` must pull in **neither** `mlx` nor `torch`. Backends import their library lazily (inside `load`/method bodies), never at module top level. Enforced by `tests/test_spec_no_mlx_import.py`-style import guards.
- **The deterministic core's behavior is an invariant.** Pure functions (`_strip_thinking`, `distinct_trigram_ratio`, perplexity aggregation, `print_table`, `_load_reference_text`) must remain byte-identical to their pre-refactor behavior, pinned by golden tests captured in Phase A before any orchestration is rewritten.
- **`token_logprobs` is the numeric seam.** It returns per-token log-probabilities as plain Python `float`s: for input `ids`, the log-prob of `ids[i]` given `ids[:i]` for `i` in `1..len(ids)-1` (so `len(ids) - 1` floats). The core owns tokenization, windowing (`ids[:window]`), and the `mean → exp` perplexity aggregation. Backends own only the forward pass + `log_softmax` + gather.
- **`Backend.stream` yields per-token text.** One `str` per generated token; `"".join(...)` reconstructs the generation. The core counts tokens by counting yields.
- **No new CI.** A Linux CI workflow is deferred (would be a later Phase F). Not in this plan.
- **No full-model golden output; no cross-backend numeric-parity gate.** Backend primitives get shape/plausibility smoke tests only (`pytest.importorskip`). Cross-backend agreement, if observed, is an informal manual check.
- **`assisted` stays torch-native.** The speculative-decoding command has no MLX equivalent; it remains in the torch shim, outside the protocol.
- **`_strip_thinking` stays two distinct functions.** `litmus_core._strip_thinking(text, tokenizer) -> tuple[str, int]` (scratchpad token count, with the untagged-preamble heuristic) and `litmus_spec.strip_thinking(text) -> tuple[str, bool]` (closed-properly, tag-only) are intentionally different contracts. Do **not** unify.
- **`baseline` defaults stay MLX-flavored** (`BASELINE_MODELS` = mlx-community 4-bit repos), overridable via `--repo`. Not backend-parameterized.
- **Historical torch invocations preserved.** `litmus_cuda.py`'s `--repo/--quant/--cmd/--max-tokens/--chat/--assistant-repo` argparse surface and command set (`throughput`, `perplexity`, `decode-stability`, `assisted`) stay working; `a6000_bootstrap.sh` and results docs are not edited.
- **Every phase keeps the full `pytest` suite green.** Run `pytest -q` at each task's final step.

---

## File Structure

| Module | Responsibility (end state) |
|---|---|
| `litmus_common.py` | Model tables (`MODELS`, `BASELINE_MODELS`, `PROMPTS`, `WARMUP_PROMPT`), size parsing / target resolution, the `Backend` protocol + `get_backend()`. No MLX/torch at import. The `_mlx()` PEP-562 shim is retired once nothing imports the re-exports. |
| `litmus_core.py` (new) | Backend-agnostic perf logic: `REFERENCE_TEXT_PATH` + `_load_reference_text`, preamble regexes + `_strip_thinking`, `distinct_trigram_ratio`, unified `Run` dataclass, `compute_perplexity(backend, …)`, `run_one` / `bench_model`, `print_table`, `_single_ttft`, and the six `cmd_*` drivers. No MLX/torch at import. |
| `litmus_mlx.py` (new) | `MLXBackend` — lazy `mlx.core` / `mlx.nn` / `mlx_lm`. Absorbs `_resp_text`, the memory helpers, `_load_timed`, `stream_generate` wiring, and the MLX perplexity forward pass. |
| `litmus_torch.py` (new) | `TorchBackend` — lazy `torch` / `transformers`. Absorbs `litmus_cuda.py`'s quant load, greedy stream, torch forward pass, cuda memory helpers. Also hosts the torch-native `assisted` implementation. |
| `litmus.py` | Thin MLX-default CLI: parse args, `get_backend(args.backend)`, dispatch to `litmus_core`. Keeps its six subcommands + adds `--backend`. |
| `litmus_cuda.py` | Thin compat shim: keeps its argparse surface; `throughput`/`perplexity`/`decode-stability` → `litmus_core` + `TorchBackend`; `assisted` → `litmus_torch.cmd_assisted`. |
| `litmus_spec.py` | Model-running path uses `get_backend` + `backend.load` / `backend.stream`; adds `--backend`. Pure loader/parser/scorer untouched. |
| `pyproject.toml` | Extras `mlx` / `cuda` / `dev`; `py-modules` gains `litmus_core`, `litmus_mlx`, `litmus_torch`. |
| `tests/conftest.py` (new) | `FakeBackend` + stub-tokenizer fixtures shared across orchestration tests. |

---

## Phase A — Pure core + golden tests (MLX-free)

Extract only the functions with **zero runtime MLX dependency** into `litmus_core.py`, pin them with golden tests that run on any machine. `compute_perplexity`, `run_one`, `bench_model`, and the `cmd_*` drivers stay in `litmus.py` for now (they still call `mx`/`stream_generate`); they move in Phase B once the backend seam exists.

### Task 1: Create `litmus_core.py` with the pure helpers

**Files:**
- Create: `litmus_core.py`
- Modify: `litmus.py:53-70` (imports), and delete the moved definitions at `litmus.py:60-153` and `litmus.py:243-298`
- Test: `tests/test_core_pure.py` (added in A2)

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `REFERENCE_TEXT_PATH: str`
  - `_load_reference_text(path: str = REFERENCE_TEXT_PATH) -> str`
  - `PREAMBLE_RE`, `PARENTHETICAL_PREFIX_RE` (compiled `re.Pattern`)
  - `_strip_thinking(text: str, tokenizer) -> tuple[str, int]`
  - `distinct_trigram_ratio(tokens: list[int]) -> float`
  - `Run` dataclass (fields below)
  - `print_table(all_runs: list[Run]) -> None`

The unified `Run` renames `litmus.py`'s `size` field to `label` (backend-neutral; the value is unchanged — `litmus.py` already passes the size string as the label). Torch callers pass `useful_*` as `None`.

```python
@dataclass
class Run:
    label: str
    prompt: str
    prompt_tokens: int
    gen_tokens: int
    prefill_tps: float
    decode_tps: float
    ttft_ms: float
    peak_mem_mb: float
    sample: str
    useful_gen_tokens: Optional[int] = None
    useful_decode_tps: Optional[float] = None
```

- [ ] **Step 1: Create `litmus_core.py` with the module docstring and imports**

```python
"""Backend-agnostic core for Litmus.

Hosts every algorithm shared across backends and CLIs: reference-text loading,
reasoning-preamble stripping, the distinct-trigram decode-stability metric,
perplexity windowing + aggregation, the Run record, report rendering, and the
per-command perf drivers. Imports neither mlx nor torch — the model runtime is
reached only through a Backend (see litmus_common.get_backend).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

REFERENCE_TEXT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "reference.txt"
)

# Matches common reasoning-preamble openers seen in Bonsai 8B output.
PREAMBLE_RE = re.compile(
    r"^(Okay|Alright|Let me|So,|First,|I need to|The user|Looking at)"
)
# Matches an optional parenthetical prefix that sometimes precedes the
# preamble opener, e.g. "(150-200 words) Okay, so I need to..."
PARENTHETICAL_PREFIX_RE = re.compile(r"^\([^)]*\)\s*")
```

- [ ] **Step 2: Move `_load_reference_text` verbatim**

Move `litmus.py:88-112` (`_load_reference_text`) into `litmus_core.py` unchanged.

- [ ] **Step 3: Move `_strip_thinking` verbatim**

Move `litmus.py:115-153` (`_strip_thinking`, docstring included) into `litmus_core.py` unchanged. It already depends only on the two regexes (now in core) and `tokenizer.encode`.

- [ ] **Step 4: Add `distinct_trigram_ratio` (extracted from `cmd_decode_stability`)**

The metric is currently inline at `litmus.py:467-472`. Extract it exactly:

```python
def distinct_trigram_ratio(tokens: list[int]) -> float:
    """Fraction of distinct token trigrams (1.0 = no repetition).

    Returns nan for sequences shorter than 3 tokens. Extracted verbatim from
    the decode-stability loop so both backends and the golden tests share it.
    """
    if len(tokens) < 3:
        return float("nan")
    trigrams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
    return len(set(trigrams)) / len(trigrams)
```

- [ ] **Step 5: Move the `Run` dataclass, renaming `size` → `label`**

Move `litmus.py:73-85` into `litmus_core.py`, changing the first field from `size: str` to `label: str`. Keep all other fields (including `useful_gen_tokens`, `useful_decode_tps`) identical.

- [ ] **Step 6: Move `print_table`, renaming `size`→`label` internally**

Move `litmus.py:243-298` (`print_table`) into `litmus_core.py`. Replace every `r.size` with `r.label` and rename the local `by_size` dict/loop var to `by_label` (behavior identical — same grouping, same output bytes for the same label values). The column header text (`'size'`) stays as-is to preserve output format.

- [ ] **Step 7: Rewire `litmus.py` to import from core**

In `litmus.py`, delete the now-moved definitions (`Run` at 73-85, `_load_reference_text` at 88-112, `_strip_thinking` at 115-153, `print_table` at 243-298, and the two regexes at 64-70). Replace the `litmus_common` import block (`litmus.py:53-58`) and add a core import:

```python
from litmus_common import (
    MODELS, BASELINE_MODELS, PROMPTS, WARMUP_PROMPT,
    _resp_text, _peak_memory_mb, _reset_peak_memory, _clear_cache,
    _parse_sizes, _targets_for, _load_timed,
    mx, nn, load, stream_generate,
)
from litmus_core import (
    REFERENCE_TEXT_PATH, _load_reference_text, _strip_thinking,
    distinct_trigram_ratio, Run, print_table,
)
```

Delete `litmus.py`'s own `REFERENCE_TEXT_PATH` (now imported). In `cmd_decode_stability` replace the inline trigram block (`litmus.py:467-472`) with `distinct = distinct_trigram_ratio(tokens)`. In `run_one`/`cmd_baseline`, the `Run(size=...)` constructions become `Run(label=...)` (occurrences at `litmus.py:197-209`).

- [ ] **Step 8: Run the suite to confirm no regression**

Run: `pytest -q`
Expected: PASS (same count as before this task; on a non-MLX machine `litmus.py` still won't import, but the spec-check suite and any core tests do).

- [ ] **Step 9: Commit**

```bash
git add litmus_core.py litmus.py
git commit -m "refactor(core): extract MLX-free pure helpers into litmus_core"
```

### Task 2: Golden tests for the pure core

**Files:**
- Test: `tests/test_core_pure.py` (create)

**Interfaces:**
- Consumes: `litmus_core.{_strip_thinking, distinct_trigram_ratio, print_table, Run, _load_reference_text}`.
- Produces: nothing (test-only). Establishes the byte-strict invariant.

These run with no MLX and no model — `_strip_thinking` uses a stub tokenizer whose `encode` splits on whitespace, giving deterministic token counts.

- [ ] **Step 1: Write the golden tests**

```python
"""Byte-strict golden tests for litmus_core's pure functions.

Captured before the orchestration is rewritten against the Backend protocol.
These pin the deterministic core's behavior — the guarantee that protects the
published results-bonsai-1bit.md numbers. No MLX, no model.
"""
import math

import litmus_core as core


class StubTokenizer:
    """Whitespace tokenizer: deterministic token counts with no model."""
    def encode(self, text: str) -> list[int]:
        return [len(w) for w in text.split()]


def test_strip_thinking_closed_tag():
    tok = StubTokenizer()
    text = "<think>reasoning here</think>The answer is 42."
    useful, scratch = core._strip_thinking(text, tok)
    assert useful == "The answer is 42."
    assert scratch == len(tok.encode("<think>reasoning here</think>"))


def test_strip_thinking_unclosed_tag_is_all_scratchpad():
    tok = StubTokenizer()
    text = "<think>never closed"
    useful, scratch = core._strip_thinking(text, tok)
    assert useful == ""
    assert scratch == len(tok.encode(text))


def test_strip_thinking_untagged_preamble_heuristic():
    tok = StubTokenizer()
    text = "Okay, let me think.\n\nThe final answer."
    useful, scratch = core._strip_thinking(text, tok)
    assert useful == "The final answer."
    assert scratch == len(tok.encode("Okay, let me think.\n\n"))


def test_strip_thinking_parenthetical_prefix():
    tok = StubTokenizer()
    text = "(150-200 words) Okay, so.\n\nAnswer body."
    useful, scratch = core._strip_thinking(text, tok)
    assert useful == "Answer body."


def test_strip_thinking_no_preamble_returns_unchanged():
    tok = StubTokenizer()
    text = "A direct answer with no reasoning."
    useful, scratch = core._strip_thinking(text, tok)
    assert useful == text
    assert scratch == 0


def test_distinct_trigram_ratio_all_distinct():
    assert core.distinct_trigram_ratio([1, 2, 3, 4, 5]) == 1.0


def test_distinct_trigram_ratio_all_repeated():
    # tokens [7,7,7,7] -> trigrams (7,7,7),(7,7,7): 1 distinct / 2 = 0.5
    assert core.distinct_trigram_ratio([7, 7, 7, 7]) == 0.5


def test_distinct_trigram_ratio_too_short_is_nan():
    assert math.isnan(core.distinct_trigram_ratio([1, 2]))


def test_print_table_golden(capsys):
    runs = [
        core.Run(
            label="1.7B", prompt="Explain quantum computing", prompt_tokens=5,
            gen_tokens=64, prefill_tps=120.0, decode_tps=45.0, ttft_ms=42.0,
            peak_mem_mb=512.0, sample="Quantum computing uses qubits",
        ),
    ]
    core.print_table(runs)
    out = capsys.readouterr().out
    # Pin the structural invariants of the rendered table.
    assert "prefill t/s" in out
    assert "1.7B" in out
    assert "--- Per-size summary ---" in out
    assert "45.0" in out  # decode t/s rendered at one decimal


def test_load_reference_text_strips_gutenberg_markers(tmp_path):
    p = tmp_path / "ref.txt"
    p.write_text(
        "header junk\n*** START OF THE BOOK ***\nReal body text.\n"
        "*** END OF THE BOOK ***\nfooter junk",
        encoding="utf-8",
    )
    assert core._load_reference_text(str(p)) == "Real body text."
```

- [ ] **Step 2: Run the golden tests**

Run: `pytest tests/test_core_pure.py -v`
Expected: PASS (all cases green — this is a characterization capture, so they pass against the just-moved code).

- [ ] **Step 3: Commit**

```bash
git add tests/test_core_pure.py
git commit -m "test(core): byte-strict golden tests for pure helpers"
```

---

## Phase B — `Backend` protocol + `MLXBackend` + orchestration move

Define the seam, implement MLX behind it, move `compute_perplexity`/`run_one`/`bench_model`/`cmd_*` into the core rewritten against a `Backend`, and shrink `litmus.py` to a thin CLI. After this phase, `litmus_core` and `litmus_common` are MLX-free at import.

### Task 3: Define the `Backend` protocol + `get_backend` in `litmus_common`

**Files:**
- Modify: `litmus_common.py` (add protocol + selector near top, after the tables)
- Test: `tests/test_backend_protocol.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `class Backend(Protocol)` with `name: str`, `load`, `stream`, `token_logprobs`, `peak_memory_mb`, `reset_peak_memory`, `clear_cache`.
  - `get_backend(name: str = "auto") -> Backend`.

- [ ] **Step 1: Write the failing test**

```python
import importlib
import sys

import pytest


def test_get_backend_unknown_name_raises():
    import litmus_common
    with pytest.raises((ValueError, SystemExit)):
        litmus_common.get_backend("nonsense")


def test_importing_common_pulls_in_neither_mlx_nor_torch():
    # Fresh import must not eagerly import the heavy libs.
    for mod in ("mlx", "mlx.core", "torch", "litmus_common"):
        sys.modules.pop(mod, None)
    importlib.import_module("litmus_common")
    assert "mlx" not in sys.modules
    assert "torch" not in sys.modules


def test_get_backend_mlx_is_lazy(monkeypatch):
    # Selecting mlx imports litmus_mlx but the call itself must not fail on a
    # non-Apple box until a model op runs; construction alone is cheap.
    import litmus_common
    be = litmus_common.get_backend("mlx")
    assert be.name == "mlx"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_backend_protocol.py -v`
Expected: FAIL with `AttributeError: module 'litmus_common' has no attribute 'get_backend'`.

- [ ] **Step 3: Add the protocol + selector to `litmus_common.py`**

Insert after the `WARMUP_PROMPT` definition (`litmus_common.py:42`):

```python
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """The single model-runtime seam for Litmus.

    The core owns every algorithm; a backend owns only the library-specific
    parts: loading, token generation, one forward pass for log-probs, and
    memory telemetry.
    """
    name: str

    def load(self, repo: str, **opts) -> tuple[object, object, float]:
        """Return (model, tokenizer, load_seconds)."""

    def stream(self, model, tokenizer, prompt: str,
               max_tokens: int) -> Iterator[str]:
        """Yield one generated token's text per step."""

    def token_logprobs(self, model, tokenizer, ids: list[int]) -> list[float]:
        """Per-token log-probs as floats: logprob of ids[i] given ids[:i],
        for i in 1..len(ids)-1. Returns len(ids)-1 values."""

    def peak_memory_mb(self) -> float: ...
    def reset_peak_memory(self) -> None: ...
    def clear_cache(self) -> None: ...


def get_backend(name: str = "auto") -> "Backend":
    """Resolve a backend by name: 'mlx' | 'cuda' | 'auto'.

    'auto' picks MLX if importable, else torch. Imports only the selected
    backend's module, so importing litmus_common/litmus_core pulls in neither
    mlx nor torch.
    """
    if name == "auto":
        try:
            import mlx.core  # noqa: F401
            name = "mlx"
        except ImportError:
            name = "cuda"
    if name == "mlx":
        from litmus_mlx import MLXBackend
        return MLXBackend()
    if name == "cuda":
        from litmus_torch import TorchBackend
        return TorchBackend()
    raise SystemExit(f"unknown backend {name!r}; pick from mlx | cuda | auto")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_backend_protocol.py::test_get_backend_unknown_name_raises tests/test_backend_protocol.py::test_importing_common_pulls_in_neither_mlx_nor_torch -v`
Expected: PASS. (`test_get_backend_mlx_is_lazy` needs `litmus_mlx` from B2 — expect it to error until then.)

- [ ] **Step 5: Commit**

```bash
git add litmus_common.py tests/test_backend_protocol.py
git commit -m "feat(backend): define Backend protocol + get_backend selector"
```

### Task 4: Implement `MLXBackend` in `litmus_mlx.py`

**Files:**
- Create: `litmus_mlx.py`
- Test: `tests/test_backend_protocol.py::test_get_backend_mlx_is_lazy` (already written)

**Interfaces:**
- Consumes: `litmus_common.Backend` (structural — no import needed).
- Produces: `class MLXBackend` implementing the protocol, `name = "mlx"`.

Absorbs the MLX bodies currently in `litmus_common.py` (`_resp_text` at 81-83, `_peak_memory_mb` 86-93, `_reset_peak_memory` 96-102, `_clear_cache` 105-111, `_load_timed` 135-139) and `litmus.py`'s perplexity forward pass (`compute_perplexity` body at 319-330, minus the aggregation).

- [ ] **Step 1: Create `litmus_mlx.py`**

```python
"""MLXBackend: the Apple-Silicon adapter for the Litmus Backend protocol.

mlx.core / mlx.nn / mlx_lm are imported lazily inside method bodies so that
importing this module (e.g. via get_backend) does not require MLX until a
model op actually runs.
"""
from __future__ import annotations

import time
import warnings
from typing import Iterator

warnings.filterwarnings(
    "ignore",
    message=r".*mx\.metal\.(clear_cache|get_peak_memory|reset_peak_memory).*deprecated.*",
)


def _resp_text(resp) -> str:
    """stream_generate yields GenerationResponse in newer mlx_lm, str in older."""
    return resp.text if hasattr(resp, "text") else str(resp)


class MLXBackend:
    name = "mlx"

    def load(self, repo: str, **opts) -> tuple[object, object, float]:
        from mlx_lm import load
        t0 = time.perf_counter()
        model, tokenizer = load(repo)
        return model, tokenizer, time.perf_counter() - t0

    def stream(self, model, tokenizer, prompt: str,
               max_tokens: int) -> Iterator[str]:
        from mlx_lm import stream_generate
        for resp in stream_generate(model, tokenizer, prompt,
                                    max_tokens=max_tokens):
            yield _resp_text(resp)

    def token_logprobs(self, model, tokenizer, ids: list[int]) -> list[float]:
        import mlx.core as mx
        import mlx.nn as nn
        x = mx.array(ids)[None, :]                       # (1, T)
        logits = model(x)                                # (1, T, V)
        log_probs = nn.log_softmax(logits[:, :-1, :], axis=-1)
        targets = x[:, 1:, None]                         # (1, T-1, 1)
        gathered = mx.take_along_axis(log_probs, targets, axis=-1).squeeze(-1)
        return [float(v) for v in gathered[0].tolist()]

    def peak_memory_mb(self) -> float:
        import mlx.core as mx
        if hasattr(mx, "get_peak_memory"):
            return mx.get_peak_memory() / (1024 * 1024)
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
            return mx.metal.get_peak_memory() / (1024 * 1024)
        return 0.0

    def reset_peak_memory(self) -> None:
        import mlx.core as mx
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
            mx.metal.reset_peak_memory()

    def clear_cache(self) -> None:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
```

- [ ] **Step 2: Run the lazy-construction test**

Run: `pytest tests/test_backend_protocol.py::test_get_backend_mlx_is_lazy -v`
Expected: PASS (constructing `MLXBackend()` imports no MLX; `name == "mlx"`).

- [ ] **Step 3: Confirm import-purity still holds**

Run: `pytest tests/test_backend_protocol.py::test_importing_common_pulls_in_neither_mlx_nor_torch -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add litmus_mlx.py
git commit -m "feat(backend): MLXBackend adapter with lazy mlx imports"
```

### Task 5: Add `FakeBackend` + move `compute_perplexity` into core (split at the seam)

**Files:**
- Create: `tests/conftest.py`
- Modify: `litmus_core.py` (add `compute_perplexity`)
- Test: `tests/test_core_perplexity.py` (create)

**Interfaces:**
- Consumes: `litmus_core.Run`.
- Produces:
  - `tests/conftest.py`: `FakeBackend` class + `fake_backend` fixture.
  - `litmus_core.compute_perplexity(backend, model, tokenizer, text: str, window: int) -> float`.

- [ ] **Step 1: Create `tests/conftest.py` with `FakeBackend`**

```python
"""Shared deterministic test doubles for backend-driven core tests."""
import math

import pytest


class FakeBackend:
    """A deterministic in-memory Backend: no model, no GPU.

    - stream(): yields the words of a canned string, space-suffixed, one 'token'
      at a time (so "".join reconstructs the canned text).
    - token_logprobs(): a fixed value per position, so perplexity is exactly
      exp(-value).
    - memory: always zero.
    """
    name = "fake"

    def __init__(self, canned_text="hello world answer", per_token_logprob=-1.5,
                 encode=None):
        self.canned_text = canned_text
        self.per_token_logprob = per_token_logprob
        self._encode = encode or (lambda s: list(range(len(s.split()))))

    def load(self, repo, **opts):
        return object(), _StubTokenizer(self._encode), 0.0

    def stream(self, model, tokenizer, prompt, max_tokens):
        words = self.canned_text.split()
        for w in words[:max_tokens]:
            yield w + " "

    def token_logprobs(self, model, tokenizer, ids):
        return [self.per_token_logprob] * (len(ids) - 1)

    def peak_memory_mb(self):
        return 0.0

    def reset_peak_memory(self):
        pass

    def clear_cache(self):
        pass


class _StubTokenizer:
    def __init__(self, encode):
        self._encode = encode

    def encode(self, text):
        return self._encode(text)

    def decode(self, ids, **kw):
        return " ".join(str(i) for i in ids)


@pytest.fixture
def fake_backend():
    return FakeBackend()
```

- [ ] **Step 2: Write the failing perplexity test**

```python
import math

import litmus_core as core


def test_compute_perplexity_matches_exp_mean_nll(fake_backend):
    # token_logprobs returns -1.5 per position -> mean NLL = 1.5 -> ppl = e^1.5
    class Tok:
        def encode(self, text):
            return list(range(10))
    ppl = core.compute_perplexity(fake_backend, object(), Tok(),
                                  "any text", window=8)
    assert math.isclose(ppl, math.exp(1.5), rel_tol=1e-12)


def test_compute_perplexity_short_input_is_nan(fake_backend):
    class Tok:
        def encode(self, text):
            return [1]
    ppl = core.compute_perplexity(fake_backend, object(), Tok(),
                                  "x", window=8)
    assert math.isnan(ppl)
```

- [ ] **Step 3: Run to verify it fails**

Run: `pytest tests/test_core_perplexity.py -v`
Expected: FAIL with `AttributeError: module 'litmus_core' has no attribute 'compute_perplexity'`.

- [ ] **Step 4: Add `compute_perplexity` to `litmus_core.py`**

```python
import math


def compute_perplexity(backend, model, tokenizer, text: str,
                       window: int) -> float:
    """Teacher-forced perplexity over the first `window` tokens of `text`.

    Returns exp(mean NLL); lower is better. The core owns tokenization,
    windowing, and aggregation; the backend owns the forward pass via
    token_logprobs.
    """
    ids = tokenizer.encode(text)[:window]
    if len(ids) < 2:
        return float("nan")
    logprobs = backend.token_logprobs(model, tokenizer, ids)
    mean_nll = -sum(logprobs) / len(logprobs)
    return math.exp(mean_nll)
```

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/test_core_perplexity.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py litmus_core.py tests/test_core_perplexity.py
git commit -m "feat(core): backend-driven compute_perplexity + FakeBackend"
```

### Task 6: Move `run_one` / `bench_model` / `_single_ttft` and the six `cmd_*` drivers into core

**Files:**
- Modify: `litmus_core.py` (add orchestration + drivers), `litmus.py` (delete moved code, become thin CLI)
- Test: `tests/test_core_orchestration.py` (create)

**Interfaces:**
- Consumes: `litmus_core.{Run, print_table, compute_perplexity, distinct_trigram_ratio, _strip_thinking, _load_reference_text, REFERENCE_TEXT_PATH}`, `litmus_common.{PROMPTS, WARMUP_PROMPT, BASELINE_MODELS, _targets_for}`, a `Backend`.
- Produces (all take a leading `backend` parameter):
  - `run_one(backend, model, tokenizer, prompt, max_tokens, label, strip_thinking=False) -> Run`
  - `bench_model(backend, label, repo, max_tokens, strip_thinking) -> list[Run]`
  - `_single_ttft(backend, model, tokenizer, prompt) -> float`
  - `cmd_throughput(backend, args)`, `cmd_perplexity(backend, args)`, `cmd_prefill_scaling(backend, args)`, `cmd_decode_stability(backend, args)`, `cmd_baseline(backend, args)`, `cmd_cold_start(backend, args)`
  - `COMMANDS: dict[str, callable]` mapping the six names to the drivers.

Rewrite rules when moving each function from `litmus.py`:
- `stream_generate(model, tokenizer, p, max_tokens=n)` → `backend.stream(model, tokenizer, p, n)` (iterates plain `str` chunks; drop the `_resp_text(resp)` wrapper — `backend.stream` already yields text, so `chunks.append(resp)`).
- `_reset_peak_memory()` → `backend.reset_peak_memory()`; `_peak_memory_mb()` → `backend.peak_memory_mb()`; `_clear_cache()` → `backend.clear_cache()`.
- `_load_timed(repo)` → `backend.load(repo)`.
- `compute_perplexity(model, tokenizer, text, window)` → `compute_perplexity(backend, model, tokenizer, text, window)`.
- `run_one(..., size, ...)` param renamed to `label`; `Run(size=...)` → `Run(label=...)`.
- `_single_ttft` loops one token via `backend.stream(model, tokenizer, prompt, 1)`.

- [ ] **Step 1: Write the failing orchestration test**

```python
import types

import litmus_core as core
from conftest import FakeBackend


def test_run_one_counts_tokens_and_builds_run():
    be = FakeBackend(canned_text="one two three four")

    class Tok:
        def encode(self, text):
            return text.split()

    r = core.run_one(be, object(), Tok(), "prompt here", max_tokens=4,
                     label="fake")
    assert isinstance(r, core.Run)
    assert r.label == "fake"
    assert r.gen_tokens == 4              # four canned words yielded
    assert r.sample.startswith("one two three four")
    assert r.peak_mem_mb == 0.0


def test_cmd_perplexity_runs_end_to_end(capsys, monkeypatch, tmp_path):
    be = FakeBackend(per_token_logprob=-2.0)
    ref = tmp_path / "ref.txt"
    ref.write_text("word " * 50, encoding="utf-8")
    args = types.SimpleNamespace(
        reference_text=str(ref), ppl_window=16, repo="some/repo",
        label="fake", sizes="1.7B",
    )
    # _targets_for reads args.repo -> single (label, repo)
    core.cmd_perplexity(be, args)
    out = capsys.readouterr().out
    assert "perplexity" in out.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_core_orchestration.py -v`
Expected: FAIL with `AttributeError: module 'litmus_core' has no attribute 'run_one'`.

- [ ] **Step 3: Move + rewrite the orchestration and drivers into `litmus_core.py`**

Move these from `litmus.py`, applying the rewrite rules above, and add the imports at the top of `litmus_core.py`:

```python
import gc
import tempfile
import time

from litmus_common import (
    BASELINE_MODELS, PROMPTS, WARMUP_PROMPT, _targets_for,
)
```

Functions to move (source → core, with the mechanical rewrites):
- `run_one` (`litmus.py:160-209`) — add leading `backend` param, `size`→`label`, `_reset_peak_memory()`→`backend.reset_peak_memory()`, stream/text as above, `_peak_memory_mb()`→`backend.peak_memory_mb()`, `_strip_thinking` already in core.
- `bench_model` (`litmus.py:212-240`) — add `backend`; `_load_timed`→`backend.load`; warmup + `run_one` calls thread `backend`; `_clear_cache()`→`backend.clear_cache()`.
- `cmd_throughput` (`litmus.py:301-307`), `cmd_perplexity` (`333-353`), `cmd_prefill_scaling` (`360-401`), `cmd_decode_stability` (`408-523`), `cmd_baseline` (`530-570`), `cmd_cold_start` (`584-598`) and `_single_ttft` (`577-581`) — each gains a leading `backend` param; apply the same substitutions; in `cmd_decode_stability` the inline trigram block is already replaced by `distinct_trigram_ratio(tokens)`.
- Add the `COMMANDS` dict (mirrors `litmus.py:605-612`) mapping names → the core drivers.

- [ ] **Step 4: Reduce `litmus.py` to a thin CLI**

Replace `litmus.py`'s body so it only parses args (adding `--backend`), resolves a backend, and dispatches. The whole file becomes:

```python
"""Litmus - a local-LLM eval suite (perf: throughput, perplexity, decode
stability, prefill scaling, baseline, cold-start).

Backend-agnostic: defaults to MLX on Apple Silicon; pass --backend cuda for
the torch path (see litmus_cuda.py for the historical CUDA CLI). See
litmus_core for the shared algorithms and litmus_common for the Backend seam.
"""
from __future__ import annotations

import argparse

from litmus_common import get_backend
from litmus_core import COMMANDS, REFERENCE_TEXT_PATH


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backend", choices=["mlx", "cuda", "auto"], default="mlx",
                    help="model runtime (default: mlx)")
    ap.add_argument("--cmd", choices=list(COMMANDS), default="throughput",
                    help="which benchmark to run (default: throughput)")
    ap.add_argument("--sizes", default="1.7B,4B,8B",
                    help="comma-separated subset of 1.7B,4B,8B")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--strip-thinking", action="store_true",
                    help="(throughput) also report 'useful' tok/s after "
                         "stripping reasoning preamble")
    ap.add_argument("--ppl-window", type=int, default=1024,
                    help="(perplexity) max tokens of reference text to score")
    ap.add_argument("--reference-text", default=REFERENCE_TEXT_PATH,
                    help=f"path to reference text (default: {REFERENCE_TEXT_PATH})")
    ap.add_argument("--repo", default=None,
                    help="HF repo id to benchmark instead of a Bonsai size")
    ap.add_argument("--label", default=None,
                    help="display label for --repo")
    ap.add_argument("--chat", action="store_true",
                    help="(decode-stability) wrap prompt in the chat template")
    args = ap.parse_args()

    backend = get_backend(args.backend)
    COMMANDS[args.cmd](backend, args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the orchestration tests + full suite**

Run: `pytest tests/test_core_orchestration.py -v && pytest -q`
Expected: PASS. Core drivers now exercised end-to-end via `FakeBackend`, MLX-free.

- [ ] **Step 6: Confirm core is MLX-free at import**

Run: `python -c "import sys, litmus_core, litmus_common; assert 'mlx' not in sys.modules and 'torch' not in sys.modules; print('clean')"`
Expected: `clean`.

- [ ] **Step 7: Commit**

```bash
git add litmus_core.py litmus.py tests/test_core_orchestration.py
git commit -m "refactor(core): move perf orchestration + drivers behind Backend; thin litmus.py CLI"
```

### Task 7: Retire the `litmus_common` re-exports and MLX memory helpers

**Files:**
- Modify: `litmus_common.py` (delete `_mlx()`, `__getattr__`, `_LAZY_NAMES`, `_resp_text`, memory helpers, `_load_timed`), `tests/test_common_import.py` (update expectations)

**Interfaces:**
- Consumes: nothing.
- Produces: `litmus_common` with only tables, `_parse_sizes`, `_targets_for`, `Backend`, `get_backend`. The MLX helpers now live in `litmus_mlx.py`.

- [ ] **Step 1: Grep for remaining importers of the re-exports**

Run: `grep -rn "from litmus_common import\|litmus_common\.\(mx\|nn\|load\|stream_generate\|_resp_text\|_peak_memory\|_reset_peak\|_clear_cache\|_load_timed\)" --include=*.py .`
Expected: only `litmus_spec.py` still references `_load_timed` / `stream_generate` / `_resp_text` (fixed in Phase D) and `litmus.py` no longer does. If `litmus_spec.py` is the sole remaining user, defer its cleanup to Phase D and keep this task limited to what's unused.

- [ ] **Step 2: Delete the superseded MLX plumbing from `litmus_common.py`**

Remove `_MLX`, `_mlx()`, `_LAZY_NAMES`, `__getattr__` (`litmus_common.py:49-74`), `_resp_text` (81-83), `_peak_memory_mb`/`_reset_peak_memory`/`_clear_cache` (86-111), `_load_timed` (135-139), and the now-moot `warnings.filterwarnings` block (11-14, moved to `litmus_mlx.py`). Keep `MODELS`, `BASELINE_MODELS`, `PROMPTS`, `WARMUP_PROMPT`, `_parse_sizes`, `_targets_for`, `Backend`, `get_backend`.

> If Step 1 shows `litmus_spec.py` still imports `_load_timed`/`stream_generate`, leave **only** those two names alive here as thin lazy shims until Phase D, then delete them in Task D1. Do not break `litmus_spec.py` mid-plan.

- [ ] **Step 3: Update `tests/test_common_import.py`**

Adjust the import-purity test to assert `get_backend` exists and that `import litmus_common` still triggers no `mlx`/`torch` import. Remove any assertion that the old re-export names (`mx`, `stream_generate`) are lazily resolvable from `litmus_common`.

- [ ] **Step 4: Run the suite**

Run: `pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add litmus_common.py tests/test_common_import.py
git commit -m "refactor(common): retire _mlx() shim; MLX helpers now live in litmus_mlx"
```

---

## Phase C — `TorchBackend` + `litmus_cuda.py` shim

Fold `litmus_cuda.py`'s torch logic into `litmus_torch.py` behind the protocol; reduce `litmus_cuda.py` to a thin shim that routes the three shared commands to the core and keeps `assisted` torch-native.

### Task 8: Implement `TorchBackend` in `litmus_torch.py`

**Files:**
- Create: `litmus_torch.py`
- Test: `tests/test_torch_backend.py` (create; `importorskip("torch")`)

**Interfaces:**
- Consumes: `litmus_common.Backend` (structural).
- Produces: `class TorchBackend` (`name = "cuda"`) implementing the protocol; `load(repo, quant="bf16")` accepts the quant modes; plus `cmd_assisted(backend, args)` and helper `_chat_wrap`, migrated from `litmus_cuda.py`.

Migration map (from `litmus_cuda.py`):
- `load` ← `_load_timed` (112-147) + `_auto_load` (150-165), `quant` taken from `opts`.
- `stream` ← `_greedy_stream` (174-202), but **yield per-token text** (running-decode delta) instead of `(id, is_first)`.
- `token_logprobs` ← `compute_perplexity` forward (297-309), returning the gathered list.
- memory helpers ← 78-87.

- [ ] **Step 1: Create `litmus_torch.py` with the backend**

```python
"""TorchBackend: the CUDA/transformers adapter for the Litmus Backend protocol.

torch / transformers are imported lazily inside method bodies. Also hosts the
torch-native `assisted` (speculative decoding) command, which has no MLX
equivalent and therefore lives outside the shared core.
"""
from __future__ import annotations

import time
from typing import Iterator


class TorchBackend:
    name = "cuda"

    def load(self, repo: str, **opts) -> tuple[object, object, float]:
        import torch
        from transformers import AutoTokenizer

        quant = opts.get("quant", "bf16")
        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(repo)
        kwargs: dict = {"device_map": "cuda:0"}
        if quant == "bf16":
            kwargs["dtype"] = torch.bfloat16
        elif quant == "fp32":
            kwargs["dtype"] = torch.float32
            kwargs["device_map"] = "auto"
        elif quant == "nf4":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif quant == "prequant":
            pass
        else:
            raise SystemExit(f"unknown --quant {quant}")
        model = _auto_load(repo, kwargs)
        model.eval()
        return model, tokenizer, time.perf_counter() - t0

    def stream(self, model, tokenizer, prompt: str,
               max_tokens: int) -> Iterator[str]:
        import torch
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        eos = tokenizer.eos_token_id
        eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else eos)
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1)
            torch.cuda.synchronize()
            generated: list[int] = []
            prev_text = ""
            for _ in range(max_tokens):
                tok = next_id.item()
                generated.append(tok)
                full = tokenizer.decode(generated, skip_special_tokens=True)
                yield full[len(prev_text):]              # incremental delta
                prev_text = full
                if tok in eos_ids:
                    return
                out = model(input_ids=next_id[:, None],
                            past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_id = out.logits[:, -1, :].argmax(dim=-1)
                torch.cuda.synchronize()

    def token_logprobs(self, model, tokenizer, ids: list[int]) -> list[float]:
        import torch
        x = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            logits = model(input_ids=x).logits.float()   # (1, T, V)
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            targets = x[:, 1:, None]
            gathered = torch.gather(log_probs, -1, targets).squeeze(-1)
        return [float(v) for v in gathered[0].tolist()]

    def peak_memory_mb(self) -> float:
        import torch
        return torch.cuda.max_memory_allocated() / (1024 * 1024)

    def reset_peak_memory(self) -> None:
        import torch
        torch.cuda.reset_peak_memory_stats()

    def clear_cache(self) -> None:
        import torch
        torch.cuda.empty_cache()


def _auto_load(repo: str, kwargs: dict):
    """CausalLM first; fall back to image-text-to-text (Gemma 4 Unified)."""
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    except ValueError as causal_err:
        try:
            from transformers import AutoModelForImageTextToText
            print(f"  (AutoModelForCausalLM refused: {causal_err}")
            print("   falling back to AutoModelForImageTextToText)")
            return AutoModelForImageTextToText.from_pretrained(repo, **kwargs)
        except Exception:
            raise causal_err


def _chat_wrap(tokenizer, prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False,
        )
    except Exception as e:
        print(f"  WARNING: chat template failed ({e}); using raw prompt")
        return prompt
```

- [ ] **Step 2: Write the smoke test (shape/plausibility, not golden)**

```python
import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for TorchBackend smoke tests",
                allow_module_level=True)

from litmus_torch import TorchBackend

REPO = "sshleifer/tiny-gpt2"  # tiny, CPU/GPU-loadable stand-in


def test_stream_yields_nonempty_text():
    be = TorchBackend()
    model, tok, secs = be.load(REPO, quant="bf16")
    assert secs >= 0.0
    chunks = list(be.stream(model, tok, "Hello", max_tokens=5))
    assert chunks and all(isinstance(c, str) for c in chunks)


def test_token_logprobs_shape_and_finiteness():
    be = TorchBackend()
    model, tok, _ = be.load(REPO, quant="bf16")
    ids = tok("Hello world", return_tensors="pt").input_ids[0].tolist()
    lps = be.token_logprobs(model, tok, ids)
    assert len(lps) == len(ids) - 1
    assert all(math.isfinite(v) for v in lps)
```

- [ ] **Step 3: Run the smoke test**

Run: `pytest tests/test_torch_backend.py -v`
Expected: PASS on a CUDA box; SKIPPED elsewhere (no torch / no CUDA). Both are acceptable green states.

- [ ] **Step 4: Commit**

```bash
git add litmus_torch.py tests/test_torch_backend.py
git commit -m "feat(backend): TorchBackend adapter + torch smoke tests"
```

### Task 9: Reduce `litmus_cuda.py` to a compat shim

**Files:**
- Modify: `litmus_cuda.py` (replace implementation with a thin shim; move `cmd_assisted` + `ASSISTED_PROMPTS` + `_timed_generate` into `litmus_torch.py`)
- Test: `tests/test_cuda_shim.py` (create)

**Interfaces:**
- Consumes: `litmus_core.COMMANDS`, `litmus_common.get_backend`, `litmus_torch.cmd_assisted`.
- Produces: `litmus_cuda.main()` preserving the `--repo/--quant/--cmd/--max-tokens/--chat/--assistant-repo` surface; `args.quant` threaded into `backend.load` via a small adapter.

The shim must pass `quant` to `backend.load`. Since the core drivers call `backend.load(repo)` positionally, wrap the torch backend so its `load` defaults pick up `args.quant`. Simplest: the shim sets `backend` to a `TorchBackend` and injects quant via `functools.partial`-style binding on a per-run basis — but the core calls `backend.load(repo)` with no quant. To thread it cleanly, give the shim a tiny subclass:

- [ ] **Step 1: Move `assisted` into `litmus_torch.py`**

Move `ASSISTED_PROMPTS` (`litmus_cuda.py:412-419`), `_timed_generate` (422-441), and `cmd_assisted` (444-504) into `litmus_torch.py`. Rewrite `cmd_assisted(args)` to `cmd_assisted(backend, args)` and replace its internal `_load_timed(args.repo, args.quant)` with `backend.load(args.repo, quant=args.quant)`; keep the rest (drafter load via `_auto_load`, warmups, ratio table) intact. `_chat_wrap` is already in `litmus_torch.py`.

- [ ] **Step 2: Write the shim test**

```python
import types

import litmus_cuda


def test_shim_exposes_historical_commands():
    # The four historical commands must still be dispatchable names.
    assert set(litmus_cuda.CUDA_COMMANDS) >= {
        "throughput", "perplexity", "decode-stability", "assisted",
    }


def test_shim_quant_backend_threads_quant(monkeypatch):
    captured = {}

    class Spy:
        name = "cuda"
        def load(self, repo, **opts):
            captured["quant"] = opts.get("quant")
            return object(), object(), 0.0

    be = litmus_cuda._QuantBackend(Spy(), quant="nf4")
    be.load("some/repo")
    assert captured["quant"] == "nf4"
```

- [ ] **Step 3: Run to verify it fails**

Run: `pytest tests/test_cuda_shim.py -v`
Expected: FAIL (`litmus_cuda` has no `CUDA_COMMANDS` / `_QuantBackend` yet).

- [ ] **Step 4: Rewrite `litmus_cuda.py` as the shim**

```python
"""CUDA compat shim.

Historical CLI for the A6000/H100/GH200 rental phases. The perf algorithms now
live in litmus_core; this file only preserves the --repo/--quant/--cmd surface
that a6000_bootstrap.sh and results-bonsai-1bit.md reference. throughput /
perplexity / decode-stability route to the shared core through TorchBackend;
`assisted` stays torch-native (speculative decoding has no MLX equivalent).

Setup + usage: unchanged — see a6000_bootstrap.sh.
"""
from __future__ import annotations

import argparse

from litmus_core import COMMANDS as CORE_COMMANDS
from litmus_torch import TorchBackend, cmd_assisted


class _QuantBackend:
    """Wraps TorchBackend so the core's backend.load(repo) picks up --quant."""
    def __init__(self, inner, quant: str):
        self._inner = inner
        self._quant = quant
        self.name = inner.name

    def load(self, repo, **opts):
        opts.setdefault("quant", self._quant)
        return self._inner.load(repo, **opts)

    def __getattr__(self, item):
        return getattr(self._inner, item)


# Historical command set (subset of core + the torch-native assisted).
CUDA_COMMANDS = {
    "throughput": CORE_COMMANDS["throughput"],
    "perplexity": CORE_COMMANDS["perplexity"],
    "decode-stability": CORE_COMMANDS["decode-stability"],
    "assisted": cmd_assisted,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cmd", choices=list(CUDA_COMMANDS), default="throughput")
    ap.add_argument("--repo", required=True, help="HF repo id to benchmark")
    ap.add_argument("--label", default=None)
    ap.add_argument("--quant", choices=["bf16", "fp32", "nf4", "prequant"],
                    default="bf16")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--ppl-window", type=int, default=1024)
    ap.add_argument("--reference-text", default=None)
    ap.add_argument("--sizes", default=None)   # unused; --repo is required here
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--assistant-repo", default=None)
    args = ap.parse_args()
    if args.reference_text is None:
        from litmus_core import REFERENCE_TEXT_PATH
        args.reference_text = REFERENCE_TEXT_PATH

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — this is the rental-phase harness.")

    backend = _QuantBackend(TorchBackend(), quant=args.quant)
    CUDA_COMMANDS[args.cmd](backend, args)


if __name__ == "__main__":
    main()
```

> Note: the core `cmd_*` drivers resolve targets via `litmus_common._targets_for(args)`, which reads `args.repo`/`args.label`/`args.sizes`. The shim always supplies `--repo`, so `_targets_for` returns the single repo — matching `litmus_cuda.py`'s historical single-repo behavior.

- [ ] **Step 5: Run the shim test + full suite**

Run: `pytest tests/test_cuda_shim.py -v && pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add litmus_cuda.py litmus_torch.py tests/test_cuda_shim.py
git commit -m "refactor(cuda): reduce litmus_cuda to a compat shim over the core; assisted stays torch-native"
```

---

## Phase D — Route spec-check through `Backend`

Swap `litmus_spec.py`'s MLX-local model path for the protocol and add `--backend`. The pure loader/parser/scorer code is untouched.

### Task 10: Backend-drive `litmus_spec.py`'s model-running path

**Files:**
- Modify: `litmus_spec.py:702-789` (`_mlx_generate` + `main`)
- Test: `tests/test_spec_backend_run.py` (create)

**Interfaces:**
- Consumes: `litmus_common.{get_backend, _targets_for}`, a `Backend`.
- Produces: `litmus_spec.main()` accepting `--backend`; `run_constraints`/`run_tool_calling` unchanged (they already take a `generate_fn`).

- [ ] **Step 1: Write the failing test (runner works on a FakeBackend)**

```python
import litmus_spec


def test_run_constraints_with_fake_generate():
    # run_constraints only needs a tokenizer + generate_fn; prove the seam is
    # generate_fn-shaped so any Backend.stream can drive it.
    cases = litmus_spec.load_cases(
        litmus_spec.DEFAULT_CASES["constraints"], "constraints")

    class Tok:
        def apply_chat_template(self, msgs, **kw):
            return msgs[-1]["content"]
        def encode(self, text):
            return text.split()

    def fake_generate(prompt):
        return "ok " * 20

    result = litmus_spec.run_constraints(cases[:2], Tok(), fake_generate,
                                         enable_thinking=None)
    assert "rows" in result or isinstance(result, dict)
```

- [ ] **Step 2: Run to verify it passes or fails cleanly**

Run: `pytest tests/test_spec_backend_run.py -v`
Expected: PASS (this characterizes the existing generate_fn seam; if the result-dict key differs, adjust the assertion to the actual shape returned by `run_constraints`).

- [ ] **Step 3: Replace `_mlx_generate` + wire `--backend` in `main`**

In `litmus_spec.py`, delete `_mlx_generate` (702-707). In `main` (710-789):
- add `ap.add_argument("--backend", choices=["mlx", "cuda", "auto"], default="mlx")`;
- replace `from litmus_common import _targets_for, _load_timed, _clear_cache` with `from litmus_common import get_backend, _targets_for`;
- `backend = get_backend(args.backend)` before the target loop;
- `model, tokenizer, t_load = _load_timed(repo)` → `backend.load(repo)`;
- inside `gen`, replace `_mlx_generate(model, tokenizer, p, _budget)` with `"".join(backend.stream(model, tokenizer, p, _budget))`;
- replace the trailing `_clear_cache()` (if any in the loop) with `backend.clear_cache()`.

- [ ] **Step 4: Run the spec suite**

Run: `pytest tests/ -q -k spec`
Expected: PASS (pure spec logic untouched; the model path now routes through the protocol).

- [ ] **Step 5: If Task B5 left `_load_timed`/`stream_generate` shims in `litmus_common`, delete them now**

Run: `grep -rn "_load_timed\|stream_generate\|_resp_text" litmus_common.py`
Expected: no matches. If any remain from the B5 deferral, delete them and re-run `pytest -q`.

- [ ] **Step 6: Commit**

```bash
git add litmus_spec.py tests/test_spec_backend_run.py litmus_common.py
git commit -m "refactor(spec): route model-running path through Backend; add --backend"
```

---

## Phase E — Packaging

### Task 11: Update `pyproject.toml` extras + py-modules

**Files:**
- Modify: `pyproject.toml`
- Test: `tests/test_packaging.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: extras `mlx` / `cuda` / `dev`; `py-modules` includes the three new modules.

- [ ] **Step 1: Write the failing test**

```python
import tomllib
from pathlib import Path


def test_pyproject_lists_new_modules_and_extras():
    data = tomllib.loads(Path("pyproject.toml").read_text())
    modules = set(data["tool"]["setuptools"]["py-modules"])
    assert {"litmus_core", "litmus_mlx", "litmus_torch"} <= modules
    extras = data["project"]["optional-dependencies"]
    assert "mlx" in extras and "cuda" in extras and "dev" in extras
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_packaging.py -v`
Expected: FAIL (new modules / extras not yet listed). Adjust the `["tool"]["setuptools"]` path if the repo uses a different table (inspect `pyproject.toml` first).

- [ ] **Step 3: Edit `pyproject.toml`**

Add `litmus_core`, `litmus_mlx`, `litmus_torch` to `py-modules`, and set the extras:

```toml
[project.optional-dependencies]
mlx = ["mlx-lm"]
cuda = ["torch", "transformers", "accelerate", "bitsandbytes", "hf_transfer"]
dev = ["pytest>=8.0"]
```

(Preserve any existing extras' intent; fold the old `mlx`/`dev` contents into these.)

- [ ] **Step 4: Run the packaging test + full suite**

Run: `pytest tests/test_packaging.py -v && pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_packaging.py
git commit -m "build: mlx/cuda/dev extras + register core/mlx/torch modules"
```

---

## Self-Review

**Spec coverage** (each spec section → task):
- Shared core hosting metrics/windowing/perplexity/rendering/orchestration → Phase A (pure) + B4 (orchestration).
- `Backend` protocol as single seam for perf + evals → B1; used by perf (B4), spec (D1).
- MLX/torch as lazily-imported adapters → B2, C1 (all library imports inside method bodies).
- No duplicated torch logic; `litmus_cuda.py` a shim → C2.
- Deterministic core invariant via byte-strict goldens captured pre-refactor → A2.
- `token_logprobs` numeric seam → B3 (core aggregation) + B2/C1 (backend forward).
- `FakeBackend` for orchestration + backend-free spec runners → B3 (conftest), used in B4/D1.
- Backend primitive smoke tests → C1 (torch); MLX primitives are covered structurally by B2's lazy-construction test (real-inference MLX smoke omitted deliberately — no gate on model inference, per non-goals; add later if an Apple CI box exists).
- Non-goals honored: no CI, no full-model golden, no cross-backend parity gate, no model-repo remapping.
- Risks resolved: `_strip_thinking` stays split (Global Constraints); `assisted` torch-native (C2); `baseline` stays MLX-default (unchanged in B4); `_mlx()` shim retired (B5); torch stream fidelity via running-decode delta + shape-only smoke (C1).

**Gap noted for the executor:** the spec lists an MLX backend smoke test symmetric to the torch one. It is intentionally omitted from the task list because (a) the dev machine has no MLX and (b) non-goals forbid gating on real inference. If you later run on an Apple box, mirror `tests/test_torch_backend.py` as `tests/test_mlx_backend.py` with `importorskip("mlx_lm")`.

**Placeholder scan:** no "TBD"/"handle edge cases"/"similar to Task N" — moves cite exact source line ranges with the specific edits; new code is shown in full.

**Type consistency:** `Run.label` (not `size`) used consistently from A1 onward; `compute_perplexity(backend, model, tokenizer, text, window)` signature identical in B3 and its B4 callers; `backend.stream(model, tokenizer, prompt, max_tokens)` positional shape identical across B2/C1/B4/D1; `token_logprobs` returns `len(ids)-1` floats everywhere; `get_backend` names `mlx|cuda|auto` consistent across B1, litmus.py (B4), litmus_cuda (C2), litmus_spec (D1).
