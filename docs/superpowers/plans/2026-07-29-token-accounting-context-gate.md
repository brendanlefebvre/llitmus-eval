# Token Accounting & Context Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace approximate/arbitrary token accounting with exact reference-tokenizer counts and model-derived context limits across `llitmus-eval` (extractor, runner gate) and `loxo-llm-router` (estimator, routing threshold, `/v1/models` truthfulness).

**Architecture:** A shared `resolve_context_length()` reads native context from cached `config.json`. The eval extractor counts exact tokens of a canonical render under a fixed reference tokenizer (`mlx-community/Qwen3-14B-4bit`) and gates the corpus at fleet-max context. The runner gates each candidate at its own resolved context. The router keeps a chars-based estimator (it must stay cheap and dependency-light) but counts *every* prompt field and divides by a divisor calibrated to never underestimate on the corpus; its routing threshold is probed from the local server at startup, with explicit config as override and 60000 as last-resort default.

**Tech Stack:** Python 3.12, pytest, `huggingface_hub` + `transformers` (already installed as mlx_lm deps), FastAPI/httpx (router, already present).

**Spec:** `docs/superpowers/specs/2026-07-29-token-accounting-context-gate-design.md` (status: reviewed, decisions resolved).

## Global Constraints

- Two repos: **Part A** tasks 1–6 run in `/Users/brendanl/src/llitmus-eval`; **Part B** tasks 7–9 run in `/Users/brendanl/src/loxo-llm-router`. Task 8 lives in llitmus-eval but imports from the router. Commit to the repo you edited; never mix repos in one commit.
- Reference tokenizer is `mlx-community/Qwen3-14B-4bit`, loaded with `local_files_only=True`. **Fail loudly if not cached; never silently fall back to an estimate.** In tests, a missing tokenizer is a loud `pytest.skip`, not a pass.
- `resolve_context_length` returns `None` when unknown — callers decide the fallback explicitly. No silent defaults inside the resolver.
- The renamed case field is exactly `ref_tokens` (was `est_tokens`).
- Router estimator must **never underestimate** on the 15-case calibration corpus; divisor 3.6 is provisional until Task 8 recalibrates. Divisor value lives in one constant: `ESTIMATE_CHARS_PER_TOKEN`.
- Router threshold precedence: explicit `LOCAL_CONTEXT_LIMIT` config/env > startup probe of `{LOCAL_BASE_URL}/models` > legacy default 60000. A down local server must never block requests.
- Strata edges stay 16,000 / 40,000. `depth_weights` recompute from each walk (already true, commit `5ab12bf`) — do not hard-code weights.
- Test commands: llitmus-eval → `python -m pytest tests/ -q`; loxo-llm-router → `python -m pytest -q` from its repo root.
- llitmus-eval keeps importing `classify` from `loxo_llm_router` (intended coupling); only the `estimate_prompt_tokens` and `LOCAL_CONTEXT_LIMIT` imports are removed from the extractor.

---

## Part A — llitmus-eval

### Task 1: `resolve_context_length` in litmus_common

**Files:**
- Modify: `litmus_common.py` (append after `get_backend`, file is ~103 lines)
- Test: `tests/test_resolve_context.py` (new)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `resolve_context_length(repo: str) -> int | None` in `litmus_common` — used by Task 4 (extractor fleet max) and Task 6 (runner gate). Lazy-imports `huggingface_hub` inside the function body (the module docstring promises dependency-free import; keep that true).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_resolve_context.py`:

```python
"""resolve_context_length: native context from cached config.json, or None.

The resolver deliberately returns None instead of a default — a silent
default is what let 51k–85k-token cases score on a 40,960-context model
(spec 2026-07-29).
"""
import json

import huggingface_hub
import pytest

from litmus_common import resolve_context_length


def _fake_cache(monkeypatch, tmp_path, config: dict | None):
    """Point try_to_load_from_cache at a temp config.json (or a miss)."""
    if config is None:
        # Cache miss: the real API returns None or a sentinel object,
        # never a str. Use None.
        monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                            lambda repo_id, filename: None)
        return
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo_id, filename: str(p))


def test_top_level_max_position_embeddings(monkeypatch, tmp_path):
    _fake_cache(monkeypatch, tmp_path, {"max_position_embeddings": 40960})
    assert resolve_context_length("org/model") == 40960


def test_nested_text_config(monkeypatch, tmp_path):
    # Qwen3.6 / gemma-4 / Qwen3-VL nest it under text_config.
    _fake_cache(monkeypatch, tmp_path,
                {"text_config": {"max_position_embeddings": 262144}})
    assert resolve_context_length("org/model") == 262144


def test_top_level_wins_over_nested(monkeypatch, tmp_path):
    _fake_cache(monkeypatch, tmp_path,
                {"max_position_embeddings": 40960,
                 "text_config": {"max_position_embeddings": 262144}})
    assert resolve_context_length("org/model") == 40960


def test_neither_key_returns_none(monkeypatch, tmp_path):
    _fake_cache(monkeypatch, tmp_path, {"architectures": ["X"]})
    assert resolve_context_length("org/model") is None


def test_not_cached_returns_none(monkeypatch, tmp_path):
    _fake_cache(monkeypatch, tmp_path, None)
    assert resolve_context_length("org/model") is None


def test_unparseable_config_returns_none(monkeypatch, tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo_id, filename: str(p))
    assert resolve_context_length("org/model") is None


def test_bool_true_is_not_a_context_length(monkeypatch, tmp_path):
    # bool is an int subclass; a config with True must not resolve to 1.
    _fake_cache(monkeypatch, tmp_path, {"max_position_embeddings": True})
    assert resolve_context_length("org/model") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_resolve_context.py -q`
Expected: FAIL / ERROR with `ImportError: cannot import name 'resolve_context_length'`

- [ ] **Step 3: Implement the resolver**

Append to `litmus_common.py`:

```python
def resolve_context_length(repo: str) -> int | None:
    """Native context length for a model repo, from config.json in the HF cache.

    Precedence: top-level ``max_position_embeddings``, then
    ``text_config.max_position_embeddings`` (Qwen3.6 / gemma-4 / Qwen3-VL
    nest it). Returns None when the config is not cached, unparseable, or
    carries neither key — the caller decides the fallback. A silent default
    here is what produced the 2026-07-28 over-context scoring bug.
    """
    import json
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(repo_id=repo, filename="config.json")
    if not isinstance(path, str):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    for scope in (cfg, cfg.get("text_config") or {}):
        if not isinstance(scope, dict):
            continue
        v = scope.get("max_position_embeddings")
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_resolve_context.py -q`
Expected: 7 passed

- [ ] **Step 5: Run the full suite to catch import regressions**

Run: `python -m pytest tests/ -q`
Expected: no new failures vs. a pre-change baseline run

- [ ] **Step 6: Commit**

```bash
git add litmus_common.py tests/test_resolve_context.py
git commit -m "feat: resolve_context_length from cached config.json, None when unknown"
```

---

### Task 2: Canonical render + `count_ref_tokens` in litmus_spec

**Files:**
- Modify: `litmus_spec.py` (add after `build_replay_prompt`, which ends near line 920)
- Test: `tests/test_spec_replay.py` (append a new test class; reuse `ReplayFakeTokenizer` defined at `tests/test_spec_replay.py:654`)

**Interfaces:**
- Consumes: existing module-private helpers `_stringify_content(messages)` and `_chat(tokenizer, messages, tools=None, enable_thinking=None)` in `litmus_spec.py`.
- Produces: `canonical_ref_render(tokenizer, body: dict) -> str` and `count_ref_tokens(tokenizer, body: dict) -> int` in `litmus_spec` — used by Task 4 (extractor), Task 5 (drift test), Task 8 (calibration script). `count_ref_tokens` calls `tokenizer.encode(...)` and returns `len(...)` of the result.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spec_replay.py` (near `TestBuildReplayPrompt`; `ReplayFakeTokenizer.encode` returns a word list, so counts are word counts):

```python
class TestCanonicalRefRender:
    """Corpus-side canonical render: one fixed convention regardless of
    candidate — messages with tools forwarded natively, no thinking flag."""

    def _body(self):
        return {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ]},
            ],
            "tools": [{"type": "function",
                       "function": {"name": "read", "parameters": {}}}],
        }

    def test_forwards_tools_natively_no_thinking_flag(self):
        from litmus_spec import canonical_ref_render
        tok = ReplayFakeTokenizer()
        canonical_ref_render(tok, self._body())
        assert tok.last_tools == self._body()["tools"]
        assert tok.last_enable_thinking is None

    def test_stringifies_content_part_lists(self):
        from litmus_spec import canonical_ref_render
        tok = ReplayFakeTokenizer()
        canonical_ref_render(tok, self._body())
        assert isinstance(tok.last_messages[-1]["content"], str)
        assert "part one" in tok.last_messages[-1]["content"]

    def test_count_ref_tokens_counts_encoded_render(self):
        from litmus_spec import count_ref_tokens
        tok = ReplayFakeTokenizer()
        # Fake render is "PROMPT:" + last message content; fake encode
        # splits on whitespace. "PROMPT:part one\npart two" -> word count.
        n = count_ref_tokens(tok, self._body())
        assert n == len(tok.encode(
            tok.apply_chat_template(self._body()["messages"])))
```

Note: if the third assertion's expected value is awkward against the fake's exact join behavior, assert instead that `n` equals `len(tok.encode(canonical_ref_render(tok, body)))` — the contract is "count = len(encode(render))".

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spec_replay.py -k CanonicalRefRender -q`
Expected: FAIL with `ImportError: cannot import name 'canonical_ref_render'`

- [ ] **Step 3: Implement**

Add to `litmus_spec.py` directly after `build_replay_prompt`:

```python
def canonical_ref_render(tokenizer, body: dict) -> str:
    """Render a captured request body the corpus-canonical way.

    One fixed convention — messages with tools forwarded natively, no
    thinking flag — because a stratum is a property of the corpus, not of a
    candidate. Per-candidate rendering reality is recorded separately as
    ``prompt_tokens_fed`` (spec 2026-07-29).
    """
    messages = _stringify_content(body["messages"])
    return _chat(tokenizer, messages, tools=body.get("tools"))


def count_ref_tokens(tokenizer, body: dict) -> int:
    """Exact token count of the canonical render under ``tokenizer``."""
    return len(tokenizer.encode(canonical_ref_render(tokenizer, body)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spec_replay.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add litmus_spec.py tests/test_spec_replay.py
git commit -m "feat: canonical reference render + exact ref-token counting"
```

---

### Task 3: Rename `est_tokens` → `ref_tokens` everywhere (values unchanged)

This task is purely mechanical: rename the field across code, tests, and the on-disk case file, without changing any counted value. Value changes happen in Tasks 4–5. This ordering keeps every commit green: after this task the loader, extractor, drift check, and `cases/main_replay.jsonl` all agree on the new name while still holding the old chars//4 numbers.

**Files:**
- Modify: `litmus_spec.py:144` (dataclass field), `litmus_spec.py:305-318` (`_load_replay_line`), `litmus_spec.py:1094` (comment), `litmus_spec.py:1147` (case record key)
- Modify: `scripts/extract_main_replay.py` (the `"est_tokens"` keys in `process_pairs`' usable dict and `main()`'s `final_cases` dict, near lines 483 and 660)
- Modify: `tests/test_spec_replay.py` (sites at lines 61, 65, 79, 99, 130–141, 1003–1027), `tests/test_f1_reference_validation.py` (line 258 `ReplayCase(...)` kwarg, and the attribute reads inside `test_est_tokens_drift_check` — keep that test comparing against `estimate_prompt_tokens` for now; only the field/attr name changes), `tests/test_extract_main_replay.py` (lines 345 area comment, 607, 702, 710, 779, 791)
- Modify: `cases/main_replay.jsonl` (key rename in place)

**Interfaces:**
- Consumes: nothing new.
- Produces: `ReplayCase.ref_tokens: int`; case-file/sidecar key `"ref_tokens"`. Tasks 4–6 and 8 use this name exclusively.

- [ ] **Step 1: Rename in code and tests**

```bash
grep -rln "est_tokens" litmus_spec.py scripts/extract_main_replay.py tests/ \
  | xargs sed -i '' 's/est_tokens/ref_tokens/g'
```

Then hand-review `git diff` for strings that deserve more than a blind rename:
- `litmus_spec.py:1094` comment: rewrite to say `ref_tokens` is the extractor's reference-tokenizer count of the canonical render (not "pre-template estimate").
- `tests/test_extract_main_replay.py:345-347` comment (`test_skip_over_limit`): leave the test logic alone in this task; it still exercises the old chars//4 path until Task 4 rewrites it.
- `tests/test_f1_reference_validation.py` module/function docstrings: update the wording (`ref_tokens` drift check), keep the `estimate_prompt_tokens` comparison intact.
- Rename the test functions themselves (`test_est_tokens_must_be_int` → `test_ref_tokens_must_be_int`, `test_est_tokens_drift_check` → `test_ref_tokens_drift_check`, etc.) — sed will have done this; just confirm.

- [ ] **Step 2: Rename the key in the on-disk case file**

```bash
sed -i '' 's/"est_tokens"/"ref_tokens"/g' cases/main_replay.jsonl
```

- [ ] **Step 3: Run both repos' consumers of the name**

Run: `python -m pytest tests/ -q`
Expected: all pass (values unchanged, only names moved)

Also confirm nothing else still says `est_tokens`:

```bash
grep -rn "est_tokens" --include="*.py" --include="*.jsonl" . ; echo "exit=$?"
```

Expected: no hits (exit=1). (`docs/` may still mention the old name historically; that's fine.)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: rename est_tokens -> ref_tokens (schema break, values unchanged)"
```

---

### Task 4: Extractor — exact counting + fleet-max gate

**Files:**
- Modify: `scripts/extract_main_replay.py`:
  - imports at lines 27–31 (drop `estimate_prompt_tokens`, `LOCAL_CONTEXT_LIMIT`; keep `classify`)
  - `assign_stratum` at line 320 (drop the `LOCAL_CONTEXT_LIMIT` default)
  - `process_pairs` at line 398 (inject counting fn + limit + stats)
  - `write_meta` at line 569 (new fields)
  - `main()` at line 598 (tokenizer load, fleet max, new CLI flags)
- Test: `tests/test_extract_main_replay.py`

**Interfaces:**
- Consumes: `count_ref_tokens` (Task 2), `resolve_context_length` (Task 1).
- Produces:
  - `process_pairs(chains, qwen_entries, count_tokens, limit, stats=None) -> list[dict]` — `count_tokens: Callable[[dict], int]` takes a capture body; `stats`, if a dict, gains key `"over_limit"` (int count of pairs excluded by the gate).
  - `assign_stratum(tokens: int, limit: int) -> str | None` (no default limit).
  - `fleet_max_context(fleet: tuple[str, ...]) -> int` — max resolved context over the fleet; exits the process with a clear message if none resolve.
  - Module constants `FLEET` and `REF_TOKENIZER`.
  - `write_meta(..., tokenizer_repo: str, fleet_max: int, over_limit: int)` → meta.json gains `"tokenizer"`, `"fleet_max_context"`, `"over_limit"`.
  - CLI flags `--tokenizer` (default `REF_TOKENIZER`) and `--fleet` (comma-separated repos, default `",".join(FLEET)`).

- [ ] **Step 1: Update the test harness with a wrapper, migrate call sites**

In `tests/test_extract_main_replay.py`, add near the top (after the `emr` import):

```python
def _pp(chains, qwen_entries, count_tokens=None, limit=131072, stats=None):
    """process_pairs with a neutral counting fn: every body 'counts' as 100
    tokens (shallow) unless a test injects its own count_tokens/limit."""
    return emr.process_pairs(chains, qwen_entries,
                             count_tokens or (lambda body: 100),
                             limit, stats=stats)
```

Migrate every `emr.process_pairs(chains, X)` call site (lines 200, 217, 231, 246, 269, 294, 316, 340, 358, 372, 389, 405, 430, 458, 483, 694, 723 — re-grep, don't trust this list) to `_pp(chains, X)`.

Rewrite `test_skip_over_limit` (line ~344): delete the 240,005-char body construction; use two small captures and inject the count:

```python
    def test_skip_over_limit(self, tmp_path, capsys):
        msgs_a = [sys_msg(), user_msg("hi")]
        msgs_b = [sys_msg(), user_msg("hi"), assistant_tool_msg(), tool_msg()]
        write_capture(tmp_path, "req-20260101T000000.000000-0000.json", msgs_a)
        write_capture(tmp_path, "req-20260101T000001.000000-0001.json", msgs_b)
        rows = emr.load_captures(tmp_path)
        chains = emr.group_chains(rows)
        stats = {}
        usable = _pp(chains, [], count_tokens=lambda body: 131073,
                     limit=131072, stats=stats)
        assert usable == []
        assert stats["over_limit"] == 1
```

Any other test that fabricates giant char bodies to hit a stratum edge (grep the file for `"x" *` and `240` / `64000` style sizes): replace the char-sizing with an injected `count_tokens=lambda body: <intended token count>` and small bodies.

Add new tests:

```python
class TestFleetMaxContext:
    def test_max_over_resolvable_fleet(self, monkeypatch):
        monkeypatch.setattr(emr, "resolve_context_length",
                            lambda repo: {"a": 40960, "b": 131072}.get(repo))
        assert emr.fleet_max_context(("a", "b")) == 131072

    def test_unresolvable_repo_is_skipped(self, monkeypatch):
        monkeypatch.setattr(emr, "resolve_context_length",
                            lambda repo: 40960 if repo == "a" else None)
        assert emr.fleet_max_context(("a", "b")) == 40960

    def test_nothing_resolvable_exits(self, monkeypatch):
        monkeypatch.setattr(emr, "resolve_context_length", lambda repo: None)
        with pytest.raises(SystemExit):
            emr.fleet_max_context(("a", "b"))
```

Update the meta tests (the region around lines 779–791 that asserts meta.json contents) to pass the new `write_meta` arguments and assert `meta["tokenizer"]`, `meta["fleet_max_context"]`, `meta["over_limit"]`.

- [ ] **Step 2: Run tests to verify the new/changed ones fail**

Run: `python -m pytest tests/test_extract_main_replay.py -q`
Expected: failures — `process_pairs()` signature mismatch, `fleet_max_context` missing

- [ ] **Step 3: Implement the extractor changes**

In `scripts/extract_main_replay.py`:

Imports (top of file): drop `estimate_prompt_tokens` and `LOCAL_CONTEXT_LIMIT` from the `loxo_llm_router` import (keep `classify`); add:

```python
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from litmus_common import resolve_context_length  # noqa: E402
from litmus_spec import count_ref_tokens          # noqa: E402
```

Constants (near `REFERENCE_MODEL` at line 47):

```python
REF_TOKENIZER = "mlx-community/Qwen3-14B-4bit"

# Candidate models the corpus must be in-scope for; the corpus gate is the
# max of their native contexts (spec 2026-07-29, "fleet max").
FLEET = (
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Qwen3-4B-4bit",
    "mlx-community/Qwen3-14B-4bit",
)


def fleet_max_context(fleet: tuple[str, ...]) -> int:
    resolved = {r: resolve_context_length(r) for r in fleet}
    known = {r: c for r, c in resolved.items() if c is not None}
    for r in fleet:
        if r not in known:
            print(f"WARN: no cached config.json for {r}; excluded from "
                  f"fleet max", file=sys.stderr)
    if not known:
        sys.exit("fleet max context unresolvable: no fleet model has a "
                 "cached config.json — refusing to fall back to a magic "
                 "constant. Download at least one fleet model first.")
    return max(known.values())
```

`assign_stratum` (line 320): signature becomes `def assign_stratum(tokens: int, limit: int) -> str | None:` — body unchanged.

`process_pairs` (line 398): signature becomes

```python
def process_pairs(chains: list[list[dict]], qwen_entries: list[dict],
                  count_tokens, limit: int, stats: dict | None = None) -> list[dict]:
```

and the counting block (currently line ~472) becomes:

```python
            # Depth stratum (over-fleet-max is excluded)
            tokens = count_tokens(cap_n["body"])
            stratum = assign_stratum(tokens, limit)
            if stratum is None:
                print(f"SKIP {pair_label}: over limit ({tokens} > {limit})",
                      file=sys.stderr)
                if stats is not None:
                    stats["over_limit"] = stats.get("over_limit", 0) + 1
                continue
```

(the usable-dict key is already `ref_tokens` from Task 3).

`write_meta` (line 569): add parameters `tokenizer_repo: str, fleet_max: int, over_limit: int` and meta keys:

```python
        "tokenizer": tokenizer_repo,
        "fleet_max_context": fleet_max,
        "over_limit": over_limit,
```

`main()` (line 598): add CLI flags and wire everything:

```python
    ap.add_argument("--tokenizer", default=REF_TOKENIZER,
                    help="reference tokenizer repo for exact ref_tokens "
                         f"counts (default: {REF_TOKENIZER})")
    ap.add_argument("--fleet", default=",".join(FLEET),
                    help="comma-separated candidate repos; corpus gate is "
                         "the max of their native contexts")
```

after arg parsing:

```python
    fleet = tuple(r.strip() for r in args.fleet.split(",") if r.strip())
    limit = fleet_max_context(fleet)

    from transformers import AutoTokenizer
    try:
        ref_tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                                local_files_only=True)
    except Exception as e:  # noqa: BLE001 - any load failure is fatal
        sys.exit(f"reference tokenizer {args.tokenizer!r} not loadable from "
                 f"the local HF cache ({type(e).__name__}: {e}). Run: "
                 f"hf download {args.tokenizer} — refusing to fall back to "
                 f"an estimate.")

    def count_tokens(body: dict) -> int:
        return count_ref_tokens(ref_tok, body)
```

replace the `process_pairs` call with:

```python
    stats: dict = {}
    usable = process_pairs(chains, qwen_entries, count_tokens, limit,
                           stats=stats)
```

and the `write_meta` call with the three extra args:

```python
    meta_path = write_meta(output_path, corpus_count, newest_name,
                           final_cases, coverage, pop_by_stratum,
                           tokenizer_repo=args.tokenizer, fleet_max=limit,
                           over_limit=stats.get("over_limit", 0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_extract_main_replay.py -q` then `python -m pytest tests/ -q`
Expected: all pass. (The drift test still passes: the on-disk case file and its values are untouched in this task.)

- [ ] **Step 5: Commit**

```bash
git add scripts/extract_main_replay.py tests/test_extract_main_replay.py
git commit -m "feat(extractor): exact ref-token counting, fleet-max corpus gate"
```

---

### Task 5: Regenerate the corpus + migrate the drift check

Regeneration and drift-test migration must land together: regenerating changes `ref_tokens` values, which breaks the old `estimate_prompt_tokens` comparison in the same commit that fixes it.

**Files:**
- Regenerate: `cases/main_replay.jsonl`, `cases/main_replay.meta.json`
- Modify: `tests/test_f1_reference_validation.py` (drift check + its imports)

**Interfaces:**
- Consumes: Task 4's extractor; `count_ref_tokens` (Task 2); meta key `"tokenizer"` (Task 4).
- Produces: a corpus whose `ref_tokens` are exact reference-tokenizer counts. Downstream: nothing else reads the values symbolically; `depth_weights` recompute from meta automatically.

- [ ] **Step 1: Migrate the drift check first (it will fail until regeneration)**

In `tests/test_f1_reference_validation.py`:
- Delete `from loxo_llm_router import estimate_prompt_tokens` (line 38).
- Add `count_ref_tokens` to the `litmus_spec` import block.
- Add a module-scoped fixture:

```python
@pytest.fixture(scope="module")
def ref_tokenizer():
    """The corpus's own reference tokenizer, from meta.json; loud skip when
    it isn't cached — a silent pass would hide a broken drift check."""
    meta_path = pathlib.Path("cases/main_replay.meta.json")
    repo = "mlx-community/Qwen3-14B-4bit"
    if meta_path.exists():
        repo = json.loads(meta_path.read_text()).get("tokenizer", repo)
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            repo, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"reference tokenizer {repo} not in local HF cache "
                    f"({type(e).__name__}) — drift check needs it")
```

- Rewrite the drift test body (keep the per-case missing-capture skip list exactly as it is):

```python
def test_ref_tokens_drift_check(ref_tokenizer):
    """Each case's ref_tokens must match count_ref_tokens(reference
    tokenizer, capture body). Catches a stale case file after either the
    canonical render or the reference tokenizer changes."""
    cases = _require_cases()
    mismatches = []
    skipped = []
    for case in cases:
        if not os.path.exists(case.capture_path):
            skipped.append(case.id)
            continue
        body = _load_capture(case.capture_path)
        expected = count_ref_tokens(ref_tokenizer, body)
        if case.ref_tokens != expected:
            mismatches.append(
                f"  {case.id}: ref_tokens={case.ref_tokens} but "
                f"count_ref_tokens={expected}")
    if skipped:
        print(f"  (skipped {len(skipped)} case(s) with missing captures: "
              f"{', '.join(skipped)})")
    assert not mismatches, (
        "ref_tokens drift: case ref_tokens no longer match the reference "
        "tokenizer's count of the canonical render:\n" + "\n".join(mismatches))
```

- [ ] **Step 2: Regenerate the corpus**

Run: `python scripts/extract_main_replay.py --output cases/main_replay.jsonl`

Expected: completes; stderr shows SKIP/DROP lines; the ledger audit passes; `cases/main_replay.meta.json` now has `tokenizer`, `fleet_max_context: 131072`, `over_limit`, and a shifted `population_by_stratum` (exact counts move cases across the 16k/40k edges — expected, `depth_weights` self-correct). If it exits complaining the tokenizer isn't cached, download it first (`hf download mlx-community/Qwen3-14B-4bit`) — do not lower the gate to proceed.

Sanity-check one number: the spec records mr-012's capture at 77,908 real tokens under Qwen3-14B counting — the regenerated `ref_tokens` for that capture should land on (or very near) that value, not the old ~40k chars//4 figure.

- [ ] **Step 3: Run the suite**

Run: `python -m pytest tests/ -q`
Expected: all pass, with the drift check either passing (tokenizer cached) or visibly skipping with the loud message — on this machine it must **pass**, not skip.

- [ ] **Step 4: Commit**

```bash
git add cases/main_replay.jsonl cases/main_replay.meta.json tests/test_f1_reference_validation.py
git commit -m "feat(corpus): regenerate with exact ref_tokens; drift check vs reference tokenizer"
```

---

### Task 6: Runner gate — resolved context, not model_max_length

**Files:**
- Modify: `litmus_spec.py:1059` (`run_main_replay` signature), the gate block at `litmus_spec.py:1104-1117`, the returned dict, `write_sidecar` at `litmus_spec.py:1277`, the call site at `litmus_spec.py:1449`, the import at `litmus_spec.py:1344`
- Test: `tests/test_spec_replay.py` (the gate tests at lines 967–999 stay untouched and must keep passing — they now exercise the fallback path)

**Interfaces:**
- Consumes: `resolve_context_length` (Task 1).
- Produces: `run_main_replay(..., context_length: int | None = None)`; result dict gains `"context_length": int | None` and `"context_length_source": "config" | "model_max_length" | None`; sidecar payload carries both keys.

- [ ] **Step 1: Write the failing tests**

Append to the runner test class in `tests/test_spec_replay.py` (mirror the style of `test_runner_prompt_overflow_lands_in_errored` at line 967):

```python
    def test_runner_gate_uses_resolved_context_over_tokenizer(self, tmp_path):
        """A resolved config context must gate even when model_max_length is
        a large 'real' number — the 2026-07-28 failure: Qwen3's tokenizer
        says 131072 while the model's native context is 40960."""
        cap = _make_capture(tmp_path, "req-1.json", tools=[READ_TOOL])
        case = _replay_case(cap, acted=True)
        tok = ReplayFakeTokenizer()
        tok.model_max_length = 131072  # tokenizer's (wrong) ceiling
        result = run_main_replay(
            [case], tok,
            lambda p, max_tokens=0: '{"tool": "read", "arguments": {"filePath": "f"}}',
            native=False, context_length=3)
        assert result["cases"] == []
        err = result["errored"][0]
        assert "prompt exceeds model context" in err["error"]
        assert "> 3" in err["error"]
        assert result["context_length"] == 3
        assert result["context_length_source"] == "config"

    def test_runner_gate_falls_back_to_model_max_length(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json", tools=[READ_TOOL])
        case = _replay_case(cap, acted=True)
        tok = ReplayFakeTokenizer()
        tok.model_max_length = 3
        result = run_main_replay(
            [case], tok,
            lambda p, max_tokens=0: '{"tool": "read", "arguments": {"filePath": "f"}}',
            native=False, context_length=None)
        assert result["errored"]  # gated via fallback
        assert result["context_length"] == 3
        assert result["context_length_source"] == "model_max_length"

    def test_runner_gate_none_context_none_mml_records_null(self, tmp_path):
        cap = _make_capture(tmp_path, "req-1.json", tools=[READ_TOOL])
        case = _replay_case(cap, acted=True)
        tok = ReplayFakeTokenizer()
        tok.model_max_length = 10**9  # sentinel -> no declared limit
        result = run_main_replay(
            [case], tok,
            lambda p, max_tokens=0: '{"tool": "read", "arguments": {"filePath": "f"}}',
            native=False)
        assert result["errored"] == []
        assert result["context_length"] is None
        assert result["context_length_source"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spec_replay.py -k "runner_gate or overflow or sentinel" -q`
Expected: new tests FAIL (`unexpected keyword argument 'context_length'`); old ones still pass

- [ ] **Step 3: Implement**

`run_main_replay` signature (line 1059): add `context_length: int | None = None,` after `depth_weights`.

Before the case loop, resolve the gate once:

```python
    gate_limit: int | None = context_length
    gate_source: str | None = "config" if context_length is not None else None
    if gate_limit is None:
        # Fallback: the tokenizer's declared max, when sane. Many tokenizers
        # ship a huge sentinel (>= 10**9, transformers' "unset" convention);
        # Qwen3's is real but wrong (the YaRN ceiling, not native context) —
        # which is why a resolved config context takes precedence.
        mml = getattr(tokenizer, "model_max_length", None)
        if (isinstance(mml, int) and not isinstance(mml, bool)
                and 0 < mml < 10**9):
            gate_limit, gate_source = mml, "model_max_length"
```

Replace the in-loop gate block (currently lines ~1104–1117, the comment plus the `if prompt_tokens_fed is not None:` clause) with:

```python
            if (prompt_tokens_fed is not None and gate_limit is not None
                    and prompt_tokens_fed > gate_limit):
                raise RuntimeError(
                    f"prompt exceeds model context: "
                    f"{prompt_tokens_fed} > {gate_limit} ({gate_source})")
```

In the returned dict (find the `return {` at the end of `run_main_replay`), add:

```python
        "context_length": gate_limit,
        "context_length_source": gate_source,
```

In `write_sidecar` (line 1277), after the `payload = {...}` literal:

```python
    payload["context_length"] = result.get("context_length")
    payload["context_length_source"] = result.get("context_length_source")
```

At the import site (line 1344): `from litmus_common import get_backend, _targets_for, resolve_context_length`.

At the call site (line 1449), pass it through:

```python
                result = run_main_replay(cases, tokenizer, gen, native=native,
                                         enable_thinking=flag,
                                         depth_weights=replay_weights,
                                         context_length=resolve_context_length(repo),
                                         progress=print)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -q`
Expected: all pass, including the untouched 967–999 gate tests (error message now ends with `(model_max_length)` — if `test_runner_prompt_overflow_lands_in_errored` asserts on the exact message tail, its `"> 3"` substring assertion still holds).

- [ ] **Step 5: Commit**

```bash
git add litmus_spec.py tests/test_spec_replay.py
git commit -m "feat(runner): gate on resolved native context, record which limit was used"
```

---

## Part B — loxo-llm-router

Work in `/Users/brendanl/src/loxo-llm-router` for Tasks 7 and 9. Task 8's script and test live in llitmus-eval but change the router constant.

### Task 7: Estimator — count every prompt field

**Files:**
- Modify: `loxo_llm_router/__init__.py:442` (`estimate_prompt_tokens`)
- Test: `test_routing.py` (estimator section at lines 99–120)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `_count_prompt_chars(body: dict) -> int` (raw char count, used by Task 8's calibration) and `estimate_prompt_tokens(body) -> int` = `int(_count_prompt_chars(body) / ESTIMATE_CHARS_PER_TOKEN)`. Module constant `ESTIMATE_CHARS_PER_TOKEN = 3.6` (provisional until Task 8).

- [ ] **Step 1: Migrate and extend the estimator tests**

In `test_routing.py`, replace the three tests at lines 101–118 with:

```python
def test_estimate_counts_message_text():
    # Exact arithmetic, derived from the constant so recalibration
    # (Task 8) doesn't break the shape check.
    expected = int(40 / R.ESTIMATE_CHARS_PER_TOKEN)
    assert R.estimate_prompt_tokens(_body(text="a" * 40)) == expected


def test_estimate_counts_tool_schemas():
    no_tools = R.estimate_prompt_tokens(_body(text="hi"))
    with_tools = R.estimate_prompt_tokens(_body(text="hi", tools=[{"x": "y" * 400}]))
    assert with_tools > no_tools


def test_estimate_counts_list_content_text_parts():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "b" * 80},
        {"type": "image_url", "image_url": {"url": "data:..."}},  # not counted
    ]}]}
    assert R.estimate_prompt_tokens(body) == int(80 / R.ESTIMATE_CHARS_PER_TOKEN)


def test_estimate_counts_tool_calls_and_reasoning():
    """Regression for the 2026-07-28 blind spot: tool_calls,
    reasoning_content, and tool_call_id were 87% of mr-012's real prompt
    and counted as zero."""
    bare = {"messages": [{"role": "user", "content": "hi"}]}
    loaded = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None,
         "reasoning_content": "r" * 900,
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "read",
                                      "arguments": "{\"filePath\": \"" + "p" * 800 + "\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "c" * 700},
    ]}
    assert R.estimate_prompt_tokens(loaded) > R.estimate_prompt_tokens(bare) + (
        (900 + 800 + 700) // 5)  # loose floor: the new fields dominate


def test_count_prompt_chars_is_divisor_free():
    body = _body(text="a" * 36)
    assert R._count_prompt_chars(body) == 36
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_routing.py -k estimate -q` (plus `-k count_prompt_chars`)
Expected: FAIL — `_count_prompt_chars`/`ESTIMATE_CHARS_PER_TOKEN` missing, arithmetic mismatches

- [ ] **Step 3: Implement**

Replace `estimate_prompt_tokens` at `loxo_llm_router/__init__.py:442` with:

```python
# Chars-per-token divisor for estimate_prompt_tokens. PROVISIONAL until
# Task 8 pins it: 3.6 was hand-calibrated against Qwen3-14B-4bit counts,
# which is also the corpus reference tokenizer, so llitmus-eval's
# scripts/calibrate_router_divisor.py is expected to confirm it (never
# underestimates on the 15-case corpus). The value equaling Qwen3.6's
# version number is pure coincidence — this is an empirical ratio, not
# model-derived.
ESTIMATE_CHARS_PER_TOKEN = 3.6


def _count_prompt_chars(body: dict[str, Any]) -> int:
    """Characters of every request field the chat template renders into the
    prompt — not just content text. On agentic traffic (OpenCode),
    tool_calls/reasoning_content/tool_call_id dominated real prompt size
    (87% of mr-012) and were previously counted as zero."""
    total = 0
    for m in body.get("messages", []):
        content = m.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += len(part.get("text", ""))
        for key in ("reasoning_content", "tool_call_id", "name"):
            v = m.get(key)
            if isinstance(v, str):
                total += len(v)
        if m.get("tool_calls"):
            total += len(json.dumps(m["tool_calls"]))
    for key in ("tools", "functions"):
        if key in body:
            total += len(json.dumps(body[key]))
    return total


def estimate_prompt_tokens(body: dict[str, Any]) -> int:
    """Conservative token estimate for threshold gating: full-field char
    count over a divisor calibrated to never underestimate on the eval
    corpus. Asymmetric failure modes drive the conservatism: an undercount
    sends an over-long prompt to a local model (garbage or a crash); an
    overcount sends it to cloud (costs money, works)."""
    return int(_count_prompt_chars(body) / ESTIMATE_CHARS_PER_TOKEN)
```

- [ ] **Step 4: Run the full router suite**

Run: `python -m pytest -q`
Expected: all pass. If any other test asserted /4 arithmetic, migrate it the same way (derive expected from `R.ESTIMATE_CHARS_PER_TOKEN`).

- [ ] **Step 5: Commit**

```bash
git add loxo_llm_router/__init__.py test_routing.py
git commit -m "feat(estimator): count all prompt fields; calibrated conservative divisor"
```

---

### Task 8: Calibrate the divisor against the reference tokenizer

Runs in **llitmus-eval** (needs the corpus + canonical render) but its output is a one-line change in **loxo-llm-router**.

**Files:**
- Create: `scripts/calibrate_router_divisor.py` (llitmus-eval)
- Create: `tests/test_router_divisor_property.py` (llitmus-eval)
- Modify: `loxo_llm_router/__init__.py` (the `ESTIMATE_CHARS_PER_TOKEN` value + its comment)
- Modify: `docs/superpowers/specs/2026-07-29-token-accounting-context-gate-design.md` (Section 4 margin numbers)

**Interfaces:**
- Consumes: `count_ref_tokens` (Task 2), `load_cases` (existing, `litmus_spec.py:323`), `_count_prompt_chars` (Task 7).
- Produces: the calibrated `ESTIMATE_CHARS_PER_TOKEN` value and a standing property test that `estimate_prompt_tokens` never underestimates on the corpus.

- [ ] **Step 1: Write the property test (llitmus-eval)**

Create `tests/test_router_divisor_property.py`:

```python
"""Property: the router's estimate_prompt_tokens never underestimates the
reference-tokenizer count on the calibration corpus. Pins the divisor to
the corpus it was calibrated on; new captures may require recalibration
(scripts/calibrate_router_divisor.py)."""
import json
import os
import pathlib

import pytest

from litmus_spec import count_ref_tokens, load_cases
from loxo_llm_router import estimate_prompt_tokens

CASES = "cases/main_replay.jsonl"


@pytest.fixture(scope="module")
def ref_tokenizer():
    meta_path = pathlib.Path("cases/main_replay.meta.json")
    repo = "mlx-community/Qwen3-14B-4bit"
    if meta_path.exists():
        repo = json.loads(meta_path.read_text()).get("tokenizer", repo)
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            repo, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"reference tokenizer {repo} not in local HF cache "
                    f"({type(e).__name__})")


def test_estimator_never_underestimates_on_corpus(ref_tokenizer):
    if not os.path.exists(CASES):
        pytest.skip(f"{CASES} missing")
    cases = load_cases(CASES, "main-replay")
    undercounts, skipped = [], []
    for case in cases:
        if not os.path.exists(case.capture_path):
            skipped.append(case.id)
            continue
        with open(case.capture_path, encoding="utf-8") as f:
            body = json.load(f)
        est = estimate_prompt_tokens(body)
        real = count_ref_tokens(ref_tokenizer, body)
        if est < real:
            undercounts.append(f"  {case.id}: est={est} < real={real}")
    if skipped:
        print(f"  (skipped: {', '.join(skipped)})")
    assert not undercounts, (
        "estimator underestimates — recalibrate the divisor "
        "(scripts/calibrate_router_divisor.py):\n" + "\n".join(undercounts))
```

Note: `load_cases(path, profile)` — confirm the profile string the runner uses is `"main-replay"` by checking the `CASES_BY_PROFILE` mapping at `litmus_spec.py:1323`; the key there is `"main-replay"`.

- [ ] **Step 2: Write the calibration script**

Create `scripts/calibrate_router_divisor.py`:

```python
#!/usr/bin/env python3
"""Calibrate the router's ESTIMATE_CHARS_PER_TOKEN divisor.

For every corpus case: chars = loxo_llm_router._count_prompt_chars(body),
real = count_ref_tokens(reference tokenizer, body). The safe divisor is
min(chars/real) over the corpus, floored to 2 decimals — then
chars/divisor >= real everywhere (never underestimates). Prints a per-case
table, the recommended divisor, and worst-case under/over margins at that
divisor. Paste the value into loxo_llm_router/__init__.py.
"""
import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from litmus_spec import count_ref_tokens, load_cases  # noqa: E402
from loxo_llm_router import _count_prompt_chars       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="cases/main_replay.jsonl")
    ap.add_argument("--tokenizer",
                    default="mlx-community/Qwen3-14B-4bit")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                            local_files_only=True)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"reference tokenizer {args.tokenizer!r} not cached "
                 f"({type(e).__name__}: {e}); run: hf download {args.tokenizer}")

    rows = []
    for case in load_cases(args.cases, "main-replay"):
        with open(case.capture_path, encoding="utf-8") as f:
            body = json.load(f)
        chars = _count_prompt_chars(body)
        real = count_ref_tokens(tok, body)
        rows.append((case.id, chars, real, chars / real))

    rows.sort(key=lambda r: r[3])
    print(f"{'case':8} {'chars':>10} {'ref_tokens':>10} {'chars/tok':>10}")
    for cid, chars, real, ratio in rows:
        print(f"{cid:8} {chars:>10} {real:>10} {ratio:>10.3f}")

    divisor = math.floor(rows[0][3] * 100) / 100
    print(f"\nrecommended ESTIMATE_CHARS_PER_TOKEN = {divisor}")
    margins = [(int(chars / divisor) - real) / real
               for _, chars, real, _ in rows]
    print(f"margins at that divisor: worst under {min(margins):+.1%}, "
          f"worst over {max(margins):+.1%}")
    if min(margins) < 0:
        sys.exit("floor produced an undercount — investigate before pinning")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the calibration**

Run (llitmus-eval root): `python scripts/calibrate_router_divisor.py`
Expected: a 15-row table, a recommended divisor, non-negative worst-under margin. Record all three numbers.

- [ ] **Step 4: Pin the divisor and sync the docs**

- In `loxo_llm_router/__init__.py`: set `ESTIMATE_CHARS_PER_TOKEN` to the recommended value and rewrite its comment to state the calibration source: corpus file, tokenizer repo, date, worst under/over margins. Delete the "PROVISIONAL" wording.
- In the spec's Section 4 (llitmus-eval `docs/superpowers/specs/2026-07-29-token-accounting-context-gate-design.md`): replace the provisional 3.6/`+0.5%`/`+40.6%` numbers with the measured ones.

- [ ] **Step 5: Run both suites**

Run: `python -m pytest tests/test_router_divisor_property.py -q` (llitmus-eval — must PASS on this machine, not skip) and `python -m pytest -q` (router root; the estimator tests derive from the constant, so they pass unchanged).

- [ ] **Step 6: Commit — one commit per repo**

```bash
# llitmus-eval
git add scripts/calibrate_router_divisor.py tests/test_router_divisor_property.py docs/superpowers/specs/2026-07-29-token-accounting-context-gate-design.md
git commit -m "feat: divisor calibration script + never-underestimates property test"
# loxo-llm-router
cd /Users/brendanl/src/loxo-llm-router
git add loxo_llm_router/__init__.py
git commit -m "feat(estimator): pin divisor calibrated against the Qwen3-14B reference tokenizer"
```

---

### Task 9: Router — probed threshold, truthful /v1/models

**Files:**
- Modify: `loxo_llm_router/config.py:104-105` (+ `Config` dataclass field type at line 29)
- Modify: `loxo_llm_router/__init__.py`: constant at line 131, `pick_target` (lines ~695, ~705), `_local_pin_preflight` (line ~735), `_virtual_model_entries` local branch (line ~1141), config/health report at line 1214, new probe + accessor + startup hook near the `app = FastAPI()` at line 347
- Test: `test_routing.py`, `test_metadata.py`, new `test_context_probe.py` (router root, matching the flat-test convention)

**Interfaces:**
- Consumes: nothing from other tasks (independent of Task 7/8 — touches the threshold, not the estimator).
- Produces:
  - `Config.local_context_limit: int | None` — `None` unless explicitly set via env/TOML.
  - `LOCAL_CONTEXT_LIMIT: int | None` module global (explicit override or None).
  - `_derived_local_context: int | None` module global (probe result cache).
  - `_context_from_models_payload(payload: dict) -> int | None` (pure parser).
  - `async probe_local_context() -> None` (fills the cache; never raises).
  - `effective_local_context() -> int` — precedence: explicit > probe > `LEGACY_CONTEXT_DEFAULT = 60000`.

- [ ] **Step 1: Write the failing tests**

Create `test_context_probe.py` (router root):

```python
"""Derived local-context threshold: explicit config > startup probe of
{LOCAL_BASE_URL}/models > legacy 60000 default. A down local server must
never block routing (probe failure -> cache stays None -> fallback)."""
import loxo_llm_router as R


def test_payload_parser_context_length():
    payload = {"data": [{"id": "m", "context_length": 262144}]}
    assert R._context_from_models_payload(payload) == 262144


def test_payload_parser_max_model_len():
    payload = {"data": [{"id": "m", "max_model_len": 40960}]}
    assert R._context_from_models_payload(payload) == 40960


def test_payload_parser_min_across_entries():
    # Conservative: a server hosting several models gates at the smallest.
    payload = {"data": [{"context_length": 262144},
                        {"context_length": 40960}]}
    assert R._context_from_models_payload(payload) == 40960


def test_payload_parser_absent_or_garbage_is_none():
    assert R._context_from_models_payload({"data": [{"id": "m"}]}) is None
    assert R._context_from_models_payload({}) is None
    assert R._context_from_models_payload(
        {"data": [{"context_length": "big"}]}) is None


def test_effective_explicit_config_wins(monkeypatch):
    monkeypatch.setattr(R, "LOCAL_CONTEXT_LIMIT", 12345)
    monkeypatch.setattr(R, "_derived_local_context", 262144)
    assert R.effective_local_context() == 12345


def test_effective_probe_when_no_explicit(monkeypatch):
    monkeypatch.setattr(R, "LOCAL_CONTEXT_LIMIT", None)
    monkeypatch.setattr(R, "_derived_local_context", 262144)
    assert R.effective_local_context() == 262144


def test_effective_legacy_default_last(monkeypatch):
    monkeypatch.setattr(R, "LOCAL_CONTEXT_LIMIT", None)
    monkeypatch.setattr(R, "_derived_local_context", None)
    assert R.effective_local_context() == 60000
```

In `test_metadata.py` (the `/v1/models` local-pinned context test around line 33): the existing `monkeypatch.setattr(R, "LOCAL_CONTEXT_LIMIT", 60000)` keeps working (explicit override wins) — leave it, but add one test asserting the probed value flows through when no explicit limit is set:

```python
def test_local_pinned_tier_advertises_probed_context(monkeypatch, client):
    monkeypatch.setattr(R, "LOCAL_CONTEXT_LIMIT", None)
    monkeypatch.setattr(R, "_derived_local_context", 262144)
    r = client.get("/v1/models")
    local = [m for m in r.json()["data"]
             if m["id"].endswith("/local")]  # adjust to the actual pinned-local tier id used in this test module
    assert local and local[0]["context_length"] == 262144
```

(Adjust the tier-id selection to match how `test_metadata.py` already identifies the local-pinned tier — copy its existing lookup.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_context_probe.py -q`
Expected: FAIL — names don't exist

- [ ] **Step 3: Implement**

`loxo_llm_router/config.py`:
- Line 29: `local_context_limit: int | None`
- Lines 104–105:

```python
    _raw_limit = os.environ.get("LOCAL_CONTEXT_LIMIT") \
        or backends.get("local_context_limit")
    # None = not explicitly configured -> derive from the local server's
    # /models at startup; 60000 is only the last-resort legacy default.
    local_context_limit = int(_raw_limit) if _raw_limit is not None else None
```

`loxo_llm_router/__init__.py`:

At line 131:

```python
LOCAL_CONTEXT_LIMIT = _cfg.local_context_limit  # None unless explicitly set
LEGACY_CONTEXT_DEFAULT = 60000
_derived_local_context: int | None = None
```

Near `app = FastAPI()` (line 347):

```python
def _context_from_models_payload(payload) -> int | None:
    """Smallest declared context among the local server's models
    (context_length, vLLM's max_model_len). None when absent/garbage."""
    vals = []
    for m in (payload or {}).get("data", []) or []:
        if not isinstance(m, dict):
            continue
        v = m.get("context_length") or m.get("max_model_len")
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            vals.append(v)
    return min(vals) if vals else None


async def probe_local_context() -> None:
    """One startup probe of the local backend's /models; cached for the
    process lifetime. Failure caches None — a down local server must not
    block requests (they fall back to the explicit/legacy limit)."""
    global _derived_local_context
    try:
        async with httpx.AsyncClient(timeout=LOCAL_CONNECT_TIMEOUT) as client:
            r = await client.get(f"{LOCAL_BASE_URL}/models")
            _derived_local_context = _context_from_models_payload(r.json())
    except Exception:  # noqa: BLE001 - any failure means "unknown", never a crash
        _derived_local_context = None


@app.on_event("startup")
async def _startup_probe_local_context():
    await probe_local_context()


def effective_local_context() -> int:
    """Routing threshold precedence: explicit config > startup probe >
    legacy default. Explicit stays first so operators (and tests
    monkeypatching LOCAL_CONTEXT_LIMIT) keep a working override."""
    if LOCAL_CONTEXT_LIMIT is not None:
        return LOCAL_CONTEXT_LIMIT
    if _derived_local_context is not None:
        return _derived_local_context
    return LEGACY_CONTEXT_DEFAULT
```

Replace every routing/advertising read of `LOCAL_CONTEXT_LIMIT` with `effective_local_context()`:
- `pick_target`: both `estimate_prompt_tokens(body) > LOCAL_CONTEXT_LIMIT` comparisons (lines ~695, ~705)
- `_local_pin_preflight`: the comparison **and** the 422 message's `{LOCAL_CONTEXT_LIMIT}` interpolation (line ~735)
- `_virtual_model_entries`: the `elif vm.routing == "local": ctx = LOCAL_CONTEXT_LIMIT` branch (line ~1141) → `ctx = effective_local_context()`
- the report at line 1214: `"local_context_limit": effective_local_context(),` plus a sibling `"local_context_source": ("config-explicit" if LOCAL_CONTEXT_LIMIT is not None else "probe" if _derived_local_context is not None else "legacy-default"),`

Grep to confirm no stragglers: `grep -n "LOCAL_CONTEXT_LIMIT" loxo_llm_router/__init__.py` — remaining hits should be only the definition, `effective_local_context`, and the source-string logic.

- [ ] **Step 4: Run the full router suite**

Run: `python -m pytest -q`
Expected: all pass. The pre-existing monkeypatch sites (`test_routing.py:26,78,184,455,464,494,512`, `test_metadata.py:33`) pass unchanged because explicit-override reads the module global at call time. If any fail, fix the *call site* to use `effective_local_context()` rather than weakening a test.

- [ ] **Step 5: Confirm real-server behavior matches the recorded finding**

Already verified 2026-07-29: mlx_lm's `/v1/models` on `localhost:7979` reports **neither** `context_length` nor `max_model_len` (entries carry only `id`/`object`/`created`), so on this stack the probe derives None and the env file's explicit `LOCAL_CONTEXT_LIMIT=32768` governs. The probe stays in scope because it self-activates on a backend that does report context (e.g. vLLM's `max_model_len`). If the server is up, re-run as a spot check:

```bash
curl -s http://localhost:7979/v1/models | python3 -m json.tool | head -30
```

Expected: no context field → router log/config report shows `local_context_source: "config-explicit"` with value 32768. If the server is down, skip the spot check — do not block the task on it.

- [ ] **Step 6: Commit**

```bash
git add loxo_llm_router/config.py loxo_llm_router/__init__.py test_context_probe.py test_metadata.py
git commit -m "feat(routing): derive local context from /models probe; truthful /v1/models"
```

---

## Final verification (both repos)

- [ ] llitmus-eval: `python -m pytest tests/ -q` — all pass; drift check and divisor property test PASS (not skipped) on this machine.
- [ ] loxo-llm-router: `python -m pytest -q` — all pass.
- [ ] `grep -rn "est_tokens" --include="*.py"` in llitmus-eval — no hits.
- [ ] `grep -rn "60000" loxo_llm_router/` — hits only `LEGACY_CONTEXT_DEFAULT` and config comments.
- [ ] Smoke: `python litmus_spec.py --profile main-replay --repo mlx-community/Qwen3-4B-4bit --label smoke --max-tokens 64` (or the repo's documented invocation — check `python litmus_spec.py --help` first) and confirm the sidecar records `"context_length": 40960, "context_length_source": "config"`, with deep cases (> 40,960 ref tokens) landing in `errored` as over-context.
