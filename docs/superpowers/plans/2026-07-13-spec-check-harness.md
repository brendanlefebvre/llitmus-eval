# Spec-check Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a unified spec-check harness to Litmus that measures two model-level capabilities — tool-calling and instruction/constraint-following — via one case→validate→aggregate runner.

**Architecture:** A single new module `litmus_spec.py` holds a string-keyed validator registry, JSONL case loader, tool-call parsers, per-profile scoring, a model-agnostic runner, and CLI. Shared model-loading helpers are extracted from `litmus.py` into `litmus_common.py` and imported by both. Cases are data (JSONL) under `cases/`. The repo's first test suite (pytest) covers every pure unit against crafted strings and a fake model — no MLX weights needed for the logic tests.

**Tech Stack:** Python 3 (stdlib only for the harness logic: `json`, `re`, `dataclasses`, `argparse`), MLX / mlx_lm (only in the runner's real model path, unchanged from litmus.py), pytest (dev-only, new).

## Global Constraints

- **Zero new runtime dependencies.** Harness logic uses the Python stdlib only. No PyYAML, no external eval libs. pytest is a **dev** dependency only.
- **Case files are JSONL** (one JSON object per line), stored under `cases/`.
- **No LLM judge, no reference model.** Every check is a regex/parser producing an objective bool.
- **Report capability under a named convention**, never as convention-free. Tool-calling always reports the prompted column; the native column is added only when the model's chat template supports `tools=`.
- **Fail fast on malformed cases.** Unknown check `kind`, missing required fields, or wrong param types raise at load, before any model loads.
- **Follow the existing single-file-module pattern** (litmus.py is one file). `litmus_spec.py` is one module; tests are split by concern under `tests/`.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

## Core types (defined once, referenced by every task)

```python
# CheckResult — every constraint validator returns this
@dataclass
class CheckResult:
    passed: bool
    detail: str          # human-readable why (e.g. "found 4 bullets, expected 3")

# A constraint validator is: Callable[[str, dict], CheckResult]
# CHECKS registry entry
@dataclass
class ConstraintCheck:
    fn: "Callable[[str, dict], CheckResult]"
    params: dict         # param-name -> python type, e.g. {"n": int}

CHECKS: "dict[str, ConstraintCheck]" = {}   # kind -> ConstraintCheck

# Case types (loader output)
@dataclass
class ConstraintCase:
    id: str
    prompt: str
    checks: "list[tuple[str, dict]]"   # (kind, params), kinds all in CHECKS

@dataclass
class ToolCase:
    id: str
    prompt: str
    tools: "list[dict]"                # JSON-Schema objects
    expect: dict                       # {"tool": str|None, "arguments": dict}

# Parsed tool call (parser output)
@dataclass
class ParsedCall:
    well_formed: bool
    tool: "str | None"                 # None == explicit no-call
    arguments: "dict | None"
    detail: str                        # e.g. "ok", "truncated", "no json found"
```

These names/types are the contract between tasks. Do not rename them.

---

### Task 1: Constraint validators + registry (introduces pytest)

**Files:**
- Create: `litmus_spec.py`
- Create: `tests/test_spec_validators.py`
- Modify: `pyproject.toml` (add pytest dev dependency + config)

**Interfaces:**
- Consumes: nothing.
- Produces: `CheckResult`, `ConstraintCheck`, `CHECKS` registry, and a `register(kind, params)` decorator. Validators keyed in `CHECKS`: `exact_bullets{n:int}`, `min_words{n:int}`, `max_words{n:int}`, `all_lowercase{}`, `all_uppercase{}`, `forbidden_word{word:str}`, `required_phrase{phrase:str}`, `ends_with{phrase:str}`, `valid_json{}`, `regex_match{pattern:str}`. Each validator: `(text: str, params: dict) -> CheckResult`.

- [ ] **Step 1: Add pytest to pyproject and configure test discovery**

In `pyproject.toml`, add:
```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write the failing tests for validators**

Create `tests/test_spec_validators.py`:
```python
from litmus_spec import CHECKS, CheckResult


def run(kind, text, **params):
    return CHECKS[kind].fn(text, params)


def test_exact_bullets_pass_and_fail():
    three = "- a\n- b\n- c"
    assert run("exact_bullets", three, n=3).passed is True
    assert run("exact_bullets", "- a\n- b", n=3).passed is False
    assert run("exact_bullets", "- a\n- b\n- c\n- d", n=3).passed is False


def test_exact_bullets_counts_numbered_and_star():
    assert run("exact_bullets", "1. a\n2. b\n3. c", n=3).passed is True
    assert run("exact_bullets", "* a\n* b\n* c", n=3).passed is True


def test_min_and_max_words():
    assert run("min_words", "one two three", n=3).passed is True
    assert run("min_words", "one two", n=3).passed is False
    assert run("max_words", "one two", n=3).passed is True
    assert run("max_words", "one two three four", n=3).passed is False


def test_casing():
    assert run("all_lowercase", "hello there").passed is True
    assert run("all_lowercase", "Hello").passed is False
    assert run("all_uppercase", "HELLO 1!").passed is True
    assert run("all_uppercase", "Hello").passed is False


def test_forbidden_word_is_case_insensitive_word_boundary():
    assert run("forbidden_word", "I like cats", word="dog").passed is True
    assert run("forbidden_word", "A DOG appeared", word="dog").passed is False
    # substring that is not the whole word does not trip it
    assert run("forbidden_word", "dogma is fine", word="dog").passed is True


def test_required_phrase_and_ends_with():
    assert run("required_phrase", "the answer is 42", phrase="answer is").passed is True
    assert run("required_phrase", "nope", phrase="answer is").passed is False
    assert run("ends_with", "goodbye now  ", phrase="now").passed is True
    assert run("ends_with", "now then", phrase="now").passed is False


def test_valid_json():
    assert run("valid_json", '{"a": 1}').passed is True
    assert run("valid_json", 'not json').passed is False


def test_regex_match():
    assert run("regex_match", "abc123", pattern=r"\d{3}").passed is True
    assert run("regex_match", "abc", pattern=r"\d{3}").passed is False


def test_result_is_checkresult_with_detail():
    r = run("exact_bullets", "- a\n- b", n=3)
    assert isinstance(r, CheckResult)
    assert r.detail  # non-empty explanation
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_spec_validators.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'litmus_spec'` (or import error for `CHECKS`).

- [ ] **Step 4: Implement the registry and validators**

Create `litmus_spec.py`:
```python
"""Litmus spec-check harness — model-level tool-calling and
instruction/constraint-following capability evals.

One case -> validate -> aggregate runner. Cases are JSONL data under cases/.
No LLM judge, no reference model: every check is an objective parser.
See docs/superpowers/specs/2026-07-13-spec-check-harness-design.md.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# constraint validators + registry
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    passed: bool
    detail: str


@dataclass
class ConstraintCheck:
    fn: Callable[[str, dict], CheckResult]
    params: dict  # param name -> expected python type


CHECKS: dict[str, ConstraintCheck] = {}


def register(kind: str, params: Optional[dict] = None):
    """Register a constraint validator under `kind` with a param type spec."""
    def deco(fn: Callable[[str, dict], CheckResult]) -> Callable:
        CHECKS[kind] = ConstraintCheck(fn=fn, params=params or {})
        return fn
    return deco


_BULLET_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+\S", re.MULTILINE)


def _count_bullets(text: str) -> int:
    return len(_BULLET_RE.findall(text))


@register("exact_bullets", {"n": int})
def _exact_bullets(text: str, params: dict) -> CheckResult:
    n = params["n"]
    got = _count_bullets(text)
    return CheckResult(got == n, f"found {got} bullets, expected {n}")


@register("min_words", {"n": int})
def _min_words(text: str, params: dict) -> CheckResult:
    got = len(text.split())
    return CheckResult(got >= params["n"], f"{got} words, need >= {params['n']}")


@register("max_words", {"n": int})
def _max_words(text: str, params: dict) -> CheckResult:
    got = len(text.split())
    return CheckResult(got <= params["n"], f"{got} words, need <= {params['n']}")


@register("all_lowercase")
def _all_lowercase(text: str, params: dict) -> CheckResult:
    ok = text == text.lower()
    return CheckResult(ok, "all lowercase" if ok else "contains uppercase")


@register("all_uppercase")
def _all_uppercase(text: str, params: dict) -> CheckResult:
    ok = text == text.upper()
    return CheckResult(ok, "all uppercase" if ok else "contains lowercase")


@register("forbidden_word", {"word": str})
def _forbidden_word(text: str, params: dict) -> CheckResult:
    word = params["word"]
    hit = re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE)
    return CheckResult(hit is None, f"'{word}' {'present' if hit else 'absent'}")


@register("required_phrase", {"phrase": str})
def _required_phrase(text: str, params: dict) -> CheckResult:
    phrase = params["phrase"]
    ok = phrase.lower() in text.lower()
    return CheckResult(ok, f"'{phrase}' {'present' if ok else 'missing'}")


@register("ends_with", {"phrase": str})
def _ends_with(text: str, params: dict) -> CheckResult:
    phrase = params["phrase"]
    ok = text.rstrip().lower().endswith(phrase.lower())
    return CheckResult(ok, f"ends with '{phrase}': {ok}")


@register("valid_json")
def _valid_json(text: str, params: dict) -> CheckResult:
    try:
        json.loads(text)
        return CheckResult(True, "parsed as JSON")
    except (ValueError, TypeError) as e:
        return CheckResult(False, f"not JSON: {e}")


@register("regex_match", {"pattern": str})
def _regex_match(text: str, params: dict) -> CheckResult:
    ok = re.search(params["pattern"], text) is not None
    return CheckResult(ok, f"pattern {'matched' if ok else 'no match'}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_spec_validators.py -q`
Expected: PASS (all validator tests green).

- [ ] **Step 6: Commit**

```bash
git add litmus_spec.py tests/test_spec_validators.py pyproject.toml
git commit -m "feat: constraint validators + registry for spec-check harness

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Constraint scoring (strict / loose)

**Files:**
- Modify: `litmus_spec.py`
- Create: `tests/test_spec_scoring.py`

**Interfaces:**
- Consumes: `CHECKS`, `CheckResult`, `ConstraintCase`.
- Produces: `score_constraint_case(text: str, case: ConstraintCase) -> list[dict]` returning `[{"kind": str, "passed": bool, "detail": str}, ...]`; and `aggregate_constraints(per_case: list[list[dict]]) -> dict` returning `{"strict": float, "loose": float, "by_kind": {kind: float}, "n_cases": int}`. `strict` = fraction of cases where all checks passed; `loose` = fraction of all individual checks passed.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spec_scoring.py`:
```python
from litmus_spec import (
    ConstraintCase, score_constraint_case, aggregate_constraints,
)


def test_score_constraint_case_runs_all_checks():
    case = ConstraintCase(id="c1", prompt="p",
                          checks=[("exact_bullets", {"n": 2}), ("all_lowercase", {})])
    results = score_constraint_case("- a\n- b", case)
    assert [r["kind"] for r in results] == ["exact_bullets", "all_lowercase"]
    assert all(r["passed"] for r in results)


def test_aggregate_strict_requires_all_checks_in_a_case():
    # case A: both pass; case B: one fails
    a = [{"kind": "exact_bullets", "passed": True, "detail": ""},
         {"kind": "all_lowercase", "passed": True, "detail": ""}]
    b = [{"kind": "exact_bullets", "passed": True, "detail": ""},
         {"kind": "all_lowercase", "passed": False, "detail": ""}]
    agg = aggregate_constraints([a, b])
    assert agg["strict"] == 0.5            # 1 of 2 cases fully passed
    assert agg["loose"] == 3 / 4           # 3 of 4 checks passed
    assert agg["by_kind"]["all_lowercase"] == 0.5
    assert agg["by_kind"]["exact_bullets"] == 1.0
    assert agg["n_cases"] == 2


def test_aggregate_empty_is_safe():
    agg = aggregate_constraints([])
    assert agg["strict"] == 0.0 and agg["loose"] == 0.0 and agg["n_cases"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spec_scoring.py -q`
Expected: FAIL — `ImportError: cannot import name 'ConstraintCase'` / `score_constraint_case`.

- [ ] **Step 3: Implement the case type and scoring**

Append to `litmus_spec.py`:
```python
# ---------------------------------------------------------------------------
# case types
# ---------------------------------------------------------------------------

@dataclass
class ConstraintCase:
    id: str
    prompt: str
    checks: list  # list[tuple[str, dict]] — (kind, params)


@dataclass
class ToolCase:
    id: str
    prompt: str
    tools: list   # list[dict] JSON-Schema
    expect: dict  # {"tool": str|None, "arguments": dict}


# ---------------------------------------------------------------------------
# constraint scoring
# ---------------------------------------------------------------------------

def score_constraint_case(text: str, case: "ConstraintCase") -> list:
    out = []
    for kind, params in case.checks:
        res = CHECKS[kind].fn(text, params)
        out.append({"kind": kind, "passed": res.passed, "detail": res.detail})
    return out


def aggregate_constraints(per_case: list) -> dict:
    n_cases = len(per_case)
    strict_pass = sum(1 for checks in per_case if checks and all(c["passed"] for c in checks))
    all_checks = [c for checks in per_case for c in checks]
    loose = (sum(1 for c in all_checks if c["passed"]) / len(all_checks)) if all_checks else 0.0
    by_kind: dict = {}
    for c in all_checks:
        by_kind.setdefault(c["kind"], []).append(c["passed"])
    by_kind_rate = {k: sum(v) / len(v) for k, v in by_kind.items()}
    return {
        "strict": (strict_pass / n_cases) if n_cases else 0.0,
        "loose": loose,
        "by_kind": by_kind_rate,
        "n_cases": n_cases,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spec_scoring.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add litmus_spec.py tests/test_spec_scoring.py
git commit -m "feat: constraint strict/loose scoring + case types

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: JSONL case loader with validation

**Files:**
- Modify: `litmus_spec.py`
- Create: `tests/test_spec_loader.py`

**Interfaces:**
- Consumes: `CHECKS`, `ConstraintCase`, `ToolCase`.
- Produces: `load_cases(path: str, profile: str) -> list` (profile in `{"constraints", "tool-calling"}`); `CaseError(Exception)`. For `constraints`: each line needs `id`, `prompt`, `checks` (list of `{kind, ...params}`); every `kind` must be in `CHECKS`; each declared param must be present and match the registered python type. For `tool-calling`: each line needs `id`, `prompt`, `tools` (list), `expect` (`{"tool": str|None, "arguments": dict}`). Any violation raises `CaseError` naming the line number and reason. Loading happens before any model loads.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spec_loader.py`:
```python
import pytest
from litmus_spec import load_cases, CaseError, ConstraintCase, ToolCase


def write(tmp_path, name, *lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_load_valid_constraint_cases(tmp_path):
    path = write(
        tmp_path, "c.jsonl",
        '{"id":"a","prompt":"p","checks":[{"kind":"exact_bullets","n":3}]}',
        '{"id":"b","prompt":"q","checks":[{"kind":"all_lowercase"}]}',
    )
    cases = load_cases(path, "constraints")
    assert len(cases) == 2
    assert isinstance(cases[0], ConstraintCase)
    assert cases[0].checks == [("exact_bullets", {"n": 3})]


def test_load_valid_tool_cases(tmp_path):
    path = write(
        tmp_path, "t.jsonl",
        '{"id":"t1","prompt":"weather?","tools":[{"name":"get_weather"}],'
        '"expect":{"tool":"get_weather","arguments":{"location":"Paris"}}}',
        '{"id":"t2","prompt":"hi","tools":[{"name":"get_weather"}],'
        '"expect":{"tool":null}}',
    )
    cases = load_cases(path, "tool-calling")
    assert isinstance(cases[0], ToolCase)
    assert cases[1].expect["tool"] is None


def test_unknown_kind_raises(tmp_path):
    path = write(tmp_path, "c.jsonl",
                 '{"id":"a","prompt":"p","checks":[{"kind":"no_such_check"}]}')
    with pytest.raises(CaseError, match="no_such_check"):
        load_cases(path, "constraints")


def test_missing_field_raises(tmp_path):
    path = write(tmp_path, "c.jsonl", '{"id":"a","checks":[]}')  # no prompt
    with pytest.raises(CaseError, match="prompt"):
        load_cases(path, "constraints")


def test_bad_param_type_raises(tmp_path):
    path = write(tmp_path, "c.jsonl",
                 '{"id":"a","prompt":"p","checks":[{"kind":"exact_bullets","n":"three"}]}')
    with pytest.raises(CaseError, match="n"):
        load_cases(path, "constraints")


def test_line_number_in_error(tmp_path):
    path = write(
        tmp_path, "c.jsonl",
        '{"id":"a","prompt":"p","checks":[{"kind":"all_lowercase"}]}',
        '{"id":"b","prompt":"p","checks":[{"kind":"bogus"}]}',
    )
    with pytest.raises(CaseError, match="line 2"):
        load_cases(path, "constraints")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spec_loader.py -q`
Expected: FAIL — `ImportError: cannot import name 'load_cases'`.

- [ ] **Step 3: Implement the loader**

Append to `litmus_spec.py`:
```python
# ---------------------------------------------------------------------------
# case loader
# ---------------------------------------------------------------------------

class CaseError(Exception):
    pass


def _require(obj: dict, key: str, ln: int):
    if key not in obj:
        raise CaseError(f"line {ln}: missing required field '{key}'")
    return obj[key]


def _load_constraint_line(obj: dict, ln: int) -> "ConstraintCase":
    cid = _require(obj, "id", ln)
    prompt = _require(obj, "prompt", ln)
    raw_checks = _require(obj, "checks", ln)
    checks = []
    for chk in raw_checks:
        kind = chk.get("kind")
        if kind not in CHECKS:
            raise CaseError(f"line {ln}: unknown check kind '{kind}'")
        spec = CHECKS[kind].params
        params = {}
        for pname, ptype in spec.items():
            if pname not in chk:
                raise CaseError(f"line {ln}: check '{kind}' missing param '{pname}'")
            val = chk[pname]
            if not isinstance(val, ptype) or isinstance(val, bool) and ptype is int:
                raise CaseError(
                    f"line {ln}: check '{kind}' param '{pname}' must be "
                    f"{ptype.__name__}, got {type(val).__name__}")
            params[pname] = val
        checks.append((kind, params))
    return ConstraintCase(id=cid, prompt=prompt, checks=checks)


def _load_tool_line(obj: dict, ln: int) -> "ToolCase":
    cid = _require(obj, "id", ln)
    prompt = _require(obj, "prompt", ln)
    tools = _require(obj, "tools", ln)
    expect = _require(obj, "expect", ln)
    if "tool" not in expect:
        raise CaseError(f"line {ln}: expect missing 'tool'")
    if not isinstance(tools, list) or not tools:
        raise CaseError(f"line {ln}: 'tools' must be a non-empty list")
    return ToolCase(id=cid, prompt=prompt, tools=tools, expect=expect)


def load_cases(path: str, profile: str) -> list:
    loader = {"constraints": _load_constraint_line,
              "tool-calling": _load_tool_line}.get(profile)
    if loader is None:
        raise CaseError(f"unknown profile '{profile}'")
    cases = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError as e:
                raise CaseError(f"line {i}: invalid JSON ({e})")
            cases.append(loader(obj, i))
    return cases
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spec_loader.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add litmus_spec.py tests/test_spec_loader.py
git commit -m "feat: JSONL case loader with fail-fast validation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Tool-call output parsers (prompted + native)

**Files:**
- Modify: `litmus_spec.py`
- Create: `tests/test_spec_parsers.py`

**Interfaces:**
- Consumes: `ParsedCall`.
- Produces: `parse_prompted(text: str) -> ParsedCall` and `parse_native(text: str) -> ParsedCall`. Both extract `{tool, arguments}`. `parse_prompted` reads the first balanced JSON object (tolerating markdown fences / leading prose) shaped `{"tool": name|null, "arguments": {...}}`; a valid object with `"tool": null` is a well-formed abstention (`tool=None`). `parse_native` tries, in order: `<tool_call>…</tool_call>` (Qwen/Hermes, JSON inside, `name`/`arguments`), `<|python_tag|>` (Llama, JSON follows), then a generic balanced-JSON fallback. On no parse: `ParsedCall(well_formed=False, tool=None, arguments=None, detail=...)`. Empty/whitespace text → `detail="truncated"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spec_parsers.py`:
```python
from litmus_spec import parse_prompted, parse_native, ParsedCall


def test_prompted_plain_json():
    r = parse_prompted('{"tool": "get_weather", "arguments": {"location": "Paris"}}')
    assert r.well_formed and r.tool == "get_weather"
    assert r.arguments == {"location": "Paris"}


def test_prompted_fenced_with_preamble():
    text = 'Sure!\n```json\n{"tool": "send_email", "arguments": {"to": "a@b.c"}}\n```'
    r = parse_prompted(text)
    assert r.well_formed and r.tool == "send_email"


def test_prompted_explicit_null_is_wellformed_abstention():
    r = parse_prompted('{"tool": null, "arguments": {}}')
    assert r.well_formed is True and r.tool is None


def test_prompted_garbage_is_not_wellformed():
    r = parse_prompted("I cannot help with that.")
    assert r.well_formed is False and r.tool is None


def test_prompted_empty_is_truncated():
    r = parse_prompted("   ")
    assert r.well_formed is False and r.detail == "truncated"


def test_native_qwen_tool_call_tag():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'
    r = parse_native(text)
    assert r.well_formed and r.tool == "get_weather"
    assert r.arguments == {"location": "Paris"}


def test_native_llama_python_tag():
    text = '<|python_tag|>{"name": "get_weather", "arguments": {"location": "Rome"}}'
    r = parse_native(text)
    assert r.well_formed and r.tool == "get_weather" and r.arguments["location"] == "Rome"


def test_native_generic_json_fallback():
    text = '{"name": "search", "arguments": {"q": "cats"}}'
    r = parse_native(text)
    assert r.well_formed and r.tool == "search"


def test_native_unparseable():
    r = parse_native("no tool here")
    assert r.well_formed is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spec_parsers.py -q`
Expected: FAIL — `ImportError: cannot import name 'parse_prompted'`.

- [ ] **Step 3: Implement the parsers**

Append to `litmus_spec.py`:
```python
# ---------------------------------------------------------------------------
# tool-call parsers
# ---------------------------------------------------------------------------

@dataclass
class ParsedCall:
    well_formed: bool
    tool: Optional[str]
    arguments: Optional[dict]
    detail: str


def _first_json_object(text: str) -> Optional[dict]:
    """Return the first balanced {...} JSON object in text, or None."""
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        start = -1
    return None


def parse_prompted(text: str) -> ParsedCall:
    if not text.strip():
        return ParsedCall(False, None, None, "truncated")
    obj = _first_json_object(text)
    if obj is None or "tool" not in obj:
        return ParsedCall(False, None, None, "no {tool,arguments} object found")
    tool = obj["tool"]
    if tool is None:
        return ParsedCall(True, None, None, "explicit abstention")
    return ParsedCall(True, tool, obj.get("arguments") or {}, "ok")


def _call_from_name_obj(obj: Optional[dict], detail: str) -> ParsedCall:
    if obj is None or "name" not in obj:
        return ParsedCall(False, None, None, "no name/arguments object found")
    return ParsedCall(True, obj["name"], obj.get("arguments") or {}, detail)


def parse_native(text: str) -> ParsedCall:
    if not text.strip():
        return ParsedCall(False, None, None, "truncated")
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if m:
        return _call_from_name_obj(_first_json_object(m.group(1)), "qwen/hermes tag")
    m = re.search(r"<\|python_tag\|>(.*)", text, re.DOTALL)
    if m:
        return _call_from_name_obj(_first_json_object(m.group(1)), "llama python_tag")
    return _call_from_name_obj(_first_json_object(text), "generic json")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spec_parsers.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add litmus_spec.py tests/test_spec_parsers.py
git commit -m "feat: prompted + native tool-call output parsers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Tool-call four-dimension scoring

**Files:**
- Modify: `litmus_spec.py`
- Create: `tests/test_spec_tool_scoring.py`

**Interfaces:**
- Consumes: `ParsedCall`, `ToolCase`.
- Produces: `score_tool_call(parsed: ParsedCall, expect: dict) -> dict` returning `{"well_formed": bool, "right_tool": bool, "args_ok": bool|None, "abstained_ok": bool|None}`. **A not-well-formed parse can never earn credit** — a parse failure yields `tool=None`, which must not be mistaken for a deliberate abstention. So: for an abstention case (`expect["tool"] is None`), `abstained_ok = parsed.well_formed and parsed.tool is None` and `right_tool` equals `abstained_ok`; `args_ok` is `None`. For a real-call case, `right_tool = parsed.well_formed and parsed.tool == expect["tool"]`; `args_ok` is `None` when `right_tool` is false, otherwise true iff the keys of `expect["arguments"]` are present with equal values and there are no extra keys. Also `aggregate_tool(per_case: list[dict]) -> dict` with per-dimension pass rates (each dimension's denominator excludes `None` entries).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spec_tool_scoring.py`:
```python
from litmus_spec import ParsedCall, score_tool_call, aggregate_tool


def test_correct_call_all_pass():
    p = ParsedCall(True, "get_weather", {"location": "Paris"}, "ok")
    s = score_tool_call(p, {"tool": "get_weather", "arguments": {"location": "Paris"}})
    assert s == {"well_formed": True, "right_tool": True,
                 "args_ok": True, "abstained_ok": None}


def test_wrong_tool_makes_args_na():
    p = ParsedCall(True, "send_email", {"to": "x"}, "ok")
    s = score_tool_call(p, {"tool": "get_weather", "arguments": {"location": "Paris"}})
    assert s["right_tool"] is False and s["args_ok"] is None


def test_hallucinated_extra_arg_fails_args():
    p = ParsedCall(True, "get_weather", {"location": "Paris", "unit": "C"}, "ok")
    s = score_tool_call(p, {"tool": "get_weather", "arguments": {"location": "Paris"}})
    assert s["args_ok"] is False


def test_abstention_correct():
    p = ParsedCall(True, None, None, "abstain")
    s = score_tool_call(p, {"tool": None})
    assert s["abstained_ok"] is True and s["right_tool"] is True and s["args_ok"] is None


def test_abstention_violated_by_calling():
    p = ParsedCall(True, "get_weather", {"location": "X"}, "ok")
    s = score_tool_call(p, {"tool": None})
    assert s["abstained_ok"] is False and s["right_tool"] is False


def test_parse_failure_on_abstention_is_not_credited():
    # unparseable output also has tool=None, but it is NOT a real abstention
    p = ParsedCall(False, None, None, "no json found")
    s = score_tool_call(p, {"tool": None})
    assert s["well_formed"] is False
    assert s["abstained_ok"] is False and s["right_tool"] is False


def test_parse_failure_on_call_case_is_not_right():
    p = ParsedCall(False, None, None, "truncated")
    s = score_tool_call(p, {"tool": "get_weather", "arguments": {"location": "Paris"}})
    assert s["right_tool"] is False and s["args_ok"] is None


def test_aggregate_excludes_none_from_denominator():
    per_case = [
        {"well_formed": True, "right_tool": True, "args_ok": True, "abstained_ok": None},
        {"well_formed": True, "right_tool": False, "args_ok": None, "abstained_ok": None},
        {"well_formed": True, "right_tool": True, "args_ok": None, "abstained_ok": True},
    ]
    agg = aggregate_tool(per_case)
    assert agg["well_formed"] == 1.0
    assert agg["right_tool"] == 2 / 3
    assert agg["args_ok"] == 1.0          # only one non-None, and it passed
    assert agg["abstained_ok"] == 1.0     # only one non-None, and it passed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spec_tool_scoring.py -q`
Expected: FAIL — `ImportError: cannot import name 'score_tool_call'`.

- [ ] **Step 3: Implement tool scoring**

Append to `litmus_spec.py`:
```python
# ---------------------------------------------------------------------------
# tool-call scoring
# ---------------------------------------------------------------------------

def score_tool_call(parsed: "ParsedCall", expect: dict) -> dict:
    is_abstention = expect.get("tool") is None

    # A not-well-formed parse yields tool=None; never let that masquerade as a
    # deliberate no-call. All credit is gated on well_formed.
    abstained_ok = None
    args_ok = None
    if is_abstention:
        abstained_ok = parsed.well_formed and parsed.tool is None
        right_tool = abstained_ok
    else:
        right_tool = parsed.well_formed and parsed.tool == expect.get("tool")
        if right_tool:
            want = expect.get("arguments") or {}
            got = parsed.arguments or {}
            args_ok = (set(got.keys()) == set(want.keys())
                       and all(got.get(k) == v for k, v in want.items()))

    return {
        "well_formed": parsed.well_formed,
        "right_tool": right_tool,
        "args_ok": args_ok,
        "abstained_ok": abstained_ok,
    }


def _rate(vals: list) -> Optional[float]:
    present = [v for v in vals if v is not None]
    return (sum(1 for v in present if v) / len(present)) if present else None


def aggregate_tool(per_case: list) -> dict:
    dims = ["well_formed", "right_tool", "args_ok", "abstained_ok"]
    return {d: _rate([c[d] for c in per_case]) for d in dims}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spec_tool_scoring.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add litmus_spec.py tests/test_spec_tool_scoring.py
git commit -m "feat: four-dimension tool-call scoring + aggregation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Extract shared core into litmus_common.py

**Files:**
- Create: `litmus_common.py`
- Modify: `litmus.py:63-169` (remove moved helpers, import them instead)
- Create: `tests/test_common_import.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `litmus_common` exposing `MODELS`, `BASELINE_MODELS`, `PROMPTS`, `WARMUP_PROMPT`, `_resp_text`, `_peak_memory_mb`, `_reset_peak_memory`, `_clear_cache`, `_parse_sizes`, `_targets_for`, `_load_timed`. `litmus.py` imports these; its CLI behavior is unchanged.

- [ ] **Step 1: Write the failing test (behavior-preserving contract)**

Create `tests/test_common_import.py`:
```python
def test_common_exposes_shared_helpers():
    import litmus_common as lc
    for name in ["MODELS", "BASELINE_MODELS", "_parse_sizes",
                 "_targets_for", "_load_timed", "_clear_cache", "_resp_text"]:
        assert hasattr(lc, name), name


def test_parse_sizes_roundtrip():
    import litmus_common as lc
    assert lc._parse_sizes("1.7B,4B") == ["1.7B", "4B"]


def test_litmus_still_imports_and_reexports():
    import litmus
    assert litmus.MODELS == __import__("litmus_common").MODELS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_common_import.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'litmus_common'`.

- [ ] **Step 3: Create litmus_common.py with the moved helpers**

Create `litmus_common.py` by moving these verbatim from `litmus.py`: the MLX imports + `warnings.filterwarnings` block, `MODELS`, `BASELINE_MODELS`, `PROMPTS`, `WARMUP_PROMPT`, and the functions `_resp_text`, `_peak_memory_mb`, `_reset_peak_memory`, `_clear_cache`, `_parse_sizes`, `_targets_for`, `_load_timed`. Header:
```python
"""Shared core for Litmus: model tables, loading, and MLX memory helpers.

Imported by litmus.py (perf benchmarks) and litmus_spec.py (spec-check harness)
so the two share one loading/targeting path instead of drifting copies.
"""
from __future__ import annotations

import time
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*mx\.metal\.(clear_cache|get_peak_memory|reset_peak_memory).*deprecated.*",
)

import mlx.core as mx
from mlx_lm import load, stream_generate

# ... (MODELS, BASELINE_MODELS, PROMPTS, WARMUP_PROMPT, and the helper
#      functions listed above, moved verbatim from litmus.py) ...
```

Note: `_targets_for` references `args.repo`/`args.label`/`args.sizes` — keep it signature-identical.

- [ ] **Step 4: Rewire litmus.py to import from litmus_common**

In `litmus.py`, replace the moved definitions with a re-export import near the top (after the module docstring):
```python
from litmus_common import (
    MODELS, BASELINE_MODELS, PROMPTS, WARMUP_PROMPT,
    _resp_text, _peak_memory_mb, _reset_peak_memory, _clear_cache,
    _parse_sizes, _targets_for, _load_timed,
    mx, nn, load, stream_generate,
)
```
Adjust: `litmus_common` must also expose `nn` (add `import mlx.nn as nn` there) since `compute_perplexity` uses it. Delete the now-duplicated definitions and the duplicate MLX import block from `litmus.py`. Leave everything else (Run dataclass, PROMPTS usage, all cmd_* functions) untouched.

- [ ] **Step 5: Run the import test and litmus's own help to verify no regression**

Run: `python -m pytest tests/test_common_import.py -q && python litmus.py --help`
Expected: tests PASS; `--help` prints the usage text with no ImportError.

- [ ] **Step 6: Commit**

```bash
git add litmus_common.py litmus.py tests/test_common_import.py
git commit -m "refactor: extract shared model/loading core to litmus_common

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Elicitation — prompted preamble + native probe

**Files:**
- Modify: `litmus_spec.py`
- Create: `tests/test_spec_elicit.py`

**Interfaces:**
- Consumes: `ToolCase`.
- Produces: `build_prompted_tool_prompt(tokenizer, case: ToolCase) -> str`; `supports_native_tools(tokenizer) -> bool`; `build_native_tool_prompt(tokenizer, case: ToolCase) -> str`; `build_constraint_prompt(tokenizer, case) -> str`. All wrap via `tokenizer.apply_chat_template(...)`. The prompted builder injects a fixed system preamble listing `case.tools` and demanding a single `{"tool": name|null, "arguments": {...}}` object. `supports_native_tools` returns True iff `apply_chat_template(msgs, tools=[...])` succeeds and differs from the no-tools rendering. Tests use a fake tokenizer, no MLX.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spec_elicit.py`:
```python
import pytest
from litmus_spec import (
    ToolCase, build_prompted_tool_prompt, supports_native_tools,
    build_native_tool_prompt, build_constraint_prompt, ConstraintCase,
)


class FakeTokenizer:
    """apply_chat_template records what it got; supports tools= by default."""
    def __init__(self, tools_supported=True):
        self.tools_supported = tools_supported
        self.last_tools = None

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=False, tools=None):
        if tools is not None and not self.tools_supported:
            raise ValueError("template has no tools support")
        self.last_tools = tools
        body = " | ".join(m["content"] for m in messages)
        suffix = f" [tools={len(tools)}]" if tools else ""
        return f"<s>{body}{suffix}"


def test_prompted_prompt_has_no_tools_kwarg_and_lists_schema():
    tok = FakeTokenizer()
    case = ToolCase("t1", "weather?", [{"name": "get_weather"}],
                    {"tool": "get_weather", "arguments": {}})
    out = build_prompted_tool_prompt(tok, case)
    assert tok.last_tools is None            # prompted path never uses tools=
    assert "get_weather" in out              # schema is in the preamble
    assert "json" in out.lower()


def test_supports_native_true_and_false():
    assert supports_native_tools(FakeTokenizer(tools_supported=True)) is True
    assert supports_native_tools(FakeTokenizer(tools_supported=False)) is False


def test_native_prompt_passes_tools_kwarg():
    tok = FakeTokenizer()
    case = ToolCase("t1", "weather?", [{"name": "get_weather"}],
                    {"tool": "get_weather", "arguments": {}})
    build_native_tool_prompt(tok, case)
    assert tok.last_tools == case.tools


def test_constraint_prompt_wraps_user_message():
    tok = FakeTokenizer()
    case = ConstraintCase("c1", "list 3 fruits", [("exact_bullets", {"n": 3})])
    out = build_constraint_prompt(tok, case)
    assert "list 3 fruits" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spec_elicit.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_prompted_tool_prompt'`.

- [ ] **Step 3: Implement elicitation**

Append to `litmus_spec.py`:
```python
# ---------------------------------------------------------------------------
# elicitation
# ---------------------------------------------------------------------------

_PROMPTED_SYSTEM = (
    "You have access to these tools (JSON schemas):\n{schemas}\n\n"
    "Decide whether a tool is needed for the user's request. Respond with "
    "ONLY a single JSON object and nothing else, in exactly this shape:\n"
    '{{"tool": <tool name as a string, or null if no tool applies>, '
    '"arguments": {{<argument name>: <value>, ...}}}}'
)


def _chat(tokenizer, messages, tools=None) -> str:
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, tools=tools)


def build_prompted_tool_prompt(tokenizer, case: "ToolCase") -> str:
    schemas = json.dumps(case.tools, indent=2)
    system = _PROMPTED_SYSTEM.format(schemas=schemas)
    return _chat(tokenizer, [
        {"role": "system", "content": system},
        {"role": "user", "content": case.prompt},
    ])


def supports_native_tools(tokenizer) -> bool:
    msgs = [{"role": "user", "content": "ping"}]
    probe_tool = [{"type": "function",
                   "function": {"name": "_probe", "parameters": {}}}]
    try:
        with_tools = _chat(tokenizer, msgs, tools=probe_tool)
        without = _chat(tokenizer, msgs, tools=None)
    except Exception:
        return False
    return with_tools != without


def build_native_tool_prompt(tokenizer, case: "ToolCase") -> str:
    return _chat(tokenizer, [{"role": "user", "content": case.prompt}],
                 tools=case.tools)


def build_constraint_prompt(tokenizer, case: "ConstraintCase") -> str:
    return _chat(tokenizer, [{"role": "user", "content": case.prompt}])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spec_elicit.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add litmus_spec.py tests/test_spec_elicit.py
git commit -m "feat: prompted/native/constraint prompt elicitation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Runner core (model-agnostic, fake-model tested)

**Files:**
- Modify: `litmus_spec.py`
- Create: `tests/test_spec_runner.py`

**Interfaces:**
- Consumes: everything above; a `generate_fn(prompt: str) -> str` and a `tokenizer`.
- Produces: `run_constraints(cases, tokenizer, generate_fn) -> dict` and `run_tool_calling(cases, tokenizer, generate_fn, native: bool) -> dict`. Each returns `{"aggregate": ..., "cases": [...], "errored": [...]}`. `generate_fn` exceptions are caught per-case and recorded in `errored` (excluded from aggregate denominators). Tool runner runs the prompted column always and the native column when `native` is True; per-case record is `{"id", "prompted": {...dims}, "native": {...}|None}`. Constraint per-case record is `{"id", "checks": [...], "output_sample": str}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spec_runner.py`:
```python
from litmus_spec import (
    ConstraintCase, ToolCase, run_constraints, run_tool_calling,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=False, tools=None):
        return messages[-1]["content"]


def test_run_constraints_scores_and_aggregates():
    cases = [
        ConstraintCase("c1", "p", [("exact_bullets", {"n": 2})]),
        ConstraintCase("c2", "p", [("all_lowercase", {})]),
    ]
    # generate_fn keyed by prompt content; c1 prompt renders to "p" for both,
    # so key on case via a canned mapping instead:
    outputs = iter(["- a\n- b", "HELLO"])
    result = run_constraints(cases, FakeTokenizer(), lambda p: next(outputs))
    assert result["aggregate"]["n_cases"] == 2
    assert result["aggregate"]["strict"] == 0.5   # c1 passes, c2 fails
    assert len(result["cases"]) == 2


def test_run_constraints_records_errors():
    cases = [ConstraintCase("c1", "p", [("all_lowercase", {})])]

    def boom(_):
        raise RuntimeError("model exploded")

    result = run_constraints(cases, FakeTokenizer(), boom)
    assert result["aggregate"]["n_cases"] == 0    # errored case not counted
    assert result["errored"][0]["id"] == "c1"
    assert "exploded" in result["errored"][0]["error"]


def test_run_tool_calling_prompted_only():
    cases = [ToolCase("t1", "weather?", [{"name": "get_weather"}],
                      {"tool": "get_weather", "arguments": {"location": "Paris"}})]
    out = '{"tool":"get_weather","arguments":{"location":"Paris"}}'
    result = run_tool_calling(cases, FakeTokenizer(), lambda p: out, native=False)
    assert result["aggregate"]["prompted"]["right_tool"] == 1.0
    assert result["aggregate"]["native"] is None
    assert result["cases"][0]["native"] is None


def test_run_tool_calling_with_native_column():
    cases = [ToolCase("t1", "weather?", [{"name": "get_weather"}],
                      {"tool": "get_weather", "arguments": {"location": "Paris"}})]
    # native parser reads name/arguments; prompted reads tool/arguments — both
    # here happen to parse from the same generic-JSON shape via name/tool keys.
    out = '{"name":"get_weather","tool":"get_weather","arguments":{"location":"Paris"}}'
    result = run_tool_calling(cases, FakeTokenizer(), lambda p: out, native=True)
    assert result["aggregate"]["native"]["right_tool"] == 1.0
    assert result["aggregate"]["native_parse_failed"] == 0


def test_run_tool_calling_counts_native_parse_failures():
    cases = [ToolCase("t1", "weather?", [{"name": "get_weather"}],
                      {"tool": "get_weather", "arguments": {"location": "Paris"}})]
    # 1st generate call = prompted (parseable), 2nd = native (junk)
    calls = {"n": 0}
    def gen(prompt):
        calls["n"] += 1
        return ('{"tool":"get_weather","arguments":{"location":"Paris"}}'
                if calls["n"] == 1 else "sorry, no idea")
    result = run_tool_calling(cases, FakeTokenizer(), gen, native=True)
    assert result["aggregate"]["native_parse_failed"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spec_runner.py -q`
Expected: FAIL — `ImportError: cannot import name 'run_constraints'`.

- [ ] **Step 3: Implement the runner**

Append to `litmus_spec.py`:
```python
# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def run_constraints(cases: list, tokenizer, generate_fn) -> dict:
    per_case_checks, records, errored = [], [], []
    for case in cases:
        prompt = build_constraint_prompt(tokenizer, case)
        try:
            text = generate_fn(prompt)
        except Exception as e:  # noqa: BLE001 - report, don't crash the run
            errored.append({"id": case.id, "error": str(e)})
            continue
        checks = score_constraint_case(text, case)
        per_case_checks.append(checks)
        records.append({"id": case.id, "checks": checks,
                        "output_sample": text[:200]})
    return {"aggregate": aggregate_constraints(per_case_checks),
            "cases": records, "errored": errored}


def run_tool_calling(cases: list, tokenizer, generate_fn, native: bool) -> dict:
    prompted_scores, native_scores, records, errored = [], [], [], []
    native_parse_failed = 0
    for case in cases:
        rec = {"id": case.id, "prompted": None, "native": None}
        try:
            p_prompt = build_prompted_tool_prompt(tokenizer, case)
            p_parsed = parse_prompted(generate_fn(p_prompt))
            p_score = score_tool_call(p_parsed, case.expect)
            prompted_scores.append(p_score)
            rec["prompted"] = p_score
            if native:
                n_prompt = build_native_tool_prompt(tokenizer, case)
                n_parsed = parse_native(generate_fn(n_prompt))
                if not n_parsed.well_formed:
                    native_parse_failed += 1
                n_score = score_tool_call(n_parsed, case.expect)
                native_scores.append(n_score)
                rec["native"] = n_score
        except Exception as e:  # noqa: BLE001
            errored.append({"id": case.id, "error": str(e)})
            continue
        records.append(rec)
    return {
        "aggregate": {
            "prompted": aggregate_tool(prompted_scores),
            "native": aggregate_tool(native_scores) if native else None,
            # distinct from well_formed=False and from abstention: how many
            # native outputs the multi-format parser could not read at all.
            "native_parse_failed": native_parse_failed if native else None,
        },
        "cases": records, "errored": errored,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spec_runner.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add litmus_spec.py tests/test_spec_runner.py
git commit -m "feat: model-agnostic runner for both spec-check profiles

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Results output — human table + JSON sidecar

**Files:**
- Modify: `litmus_spec.py`
- Create: `tests/test_spec_output.py`

**Interfaces:**
- Consumes: the runner result dicts.
- Produces: `format_tool_table(label, result) -> str`; `format_constraint_table(label, result) -> str`; `write_sidecar(path, profile, model, label, convention_support, result) -> None` (writes JSON with `profile, model, label, convention_support, aggregate, n_cases, cases, errored`). Tables are plain strings (printed by the CLI); sidecar is the Loxo-facing artifact and retains full per-case detail.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spec_output.py`:
```python
import json
from litmus_spec import format_tool_table, format_constraint_table, write_sidecar


def test_tool_table_shows_prompted_and_native_and_gap():
    result = {"aggregate": {
        "prompted": {"well_formed": 0.9, "right_tool": 0.8, "args_ok": 0.7, "abstained_ok": 0.6},
        "native": {"well_formed": 1.0, "right_tool": 0.9, "args_ok": 0.8, "abstained_ok": 0.7}},
        "cases": [], "errored": []}
    table = format_tool_table("Qwen", result)
    assert "prompted" in table and "native" in table
    assert "Qwen" in table


def test_tool_table_handles_no_native():
    result = {"aggregate": {
        "prompted": {"well_formed": 0.7, "right_tool": 0.5, "args_ok": 0.4, "abstained_ok": 0.4},
        "native": None}, "cases": [], "errored": []}
    table = format_tool_table("Bonsai", result)
    assert "no native" in table.lower()


def test_constraint_table_shows_strict_loose():
    result = {"aggregate": {"strict": 0.8, "loose": 0.9, "by_kind": {"all_lowercase": 1.0},
                            "n_cases": 5}, "cases": [], "errored": []}
    table = format_constraint_table("M", result)
    assert "strict" in table.lower() and "0.8" in table


def test_write_sidecar_roundtrips(tmp_path):
    result = {"aggregate": {"strict": 0.5, "loose": 0.5, "by_kind": {}, "n_cases": 2},
              "cases": [{"id": "c1"}], "errored": []}
    path = tmp_path / "out.json"
    write_sidecar(str(path), "constraints", "org/model", "model",
                  {"prompted": True, "native": False}, result)
    data = json.loads(path.read_text())
    assert data["profile"] == "constraints"
    assert data["convention_support"]["native"] is False
    assert data["n_cases"] == 2
    assert data["cases"][0]["id"] == "c1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_spec_output.py -q`
Expected: FAIL — `ImportError: cannot import name 'format_tool_table'`.

- [ ] **Step 3: Implement output formatting**

Append to `litmus_spec.py`:
```python
# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "  - "


def format_tool_table(label: str, result: dict) -> str:
    agg = result["aggregate"]
    p = agg["prompted"]
    lines = [f"{'model':<20} {'conv':<9} {'well':>5} {'right':>6} {'args':>5} {'abst':>5}"]
    lines.append(f"{label:<20} {'prompted':<9} "
                 f"{_fmt(p['well_formed']):>5} {_fmt(p['right_tool']):>6} "
                 f"{_fmt(p['args_ok']):>5} {_fmt(p['abstained_ok']):>5}")
    n = agg["native"]
    if n is None:
        lines.append(f"{'':<20} {'native':<9}  (no native tool support)")
    else:
        gap = (p["right_tool"] or 0) - (n["right_tool"] or 0)
        lines.append(f"{'':<20} {'native':<9} "
                     f"{_fmt(n['well_formed']):>5} {_fmt(n['right_tool']):>6} "
                     f"{_fmt(n['args_ok']):>5} {_fmt(n['abstained_ok']):>5}"
                     f"   gap(right)={gap:+.2f}")
    return "\n".join(lines)


def format_constraint_table(label: str, result: dict) -> str:
    agg = result["aggregate"]
    lines = [f"{label}:  strict={agg['strict']:.2f}  loose={agg['loose']:.2f}  "
             f"(n={agg['n_cases']})"]
    if agg["by_kind"]:
        lines.append("  by kind: " + "  ".join(
            f"{k}={v:.2f}" for k, v in sorted(agg["by_kind"].items())))
    if result["errored"]:
        lines.append(f"  errored: {len(result['errored'])} case(s)")
    return "\n".join(lines)


def write_sidecar(path: str, profile: str, model: str, label: str,
                  convention_support: dict, result: dict) -> None:
    payload = {
        "profile": profile,
        "model": model,
        "label": label,
        "convention_support": convention_support,
        "aggregate": result["aggregate"],
        "n_cases": result["aggregate"].get("n_cases",
                                           len(result.get("cases", []))),
        "cases": result.get("cases", []),
        "errored": result.get("errored", []),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_spec_output.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add litmus_spec.py tests/test_spec_output.py
git commit -m "feat: human tables + JSON sidecar output

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: CLI, MLX wiring, and authored case files

**Files:**
- Modify: `litmus_spec.py`
- Create: `cases/tool_calling.jsonl`
- Create: `cases/constraints.jsonl`
- Create: `tests/test_spec_cases_files.py`
- Modify: `README.md` (document the new command)

**Interfaces:**
- Consumes: everything above, plus `litmus_common._targets_for`, `_load_timed`, `stream_generate`, `_clear_cache`.
- Produces: `main()` with argparse: `--profile {tool-calling,constraints}` (required), `--cases <path>` (default per profile), `--repo`/`--sizes`/`--label` (via `_targets_for`), `--max-tokens` (default 128 tool / 512 constraints), `--out-dir` (sidecar destination, default `.`). Real generation via `_mlx_generate(model, tokenizer, prompt, max_tokens)` built on `stream_generate`.

- [ ] **Step 1: Author the case files**

Create `cases/constraints.jsonl` (12 cases; here are the first three — author 12 total spanning every `kind` in the registry):
```
{"id":"constr-01","prompt":"List exactly three primary colors, one per line as a bulleted list, all lowercase.","checks":[{"kind":"exact_bullets","n":3},{"kind":"all_lowercase"}]}
{"id":"constr-02","prompt":"Reply with a single JSON object mapping the keys name and age to example values. Output only the JSON.","checks":[{"kind":"valid_json"}]}
{"id":"constr-03","prompt":"Write two sentences about the ocean. Do not use the word water.","checks":[{"kind":"forbidden_word","word":"water"},{"kind":"max_words","n":60}]}
```

Create `cases/tool_calling.jsonl` (10 cases incl. at least 2 abstentions; first three shown — author 10 total):
```
{"id":"tool-01","prompt":"What's the weather in Paris right now?","tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"expect":{"tool":"get_weather","arguments":{"location":"Paris"}}}
{"id":"tool-02","prompt":"Email alice@example.com the subject 'Hi' and body 'Hello'.","tools":[{"type":"function","function":{"name":"get_weather","description":"weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}},{"type":"function","function":{"name":"send_email","description":"Send an email","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}},"required":["to","subject","body"]}}}],"expect":{"tool":"send_email","arguments":{"to":"alice@example.com","subject":"Hi","body":"Hello"}}}
{"id":"tool-03","prompt":"What is the capital of France?","tools":[{"type":"function","function":{"name":"get_weather","description":"weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"expect":{"tool":null}}
```

- [ ] **Step 2: Write the failing test (case files load and validate)**

Create `tests/test_spec_cases_files.py`:
```python
from litmus_spec import load_cases


def test_shipped_constraint_cases_load():
    cases = load_cases("cases/constraints.jsonl", "constraints")
    assert len(cases) >= 10
    kinds = {k for c in cases for k, _ in c.checks}
    # every registry kind is exercised by at least one shipped case
    from litmus_spec import CHECKS
    assert set(CHECKS) <= kinds


def test_shipped_tool_cases_load_with_abstentions():
    cases = load_cases("cases/tool_calling.jsonl", "tool-calling")
    assert len(cases) >= 8
    assert sum(1 for c in cases if c.expect["tool"] is None) >= 2
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_spec_cases_files.py -q`
Expected: FAIL — either files missing, `<10 cases`, or not every `kind` is exercised (author more cases until green — every `kind` in the registry must appear).

- [ ] **Step 4: Implement the CLI and MLX generation**

Append to `litmus_spec.py`:
```python
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import argparse
import os

from litmus_common import _targets_for, _load_timed, _clear_cache, stream_generate

DEFAULT_CASES = {
    "tool-calling": "cases/tool_calling.jsonl",
    "constraints": "cases/constraints.jsonl",
}
DEFAULT_MAX_TOKENS = {"tool-calling": 128, "constraints": 512}


def _mlx_generate(model, tokenizer, prompt: str, max_tokens: int) -> str:
    from litmus_common import _resp_text
    chunks = []
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        chunks.append(_resp_text(resp))
    return "".join(chunks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True,
                    choices=["tool-calling", "constraints"])
    ap.add_argument("--cases", default=None, help="JSONL case file (default per profile)")
    ap.add_argument("--sizes", default="1.7B,4B,8B")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    cases_path = args.cases or DEFAULT_CASES[args.profile]
    max_tokens = args.max_tokens or DEFAULT_MAX_TOKENS[args.profile]
    cases = load_cases(cases_path, args.profile)   # fail-fast before loading a model
    print(f"loaded {len(cases)} cases from {cases_path}")

    for label, repo in _targets_for(args):
        print(f"\n=== {label}: loading {repo} ===")
        model, tokenizer, t_load = _load_timed(repo)
        print(f"loaded in {t_load:.1f}s")
        gen = lambda p: _mlx_generate(model, tokenizer, p, max_tokens)

        if args.profile == "constraints":
            result = run_constraints(cases, tokenizer, gen)
            print(format_constraint_table(label, result))
            conv = {"prompted": True, "native": False}
        else:
            native = supports_native_tools(tokenizer)
            result = run_tool_calling(cases, tokenizer, gen, native=native)
            print(format_tool_table(label, result))
            conv = {"prompted": True, "native": native}

        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
        out_path = os.path.join(args.out_dir, f"results_{args.profile}_{safe}.json")
        write_sidecar(out_path, args.profile, repo, label, conv, result)
        print(f"sidecar written: {out_path}")

        del model, tokenizer
        _clear_cache()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the case-file test and the CLI's argparse to verify**

Run: `python -m pytest tests/test_spec_cases_files.py -q && python litmus_spec.py --help`
Expected: tests PASS; `--help` prints usage listing `--profile`. (No model is loaded by `--help`.)

- [ ] **Step 6: Document the command in README.md**

Under the Quick start section of `README.md`, add:
```markdown
### Spec-check (capability evals)

Objective, fixed-input checks of tool-calling and instruction-following:

    python litmus_spec.py --profile constraints --repo <hf-org/model-mlx>
    python litmus_spec.py --profile tool-calling --repo <hf-org/model-mlx>

Tool-calling reports a prompted-JSON column for every model plus a native
`tools=` column where the chat template supports it (and the gap between them).
Constraint-following reports IFEval-style strict/loose accuracy. Each run writes
a `results_<profile>_<label>.json` sidecar for downstream (Loxo) consumption.
```

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS (all spec-check + common tests green).

- [ ] **Step 8: Commit**

```bash
git add litmus_spec.py cases/ tests/test_spec_cases_files.py README.md
git commit -m "feat: spec-check CLI, MLX wiring, and authored case files

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Post-implementation (human-run, not part of TDD tasks)

These require MLX weights on the Mac and are run by Brendan, not the plan executor:

- **Smoke run** against a small stock model with native tools (e.g.
  `mlx-community/Qwen2.5-3B-Instruct-4bit`) for `--profile tool-calling` and a
  Bonsai 1-bit for the prompted-only path — confirm both columns populate and the
  gap prints.
- **Sanity-read a sidecar** to confirm per-case detail is captured.
- Open the PR from `feat/spec-check-harness`.

## Self-review notes

- **Spec coverage:** module layout (Task 6), data-driven JSONL cases + registry
  (Tasks 1,3), both tool conventions + gap (Tasks 4,5,7,8,9), four dimensions
  (Task 5), constraint strict/loose (Task 2), IFEval-style vocabulary (Task 1),
  our-cases-now (Task 10), human table + JSON sidecar (Task 9), fail-fast loader
  (Task 3), errored-case accounting (Task 8), native parse-failure detail
  (Task 4, via `ParsedCall.detail`), first pytest suite (all tasks). Vendored
  IFEval subset is explicitly deferred in the spec — no task, by design.
- **Naming consistency:** `CheckResult`, `ConstraintCase`, `ToolCase`,
  `ParsedCall`, `CHECKS`, `score_constraint_case`, `aggregate_constraints`,
  `score_tool_call`, `aggregate_tool`, `run_constraints`, `run_tool_calling`,
  `build_*_prompt`, `supports_native_tools`, `format_*_table`, `write_sidecar`
  are used identically across defining and consuming tasks.
