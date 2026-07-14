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
    if not isinstance(raw_checks, list):
        raise CaseError(f"line {ln}: 'checks' must be a list")
    checks = []
    for chk in raw_checks:
        if not isinstance(chk, dict):
            raise CaseError(f"line {ln}: each check must be an object")
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
    if not isinstance(expect, dict):
        raise CaseError(f"line {ln}: 'expect' must be an object")
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
            if not isinstance(obj, dict):
                raise CaseError(f"line {i}: top-level value must be a JSON object")
            cases.append(loader(obj, i))
    return cases


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
    """Return the first balanced {...} JSON object in text, or None.

    Brace counting ignores { and } that occur inside JSON string literals
    (respecting backslash escapes), so a brace in an argument's string value
    does not corrupt the balance.
    """
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
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
