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
