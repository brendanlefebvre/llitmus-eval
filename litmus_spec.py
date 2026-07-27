"""Litmus spec-check harness — model-level tool-calling and
instruction/constraint-following capability evals.

One case -> validate -> aggregate runner. Cases are JSONL data under cases/.
No LLM judge, no reference model: every check is an objective parser.
"""
from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
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

_NO_ANSWER = "no answer: thinking budget exhausted (unclosed <think>)"


def score_constraint_case(text: str, case: "ConstraintCase",
                          closed: bool = True) -> list:
    """Score `text` against a case's checks.

    `closed=False` means the model never finished reasoning, so there is no
    answer. Those cases must fail explicitly rather than be scored on the empty
    string: prohibition-style checks (all_lowercase, max_words, forbidden_word,
    all_uppercase) are *vacuously satisfied* by empty text, which would reward a
    runaway reasoner with a pass.
    """
    if not closed:
        return [{"kind": kind, "passed": False, "detail": _NO_ANSWER}
                for kind, _ in case.checks]
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
            if kind == "regex_match" and pname == "pattern":
                try:
                    re.compile(val)
                except re.error as e:
                    raise CaseError(
                        f"line {ln}: check 'regex_match' param 'pattern' "
                        f"is not a valid regex: {e}")
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
    # chore: reuses the constraints loader/runner. Checks are compliance-only
    # (length, format, forbidden prefixes); current cases saturate at
    # strict=1.00 for both think and no-think on a 14B model, so the score
    # does not discriminate between models — routing for this class falls to
    # cost, not adequacy.
    loader = {"constraints": _load_constraint_line,
              "chore": _load_constraint_line,
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
    attempted: bool = False  # a tool-call was attempted: a structure was present, or a call parsed


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
    return ParsedCall(True, tool, obj.get("arguments") or {}, "ok", attempted=True)


def _call_from_name_obj(obj: Optional[dict], detail: str,
                        attempted: Optional[bool]) -> ParsedCall:
    # attempted: True forces an attempt (a structure marker was present); a
    # falsy/None value means "attempted only if an object actually parsed".
    if obj is None or "name" not in obj:
        return ParsedCall(False, None, None, "no name/arguments object found",
                          attempted=bool(attempted))
    return ParsedCall(True, obj["name"], obj.get("arguments") or {}, detail, attempted=True)


def parse_native(text: str) -> ParsedCall:
    if not text.strip():
        return ParsedCall(False, None, None, "truncated")
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if m:
        return _call_from_name_obj(_first_json_object(m.group(1)), "qwen/hermes tag", attempted=True)
    m = re.search(r"<\|python_tag\|>(.*)", text, re.DOTALL)
    if m:
        return _call_from_name_obj(_first_json_object(m.group(1)), "llama python_tag", attempted=True)
    # No complete tag pair. An unclosed opening marker still signals an attempted
    # call (models don't always emit the closing tag). Absent any marker, a
    # generic JSON object counts as an attempt only if it actually parses.
    structure = ("<tool_call>" in text) or ("<|python_tag|>" in text)
    return _call_from_name_obj(_first_json_object(text), "generic json", attempted=structure)


# ---------------------------------------------------------------------------
# tool-call scoring
# ---------------------------------------------------------------------------

def score_tool_call(parsed: "ParsedCall", expect: dict, native: bool = False,
                    closed: bool = True) -> dict:
    is_abstention = expect.get("tool") is None
    if not closed:
        # The model never finished reasoning, so there is no answer to score.
        # This must bypass the native-abstention path below, which infers a
        # deliberate no-call from the *absence* of tool-call structure — an
        # empty answer has no structure either, and would be credited as a
        # correct abstention for never having answered at all.
        return {"well_formed": False, "right_tool": False, "args_ok": None,
                "abstained_ok": False if is_abstention else None}
    abstained_ok = None
    args_ok = None
    if is_abstention:
        if native:
            # Native FC has no explicit no-call token. A genuine abstention means
            # NO tool-call structure was attempted; a structure that failed to
            # parse is an attempted (failed) call, not an abstention.
            abstained_ok = (not parsed.attempted) and parsed.tool is None
            well_formed = True if abstained_ok else parsed.well_formed
        else:
            # Prompted convention asks for an explicit {"tool": null}; a garbage
            # (not well_formed) output must not be credited as an abstention.
            abstained_ok = parsed.well_formed and parsed.tool is None
            well_formed = parsed.well_formed
        right_tool = abstained_ok
    else:
        well_formed = parsed.well_formed
        right_tool = parsed.well_formed and parsed.tool == expect.get("tool")
        if right_tool:
            want = expect.get("arguments") or {}
            got = parsed.arguments or {}
            args_ok = (set(got.keys()) == set(want.keys())
                       and all(got.get(k) == v for k, v in want.items()))
    return {
        "well_formed": well_formed,
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


def strip_thinking(text: str) -> tuple[str, bool]:
    """Split a reasoning preamble off the front of `text`.

    Returns (answer, closed). Only the explicit ``<think>…</think>`` tag is
    honoured — unlike ``litmus.py``'s throughput-side counterpart, there is no
    untagged prose-preamble heuristic here. That heuristic splits on the first
    blank line, which is fine for estimating "useful t/s" but would silently
    swallow half a legitimate answer during scoring (a ``min_words`` or
    ``exact_bullets`` case would fail for the harness's reasons, not the
    model's). Scoring gets the conservative path: no tag, no strip.

    An opened ``<think>`` with no closing tag means the model burned its whole
    budget reasoning and never committed to an answer. That returns ("", False)
    so it scores as a miss — which is the honest result, not a harness artifact.
    """
    close_idx = text.find("</think>")
    if close_idx != -1:
        return text[close_idx + len("</think>"):].lstrip(), True
    if text.lstrip().startswith("<think>"):
        return "", False
    return text, True


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


def _chat(tokenizer, messages, tools=None, enable_thinking=None) -> str:
    # Only forward enable_thinking when a mode was explicitly requested: older
    # tokenizers predating thinking modes reject the unknown kwarg outright.
    kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, tools=tools, **kw)


def supports_thinking(tokenizer) -> bool:
    """True when the chat template actually honours ``enable_thinking``.

    Probes by rendering the same messages both ways and comparing, mirroring
    `supports_native_tools`. A template that ignores the flag renders
    identically and reports False, so non-thinking models keep a single column.
    """
    msgs = [{"role": "user", "content": "ping"}]
    try:
        default = _chat(tokenizer, msgs)
        no_think = _chat(tokenizer, msgs, enable_thinking=False)
    except Exception:
        return False
    return default != no_think


def build_prompted_tool_prompt(tokenizer, case: "ToolCase",
                               enable_thinking=None) -> str:
    schemas = json.dumps(case.tools, indent=2)
    system = _PROMPTED_SYSTEM.format(schemas=schemas)
    return _chat(tokenizer, [
        {"role": "system", "content": system},
        {"role": "user", "content": case.prompt},
    ], enable_thinking=enable_thinking)


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


def build_native_tool_prompt(tokenizer, case: "ToolCase",
                             enable_thinking=None) -> str:
    return _chat(tokenizer, [{"role": "user", "content": case.prompt}],
                 tools=case.tools, enable_thinking=enable_thinking)


def build_constraint_prompt(tokenizer, case: "ConstraintCase",
                            enable_thinking=None) -> str:
    return _chat(tokenizer, [{"role": "user", "content": case.prompt}],
                 enable_thinking=enable_thinking)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def run_constraints(cases: list, tokenizer, generate_fn,
                    enable_thinking=None) -> dict:
    per_case_checks, records, errored = [], [], []
    for case in cases:
        prompt = build_constraint_prompt(tokenizer, case,
                                         enable_thinking=enable_thinking)
        try:
            raw = generate_fn(prompt)
        except Exception as e:  # noqa: BLE001 - report, don't crash the run
            errored.append({"id": case.id, "error": str(e)})
            continue
        # Score the answer, not the scratchpad. A no-op for non-thinking models.
        text, closed = strip_thinking(raw)
        checks = score_constraint_case(text, case, closed=closed)
        per_case_checks.append(checks)
        records.append({"id": case.id, "checks": checks,
                        "output_sample": text[:200],
                        "thinking_unclosed": not closed})
    return {"aggregate": aggregate_constraints(per_case_checks),
            "cases": records, "errored": errored}


def run_tool_calling(cases: list, tokenizer, generate_fn, native: bool,
                     enable_thinking=None) -> dict:
    prompted_scores, native_scores, records, errored = [], [], [], []
    native_parse_failed = 0
    for case in cases:
        rec = {"id": case.id, "prompted": None, "native": None,
               "prompted_output": None, "native_output": None}
        try:
            p_prompt = build_prompted_tool_prompt(tokenizer, case,
                                                  enable_thinking=enable_thinking)
            # Score the answer, not the scratchpad: reasoning models weigh
            # candidate calls as JSON inside <think>, and parse_prompted takes
            # the first JSON object it finds. A no-op for non-thinking models.
            p_text, p_closed = strip_thinking(generate_fn(p_prompt))
            p_score = score_tool_call(parse_prompted(p_text), case.expect,
                                      closed=p_closed)
            n_text = None
            n_score = None
            n_failed = False
            n_closed = True
            if native:
                n_prompt = build_native_tool_prompt(tokenizer, case,
                                                    enable_thinking=enable_thinking)
                n_text, n_closed = strip_thinking(generate_fn(n_prompt))
                n_parsed = parse_native(n_text)
                n_failed = n_parsed.attempted and not n_parsed.well_formed
                n_score = score_tool_call(n_parsed, case.expect, native=True,
                                          closed=n_closed)
        except Exception as e:  # noqa: BLE001 - report, don't crash the run
            errored.append({"id": case.id, "error": str(e)})
            continue
        prompted_scores.append(p_score)
        rec["prompted"] = p_score
        rec["prompted_output"] = p_text[:200]
        rec["thinking_unclosed"] = not p_closed
        if native:
            native_scores.append(n_score)
            rec["native"] = n_score
            rec["native_output"] = n_text[:200]
            rec["native_thinking_unclosed"] = not n_closed
            if n_failed:
                native_parse_failed += 1
        records.append(rec)
    return {
        "aggregate": {
            "prompted": aggregate_tool(prompted_scores),
            "native": aggregate_tool(native_scores) if native else None,
            "native_parse_failed": native_parse_failed if native else None,
        },
        "cases": records, "errored": errored,
    }


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


def _headline(result: dict, profile: str) -> float:
    """The one number a thinking/no-thinking comparison turns on."""
    agg = result["aggregate"]
    if profile in ("constraints", "chore"):
        return agg["strict"] or 0.0
    return (agg["prompted"] or {}).get("right_tool") or 0.0


def format_thinking_gap(label: str, per_mode: dict, profile: str,
                        tokens_per_case: dict) -> str:
    """Report what thinking bought, and what it cost.

    The delta alone is only half a routing decision: +0.08 accuracy for 18x the
    tokens is a different call than +0.08 for free. Loxo needs both, so both go
    on the line.
    """
    off, on = per_mode["no-think"], per_mode["think"]
    gap = _headline(on, profile) - _headline(off, profile)
    metric = "strict" if profile in ("constraints", "chore") else "right_tool"
    t_off = tokens_per_case.get("no-think") or 0.0
    t_on = tokens_per_case.get("think") or 0.0
    cost = f"{t_off:.0f} -> {t_on:.0f} tok/case"
    if t_off > 0:
        cost += f" ({t_on / t_off:.1f}x)"
    return (f"{label}: thinking gap({metric})={gap:+.2f}  "
            f"[no-think={_headline(off, profile):.2f}  "
            f"think={_headline(on, profile):.2f}]  cost: {cost}")


def write_sidecar(path: str, profile: str, model: str, label: str,
                  convention_support: dict, result: dict,
                  modes: Optional[dict] = None,
                  tokens_per_case: Optional[dict] = None) -> None:
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
    # Top-level keys stay the primary (thinking, where supported) result so the
    # sidecar contract holds; per-mode detail rides alongside for Loxo, which
    # needs the accuracy/token trade-off to route.
    if modes:
        payload["modes"] = {
            m: {"aggregate": r["aggregate"], "cases": r.get("cases", []),
                "errored": r.get("errored", []),
                "mean_tokens_per_case": (tokens_per_case or {}).get(m)}
            for m, r in modes.items()
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import argparse
import os

DEFAULT_CASES = {
    "tool-calling": "cases/tool_calling.jsonl",
    "constraints": "cases/constraints.jsonl",
    "chore": "cases/chore.jsonl",
}
DEFAULT_MAX_TOKENS = {"tool-calling": 128, "constraints": 512, "chore": 64}

# Reasoning models need room for the scratchpad on top of the answer. Measured
# on Ternary-Bonsai-27B-mlx-2bit (2026-07-15): the <think> preamble alone runs
# 66-1084 tok across the tool-calling cases and 165-1445 across the constraint
# cases. These ceilings clear 11/12 and 14/14 respectively. A case that still
# overruns is a genuine non-termination, not a truncation artifact (tool-12
# loops "I will output the JSON / One last check" past 8192), and scores as a
# miss. max_tokens is a ceiling, not a target: non-thinking models stop at EOS
# well short of it, so raising this does not perturb their scores.
THINKING_MAX_TOKENS = {"tool-calling": 2048, "constraints": 1536, "chore": 4096}


def main() -> None:
    from litmus_common import get_backend, _targets_for

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True,
                    choices=["tool-calling", "constraints", "chore"])
    ap.add_argument("--cases", default=None, help="JSONL case file (default per profile)")
    ap.add_argument("--sizes", default="1.7B,4B,8B")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--backend", choices=["mlx", "cuda", "auto"], default="mlx")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="ceiling for the non-thinking column")
    ap.add_argument("--think-max-tokens", type=int, default=None,
                    help="ceiling for the thinking column (needs room for the "
                         "<think> scratchpad on top of the answer)")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    cases_path = args.cases or DEFAULT_CASES[args.profile]
    max_tokens = args.max_tokens or DEFAULT_MAX_TOKENS[args.profile]
    think_max_tokens = (args.think_max_tokens
                        or THINKING_MAX_TOKENS[args.profile])
    cases = load_cases(cases_path, args.profile)   # fail-fast before loading a model
    print(f"loaded {len(cases)} cases from {cases_path}")

    backend = get_backend(args.backend)
    for label, repo in _targets_for(args):
        print(f"\n=== {label}: loading {repo} ===")
        model, tokenizer, t_load = backend.load(repo)
        print(f"loaded in {t_load:.1f}s")

        thinking = supports_thinking(tokenizer)
        native = (args.profile == "tool-calling"
                  and supports_native_tools(tokenizer))
        conv = {"prompted": True, "native": native, "thinking": thinking}

        # Thinking-capable models are scored both ways: the gap between them is
        # what tells a router whether reasoning is worth its token cost for a
        # task class. Models without a thinking mode keep their single column.
        if thinking:
            modes = [("no-think", False, max_tokens),
                     ("think", True, think_max_tokens)]
        else:
            modes = [("default", None, max_tokens)]
        print(f"thinking mode: {'supported' if thinking else 'not supported'}"
              f" | native tools: {'yes' if native else 'no'}")

        per_mode, tokens_per_case = {}, {}
        for mode, flag, budget in modes:
            tally = {"tokens": 0, "calls": 0}

            def gen(p, _budget=budget, _tally=tally,
                    _model=model, _tok=tokenizer):
                text = "".join(backend.stream(_model, _tok, p, _budget))
                _tally["tokens"] += len(_tok.encode(text))
                _tally["calls"] += 1
                return text

            print(f"\n--- {label} [{mode}] (max_tokens={budget}) ---")
            if args.profile in ("constraints", "chore"):
                result = run_constraints(cases, tokenizer, gen,
                                         enable_thinking=flag)
                print(format_constraint_table(f"{label} [{mode}]", result))
            else:
                result = run_tool_calling(cases, tokenizer, gen, native=native,
                                          enable_thinking=flag)
                print(format_tool_table(f"{label} [{mode}]", result))
            per_mode[mode] = result
            tokens_per_case[mode] = (tally["tokens"] / tally["calls"]
                                     if tally["calls"] else 0.0)

        if thinking:
            print("\n" + format_thinking_gap(label, per_mode, args.profile,
                                             tokens_per_case))
        primary = per_mode.get("think") or per_mode["default"]

        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
        out_path = os.path.join(args.out_dir, f"results_{args.profile}_{safe}.json")
        write_sidecar(out_path, args.profile, repo, label, conv, primary,
                      modes=per_mode, tokens_per_case=tokens_per_case)
        print(f"sidecar written: {out_path}")

        del model, tokenizer
        gc.collect()
        backend.clear_cache()


if __name__ == "__main__":
    main()
