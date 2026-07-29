"""Litmus spec-check harness — model-level tool-calling and
instruction/constraint-following capability evals.

One case -> validate -> aggregate runner. Cases are JSONL data under cases/.
No LLM judge, no reference model: every check is an objective parser.
"""
from __future__ import annotations

import gc
import json
import os
import re
import time
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


@dataclass
class ReplayCase:
    id: str
    capture_path: str
    chain_id: str
    depth_stratum: str          # "shallow" | "mid" | "deep"
    est_tokens: int
    reference: dict             # {acted: bool, tools: list[str], arguments: list[dict]}
    reference_model: Optional[str] = None  # sourced from the case file


# ---------------------------------------------------------------------------
# constraint scoring
# ---------------------------------------------------------------------------

_NO_ANSWER = "no answer: thinking budget exhausted (unclosed <think>)"


def normalize_output(text: str) -> str:
    """Clean generated text the way a downstream consumer would before using it.

    The discipline: if the consumer of this output would do this anyway, it is
    normalization. If not, it is score inflation. Specifically:
    - strip surrounding whitespace
    - strip a MATCHED pair of wrapping quotes (double or single), not stray
      quote characters elsewhere in the string
    - strip matched wrapping backticks or triple-backtick code fences

    Does NOT strip a "Title:" prefix — forbidden_word exists to catch exactly
    that, and removing it would launder a real defect.
    """
    s = text.strip()
    if len(s) >= 2:
        if s.startswith("`" * 3) and s.endswith("`" * 3):
            s = s[3:-3].strip()
        elif s[0] == s[-1] and s[0] in ('"', "'", "`"):
            s = s[1:-1].strip()
    return s


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


def aggregate_constraints(per_case: list,
                          per_case_normalized: Optional[list] = None) -> dict:
    n_cases = len(per_case)
    strict_pass = sum(1 for checks in per_case if checks and all(c["passed"] for c in checks))
    all_checks = [c for checks in per_case for c in checks]
    loose = (sum(1 for c in all_checks if c["passed"]) / len(all_checks)) if all_checks else 0.0
    by_kind: dict = {}
    for c in all_checks:
        by_kind.setdefault(c["kind"], []).append(c["passed"])
    by_kind_rate = {k: sum(v) / len(v) for k, v in by_kind.items()}
    result = {
        "strict": (strict_pass / n_cases) if n_cases else 0.0,
        "loose": loose,
        "by_kind": by_kind_rate,
        "n_cases": n_cases,
    }
    if per_case_normalized is not None:
        sn_pass = sum(1 for checks in per_case_normalized
                      if checks and all(c["passed"] for c in checks))
        result["strict_normalized"] = (sn_pass / len(per_case_normalized)
                                       if per_case_normalized else 0.0)
    return result


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


_REPLAY_STRATA = ("shallow", "mid", "deep")


def _load_replay_line(obj: dict, ln: int) -> "ReplayCase":
    cid = _require(obj, "id", ln)
    if not isinstance(cid, str):
        raise CaseError(f"line {ln}: 'id' must be a string")
    capture_path = _require(obj, "capture_path", ln)
    if not isinstance(capture_path, str):
        raise CaseError(f"line {ln}: 'capture_path' must be a string")
    if not os.path.exists(capture_path):
        raise CaseError(f"line {ln}: 'capture_path' does not exist: {capture_path}")
    chain_id = _require(obj, "chain_id", ln)
    if not isinstance(chain_id, str):
        raise CaseError(f"line {ln}: 'chain_id' must be a string")
    depth_stratum = _require(obj, "depth_stratum", ln)
    if depth_stratum not in _REPLAY_STRATA:
        raise CaseError(
            f"line {ln}: 'depth_stratum' must be one of "
            f"{', '.join(_REPLAY_STRATA)}, got {depth_stratum!r}")
    est_tokens = _require(obj, "est_tokens", ln)
    if not isinstance(est_tokens, int) or isinstance(est_tokens, bool):
        raise CaseError(f"line {ln}: 'est_tokens' must be an integer")
    reference = _require(obj, "reference", ln)
    if not isinstance(reference, dict):
        raise CaseError(f"line {ln}: 'reference' must be an object")
    if "acted" not in reference or not isinstance(reference["acted"], bool):
        raise CaseError(f"line {ln}: reference.acted must be a boolean")
    if not isinstance(reference.get("tools"), list):
        raise CaseError(f"line {ln}: reference.tools must be a list")
    if not isinstance(reference.get("arguments"), list):
        raise CaseError(f"line {ln}: reference.arguments must be a list")
    return ReplayCase(id=cid, capture_path=capture_path, chain_id=chain_id,
                      depth_stratum=depth_stratum, est_tokens=est_tokens,
                      reference=reference,
                      reference_model=obj.get("reference_model"))


def load_cases(path: str, profile: str) -> list:
    # chore: reuses the constraints loader/runner. Checks are compliance-only
    # (length, format, forbidden prefixes). The profile discriminates: 14B
    # saturates at strict=1.00 both modes, 4B drops to 0.83 in no-think
    # (word-count overflow) but 1.00 with thinking, 1B Llama scores 0.00
    # (wraps titles in quotes). Floor is between 1B and 4B. An earlier claim
    # that the profile "does not discriminate" was drawn from the 14B alone
    # and was wrong.
    loader = {"constraints": _load_constraint_line,
              "chore": _load_constraint_line,
              "tool-calling": _load_tool_line,
              "main-replay": _load_replay_line}.get(profile)
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
    # TODO: Only recognises JSON inside the tool_call tag pair and the
    # python_tag marker. The function=/parameter= XML family used by some
    # serving harnesses is not parsed. Moot for increment 1 (no model runs
    # native), but needs addressing before native replay.
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


# ---------------------------------------------------------------------------
# main-replay tier-0 scoring
# ---------------------------------------------------------------------------

# JSON schema "type" -> python type. Note: python's bool is a subclass of int,
# so integer/number checks must explicitly reject bools (see _check_args_schema).
_JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _capture_tool_names(capture_tools: list) -> set:
    names = set()
    for t in capture_tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function", {})
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str):
                names.add(name)
    return names


def _capture_tool_params(capture_tools: list, tool_name: str) -> Optional[dict]:
    """Return the JSON-schema `parameters` object for `tool_name`, or None."""
    for t in capture_tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function", {})
        if isinstance(fn, dict) and fn.get("name") == tool_name:
            params = fn.get("parameters", {})
            return params if isinstance(params, dict) else {}
    return None


def _check_args_schema(args, properties, required) -> bool:
    """Schema validation: required keys present, no hallucinated keys, types match.

    Deliberately NOT exact-value equality (unlike score_tool_call): at a real
    mid-session decision point there is no single correct argument value, but
    there is exactly one schema.
    """
    if not isinstance(args, dict):
        return False
    for r in (required or []):
        if r not in args:
            return False
    props = properties or {}
    # no hallucinated keys
    for k in args:
        if k not in props:
            return False
    # types match
    for k, v in args.items():
        schema = props.get(k, {})
        if not isinstance(schema, dict):
            continue
        t = schema.get("type")
        if t is None:
            continue
        types = t if isinstance(t, list) else [t]
        matched = False
        for ti in types:
            expected = _JSON_TYPE_MAP.get(ti)
            if expected is None:
                # unknown type name — don't fail on what we can't check
                matched = True
                break
            # bool is a subclass of int in python; JSON integer/number must
            # reject a bool value.
            if ti in ("integer", "number") and isinstance(v, bool):
                continue
            if isinstance(v, expected):
                matched = True
                break
        if not matched:
            return False
    return True


def score_replay_call(parsed: "ParsedCall", case: "ReplayCase",
                      capture_tools: list, closed: bool = True) -> dict:
    """Tier-0 mechanical validity for a main-replay case.

    Returns:
        acted_ok, well_formed, tool_exists (bool), args_schema_ok (bool|None),
        action_valid (bool).
    """
    if not closed:
        # thinking budget exhausted — nothing was produced to check. acted_ok
        # is False (it did not act); the remaining dimensions are not
        # applicable (no action was produced), so they are None per the
        # dimension-applicability rule.
        return {"acted_ok": False, "well_formed": None, "tool_exists": None,
                "args_schema_ok": None, "action_valid": False}

    ref = case.reference
    ref_acted = bool(ref.get("acted"))
    candidate_attempted = parsed.attempted  # a structure was present or a call parsed
    candidate_acted = parsed.tool is not None  # a well-formed tool call was made

    if not candidate_attempted:
        # Model chose prose, never attempted a call. acted_ok: did the model
        # do the right thing (act vs prose)? well_formed/tool_exists/
        # args_schema_ok are not applicable (no action to check).
        acted_ok = not ref_acted  # correct iff reference also didn't act
        return {"acted_ok": acted_ok, "well_formed": None,
                "tool_exists": None, "args_schema_ok": None,
                "action_valid": acted_ok}

    # Model attempted a call (well-formed or not). acted_ok is correct iff
    # the reference acted (the model tried to act when it should).
    acted_ok = ref_acted
    well_formed = parsed.well_formed  # meaningful: did the attempt parse?

    if not candidate_acted:
        # Attempted but failed to produce a well-formed call.
        return {"acted_ok": acted_ok, "well_formed": well_formed,
                "tool_exists": None, "args_schema_ok": None,
                "action_valid": False}  # not well_formed → not valid

    tool_names = _capture_tool_names(capture_tools)
    tool_exists = parsed.tool in tool_names

    args_schema_ok: Optional[bool]
    if not tool_exists:
        # tool not in the capture's tools — can't validate its schema.
        args_schema_ok = None
    else:
        params = _capture_tool_params(capture_tools, parsed.tool)
        if params is None:
            args_schema_ok = None
        else:
            args_schema_ok = _check_args_schema(
                parsed.arguments or {}, params.get("properties"),
                params.get("required"))

    action_valid = acted_ok and well_formed and tool_exists
    if args_schema_ok is not None:
        action_valid = action_valid and args_schema_ok
    return {"acted_ok": acted_ok, "well_formed": well_formed,
            "tool_exists": tool_exists, "args_schema_ok": args_schema_ok,
            "action_valid": action_valid}


def load_replay_meta(cases_path: str) -> dict | None:
    """Read the extractor's ``<stem>.meta.json`` sibling of a case file.

    Returns the meta dict when present and its depth_weights are usable
    (strata keys, numeric values, positive sum), else None — and with no
    weights the aggregate reports action_valid_weighted as None with
    depth_weights_source "missing", per the applicability rule: a number we
    cannot compute is None, never a stale stand-in.
    """
    meta_path = os.path.splitext(cases_path)[0] + ".meta.json"
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    weights = meta.get("depth_weights")
    if (isinstance(weights, dict)
            and set(weights) <= set(_REPLAY_STRATA)
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and v >= 0 for v in weights.values())
            and sum(weights.values()) > 0):
        return meta
    return None


def aggregate_replay(per_case: list, depth_weights: dict | None = None) -> dict:
    """Aggregate main-replay tier-0 results.

    `per_case` is a list of score dicts (from score_replay_call), each
    optionally carrying `depth_stratum` and `chain_id` keys for breakdowns.

    The headline `action_valid_weighted` is Σ stratum_rate × depth_weight over
    in-scope strata (those with cases), renormalized by the weight those strata
    carry — an absent stratum narrows the claim instead of silently dragging
    the number toward zero (a perfect shallow-only sample is 1.0, not 0.075).
    `depth_weight_coverage` (Σ weight over strata with cases) reports how much
    of the traffic mix the number represents; 1.0 when all three strata have
    cases. No unweighted pooled number exists — a pooled rate over an equal-N
    sample describes a traffic mix that does not exist. Per-stratum and
    per-chain rates are unweighted (passes/n); only the top-level number is
    weighted.
    """
    n_cases = len(per_case)
    dims = ["acted_ok", "well_formed", "tool_exists", "args_schema_ok"]
    by_dimension = {}
    for d in dims:
        vals = [c.get(d) for c in per_case]
        present = [v for v in vals if v is not None]
        rate = (sum(1 for v in present if v) / len(present)) if present else None
        by_dimension[d] = {"rate": rate, "n_applicable": len(present)}

    by_depth: dict = {}
    for stratum in _REPLAY_STRATA:
        rows = [c for c in per_case if c.get("depth_stratum") == stratum]
        n = len(rows)
        if n:
            rate = sum(1 for c in rows if c.get("action_valid")) / n
        else:
            rate = 0.0
        by_depth[stratum] = {"action_valid": rate, "n": n}

    # Weighted headline: Σ stratum_rate × depth_weight over strata with cases,
    # renormalized over the weight actually present. Coverage is reported
    # alongside so consumers can see how much of the mix the number spans.
    if depth_weights is None:
        # No observed weights: the weighted headline is not computable.
        # None, never a stale stand-in — the same applicability rule the
        # per-case dimensions follow.
        action_valid_weighted = None
        weight_present = None
        weights_used = None
        weights_source = "missing"
    else:
        weights_used = depth_weights
        weights_source = "corpus-meta"
        weighted_sum = 0.0
        weight_present = 0.0
        for stratum, weight in weights_used.items():
            d = by_depth.get(stratum) or {}
            if d.get("n", 0):
                weighted_sum += d["action_valid"] * weight
                weight_present += weight
        action_valid_weighted = (
            (weighted_sum / weight_present) if weight_present else 0.0)

    by_chain: dict = {}
    for c in per_case:
        cid = c.get("chain_id")
        if cid is None:
            continue
        by_chain.setdefault(cid, []).append(c.get("action_valid"))
    by_chain = {cid: {"action_valid": (sum(1 for v in vals if v)
                                       / len(vals) if vals else 0.0),
                      "n": len(vals)}
                for cid, vals in by_chain.items()}

    # Derived from the per-case records (which carry the case file's value):
    # unanimous -> that value, mixed -> sorted list, absent -> None.
    ref_models = sorted({c["reference_model"] for c in per_case
                         if c.get("reference_model")})
    if not ref_models:
        reference_model = None
    elif len(ref_models) == 1:
        reference_model = ref_models[0]
    else:
        reference_model = ref_models

    return {
        "action_valid_weighted": action_valid_weighted,
        "depth_weight_coverage": weight_present,
        "by_dimension": by_dimension,
        "by_depth": by_depth,
        "by_chain": by_chain,
        "reference_model": reference_model,
        "depth_weights": dict(weights_used) if weights_used else None,
        "depth_weights_source": weights_source,
        "n_cases": n_cases,
    }


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


def _stringify_content(messages: list) -> list:
    """Join list-of-parts message content into plain strings.

    Real captures carry OpenAI-style content arrays
    (``[{"type": "text", "text": ...}, ...]``); chat templates assume
    strings and crash on lists (Qwen3: ``'list object' has no attribute
    'startswith'``). Every serving layer joins text parts before
    templating, so this is render-faithfulness, not body mutation — the
    capture on disk is untouched.
    """
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            m = dict(m)
            m["content"] = "\n".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text")
        out.append(m)
    return out


def build_replay_prompt(tokenizer, case: "ReplayCase", native: bool,
                        enable_thinking=None) -> str:
    """Build the prompt from the captured request body.

    The prompt IS the captured request: its messages array (which already
    contains the session's system prompt) is applied to the chat template
    directly (content-part lists joined to strings — see
    _stringify_content). For native mode the captured `tools` array is
    forwarded byte-verbatim. For prompted mode the ``_PROMPTED_SYSTEM``
    convention plus the request's own tool schemas are appended as a final
    system message — the sole deliberate deviation from body-verbatim — so
    the model sees the JSON shape it is graded on (spec, 2026-07-28).
    """
    with open(case.capture_path, encoding="utf-8") as f:
        body = json.load(f)
    messages = _stringify_content(body["messages"])
    if native:
        tools = body.get("tools")
        return _chat(tokenizer, messages, tools=tools,
                     enable_thinking=enable_thinking)
    # Non-native: append the prompted convention + tool schemas as a
    # final system message so the model sees the format it is graded on.
    # Sole deviation from body-verbatim (spec, 2026-07-28).
    tools = body.get("tools") or []
    schemas = json.dumps(tools, indent=2)
    convention = _PROMPTED_SYSTEM.format(schemas=schemas)
    messages = list(messages)  # don't mutate the captured list
    messages.append({"role": "system", "content": convention})
    return _chat(tokenizer, messages, tools=None,
                 enable_thinking=enable_thinking)


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


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def run_constraints(cases: list, tokenizer, generate_fn,
                    enable_thinking=None) -> dict:
    per_case_checks, per_case_norm, records, errored = [], [], [], []
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
        # Re-score against the consumer-normalized text to isolate cosmetic
        # failures (wrapping quotes, code fences) from real defects.
        norm_text = normalize_output(text)
        norm_checks = score_constraint_case(norm_text, case, closed=closed)
        per_case_norm.append(norm_checks)
        records.append({"id": case.id, "checks": checks,
                        "normalized_checks": norm_checks,
                        "output_sample": text[:200],
                        "thinking_unclosed": not closed})
    return {"aggregate": aggregate_constraints(per_case_checks,
                                               per_case_norm),
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


def _fmt_secs(s: float) -> str:
    """Compact duration: 42s, 3m20s, 1h04m."""
    s = int(round(s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _replay_progress_line(i: int, total: int, case, outcome: str,
                          case_secs: float, remaining: list,
                          stratum_secs: dict, detail: str = "") -> str:
    """Format one per-case progress line.

    Carries the stratum because per-case cost varies by several multiples
    across strata — seeing which stratum a slow case belonged to is most of
    the diagnostic value, and it makes the ETA's variance legible rather than
    mysterious.
    """
    eta = _replay_eta_secs(remaining, stratum_secs)
    eta_s = f"  eta ~{_fmt_secs(eta)}" if eta is not None else ""
    tail = f"  {detail}" if detail else ""
    return (f"[{i:>3}/{total}] {getattr(case, 'id', '?'):<8} "
            f"{getattr(case, 'depth_stratum', '?'):<8} {outcome:<14} "
            f"{_fmt_secs(case_secs):>7}{eta_s}{tail}")


def _replay_eta_secs(remaining: list, stratum_secs: dict) -> Optional[float]:
    """Project remaining wall-clock, priced per stratum.

    Deep cases carry 40-60k prompt tokens against a shallow case's <16k, so
    they cost several times more. A pooled mean projected onto a stratified
    queue is therefore wrong by whatever the remaining mix happens to be —
    badly optimistic while the shallow cases are being drained, badly
    pessimistic afterwards. Each remaining case is priced by its own
    stratum's observed mean, falling back to the pooled mean for a stratum
    that has not completed a case yet. Returns None before anything has
    finished, rather than inventing a number from no data.
    """
    observed = [x for v in stratum_secs.values() for x in v]
    if not observed:
        return None
    pooled_mean = sum(observed) / len(observed)
    total = 0.0
    for case in remaining:
        samples = stratum_secs.get(getattr(case, "depth_stratum", None))
        total += (sum(samples) / len(samples)) if samples else pooled_mean
    return total


def run_main_replay(cases: list, tokenizer, generate_fn, native: bool,
                    enable_thinking=None, depth_weights: dict | None = None,
                    progress: Optional[Callable[[str], None]] = None) -> dict:
    """Replay captured LLM request bodies through a local model and score tier-0.

    The prompt is the captured request body verbatim by reference: the runner
    reads the capture file for `messages`, `tools`, and `max_tokens`, applies
    the chat template, and generates with the captured `max_tokens` (32000) —
    NOT a tight action budget. This is a deliberate parameter decision per the
    spec: the candidate gets the same token room the serving model had.

    Unlike run_tool_calling, each case runs in a single convention (native if
    the model supports tools, prompted otherwise) — there is no paired column.

    `progress` is an optional sink called with one formatted line per case as
    it completes (the CLI passes `print`; tests leave it None so the library
    stays quiet). A full matrix is tens of minutes to hours of wall-clock and
    previously emitted nothing until a mode finished, so a stalled run was
    indistinguishable from a slow one.
    """
    per_case_scores, records, errored = [], [], []
    total = len(cases)
    stratum_secs: dict = {}
    for i, case in enumerate(cases, 1):
        t_case = time.monotonic()
        try:
            with open(case.capture_path, encoding="utf-8") as f:
                body = json.load(f)
            max_tokens = int(body.get("max_tokens", 32000))
            capture_tools = body.get("tools") or []
            prompt = build_replay_prompt(tokenizer, case, native,
                                         enable_thinking=enable_thinking)
            # Truncation gate (F3a): record the token count the model is
            # actually fed by encoding the rendered prompt. This is the only
            # number that can surface silent truncation on deep cases (40–60k
            # tokens) — est_tokens is the extractor's pre-template estimate
            # and can't see template overhead or context-window clipping. A
            # tokenizer without encode() (e.g. a stub) records None plus a
            # tokens_fed_error note on the case record rather than crashing
            # the run — a silent None would hide a broken gate.
            tokens_fed_error = None
            try:
                prompt_tokens_fed = len(tokenizer.encode(prompt))
            except Exception as te:  # noqa: BLE001 - surface, don't crash
                prompt_tokens_fed = None
                tokens_fed_error = f"{type(te).__name__}: {te}"
            # Minimum gate: if the tokenizer declares a sane context length
            # and the rendered prompt exceeds it, scoring would grade a
            # silently-truncated generation. Error the case instead (raising
            # routes it through the same errored path as any other failure).
            # Many tokenizers use a huge sentinel model_max_length; treat
            # anything >= 10**9 as "no declared limit".
            if prompt_tokens_fed is not None:
                mml = getattr(tokenizer, "model_max_length", None)
                if (isinstance(mml, int) and not isinstance(mml, bool)
                        and 0 < mml < 10**9 and prompt_tokens_fed > mml):
                    raise RuntimeError(
                        f"prompt exceeds model context: "
                        f"{prompt_tokens_fed} > {mml}")
            raw = generate_fn(prompt, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001 - report, don't crash the run
            errored.append({"id": case.id, "error": str(e)})
            if progress is not None:
                # Errored cases are deliberately NOT fed into stratum_secs: a
                # case that failed to render consumed no generation time, and
                # averaging it in would make the ETA optimistic in precisely
                # the run where things are going wrong.
                progress(_replay_progress_line(
                    i, total, case, "ERROR", time.monotonic() - t_case,
                    cases[i:], stratum_secs, detail=str(e)[:60]))
            continue
        case_secs = time.monotonic() - t_case
        text, closed = strip_thinking(raw)
        parsed = parse_native(text) if native else parse_prompted(text)
        score = score_replay_call(parsed, case, capture_tools, closed=closed)
        # Attach depth_stratum, chain_id, reference_model so aggregate_replay
        # can break down by stratum, chain, and report the reference model.
        score_with_depth = dict(score)
        score_with_depth["depth_stratum"] = case.depth_stratum
        score_with_depth["chain_id"] = case.chain_id
        score_with_depth["reference_model"] = case.reference_model
        per_case_scores.append(score_with_depth)
        rec = {
            "id": case.id, "chain_id": case.chain_id,
            "depth_stratum": case.depth_stratum, "native": native,
            "score": score, "output_sample": text[:200],
            "thinking_unclosed": not closed,
            "prompt_tokens_fed": prompt_tokens_fed,
            "est_tokens": case.est_tokens,
        }
        if tokens_fed_error is not None:
            rec["tokens_fed_error"] = tokens_fed_error
        rec["case_secs"] = round(case_secs, 2)
        records.append(rec)
        stratum_secs.setdefault(case.depth_stratum, []).append(case_secs)
        if progress is not None:
            av = score.get("action_valid")
            outcome = "valid" if av else ("unclosed-think" if not closed
                                          else "invalid")
            progress(_replay_progress_line(
                i, total, case, outcome, case_secs, cases[i:], stratum_secs))
    return {"aggregate": aggregate_replay(per_case_scores,
                                          depth_weights=depth_weights),
            "cases": records, "errored": errored}


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


def format_replay_table(label: str, result: dict) -> str:
    agg = result["aggregate"]
    avw = agg.get("action_valid_weighted")
    headline = (f"{label}:  action_valid_weighted="
                + (f"{avw:.2f}" if avw is not None
                   else "n/a (weights missing)")
                + f"  (n={agg['n_cases']})")
    coverage = agg.get("depth_weight_coverage")
    if coverage is not None and coverage < 1.0:
        # The renormalized headline only speaks for the strata that have
        # cases — flag how much of the traffic mix that is.
        headline += f"  coverage={coverage:.2f} — missing strata"
    lines = [headline]
    dims = agg.get("by_dimension") or {}
    if dims:
        parts = []
        for k, v in dims.items():
            if isinstance(v, dict):
                rate = v.get("rate")
                n = v.get("n_applicable", 0)
                parts.append(f"{k}={_fmt(rate)}(n={n})")
            else:
                parts.append(f"{k}={_fmt(v)}")
        lines.append("  by dimension: " + "  ".join(parts))
    by_depth = agg.get("by_depth") or {}
    if by_depth:
        parts = []
        for s in ("shallow", "mid", "deep"):
            d = by_depth.get(s) or {}
            parts.append(f"{s}={_fmt(d.get('action_valid'))}"
                         f"(n={d.get('n', 0)})")
        lines.append("  by depth: " + "  ".join(parts))
    if result["errored"]:
        lines.append(f"  errored: {len(result['errored'])} case(s)")
    return "\n".join(lines)


def _headline(result: dict, profile: str) -> float:
    """The one number a thinking/no-thinking comparison turns on."""
    agg = result["aggregate"]
    if profile in ("constraints", "chore"):
        return agg["strict"] or 0.0
    if profile == "main-replay":
        # May be None when depth weights were missing; callers must guard.
        return agg.get("action_valid_weighted")
    return (agg["prompted"] or {}).get("right_tool") or 0.0


def format_thinking_gap(label: str, per_mode: dict, profile: str,
                        tokens_per_case: dict) -> str:
    """Report what thinking bought, and what it cost.

    The delta alone is only half a routing decision: +0.08 accuracy for 18x the
    tokens is a different call than +0.08 for free. Loxo needs both, so both go
    on the line.
    """
    off, on = per_mode["no-think"], per_mode["think"]
    h_off, h_on = _headline(off, profile), _headline(on, profile)
    metric = ("strict" if profile in ("constraints", "chore")
              else "action_valid_weighted" if profile == "main-replay"
              else "right_tool")
    t_off = tokens_per_case.get("no-think") or 0.0
    t_on = tokens_per_case.get("think") or 0.0
    cost = f"{t_off:.0f} -> {t_on:.0f} tok/case"
    if t_off > 0:
        cost += f" ({t_on / t_off:.1f}x)"
    if h_off is None or h_on is None:
        return (f"{label}: thinking gap({metric})=n/a (weights missing)  "
                f"cost: {cost}")
    return (f"{label}: thinking gap({metric})={h_on - h_off:+.2f}  "
            f"[no-think={h_off:.2f}  think={h_on:.2f}]  cost: {cost}")


def write_sidecar(path: str, profile: str, model: str, label: str,
                  convention_support: dict, result: dict,
                  modes: Optional[dict] = None,
                  tokens_per_case: Optional[dict] = None,
                  cost: Optional[dict] = None) -> None:
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
    if cost:
        payload["load_ms"] = cost.get("load_ms")
        payload["peak_memory_mb"] = cost.get("peak_memory_mb")
    # Top-level keys stay the primary (thinking, where supported) result so the
    # sidecar contract holds; per-mode detail rides alongside for Loxo, which
    # needs the accuracy/token trade-off to route.
    if modes:
        payload["modes"] = {
            m: {"aggregate": r["aggregate"], "cases": r.get("cases", []),
                "errored": r.get("errored", []),
                "mean_tokens_per_case": (tokens_per_case or {}).get(m),
                "median_latency_ms": (cost or {}).get("median_latency_ms", {}).get(m),
                "latencies_ms": (cost or {}).get("latencies_ms", {}).get(m),
                "tokens_per_second": (cost or {}).get("tokens_per_second", {}).get(m),
                "peak_memory_mb": (cost or {}).get("peak_memory_mb"),
                "load_ms": (cost or {}).get("load_ms")}
            for m, r in modes.items()
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

import argparse
import statistics

DEFAULT_CASES = {
    "tool-calling": "cases/tool_calling.jsonl",
    "constraints": "cases/constraints.jsonl",
    "chore": "cases/chore.jsonl",
    "main-replay": "cases/main_replay.jsonl",
}
DEFAULT_MAX_TOKENS = {"tool-calling": 128, "constraints": 512, "chore": 64}
# main-replay is deliberately absent: max_tokens comes from the captured body
# (32000), NOT from the CLI. The runner reads body["max_tokens"] per case.

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
                    choices=["tool-calling", "constraints", "chore",
                             "main-replay"])
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
    if args.profile == "main-replay":
        # max_tokens comes from the captured body (32000), not the CLI. The
        # budgets below are only display placeholders; the per-case budget is
        # read by run_main_replay from each capture file.
        max_tokens = None
        think_max_tokens = None
    else:
        max_tokens = args.max_tokens or DEFAULT_MAX_TOKENS[args.profile]
        think_max_tokens = (args.think_max_tokens
                            or THINKING_MAX_TOKENS[args.profile])
    cases = load_cases(cases_path, args.profile)   # fail-fast before loading a model
    print(f"loaded {len(cases)} cases from {cases_path}")
    replay_weights = None
    if args.profile == "main-replay":
        replay_meta = load_replay_meta(cases_path)
        if replay_meta is not None:
            replay_weights = replay_meta["depth_weights"]
            print(f"depth weights from corpus meta: {replay_weights}")
        else:
            print("no usable .meta.json beside case file — depth weights "
                  "missing, action_valid_weighted will be null "
                  "(depth_weights_source=missing)")

    backend = get_backend(args.backend)
    for label, repo in _targets_for(args):
        print(f"\n=== {label}: loading {repo} ===")
        backend.reset_peak_memory()
        model, tokenizer, t_load = backend.load(repo)
        print(f"loaded in {t_load:.1f}s")

        thinking = supports_thinking(tokenizer)
        native = (args.profile in ("tool-calling", "main-replay")
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
        median_latency_ms, latencies_ms, tokens_per_second = {}, {}, {}
        for mode, flag, budget in modes:
            tally = {"tokens": 0, "calls": 0, "latencies": [], "gen_seconds": 0.0}

            if args.profile == "main-replay":
                # main-replay's budget comes from each capture's body
                # (max_tokens=32000), not the CLI mode budget. The gen closure
                # accepts max_tokens as a kwarg from run_main_replay.
                def gen(p, max_tokens=32000, _tally=tally,
                        _model=model, _tok=tokenizer):
                    t0 = time.perf_counter()
                    text = "".join(backend.stream(_model, _tok, p, max_tokens))
                    elapsed = time.perf_counter() - t0
                    _tally["latencies"].append(elapsed * 1000)
                    _tally["gen_seconds"] += elapsed
                    _tally["tokens"] += len(_tok.encode(text))
                    _tally["calls"] += 1
                    return text
            else:
                def gen(p, _budget=budget, _tally=tally,
                        _model=model, _tok=tokenizer):
                    t0 = time.perf_counter()
                    text = "".join(backend.stream(_model, _tok, p, _budget))
                    elapsed = time.perf_counter() - t0
                    _tally["latencies"].append(elapsed * 1000)
                    _tally["gen_seconds"] += elapsed
                    _tally["tokens"] += len(_tok.encode(text))
                    _tally["calls"] += 1
                    return text

            budget_display = "capture (per-case)" if args.profile == "main-replay" else budget
            print(f"\n--- {label} [{mode}] (max_tokens={budget_display}) ---")
            if args.profile in ("constraints", "chore"):
                result = run_constraints(cases, tokenizer, gen,
                                         enable_thinking=flag)
                print(format_constraint_table(f"{label} [{mode}]", result))
            elif args.profile == "main-replay":
                result = run_main_replay(cases, tokenizer, gen, native=native,
                                         enable_thinking=flag,
                                         depth_weights=replay_weights,
                                         progress=print)
                print(format_replay_table(f"{label} [{mode}]", result))
            else:
                result = run_tool_calling(cases, tokenizer, gen, native=native,
                                          enable_thinking=flag)
                print(format_tool_table(f"{label} [{mode}]", result))
            per_mode[mode] = result
            tokens_per_case[mode] = (tally["tokens"] / tally["calls"]
                                     if tally["calls"] else 0.0)
            latencies_ms[mode] = [round(x, 1) for x in tally["latencies"]]
            median_latency_ms[mode] = (statistics.median(tally["latencies"])
                                       if tally["latencies"] else 0.0)
            tokens_per_second[mode] = (tally["tokens"] / tally["gen_seconds"]
                                       if tally["gen_seconds"] > 0 else 0.0)
            print(f"  median latency: {median_latency_ms[mode]:.0f} ms  "
                  f"tok/sec: {tokens_per_second[mode]:.1f}")

        peak_mb = backend.peak_memory_mb()

        if thinking:
            print("\n" + format_thinking_gap(label, per_mode, args.profile,
                                             tokens_per_case))
        primary = per_mode.get("think") or per_mode["default"]

        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
        out_path = os.path.join(args.out_dir, f"results_{args.profile}_{safe}.json")
        cost = {"median_latency_ms": median_latency_ms,
                "latencies_ms": latencies_ms,
                "tokens_per_second": tokens_per_second,
                "peak_memory_mb": peak_mb,
                "load_ms": round(t_load * 1000)}
        write_sidecar(out_path, args.profile, repo, label, conv, primary,
                      modes=per_mode, tokens_per_case=tokens_per_case,
                      cost=cost)
        print(f"sidecar written: {out_path}  (peak: {peak_mb:.0f} MB)")

        del model, tokenizer
        gc.collect()
        backend.clear_cache()


if __name__ == "__main__":
    main()
