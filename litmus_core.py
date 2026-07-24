"""Backend-agnostic core for Litmus.

Hosts every algorithm shared across backends and CLIs: reference-text loading,
reasoning-preamble stripping, the distinct-trigram decode-stability metric,
perplexity windowing + aggregation, the Run record, report rendering, and the
per-command perf drivers. Imports neither mlx nor torch — the model runtime is
reached only through a Backend (see litmus_common.get_backend).
"""
from __future__ import annotations

import gc
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

from litmus_common import (
    BASELINE_MODELS, PROMPTS, WARMUP_PROMPT, _targets_for,
)

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


def _load_reference_text(path: str = REFERENCE_TEXT_PATH) -> str:
    """Load and strip a Project Gutenberg plain-text file.

    The script never embeds prose — the user downloads Gatsby once. See the
    module docstring for the curl command.
    """
    if not os.path.exists(path):
        raise SystemExit(
            f"reference text not found at {path}.\n"
            "Download The Great Gatsby from Project Gutenberg once:\n"
            "  curl -L https://www.gutenberg.org/cache/epub/64317/pg64317.txt "
            f"-o {path}"
        )
    with open(path, encoding="utf-8") as f:
        text = f.read()
    # Strip Gutenberg header/footer if present.
    start = text.find("*** START OF")
    if start != -1:
        nl = text.find("\n", start)
        if nl != -1:
            text = text[nl + 1:]
    end = text.find("*** END OF")
    if end != -1:
        text = text[:end]
    return text.strip()


def _strip_thinking(text: str, tokenizer) -> tuple[str, int]:
    """Remove a reasoning preamble from the front of `text`.

    Returns (useful_text, scratchpad_token_count). Two detection paths:

    1. Explicit reasoning block delimited by a literal ``</think>`` tag
       (LFM2.5, DeepSeek-R1, QwQ, …). Everything up to and including the
       closing tag is scratchpad; the remainder is the answer. An opened
       ``<think>`` with no closing tag means the model ran out of budget
       mid-reasoning — the whole generation is treated as scratchpad.
    2. Fallback heuristic for untagged models: a prose preamble separated
       from the answer by the first ``\\n\\n`` blank-line break.

    If nothing is detected, returns (text, 0). Token counts are a heuristic
    — re-encoding a segment may not match the streaming count exactly, but
    it's close enough for a rough "useful t/s".
    """
    stripped = text.lstrip()

    # Path 1: explicit <think>…</think> tag — preferred when present.
    close_idx = text.find("</think>")
    if close_idx != -1:
        end = close_idx + len("</think>")
        return text[end:].lstrip(), len(tokenizer.encode(text[:end]))
    if stripped.startswith("<think>"):
        # Opened a think block but never closed it — whole output is scratchpad.
        return "", len(tokenizer.encode(text))

    # Path 2: untagged prose-preamble heuristic.
    after_paren = PARENTHETICAL_PREFIX_RE.sub("", stripped, count=1)
    if not PREAMBLE_RE.match(after_paren):
        return text, 0
    parts = text.split("\n\n", 1)
    if len(parts) < 2:
        # Preamble never ended — whole output is scratchpad.
        return "", len(tokenizer.encode(text))
    head, tail = parts
    scratchpad_tokens = len(tokenizer.encode(head + "\n\n"))
    return tail, scratchpad_tokens


def distinct_trigram_ratio(tokens: list[int]) -> float:
    """Fraction of distinct token trigrams (1.0 = no repetition).

    Returns nan for sequences shorter than 3 tokens. Extracted verbatim from
    the decode-stability loop so both backends and the golden tests share it.
    """
    if len(tokens) < 3:
        return float("nan")
    trigrams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
    return len(set(trigrams)) / len(trigrams)


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


def print_table(all_runs: list[Run]) -> None:
    has_useful = any(r.useful_decode_tps is not None for r in all_runs)
    width = 122 if has_useful else 110

    print("\n" + "=" * width)
    header = (
        f"{'size':<14} {'prompt':<40} {'p_tok':>6} {'g_tok':>6} "
        f"{'prefill t/s':>12} {'decode t/s':>12} {'TTFT ms':>10} {'peak MB':>10}"
    )
    if has_useful:
        header += f" {'useful t/s':>12}"
    print(header)
    print("-" * width)
    for r in all_runs:
        line = (
            f"{r.label:<14} {r.prompt:<40} {r.prompt_tokens:>6} {r.gen_tokens:>6} "
            f"{r.prefill_tps:>12.1f} {r.decode_tps:>12.1f} "
            f"{r.ttft_ms:>10.1f} {r.peak_mem_mb:>10.0f}"
        )
        if has_useful:
            u = r.useful_decode_tps
            line += f" {u:>12.1f}" if u is not None else f" {'-':>12}"
        print(line)

    print("\n--- Per-size summary ---")
    header = (
        f"{'size':<14} {'avg decode t/s':>16} {'avg prefill t/s':>18} "
        f"{'avg TTFT ms':>14} {'peak MB':>10}"
    )
    if has_useful:
        header += f" {'avg useful t/s':>16}"
    print(header)
    by_label: dict[str, list[Run]] = {}
    for r in all_runs:
        by_label.setdefault(r.label, []).append(r)
    for label, runs in by_label.items():
        avg_dec = sum(r.decode_tps for r in runs) / len(runs)
        avg_pre = sum(r.prefill_tps for r in runs) / len(runs)
        avg_ttft = sum(r.ttft_ms for r in runs) / len(runs)
        peak = max(r.peak_mem_mb for r in runs)
        line = (
            f"{label:<14} {avg_dec:>16.1f} {avg_pre:>18.1f} "
            f"{avg_ttft:>14.1f} {peak:>10.0f}"
        )
        if has_useful:
            useful_vals = [r.useful_decode_tps for r in runs if r.useful_decode_tps is not None]
            if useful_vals:
                avg_useful = sum(useful_vals) / len(useful_vals)
                line += f" {avg_useful:>16.1f}"
            else:
                line += f" {'-':>16}"
        print(line)

    print("\n--- Sample outputs (first 80 chars) ---")
    for r in all_runs:
        print(f"  [{r.label}] {r.prompt[:30]:<30} -> {r.sample}")


# ---------------------------------------------------------------------------
# cmd: throughput
# ---------------------------------------------------------------------------

def run_one(
    backend,
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    label: str,
    strip_thinking: bool = False,
) -> Run:
    backend.reset_peak_memory()

    n_prompt = len(tokenizer.encode(prompt))
    t_start = time.perf_counter()
    first_token_t: Optional[float] = None
    chunks: list[str] = []
    gen_tokens = 0

    for resp in backend.stream(model, tokenizer, prompt, max_tokens):
        if first_token_t is None:
            first_token_t = time.perf_counter()
        chunks.append(resp)
        gen_tokens += 1

    t_end = time.perf_counter()
    if first_token_t is None:
        first_token_t = t_end

    prefill_s = first_token_t - t_start
    decode_s = max(t_end - first_token_t, 1e-9)
    full_text = "".join(chunks)

    useful_gen: Optional[int] = None
    useful_tps: Optional[float] = None
    if strip_thinking:
        _, scratchpad_tokens = _strip_thinking(full_text, tokenizer)
        useful_gen = max(gen_tokens - scratchpad_tokens, 0)
        useful_tps = useful_gen / decode_s

    return Run(
        label=label,
        prompt=prompt[:40],
        prompt_tokens=n_prompt,
        gen_tokens=gen_tokens,
        prefill_tps=(n_prompt / prefill_s) if prefill_s > 0 else 0.0,
        decode_tps=gen_tokens / decode_s,
        ttft_ms=prefill_s * 1000.0,
        peak_mem_mb=backend.peak_memory_mb(),
        sample=full_text[:80].replace("\n", " "),
        useful_gen_tokens=useful_gen,
        useful_decode_tps=useful_tps,
    )


def bench_model(
    backend, label: str, repo: str, max_tokens: int, strip_thinking: bool
) -> list[Run]:
    print(f"\n=== {label}: loading {repo} ===")
    model, tokenizer, t_load = backend.load(repo)
    print(f"loaded in {t_load:.1f}s")

    # Warmup pass — discard. First call pays kernel-compile + cache costs.
    print("warmup...")
    for _ in backend.stream(model, tokenizer, WARMUP_PROMPT, 8):
        pass

    runs: list[Run] = []
    for prompt in PROMPTS:
        r = run_one(backend, model, tokenizer, prompt, max_tokens, label,
                    strip_thinking)
        runs.append(r)
        extra = ""
        if r.useful_decode_tps is not None:
            extra = f"  useful {r.useful_decode_tps:>6.1f} t/s"
        print(
            f"  {prompt[:36]:<36} {r.decode_tps:>6.1f} tok/s "
            f"prefill {r.prefill_tps:>6.1f} t/s  TTFT {r.ttft_ms:>6.1f} ms  "
            f"peak {r.peak_mem_mb:>6.0f} MB{extra}"
        )

    del model, tokenizer
    gc.collect()
    backend.clear_cache()
    return runs


def cmd_throughput(backend, args) -> None:
    all_runs: list[Run] = []
    for label, repo in _targets_for(args):
        all_runs.extend(
            bench_model(backend, label, repo, args.max_tokens, args.strip_thinking)
        )
    print_table(all_runs)


# ---------------------------------------------------------------------------
# cmd: perplexity
# ---------------------------------------------------------------------------

def cmd_perplexity(backend, args) -> None:
    text = _load_reference_text(args.reference_text)
    print(f"loaded reference text: {len(text)} chars")
    print(f"perplexity window: {args.ppl_window} tokens\n")

    results: list[tuple[str, float]] = []
    for label, repo in _targets_for(args):
        print(f"=== {label}: loading {repo} ===")
        model, tokenizer, t_load = backend.load(repo)
        print(f"loaded in {t_load:.1f}s")
        ppl = compute_perplexity(backend, model, tokenizer, text, args.ppl_window)
        print(f"  perplexity: {ppl:.3f}\n")
        results.append((label, ppl))
        del model, tokenizer
        gc.collect()
        backend.clear_cache()

    print("--- Perplexity summary (lower is better) ---")
    print(f"{'label':<32} {'perplexity':>12}")
    for label, ppl in results:
        print(f"{label:<32} {ppl:>12.3f}")


# ---------------------------------------------------------------------------
# cmd: prefill-scaling
# ---------------------------------------------------------------------------

def cmd_prefill_scaling(backend, args) -> None:
    lengths = [10, 50, 200, 500, 1000]
    ref = _load_reference_text(args.reference_text)

    for label, repo in _targets_for(args):
        print(f"\n=== {label}: prefill scaling ===")
        model, tokenizer, t_load = backend.load(repo)
        print(f"loaded in {t_load:.1f}s")

        for _ in backend.stream(model, tokenizer, WARMUP_PROMPT, 4):
            pass

        ref_ids = tokenizer.encode(ref)
        if len(ref_ids) < max(lengths):
            raise SystemExit(
                f"reference text has only {len(ref_ids)} tokens, need "
                f"{max(lengths)}. Download the full Gutenberg text."
            )

        print(f"  {'n_tokens':>10} {'prefill t/s':>14} {'TTFT ms':>12}")
        for n in lengths:
            ids = ref_ids[:n]
            prompt = tokenizer.decode(ids)
            # Re-encode to get the actual token count after decode round-trip.
            actual_n = len(tokenizer.encode(prompt))
            t0 = time.perf_counter()
            first = None
            for _ in backend.stream(model, tokenizer, prompt, 1):
                if first is None:
                    first = time.perf_counter()
                    break
            if first is None:
                first = time.perf_counter()
            prefill_s = max(first - t0, 1e-9)
            print(
                f"  {actual_n:>10} {actual_n / prefill_s:>14.1f} "
                f"{prefill_s * 1000:>12.1f}"
            )

        del model, tokenizer
        gc.collect()
        backend.clear_cache()


# ---------------------------------------------------------------------------
# cmd: decode-stability
# ---------------------------------------------------------------------------

def cmd_decode_stability(backend, args) -> None:
    window = 128
    total = args.max_tokens if args.max_tokens >= window else 1024
    if args.max_tokens < window:
        print(f"(bumping max-tokens to {total} so windows fit)")

    prompt = (
        "Write a long, detailed essay about the history of the telescope, "
        "covering its invention, major improvements, and scientific impact."
    )

    for label, repo in _targets_for(args):
        mode = "chat" if args.chat else "raw"
        print(f"\n=== {label}: decode stability over {total} tokens ({mode}) ===")
        model, tokenizer, t_load = backend.load(repo)
        print(f"loaded in {t_load:.1f}s")
        backend.reset_peak_memory()

        # Optionally wrap the prompt in the model's chat template. Without this
        # an instruct-tuned model continues the prompt as text rather than
        # responding to it.
        if args.chat:
            try:
                actual_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception as e:
                print(f"  WARNING: chat template failed ({e}); using raw prompt")
                actual_prompt = prompt
        else:
            actual_prompt = prompt

        for _ in backend.stream(model, tokenizer, WARMUP_PROMPT, 4):
            pass

        per_window_tps: list[float] = []
        per_window_mem: list[float] = []
        chunks: list[str] = []
        gen = 0
        t_prev = None
        t_first = None

        for resp in backend.stream(model, tokenizer, actual_prompt, total):
            if t_prev is None:
                t_prev = time.perf_counter()
                t_first = t_prev
            chunks.append(resp)
            gen += 1
            if gen % window == 0:
                t_now = time.perf_counter()
                per_window_tps.append(window / max(t_now - t_prev, 1e-9))
                per_window_mem.append(backend.peak_memory_mb())
                t_prev = t_now
        t_last = time.perf_counter()

        full_text = "".join(chunks)
        tokens = tokenizer.encode(full_text)
        distinct = distinct_trigram_ratio(tokens)

        print(f"  windows: {len(per_window_tps)} x {window} tokens each")
        print(
            "  window tok/s: "
            + " ".join(f"{t:6.1f}" for t in per_window_tps)
        )
        print(
            "  peak MB:      "
            + " ".join(f"{m:6.0f}" for m in per_window_mem)
        )
        if per_window_tps:
            first, last = per_window_tps[0], per_window_tps[-1]
            pct = (last - first) / first * 100 if first > 0 else 0
            direction = "slowdown" if pct < 0 else "speedup"
            print(f"  first->last tok/s: {pct:+.1f}% ({direction})")
        print(f"  distinct-trigram ratio: {distinct:.3f} (1.0 = no repetition)")

        if args.strip_thinking:
            useful_text, scratch_tokens = _strip_thinking(full_text, tokenizer)
            useful_tokens = len(tokenizer.encode(useful_text)) if useful_text else 0
            elapsed = (t_last - t_first) if t_first is not None else float("nan")
            useful_tps = useful_tokens / max(elapsed, 1e-9)
            useful_pct = (useful_tokens / gen * 100) if gen else 0.0
            print(
                f"  strip-thinking: {scratch_tokens} scratchpad tok, "
                f"{useful_tokens} useful tok ({useful_pct:.0f}% useful)"
            )
            print(
                f"  useful tok/s: {useful_tps:.1f} "
                "(answer tokens / decode time, excludes reasoning)"
            )

        # Dump full generated text and show head/tail inline. Critical when
        # the trigram ratio collapses — without seeing the actual loop content,
        # you can't tell whether the model is repeating prose or stuck on a
        # single token.
        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
        out_path = os.path.join(
            tempfile.gettempdir(), f"decode_stability_{safe_label}.txt"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        head = full_text[:300].replace("\n", " ⏎ ")
        tail = full_text[-300:].replace("\n", " ⏎ ")
        print(f"  full output saved: {out_path} ({len(full_text)} chars)")
        print(f"  head: {head}")
        print(f"  tail: {tail}")

        del model, tokenizer
        gc.collect()
        backend.clear_cache()


# ---------------------------------------------------------------------------
# cmd: baseline
# ---------------------------------------------------------------------------

def cmd_baseline(backend, args) -> None:
    """Run stock 4-bit Llama 3.2 1B & 3B through throughput + perplexity."""
    try:
        ref = _load_reference_text(args.reference_text)
    except SystemExit as e:
        print(f"(skipping perplexity: {e})")
        ref = None

    all_runs: list[Run] = []
    ppls: list[tuple[str, float]] = []

    for name, repo in BASELINE_MODELS.items():
        print(f"\n=== {name}: loading {repo} ===")
        model, tokenizer, t_load = backend.load(repo)
        print(f"loaded in {t_load:.1f}s")

        for _ in backend.stream(model, tokenizer, WARMUP_PROMPT, 8):
            pass

        for prompt in PROMPTS:
            r = run_one(backend, model, tokenizer, prompt, args.max_tokens,
                        name, False)
            all_runs.append(r)
            print(
                f"  {prompt[:36]:<36} {r.decode_tps:>6.1f} tok/s "
                f"prefill {r.prefill_tps:>6.1f} t/s  peak {r.peak_mem_mb:>6.0f} MB"
            )

        if ref is not None:
            ppl = compute_perplexity(backend, model, tokenizer, ref,
                                     args.ppl_window)
            print(f"  perplexity: {ppl:.3f}")
            ppls.append((name, ppl))

        del model, tokenizer
        gc.collect()
        backend.clear_cache()

    print_table(all_runs)
    if ppls:
        print("\n--- Baseline perplexity ---")
        for name, ppl in ppls:
            print(f"  {name:<20} {ppl:.3f}")


# ---------------------------------------------------------------------------
# cmd: cold-start
# ---------------------------------------------------------------------------

def _single_ttft(backend, model, tokenizer, prompt: str) -> float:
    t0 = time.perf_counter()
    for _ in backend.stream(model, tokenizer, prompt, 1):
        return time.perf_counter() - t0
    return time.perf_counter() - t0


def cmd_cold_start(backend, args) -> None:
    print(f"{'label':<24} {'load s':>10} {'cold TTFT ms':>14} "
          f"{'warm TTFT ms':>14} {'delta ms':>12}")
    for label, repo in _targets_for(args):
        model, tokenizer, t_load = backend.load(repo)
        cold_ttft = _single_ttft(backend, model, tokenizer, WARMUP_PROMPT)
        warm_ttft = _single_ttft(backend, model, tokenizer, WARMUP_PROMPT)
        delta = (cold_ttft - warm_ttft) * 1000
        print(
            f"{label:<24} {t_load:>10.2f} {cold_ttft * 1000:>14.1f} "
            f"{warm_ttft * 1000:>14.1f} {delta:>12.1f}"
        )
        del model, tokenizer
        gc.collect()
        backend.clear_cache()


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

COMMANDS = {
    "throughput": cmd_throughput,
    "perplexity": cmd_perplexity,
    "prefill-scaling": cmd_prefill_scaling,
    "decode-stability": cmd_decode_stability,
    "baseline": cmd_baseline,
    "cold-start": cmd_cold_start,
}
