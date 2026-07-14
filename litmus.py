"""
Litmus - a local-LLM eval suite for MLX inference on Apple Silicon
(decode-stability, perplexity, throughput, cold-start, time-to-first-token).
The measurement layer of the Loxo eval-driven router.

v0 benchmarks PrismML Bonsai 1-bit MLX models. Point it at any MLX repo with --repo.

Loads each Bonsai size in turn, runs a fixed prompt set through it, and
prints prefill throughput, decode throughput, time-to-first-token, and
peak GPU memory. Designed to give apples-to-apples numbers on whatever
Mac you run it on.

Setup (one-time, recommended in a venv since the PrismML fork pins MLX):
    python -m venv .venv-litmus && source .venv-litmus/bin/activate
    pip install mlx-lm psutil
    pip install "mlx @ git+https://github.com/PrismML-Eng/mlx.git@prism"

Reference text (required for `perplexity` and `prefill-scaling`):
    The Great Gatsby is public domain in the US since 2021. Download once:
        curl -L https://www.gutenberg.org/cache/epub/64317/pg64317.txt \
             -o reference.txt

Usage:
    python litmus.py                           # throughput (default)
    python litmus.py --strip-thinking          # report useful t/s
    python litmus.py --cmd perplexity
    python litmus.py --cmd prefill-scaling
    python litmus.py --cmd decode-stability --max-tokens 1024
    python litmus.py --cmd baseline
    python litmus.py --cmd cold-start
    python litmus.py --sizes 1.7B,4B

Models live at:
    https://huggingface.co/prism-ml/Bonsai-1.7B-mlx-1bit
    https://huggingface.co/prism-ml/Bonsai-4B-mlx-1bit
    https://huggingface.co/prism-ml/Bonsai-8B-mlx-1bit

Note: stock mlx_lm does not yet have 1-bit g128 kernels. Without the
PrismML fork installed, `load()` will fail with an "invalid quant value"
error.
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

from litmus_common import (
    MODELS, BASELINE_MODELS, PROMPTS, WARMUP_PROMPT,
    _resp_text, _peak_memory_mb, _reset_peak_memory, _clear_cache,
    _parse_sizes, _targets_for, _load_timed,
    mx, nn, load, stream_generate,
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


@dataclass
class Run:
    size: str
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


# ---------------------------------------------------------------------------
# cmd: throughput
# ---------------------------------------------------------------------------

def run_one(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    size: str,
    strip_thinking: bool = False,
) -> Run:
    _reset_peak_memory()

    n_prompt = len(tokenizer.encode(prompt))
    t_start = time.perf_counter()
    first_token_t: Optional[float] = None
    chunks: list[str] = []
    gen_tokens = 0

    for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        if first_token_t is None:
            first_token_t = time.perf_counter()
        chunks.append(_resp_text(resp))
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
        size=size,
        prompt=prompt[:40],
        prompt_tokens=n_prompt,
        gen_tokens=gen_tokens,
        prefill_tps=(n_prompt / prefill_s) if prefill_s > 0 else 0.0,
        decode_tps=gen_tokens / decode_s,
        ttft_ms=prefill_s * 1000.0,
        peak_mem_mb=_peak_memory_mb(),
        sample=full_text[:80].replace("\n", " "),
        useful_gen_tokens=useful_gen,
        useful_decode_tps=useful_tps,
    )


def bench_model(
    size: str, repo: str, max_tokens: int, strip_thinking: bool
) -> list[Run]:
    print(f"\n=== {size}: loading {repo} ===")
    model, tokenizer, t_load = _load_timed(repo)
    print(f"loaded in {t_load:.1f}s")

    # Warmup pass — discard. First call pays kernel-compile + cache costs.
    print("warmup...")
    for _ in stream_generate(model, tokenizer, WARMUP_PROMPT, max_tokens=8):
        pass

    runs: list[Run] = []
    for prompt in PROMPTS:
        r = run_one(model, tokenizer, prompt, max_tokens, size, strip_thinking)
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
    _clear_cache()
    return runs


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
            f"{r.size:<14} {r.prompt:<40} {r.prompt_tokens:>6} {r.gen_tokens:>6} "
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
    by_size: dict[str, list[Run]] = {}
    for r in all_runs:
        by_size.setdefault(r.size, []).append(r)
    for size, runs in by_size.items():
        avg_dec = sum(r.decode_tps for r in runs) / len(runs)
        avg_pre = sum(r.prefill_tps for r in runs) / len(runs)
        avg_ttft = sum(r.ttft_ms for r in runs) / len(runs)
        peak = max(r.peak_mem_mb for r in runs)
        line = (
            f"{size:<14} {avg_dec:>16.1f} {avg_pre:>18.1f} "
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
        print(f"  [{r.size}] {r.prompt[:30]:<30} -> {r.sample}")


def cmd_throughput(args) -> None:
    all_runs: list[Run] = []
    for label, repo in _targets_for(args):
        all_runs.extend(
            bench_model(label, repo, args.max_tokens, args.strip_thinking)
        )
    print_table(all_runs)


# ---------------------------------------------------------------------------
# cmd: perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(model, tokenizer, text: str, window: int) -> float:
    """Teacher-forced perplexity over the first `window` tokens of `text`.

    Returns exp(mean NLL). Lower is better.
    """
    ids = tokenizer.encode(text)
    ids = ids[:window]
    if len(ids) < 2:
        return float("nan")
    x = mx.array(ids)[None, :]  # (1, T)
    logits = model(x)  # (1, T, V)
    log_probs = nn.log_softmax(logits[:, :-1, :], axis=-1)
    targets = x[:, 1:, None]  # (1, T-1, 1)
    gathered = mx.take_along_axis(log_probs, targets, axis=-1).squeeze(-1)
    nll = -gathered  # (1, T-1)
    mean_nll = mx.mean(nll)
    return float(mx.exp(mean_nll).item())


def cmd_perplexity(args) -> None:
    text = _load_reference_text(args.reference_text)
    print(f"loaded reference text: {len(text)} chars")
    print(f"perplexity window: {args.ppl_window} tokens\n")

    results: list[tuple[str, float]] = []
    for label, repo in _targets_for(args):
        print(f"=== {label}: loading {repo} ===")
        model, tokenizer, t_load = _load_timed(repo)
        print(f"loaded in {t_load:.1f}s")
        ppl = compute_perplexity(model, tokenizer, text, args.ppl_window)
        print(f"  perplexity: {ppl:.3f}\n")
        results.append((label, ppl))
        del model, tokenizer
        gc.collect()
        _clear_cache()

    print("--- Perplexity summary (lower is better) ---")
    print(f"{'label':<32} {'perplexity':>12}")
    for label, ppl in results:
        print(f"{label:<32} {ppl:>12.3f}")


# ---------------------------------------------------------------------------
# cmd: prefill-scaling
# ---------------------------------------------------------------------------

def cmd_prefill_scaling(args) -> None:
    lengths = [10, 50, 200, 500, 1000]
    ref = _load_reference_text(args.reference_text)

    for label, repo in _targets_for(args):
        print(f"\n=== {label}: prefill scaling ===")
        model, tokenizer, t_load = _load_timed(repo)
        print(f"loaded in {t_load:.1f}s")

        for _ in stream_generate(model, tokenizer, WARMUP_PROMPT, max_tokens=4):
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
            for _ in stream_generate(model, tokenizer, prompt, max_tokens=1):
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
        _clear_cache()


# ---------------------------------------------------------------------------
# cmd: decode-stability
# ---------------------------------------------------------------------------

def cmd_decode_stability(args) -> None:
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
        model, tokenizer, t_load = _load_timed(repo)
        print(f"loaded in {t_load:.1f}s")
        _reset_peak_memory()

        # Optionally wrap the prompt in the model's chat template. Without this
        # an instruct-tuned model continues the prompt as text rather than
        # responding to it — see qwen3_handoff.md for the failure mode this
        # was added to address.
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

        for _ in stream_generate(model, tokenizer, WARMUP_PROMPT, max_tokens=4):
            pass

        per_window_tps: list[float] = []
        per_window_mem: list[float] = []
        chunks: list[str] = []
        gen = 0
        t_prev = None
        t_first = None

        for resp in stream_generate(model, tokenizer, actual_prompt, max_tokens=total):
            if t_prev is None:
                t_prev = time.perf_counter()
                t_first = t_prev
            chunks.append(_resp_text(resp))
            gen += 1
            if gen % window == 0:
                t_now = time.perf_counter()
                per_window_tps.append(window / max(t_now - t_prev, 1e-9))
                per_window_mem.append(_peak_memory_mb())
                t_prev = t_now
        t_last = time.perf_counter()

        full_text = "".join(chunks)
        tokens = tokenizer.encode(full_text)
        if len(tokens) >= 3:
            trigrams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
            distinct = len(set(trigrams)) / len(trigrams)
        else:
            distinct = float("nan")

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
        # you can't tell whether the model is repeating prose, repeating a
        # `<think>` reasoning preamble, or stuck on a single token.
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
        _clear_cache()


# ---------------------------------------------------------------------------
# cmd: baseline
# ---------------------------------------------------------------------------

def cmd_baseline(args) -> None:
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
        model, tokenizer, t_load = _load_timed(repo)
        print(f"loaded in {t_load:.1f}s")

        for _ in stream_generate(model, tokenizer, WARMUP_PROMPT, max_tokens=8):
            pass

        for prompt in PROMPTS:
            r = run_one(model, tokenizer, prompt, args.max_tokens, name, False)
            all_runs.append(r)
            print(
                f"  {prompt[:36]:<36} {r.decode_tps:>6.1f} tok/s "
                f"prefill {r.prefill_tps:>6.1f} t/s  peak {r.peak_mem_mb:>6.0f} MB"
            )

        if ref is not None:
            ppl = compute_perplexity(model, tokenizer, ref, args.ppl_window)
            print(f"  perplexity: {ppl:.3f}")
            ppls.append((name, ppl))

        del model, tokenizer
        gc.collect()
        _clear_cache()

    print_table(all_runs)
    if ppls:
        print("\n--- Baseline perplexity ---")
        for name, ppl in ppls:
            print(f"  {name:<20} {ppl:.3f}")


# ---------------------------------------------------------------------------
# cmd: cold-start
# ---------------------------------------------------------------------------

def _single_ttft(model, tokenizer, prompt: str) -> float:
    t0 = time.perf_counter()
    for _ in stream_generate(model, tokenizer, prompt, max_tokens=1):
        return time.perf_counter() - t0
    return time.perf_counter() - t0


def cmd_cold_start(args) -> None:
    print(f"{'label':<24} {'load s':>10} {'cold TTFT ms':>14} "
          f"{'warm TTFT ms':>14} {'delta ms':>12}")
    for label, repo in _targets_for(args):
        model, tokenizer, t_load = _load_timed(repo)
        cold_ttft = _single_ttft(model, tokenizer, WARMUP_PROMPT)
        warm_ttft = _single_ttft(model, tokenizer, WARMUP_PROMPT)
        delta = (cold_ttft - warm_ttft) * 1000
        print(
            f"{label:<24} {t_load:>10.2f} {cold_ttft * 1000:>14.1f} "
            f"{warm_ttft * 1000:>14.1f} {delta:>12.1f}"
        )
        del model, tokenizer
        gc.collect()
        _clear_cache()


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cmd",
        choices=list(COMMANDS),
        default="throughput",
        help="which benchmark to run (default: throughput)",
    )
    ap.add_argument(
        "--sizes",
        default="1.7B,4B,8B",
        help="comma-separated subset of 1.7B,4B,8B",
    )
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument(
        "--strip-thinking",
        action="store_true",
        help="(throughput) also report 'useful' tok/s after stripping reasoning preamble",
    )
    ap.add_argument(
        "--ppl-window",
        type=int,
        default=1024,
        help="(perplexity) max tokens of reference text to score",
    )
    ap.add_argument(
        "--reference-text",
        default=REFERENCE_TEXT_PATH,
        help=f"path to reference text file (default: {REFERENCE_TEXT_PATH})",
    )
    ap.add_argument(
        "--repo",
        default=None,
        help="HF repo id to benchmark instead of a Bonsai size; overrides --sizes",
    )
    ap.add_argument(
        "--label",
        default=None,
        help="display label for --repo (default: last path segment of the repo)",
    )
    ap.add_argument(
        "--chat",
        action="store_true",
        help="(decode-stability) wrap prompt in the model's chat template "
        "so instruct-tuned models respond instead of continuing the prompt",
    )
    args = ap.parse_args()

    COMMANDS[args.cmd](args)


if __name__ == "__main__":
    main()
