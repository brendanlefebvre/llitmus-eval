"""
CUDA port of litmus.py for the A6000/H100/GH200 rental phases.

Mirrors three subcommands from the MLX harness (throughput, perplexity,
decode-stability) on an HF transformers backend so results are directly
comparable with results-bonsai-1bit.md. Same prompts, same Gatsby
reference, same 128-token windows, same distinct-trigram metric, same
head/tail dump. Greedy decoding throughout (matches mlx_lm's default).

Setup (on the rented instance — see a6000_bootstrap.sh):
    pip install torch transformers accelerate bitsandbytes hf_transfer psutil
    export HF_HUB_ENABLE_HF_TRANSFER=1
    curl -L https://www.gutenberg.org/cache/epub/64317/pg64317.txt \
         -o reference.txt

Usage:
    python litmus_cuda.py --repo unsloth/Llama-3.2-3B-Instruct \
        --quant nf4 --cmd perplexity                      # sanity gate (expect ~6.5)
    python litmus_cuda.py --repo google/gemma-4-12B-it \
        --quant bf16 --cmd decode-stability --max-tokens 1024          # raw mode
    python litmus_cuda.py --repo google/gemma-4-12B-it \
        --quant bf16 --cmd decode-stability --max-tokens 1024 --chat   # chat mode
    python litmus_cuda.py --repo cyankiwi/gemma-4-31B-it-AWQ-4bit \
        --quant prequant --cmd decode-stability --max-tokens 1024
    python litmus_cuda.py --repo google/gemma-4-12B-it \
        --quant nf4 --cmd throughput

Quant modes:
    bf16     — full-precision bfloat16 load
    nf4      — on-the-fly bitsandbytes 4-bit NF4 (downloads the bf16 weights)
    prequant — repo is already quantized (AWQ/GPTQ/bnb); load as-is
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import torch

PROMPTS = [
    "Explain quantum computing in two sentences.",
    "Write a haiku about a Mac Mini.",
    "What is the difference between a hash table and a B-tree?",
    "Summarize the plot of Hamlet in one paragraph.",
    "def fibonacci(n):",
]

WARMUP_PROMPT = "Hello."

REFERENCE_TEXT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "reference.txt"
)


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


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _peak_memory_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _reset_peak_memory() -> None:
    torch.cuda.reset_peak_memory_stats()


def _clear_cache() -> None:
    torch.cuda.empty_cache()


def _load_reference_text(path: str = REFERENCE_TEXT_PATH) -> str:
    """Load and strip a Project Gutenberg plain-text file (same as MLX harness)."""
    if not os.path.exists(path):
        raise SystemExit(
            f"reference text not found at {path}.\n"
            "Download The Great Gatsby from Project Gutenberg once:\n"
            "  curl -L https://www.gutenberg.org/cache/epub/64317/pg64317.txt "
            f"-o {path}"
        )
    with open(path, encoding="utf-8") as f:
        text = f.read()
    start = text.find("*** START OF")
    if start != -1:
        nl = text.find("\n", start)
        if nl != -1:
            text = text[nl + 1:]
    end = text.find("*** END OF")
    if end != -1:
        text = text[:end]
    return text.strip()


def _load_timed(repo: str, quant: str) -> tuple[object, object, float]:
    """Load model+tokenizer with the requested quant mode. Returns load seconds."""
    from transformers import AutoTokenizer

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(repo)

    kwargs: dict = {"device_map": "cuda:0"}
    if quant == "bf16":
        kwargs["dtype"] = torch.bfloat16
    elif quant == "fp32":
        # Numerics-discriminator mode: full fp32 weights, spilled across
        # GPU + CPU RAM via accelerate (fp32 31B-class ~126GB needs the
        # Grace side). Slow — use for single-forward measurements
        # (perplexity), not generation loops.
        kwargs["dtype"] = torch.float32
        kwargs["device_map"] = "auto"
    elif quant == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quant == "prequant":
        # AWQ/GPTQ/bnb checkpoints carry their own quantization_config;
        # from_pretrained picks it up as long as the right backend package
        # is installed (autoawq / gptqmodel / bitsandbytes).
        pass
    else:
        raise SystemExit(f"unknown --quant {quant}")

    model = _auto_load(repo, kwargs)
    model.eval()
    return model, tokenizer, time.perf_counter() - t0


def _auto_load(repo: str, kwargs: dict):
    """Try CausalLM first; fall back to multimodal classes (Gemma 4 Unified
    checkpoints register as image-text-to-text architectures)."""
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


def _encode(tokenizer, text: str) -> torch.Tensor:
    """(1, T) input ids on cuda, with the tokenizer's default special tokens
    (mlx_lm's tokenizer.encode also adds them by default)."""
    return tokenizer(text, return_tensors="pt").input_ids.to("cuda")


@torch.inference_mode()
def _greedy_stream(model, tokenizer, prompt: str, max_tokens: int):
    """Manual greedy decode loop. Yields (token_id, is_first) per generated
    token so callers can do per-token timing, mirroring mlx_lm's
    stream_generate. Stops at EOS or max_tokens.
    """
    ids = _encode(tokenizer, prompt)
    eos_ids = tokenizer.eos_token_id
    if eos_ids is None:
        eos_ids = []
    elif isinstance(eos_ids, int):
        eos_ids = [eos_ids]

    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(dim=-1)
    torch.cuda.synchronize()

    for i in range(max_tokens):
        tok = next_id.item()
        yield tok, i == 0
        if tok in eos_ids:
            return
        out = model(
            input_ids=next_id[:, None], past_key_values=past, use_cache=True
        )
        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1)
        torch.cuda.synchronize()


def _chat_wrap(tokenizer, prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as e:
        print(f"  WARNING: chat template failed ({e}); using raw prompt")
        return prompt


# ---------------------------------------------------------------------------
# cmd: throughput
# ---------------------------------------------------------------------------

def run_one(model, tokenizer, prompt: str, max_tokens: int, label: str) -> Run:
    _reset_peak_memory()
    n_prompt = _encode(tokenizer, prompt).shape[1]

    t_start = time.perf_counter()
    first_token_t: Optional[float] = None
    tok_ids: list[int] = []

    for tok, is_first in _greedy_stream(model, tokenizer, prompt, max_tokens):
        if is_first:
            first_token_t = time.perf_counter()
        tok_ids.append(tok)

    t_end = time.perf_counter()
    if first_token_t is None:
        first_token_t = t_end

    prefill_s = first_token_t - t_start
    decode_s = max(t_end - first_token_t, 1e-9)
    full_text = tokenizer.decode(tok_ids, skip_special_tokens=True)

    return Run(
        label=label,
        prompt=prompt[:40],
        prompt_tokens=n_prompt,
        gen_tokens=len(tok_ids),
        prefill_tps=(n_prompt / prefill_s) if prefill_s > 0 else 0.0,
        decode_tps=len(tok_ids) / decode_s,
        ttft_ms=prefill_s * 1000.0,
        peak_mem_mb=_peak_memory_mb(),
        sample=full_text[:80].replace("\n", " "),
    )


def cmd_throughput(args) -> None:
    label = args.label or args.repo.split("/")[-1]
    print(f"\n=== {label}: loading {args.repo} ({args.quant}) ===")
    model, tokenizer, t_load = _load_timed(args.repo, args.quant)
    print(f"loaded in {t_load:.1f}s")

    print("warmup...")
    for _ in _greedy_stream(model, tokenizer, WARMUP_PROMPT, 8):
        pass

    runs: list[Run] = []
    for prompt in PROMPTS:
        r = run_one(model, tokenizer, prompt, args.max_tokens, label)
        runs.append(r)
        print(
            f"  {prompt[:36]:<36} {r.decode_tps:>6.1f} tok/s "
            f"prefill {r.prefill_tps:>6.1f} t/s  TTFT {r.ttft_ms:>6.1f} ms  "
            f"peak {r.peak_mem_mb:>6.0f} MB"
        )

    print("\n--- Summary ---")
    avg_dec = sum(r.decode_tps for r in runs) / len(runs)
    avg_pre = sum(r.prefill_tps for r in runs) / len(runs)
    avg_ttft = sum(r.ttft_ms for r in runs) / len(runs)
    peak = max(r.peak_mem_mb for r in runs)
    print(
        f"{label:<28} decode {avg_dec:.1f} t/s  prefill {avg_pre:.1f} t/s  "
        f"TTFT {avg_ttft:.1f} ms  peak {peak:.0f} MB"
    )
    print("\n--- Sample outputs (first 80 chars) ---")
    for r in runs:
        print(f"  {r.prompt[:30]:<30} -> {r.sample}")

    del model, tokenizer
    gc.collect()
    _clear_cache()


# ---------------------------------------------------------------------------
# cmd: perplexity
# ---------------------------------------------------------------------------

@torch.inference_mode()
def compute_perplexity(model, tokenizer, text: str, window: int) -> float:
    """Teacher-forced perplexity over the first `window` tokens of `text`.
    Same definition as the MLX harness: exp(mean NLL). Lower is better."""
    ids = _encode(tokenizer, text)[:, :window]
    if ids.shape[1] < 2:
        return float("nan")
    logits = model(input_ids=ids).logits.float()  # (1, T, V)
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = ids[:, 1:, None]  # (1, T-1, 1)
    gathered = torch.gather(log_probs, -1, targets).squeeze(-1)
    mean_nll = (-gathered).mean()
    return float(torch.exp(mean_nll).item())


def cmd_perplexity(args) -> None:
    text = _load_reference_text(args.reference_text)
    print(f"loaded reference text: {len(text)} chars")
    print(f"perplexity window: {args.ppl_window} tokens\n")

    label = args.label or args.repo.split("/")[-1]
    print(f"=== {label}: loading {args.repo} ({args.quant}) ===")
    model, tokenizer, t_load = _load_timed(args.repo, args.quant)
    print(f"loaded in {t_load:.1f}s")
    ppl = compute_perplexity(model, tokenizer, text, args.ppl_window)
    print(f"  perplexity: {ppl:.3f}")

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

    label = args.label or args.repo.split("/")[-1]
    mode = "chat" if args.chat else "raw"
    print(f"\n=== {label}: decode stability over {total} tokens ({mode}, {args.quant}) ===")
    model, tokenizer, t_load = _load_timed(args.repo, args.quant)
    print(f"loaded in {t_load:.1f}s")
    _reset_peak_memory()

    actual_prompt = _chat_wrap(tokenizer, prompt) if args.chat else prompt

    for _ in _greedy_stream(model, tokenizer, WARMUP_PROMPT, 4):
        pass

    per_window_tps: list[float] = []
    per_window_mem: list[float] = []
    tok_ids: list[int] = []
    gen = 0
    t_prev = None

    for tok, _is_first in _greedy_stream(model, tokenizer, actual_prompt, total):
        if t_prev is None:
            t_prev = time.perf_counter()
        tok_ids.append(tok)
        gen += 1
        if gen % window == 0:
            t_now = time.perf_counter()
            per_window_tps.append(window / max(t_now - t_prev, 1e-9))
            per_window_mem.append(_peak_memory_mb())
            t_prev = t_now

    full_text = tokenizer.decode(tok_ids, skip_special_tokens=True)
    tokens = tokenizer.encode(full_text)
    if len(tokens) >= 3:
        trigrams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
        distinct = len(set(trigrams)) / len(trigrams)
    else:
        distinct = float("nan")

    print(f"  windows: {len(per_window_tps)} x {window} tokens each")
    print("  window tok/s: " + " ".join(f"{t:6.1f}" for t in per_window_tps))
    print("  peak MB:      " + " ".join(f"{m:6.0f}" for m in per_window_mem))
    if per_window_tps:
        first, last = per_window_tps[0], per_window_tps[-1]
        pct = (last - first) / first * 100 if first > 0 else 0
        direction = "slowdown" if pct < 0 else "speedup"
        print(f"  first->last tok/s: {pct:+.1f}% ({direction})")
    print(f"  distinct-trigram ratio: {distinct:.3f} (1.0 = no repetition)")

    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{label}_{args.quant}_{mode}")
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
# cmd: assisted (speculative decoding with a drafter / assistant model)
# ---------------------------------------------------------------------------

ASSISTED_PROMPTS = [
    ("essay", "Write a long, detailed essay about the history of the "
              "telescope, covering its invention, major improvements, and "
              "scientific impact."),
    ("code", "Write a complete Python implementation of an LRU cache with "
             "get/put methods, full docstrings, and a small test block."),
    ("qa", "What is the difference between a hash table and a B-tree?"),
]


@torch.inference_mode()
def _timed_generate(model, tokenizer, prompt: str, max_tokens: int,
                    assistant_model=None) -> tuple[int, float]:
    """Run model.generate (greedy), optionally speculative via
    assistant_model. Returns (new_tokens, seconds)."""
    ids = _encode(tokenizer, prompt)
    kwargs = dict(
        input_ids=ids,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    if assistant_model is not None:
        kwargs["assistant_model"] = assistant_model
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**kwargs)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return out.shape[1] - ids.shape[1], dt


def cmd_assisted(args) -> None:
    """Baseline vs assisted (speculative) decode throughput.

    The decision metric is the end-to-end speedup ratio; verifier
    guarantees output equivalence under greedy decoding, so quality is
    fixed by construction. transformers does not expose acceptance-rate
    counters publicly — speedup is what we can measure honestly.
    """
    if not args.assistant_repo:
        raise SystemExit("--assistant-repo is required for --cmd assisted")

    label = args.label or args.repo.split("/")[-1]
    print(f"\n=== {label}: assisted generation "
          f"(drafter: {args.assistant_repo}) ===")
    model, tokenizer, t_load = _load_timed(args.repo, args.quant)
    print(f"target loaded in {t_load:.1f}s")

    from transformers import AutoModelForCausalLM
    t0 = time.perf_counter()
    try:
        assistant = AutoModelForCausalLM.from_pretrained(
            args.assistant_repo, dtype=torch.bfloat16,
            device_map="cuda:0",
        )
    except ValueError:
        # MTP/assistant-specific archs may need their Auto class resolved
        # from the config; fall back to AutoModel-with-trust of the config's
        # declared architecture via the generic loader.
        assistant = _auto_load(args.assistant_repo,
                               {"dtype": torch.bfloat16,
                                "device_map": "cuda:0"})
    assistant.eval()
    print(f"assistant loaded in {time.perf_counter() - t0:.1f}s")

    # Warmup both paths (kernel compile, cache alloc)
    _timed_generate(model, tokenizer, WARMUP_PROMPT, 8)
    _timed_generate(model, tokenizer, WARMUP_PROMPT, 8, assistant)

    print(f"\n{'prompt':<8} {'base t/s':>10} {'assisted t/s':>14} "
          f"{'speedup':>9} {'tokens b/a':>12}")
    ratios = []
    for tag, prompt in ASSISTED_PROMPTS:
        actual = _chat_wrap(tokenizer, prompt) if args.chat else prompt
        n_base, s_base = _timed_generate(model, tokenizer, actual,
                                         args.max_tokens)
        n_asst, s_asst = _timed_generate(model, tokenizer, actual,
                                         args.max_tokens, assistant)
        tps_base = n_base / max(s_base, 1e-9)
        tps_asst = n_asst / max(s_asst, 1e-9)
        ratio = tps_asst / max(tps_base, 1e-9)
        ratios.append(ratio)
        print(f"{tag:<8} {tps_base:>10.1f} {tps_asst:>14.1f} "
              f"{ratio:>8.2f}x {n_base:>5}/{n_asst}")

    print(f"\nmean speedup: {sum(ratios)/len(ratios):.2f}x "
          f"(greedy; output token counts may differ slightly if EOS "
          f"timing shifts)")

    del model, tokenizer, assistant
    gc.collect()
    _clear_cache()


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

COMMANDS = {
    "throughput": cmd_throughput,
    "perplexity": cmd_perplexity,
    "decode-stability": cmd_decode_stability,
    "assisted": cmd_assisted,
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
        "--repo",
        required=True,
        help="HF repo id to benchmark",
    )
    ap.add_argument(
        "--label",
        default=None,
        help="display label for --repo (default: last path segment)",
    )
    ap.add_argument(
        "--quant",
        choices=["bf16", "fp32", "nf4", "prequant"],
        default="bf16",
        help="bf16 | fp32 (device_map=auto, CPU-spill; for ppl numerics checks) | "
             "nf4 (on-the-fly bnb) | prequant (repo already AWQ/GPTQ/bnb)",
    )
    ap.add_argument("--max-tokens", type=int, default=128)
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
        "--chat",
        action="store_true",
        help="(decode-stability/assisted) wrap prompt in the model's chat template",
    )
    ap.add_argument(
        "--assistant-repo",
        default=None,
        help="(assisted) HF repo id of the drafter/assistant model",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — this is the rental-phase harness.")

    COMMANDS[args.cmd](args)


if __name__ == "__main__":
    main()
