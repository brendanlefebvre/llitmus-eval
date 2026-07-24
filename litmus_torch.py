"""TorchBackend: the CUDA/transformers adapter for the Litmus Backend protocol.

torch / transformers are imported lazily inside method bodies. Also hosts the
torch-native `assisted` (speculative decoding) command, which has no MLX
equivalent and therefore lives outside the shared core.
"""
from __future__ import annotations

import gc
import time
from typing import Iterator

WARMUP_PROMPT = "Hello."


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


ASSISTED_PROMPTS = [
    ("essay", "Write a long, detailed essay about the history of the "
              "telescope, covering its invention, major improvements, and "
              "scientific impact."),
    ("code", "Write a complete Python implementation of an LRU cache with "
             "get/put methods, full docstrings, and a small test block."),
    ("qa", "What is the difference between a hash table and a B-tree?"),
]


def _timed_generate(model, tokenizer, prompt: str, max_tokens: int,
                    assistant_model=None) -> tuple[int, float]:
    """Run model.generate (greedy), optionally speculative via
    assistant_model. Returns (new_tokens, seconds)."""
    import torch

    ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    kwargs = dict(
        input_ids=ids,
        max_new_tokens=max_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    if assistant_model is not None:
        kwargs["assistant_model"] = assistant_model
    with torch.inference_mode():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**kwargs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return out.shape[1] - ids.shape[1], dt


def cmd_assisted(backend, args) -> None:
    """Baseline vs assisted (speculative) decode throughput.

    The decision metric is the end-to-end speedup ratio; verifier
    guarantees output equivalence under greedy decoding, so quality is
    fixed by construction. transformers does not expose acceptance-rate
    counters publicly — speedup is what we can measure honestly.
    """
    import torch

    if not args.assistant_repo:
        raise SystemExit("--assistant-repo is required for --cmd assisted")

    label = args.label or args.repo.split("/")[-1]
    print(f"\n=== {label}: assisted generation "
          f"(drafter: {args.assistant_repo}) ===")
    model, tokenizer, t_load = backend.load(args.repo, quant=args.quant)
    print(f"target loaded in {t_load:.1f}s")

    from transformers import AutoModelForCausalLM
    t0 = time.perf_counter()
    try:
        assistant = AutoModelForCausalLM.from_pretrained(
            args.assistant_repo, dtype=torch.bfloat16,
            device_map="cuda:0",
        )
    except ValueError:
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
    backend.clear_cache()
