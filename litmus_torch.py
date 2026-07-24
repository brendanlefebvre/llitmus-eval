"""TorchBackend: the CUDA/transformers adapter for the Litmus Backend protocol.

torch / transformers are imported lazily inside method bodies. Also hosts the
torch-native `assisted` (speculative decoding) command, which has no MLX
equivalent and therefore lives outside the shared core.
"""
from __future__ import annotations

import time
from typing import Iterator


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
