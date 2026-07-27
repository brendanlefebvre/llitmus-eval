# Litmus

A small, local-first **LLM evaluation suite** for MLX inference on Apple Silicon.
Litmus measures the things that actually decide whether a local model is usable:

- **decode-stability** — does the model stay coherent over long generations, or collapse into loops? (chat-templated for instruct models)
- **perplexity** — held-out language-modeling quality on a fixed reference text
- **throughput** — prefill and decode tokens/sec, apples-to-apples on your hardware
- **cold-start** — load + time-to-first-token
- **prefill-scaling** — how prefill throughput behaves as context grows

Litmus is the measurement layer of the **Loxo** eval-driven router: the same numbers
that characterize a model here are what a router should use to decide *which model to
send a task to*. Measured capability in, routing decisions out.

## Status

**v0.** The harness currently targets **PrismML Bonsai 1-bit MLX models**
(`prism-ml/Bonsai-{1.7B,4B,8B}-mlx-1bit`) — that's what it was first built to
characterize (full findings in [`results-bonsai-1bit.md`](results-bonsai-1bit.md)).
Any Hugging Face MLX repo can be pointed at it with `--repo`; generalizing the model
set is the near-term direction.

## Quick start

```
python -m venv .venv-litmus && source .venv-litmus/bin/activate
pip install mlx-lm psutil
# 1-bit models need the PrismML MLX fork (stock mlx_lm lacks 1-bit g128 kernels):
pip install "mlx @ git+https://github.com/PrismML-Eng/mlx.git@prism"

# reference corpus for perplexity / prefill-scaling (public-domain Gatsby):
curl -L https://www.gutenberg.org/cache/epub/64317/pg64317.txt -o reference.txt

python litmus.py                                  # throughput (default)
python litmus.py --cmd decode-stability --chat    # coherence over a long generation
python litmus.py --cmd perplexity
python litmus.py --repo <hf-org/model-mlx>        # any MLX repo, not just Bonsai
```

`litmus_cuda.py` is the CUDA/PyTorch counterpart for rented-GPU runs (same prompts,
comparable numbers).

### Spec-check (capability evals)

Objective, fixed-input checks of tool-calling and instruction-following:

    python litmus_spec.py --profile constraints --repo <hf-org/model-mlx>
    python litmus_spec.py --profile tool-calling --repo <hf-org/model-mlx>
    python litmus_spec.py --profile chore --repo <hf-org/model-mlx>

Tool-calling reports a prompted-JSON column for every model plus a native
`tools=` column where the chat template supports it (and the gap between them).
Constraint-following reports IFEval-style strict/loose accuracy. Each run writes
a `results_<profile>_<label>.json` sidecar for downstream (Loxo) consumption.

The `chore` profile measures thread-title generation (OpenCode's title-generator
prompt). Its checks are compliance-only (length, format, forbidden prefixes).
Current cases saturate at strict=1.00 for both think and no-think modes on a
14B model, so the score does not discriminate between models — routing for
this class falls to cost, not adequacy.

## Why "Litmus"

A litmus test reveals a property with one clean check. That's the job: apply the
suite, read off whether a model is fit for purpose. (The package ships as
`llitmus-eval` — a nod to the LLM it tests.)
