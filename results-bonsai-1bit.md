# Bonsai 1-bit MLX: Full Benchmark Results

A complete characterization of PrismML's Bonsai 1-bit quantized models
(1.7B / 4B / 8B) on Apple Silicon, with stock 4-bit Llama 3.2 baselines
for comparison. Produced by `scripts/litmus.py`.

## Test configuration

- **Hardware:** Mac Mini M4 Pro, 24 GB unified memory
- **Framework:** `mlx_lm` on the PrismML `mlx` fork (needed for 1-bit g128 kernels)
- **Bonsai models:** `prism-ml/Bonsai-{1.7B,4B,8B}-mlx-1bit`
- **Baseline models:** `mlx-community/Llama-3.2-{1B,3B}-Instruct-4bit`
- **Reference text** (perplexity, prefill-scaling): The Great Gatsby opening,
  fetched from Project Gutenberg (public domain in the US since 2021),
  stored at `scripts/reference.txt`.

## Executive verdict

**Bonsai 1-bit is a RAM-constrained play, and little else.** At every weight
class we tested, stock 4-bit Llama 3.2 either matches or beats the
corresponding Bonsai model on quality and throughput, usually by wide
margins. Bonsai only wins on memory footprint, and only in a narrow window.

Use Bonsai when:
- You must fit inference into 1.3-1.7 GB of RAM, and a good 3B-class
  model in 1.8 GB is out of reach.
- The workload tolerates medium-quality generation (brainstorming, rough
  drafts) and does not need factual precision, code correctness, or
  structured output.

Use stock 4-bit Llama 3.2 3B (or similar) when:
- You have 1.8 GB+ of RAM available. This is the right default for
  almost every local-inference use case on Apple Silicon today.

Avoid Bonsai 1.7B entirely. It collapses into a repetition loop on long
outputs and produces degenerate text on most open-ended prompts at any
budget.

## Throughput

Default `throughput` subcommand, 5 prompts × 128 tokens, averaged.
Numbers are per-size averages.

| Model                | decode t/s | prefill t/s | TTFT (ms) | peak MB |
|----------------------|-----------:|------------:|----------:|--------:|
| Bonsai 1.7B          |     302.6  |         —   |       —   |    320  |
| Bonsai 4B            |     128.8  |         —   |       —   |    686  |
| Bonsai 8B            |     114.7  |         —   |       —   |   1321  |
| Llama 3.2 1B 4-bit   |     281.6  |      124.5  |      86.7 |    703  |
| Llama 3.2 3B 4-bit   |     119.6  |       95.0  |     105.9 |   1830  |

**Observations:**
- Bonsai matches PrismML's published M4 Pro 48 GB marketing within ~12 %
  on the 4B and 8B (8B shortfall is likely memory-bandwidth related on
  24 GB vs 48 GB).
- Bonsai 4B and Llama 3.2 3B 4-bit are throughput-tied. Bonsai 1.7B and
  Llama 3.2 1B 4-bit are throughput-tied. Bonsai does not win on speed.
- Bonsai does win decisively on RAM: 1.7B at 320 MB vs Llama 1B at 703 MB,
  8B at 1321 MB vs Llama 3B at 1830 MB.

## Perplexity

`perplexity` subcommand, `exp(mean NLL)` over the first 1024 tokens of
the reference text under teacher forcing. Lower is better. Typical
unquantized English-prose perplexity for small LLMs is 6–10.

| Model                | perplexity |
|----------------------|-----------:|
| Bonsai 1.7B          |      48.00 |
| Bonsai 4B            |      29.22 |
| Bonsai 8B            |      10.56 |
| Llama 3.2 1B 4-bit   |      20.61 |
| Llama 3.2 3B 4-bit   |       6.48 |

**Observations:**
- Bonsai 8B (10.56) is the only Bonsai size in the normal range. It is
  still **63 % worse** than Llama 3.2 3B 4-bit (6.48), despite having
  nearly 3× the parameters.
- Bonsai 4B (29.22) is dramatically worse than Llama 1B 4-bit (20.61) at
  4× the parameter count.
- Bonsai 1.7B (48.00) is effectively broken on neutral prose — 2.3× worse
  than the 1B-class baseline.
- The 4.5× spread across Bonsai sizes is steeper than the throughput
  spread, indicating a hard quality cliff below ~3B parameters under
  1-bit quantization.

## Decode stability

`decode-stability --max-tokens 1024` subcommand. One long generation per
model, measured in eight 128-token windows. The distinct-trigram ratio
measures repetition in the model's own output (1.0 = fully diverse,
0.0 = fully repeating).

| Model       | first→last t/s | slowdown % | trigram ratio | peak MB growth |
|-------------|---------------:|-----------:|--------------:|---------------:|
| Bonsai 1.7B |   308.5 → 233.8 |     −24.2 % |     **0.059** |    397 → 506   |
| Bonsai 4B   |   129.9 → 118.1 |      −9.1 % |     **0.924** |    731 → 854   |
| Bonsai 8B   |   115.8 → 108.0 |      −6.7 % |     **0.795** |   1381 → 1497  |

**Observations:**
- **The 1.7B's 0.059 trigram ratio is catastrophic.** 94 % of its trigrams
  are repeats — it collapses into a tight loop and grinds the same
  handful of tokens into a paste. This is the hard numerical boundary of
  the 1-bit quality cliff and the reason the 1.7B is unusable.
- Decode speed degrades with KV cache growth, as expected. Larger models
  degrade less as a percentage because their baseline is slower, so cache
  overhead is a smaller fraction of total work.
- **Counter-intuitive: Bonsai 4B has higher diversity than 8B** (0.924 vs
  0.795), despite 8B having much better perplexity. The 4B generates
  creative text that does not match reference distributions; the 8B
  matches distributions better but converges on common phrases more.
  Implication: 4B may be better than its perplexity number suggests for
  open-ended generation workloads (brainstorming, rough drafts).
- KV cache growth is linear and expected: ~0.12 MB per token generated
  for both the 4B and 8B.

## Prefill scaling

`prefill-scaling` subcommand. Synthetic prompts of 10, 50, 200, 500, and
1000 tokens, measured as prefill throughput and time-to-first-token.

### Bonsai 1.7B
| n tokens | prefill t/s | TTFT (ms) |
|---------:|------------:|----------:|
|       10 |       135.0 |      74.1 |
|       50 |       590.6 |      84.7 |
|      200 |      1223.5 |     163.5 |
|      500 |      1561.1 |     320.3 |
|     1000 |      1729.2 |     578.3 |

### Bonsai 4B
| n tokens | prefill t/s | TTFT (ms) |
|---------:|------------:|----------:|
|       10 |        90.6 |     110.3 |
|       50 |       349.0 |     143.3 |
|      200 |       573.3 |     348.9 |
|      500 |       688.3 |     726.4 |
|     1000 |       700.3 |    1428.0 |

### Bonsai 8B
| n tokens | prefill t/s | TTFT (ms) |
|---------:|------------:|----------:|
|       10 |        86.8 |     115.2 |
|       50 |       229.7 |     217.7 |
|      200 |       333.2 |     600.3 |
|      500 |       381.8 |    1309.6 |
|     1000 |       389.2 |    2569.1 |

**Observations:**
- **Short prompts are catastrophically inefficient across all sizes.** At
  10 tokens every model does under 135 t/s, because fixed per-call launch
  overhead dominates. The 4B and 8B are actually *slower* than the 1.7B
  at 10 tokens because their fixed overhead is a larger fraction of the
  tiny workload. **Lesson: batching short queries into longer prompts
  yields 5–20× better prefill throughput.**
- The 1.7B has not yet saturated even at 1000 tokens — it continues
  climbing past 1700 t/s. The 4B plateaus around 700 t/s. The 8B
  plateaus earliest, around 390 t/s. Larger models saturate earlier
  because each token is more compute.
- **8B TTFT at 1000 tokens is 2.57 seconds.** This is significant for
  interactive chat use. Every 100 tokens of system prompt adds ~260 ms
  to first-token latency on the 8B; on the 4B it adds ~143 ms.
- Prefill-to-decode ratios: 1.7B ~5.7×, 4B ~5.5×, 8B ~3.4×. The 8B's
  lower ratio confirms it is more memory-bandwidth-bound than
  compute-bound, consistent with the 24 GB vs 48 GB marketing shortfall.

## Cold start

`cold-start` subcommand. Model load time plus cold and warm
time-to-first-token. Weights were already downloaded and on disk.

| Model       | load (s) | cold TTFT (ms) | warm TTFT (ms) | delta (ms) |
|-------------|---------:|---------------:|---------------:|-----------:|
| Bonsai 1.7B |     0.48 |           67.6 |           88.5 |      −20.9 |
| Bonsai 4B   |     0.36 |           96.9 |           91.1 |       +5.7 |
| Bonsai 8B   |     0.39 |          107.4 |           98.1 |       +9.3 |

**Observations:**
- **There is no cold-start penalty worth measuring.** Load is sub-half-second
  across all sizes (mmap + cached weights — MLX faults pages in on demand).
  Cold-to-warm deltas are within measurement noise. The 1.7B showing cold
  faster than warm (−20.9 ms) proves this — it's jitter.
- No warmup pass is needed to get good first-response latency on this setup.
- **Zero-to-first-token from a completely cold state:**
  - Bonsai 1.7B: 0.48 s + 68 ms = **550 ms**
  - Bonsai 4B: 0.36 s + 97 ms = **460 ms**
  - Bonsai 8B: 0.39 s + 107 ms = **500 ms**
- These numbers are competitive with a typical remote-API round-trip
  (300–800 ms), and you get privacy and no network dependency. The 4B
  has the best cold-path latency of the three.

## Strip-thinking: the "thinking tax"

`throughput --strip-thinking` on the 8B, 5 prompts × 256 tokens. The
heuristic detects reasoning-preamble openers (e.g. "Okay, so the user
wants…"), optionally preceded by a parenthetical prefix (e.g.
"(150-200 words) Okay, so I need to…"), and subtracts scratchpad tokens
from the effective decode count. If preamble is detected but no `\n\n`
break to a real answer is present, the entire generation is treated as
scratchpad.

| Prompt                      | raw decode t/s | useful decode t/s |
|-----------------------------|---------------:|------------------:|
| Explain quantum computing   |          116.7 |             116.7 |
| Write a haiku               |          116.6 |             116.1 |
| Hash table vs B-tree        |          116.7 |             116.7 |
| Summarize Hamlet            |          116.1 |           **0.0** |
| `def fibonacci(n):`         |          116.2 |             116.2 |
| **Average**                 |      **116.5** |          **93.1** |

**Observations:**
- **The Hamlet prompt produced zero useful tokens at 256 tokens of budget.**
  The entire generation was the reasoning preamble
  ("(150-200 words) Okay, so I need to summarize the plot of Hamlet…"),
  which never reached an actual answer. This is the cleanest illustration
  of the failure mode: **the 8B can spend an entire typical-budget
  generation thinking about how to answer without ever answering.**
- 3 of 5 prompts (quantum, hash table, fibonacci) had no preamble at all.
  Preamble emission is not universal — roughly half of open-ended prompts
  trigger it.
- **The haiku result is a heuristic false negative.** Its first 80 chars
  are still pure preamble at 256 tokens, but a `\n\n` break somewhere in
  the generation caused the heuristic to classify only a few tokens as
  scratchpad. Without the full text we cannot see where.
- **Effective throughput on the 5-prompt mix is ~20 % lower than raw**
  (93.1 vs 116.5 t/s) once you account for wasted scratchpad tokens. This
  is the right number to use when sizing the 8B for chat workloads.
- The 1.7B and 4B do not emit a reasoning preamble and are unaffected by
  `--strip-thinking`.

## Practical decision tree

**"Should I build X on Bonsai?"**

1. **Is it a precision task?** (code generation, RAG, structured output,
   factual recall, anything where accuracy matters)
   → No. Use Llama 3.2 3B 4-bit. Its 4.5× better perplexity is decisive.
2. **Is it open-ended generation?** (brainstorming, rough drafts, creative
   text, anything where "plausible-ish text fast" is the goal)
   → **Bonsai 4B is viable.** Its high output diversity (0.924 trigram
   ratio) makes it better for this than its 29.2 perplexity number
   suggests.
3. **Is it chat with typical 128-256 token budgets?**
   → Avoid the 8B. Its ~20 % thinking tax plus 2.57 s TTFT on 1000-token
   system prompts makes it slow and unreliable. Use the 4B or Llama 3B.
4. **Are you RAM-constrained below 1.8 GB?**
   → Bonsai 8B is the best option in the 1.3–1.7 GB window. Below 1.3 GB,
   take Bonsai 4B. Below 700 MB, you are choosing between Bonsai 1.7B
   (fast but broken) and nothing — prefer nothing.
5. **Never use the 1.7B.** Its 0.059 trigram ratio means it collapses into
   a loop on long outputs, and its perplexity of 48 means it produces
   incoherent text on short ones.

## Cross-family comparison: Qwen3 / QwQ / Gemma 4 (update 2026-04-08)

Follow-up sweeps ran the same harness against several mainstream 4-bit
models on the same Mac Mini M4 Pro 24 GB. The goal was to answer two
practical questions that the original Bonsai sweep left open: "what
should you *actually* reach for on this hardware?" and "how does the
dense-vs-MoE tradeoff play out at the 24 GB ceiling?" Full per-sweep
handoff documents live alongside this file at
`scripts/qwen3_handoff.md` and `scripts/qwq_handoff.md`. Both untracked
by preference; this section is the committed summary.

### Models added

| label               | repo                                         | architecture           |
|---------------------|----------------------------------------------|------------------------|
| Qwen3-4B            | `mlx-community/Qwen3-4B-4bit`                | Dense 4B               |
| Qwen3-14B           | `mlx-community/Qwen3-14B-4bit`               | Dense 14B              |
| Qwen3-30B-A3B       | `mlx-community/Qwen3-30B-A3B-4bit`           | MoE 30B total / 3B active |
| QwQ-32B             | `mlx-community/QwQ-32B-4bit`                 | Dense 32B, reasoning-first |
| Gemma 4 31B         | `mlx-community/gemma-4-31b-it-4bit`          | Dense 31B              |
| Gemma 4 26B-A4B     | `mlx-community/gemma-4-26b-a4b-it-4bit`      | MoE 26B total / 4B active |

All runs used `litmus.py --repo ... --label ...` (harness
generic-model flag, commit `03d669a`). Decode-stability runs on
instruct-tuned models used `--chat` (commit `2b1c1c1`) to apply the
tokenizer chat template. **Without `--chat`, raw-prompt greedy-decode on
instruct-tuned models collapses into degenerate loops** — this is the
most important methodology lesson from the post-Bonsai sweeps and is
covered in more detail in the "Methodology lessons" subsection below.

### Throughput, memory, and decode-stability (chat mode)

| model                 | decode t/s | peak MB | trigram (chat) | slowdown tail |
|-----------------------|-----------:|--------:|---------------:|--------------:|
| Qwen3-4B              |       92.0 |   ~2300 |          0.894 |         —     |
| Qwen3-14B             |       29.8 |   ~8000 |          0.928 |         —     |
| **Qwen3-30B-A3B**     |   **89.7** | **~16400** |      **0.878** |         —     |
| QwQ-32B               |       13.4 |   17660 |          0.839 |         −6.6% |
| Gemma 4 31B dense     |       13.8 |   17497 |          0.870 |         −7.6% |
| **Gemma 4 26B-A4B**   |   **75.5** | **14062** |      **0.892** |         −7.6% |
| *(reference)* Llama 3.2 3B 4-bit | 119.6 |   1830 |          —     |         —     |

**Observations:**
- **Qwen3-30B-A3B is the 24 GB Mac Mini champion.** It decodes at
  near-4B-dense speed (89.7 t/s, matching Qwen3-4B within 2.5 %),
  quantized 30B total params sit in ~16.4 GB of unified memory, and
  chat-mode trigram diversity is healthy at 0.878. This is the model to
  reach for by default for quality-sensitive work on this hardware.
- **Dense ~32B 4-bit all hits the same ~13 t/s bandwidth ceiling.**
  QwQ-32B and Gemma 4 31B dense both land at 13.4–13.8 t/s. This is the
  M4 Pro memory-bandwidth floor for dense models in this size class. If
  you want faster than 13 t/s at the 32B-class scale, you need an MoE.
- **The MoE advantage is real and replicates across families.**
  Qwen3-30B-A3B decodes 3× faster than Qwen3-14B dense (89.7 vs 29.8) at
  a better perplexity. Gemma 4 26B-A4B decodes 5.2× faster than Gemma 4
  31B dense (75.5 vs 13.8) with slightly better chat-mode trigram
  diversity. The "MoE runs at active-param speed with close-to-total-
  param quality" claim holds empirically.
- **Gemma 4 26B-A4B has the smallest 30B-class footprint at 14.1 GB.**
  That's 2.3 GB less than Qwen3-30B-A3B — useful when you need to hold
  longer context, more KV cache, or run a second small model alongside.

### Perplexity on The Great Gatsby (1024-token window)

| model                 | perplexity   | notes                                  |
|-----------------------|-------------:|----------------------------------------|
| Llama 3.2 3B 4-bit    |         6.48 | sanity anchor, matches original sweep  |
| Qwen3-4B              |        19.50 | base 4B instruct                       |
| Qwen3-14B             |         3.89 | dense 14B instruct                     |
| Qwen3-30B-A3B         |         2.98 | MoE, beats 14B dense                   |
| **QwQ-32B**           |     **1.705** | **dense 32B reasoning-first, best of all** |
| Gemma 4 31B dense     |          N/A | raw-input pathway broken (measured 1408) |
| Gemma 4 26B-A4B       |          N/A | raw-input pathway broken (measured 536)  |

**Observations:**
- **QwQ-32B has the lowest measured Gatsby perplexity (1.705)**,
  contradicting the a-priori expectation that reasoning-tuning would
  inflate prose perplexity. The working conclusion is that dense active
  parameter count dominates any reasoning-tuning drift — 32B dense
  beats 3B-active MoE on raw LM capacity by a wide enough margin that
  the tuning regime is second-order. Do not cite the
  "reasoning-tuning tax on prose perplexity" framing without
  replicating the check on a new model family.
- **Gemma 4 4-bit perplexity is marked N/A, not "catastrophic" or
  "broken in the harness."** Both variants produce perplexity numbers
  (1408 and 536) that are ~30–200× worse than any other model we've
  measured. The harness was sanity-checked on Llama 3.2 3B during the
  Gemma sweep and returned 6.484, matching the original Bonsai-sweep
  measurement of 6.48 to three decimals. The harness is fine. What
  this means is covered in "The Gemma 4 4-bit finding" below.

### Decision tree for the 24 GB Mac Mini

**"Which model should I reach for today?"**

1. **Quality-sensitive work, interactive latency matters**
   → **Qwen3-30B-A3B MoE.** 89.7 t/s decode, 2.98 Gatsby perplexity,
   dual-mode (raw + chat), 16.4 GB peak. The default.
2. **Quality-sensitive work, context/KV-cache headroom matters more
   than peak speed**
   → **Gemma 4 26B-A4B MoE.** 75.5 t/s decode, 14.1 GB peak (2.3 GB
   more headroom than Qwen3-30B-A3B), 0.892 trigram diversity.
   **Chat-template-only** — see the Gemma section below.
3. **Long-horizon analytical writing, always-on reasoning, ~13 t/s
   acceptable**
   → **QwQ-32B.** Lowest measured perplexity (1.705), dense 32B raw
   capacity, reasoning preserved in output. Not interactive at 13.4
   t/s but worth the wait for hard problems.
4. **Low-latency short-context chat, moderate quality**
   → **Qwen3-4B dense.** 92.0 t/s, 2.3 GB peak, trigram 0.894. Almost
   as fast as Bonsai 4B at dramatically better quality.
5. **Memory under 4 GB**
   → Qwen3-4B if 2.3 GB fits; otherwise the Bonsai decision tree above
   (1.3–1.7 GB window).
6. **Memory under 700 MB**
   → Nothing on this list qualifies. Bonsai 1.7B is the only option and
   it's broken. Don't.
7. **Memory under 320 MB**
   → Bonsai 1.7B (broken). You are better off not running inference.

### The Gemma 4 4-bit finding

All three measurement pathways for both Gemma 4 4-bit variants tell the
same asymmetric story:

| pathway                   | 31B dense              | 26B-A4B MoE              |
|---------------------------|------------------------|--------------------------|
| raw throughput generation | `額額額`, `STTTTT`, `(n):(n):` | `"The Mac Mini is a great tool."×N`, `(n) (n) (n)`, `"The target is to be in one paragraph."×N` |
| raw Gatsby perplexity     | 1408                   | 536                      |
| chat-templated generation | clean, 0.870 trigram   | clean, 0.892 trigram     |

Both variants produce catastrophic output on raw prompts, catastrophic
perplexity on raw Gatsby, and **clean, coherent chat-templated output
with healthy trigram diversity.** No other model in this sweep shows
this asymmetry. Qwen3, QwQ, Llama, and Bonsai all produce sensible
outputs and sensible perplexities on raw input.

**Working hypothesis:** 4-bit quantization preserves Gemma 4's chat
pathway (which is what RLHF optimized hardest) at the cost of the
base-LM raw-continuation pathway. The MoE fails into real English
phrase repetition while the dense fails into single-token loops —
possibly because MoE routing can dodge to less-quantization-damaged
experts on any given token. Speculative on the mechanism; the
characterization itself is solid.

**Operational rule: Gemma 4 4-bit is chat-template-only.** There is no
fallback to raw-mode inference. If you deploy Gemma 4 4-bit anywhere,
the prompt must go through `tokenizer.apply_chat_template(...,
add_generation_prompt=True)` before reaching the model. This is
enforced by `litmus.py --chat` on the decode-stability
subcommand and is what produced the clean results in the table above.

### Methodology lessons

The post-Bonsai sweeps surfaced four lessons that will carry forward
into any future model evaluation on this hardware. They're recorded
here so the next sweep doesn't have to rediscover them:

1. **Always pass `--chat` on decode-stability for instruct-tuned
   models.** Raw-prompt free-generation tests on instruct-tuned models
   are diagnostically meaningless: small dense models fail loudly with
   truncated meta-thinking, large dense models partially escape, MoEs
   collapse silently into single-phrase loops under greedy sampling.
   Without the chat template, the harness rank-orders models by their
   ability to recover from a fundamentally unfair test.
   Externally validated by mlx-lm maintainer Angelos Katharopoulos in
   `ml-explore/mlx-lm#1123`, which independently delivered the same
   diagnosis for Gemma 4 on the same day the Qwen3 sweep rediscovered
   it.
2. **Dense active-parameter count dominates tuning regime for raw-
   prose perplexity.** The a-priori expectation that reasoning-tuning
   would inflate prose perplexity (QwQ vs Qwen3) failed to replicate:
   QwQ-32B scored 1.705 on Gatsby, dramatically *better* than
   Qwen3-30B-A3B's 2.98. Do not cite a "reasoning-tuning tax on prose"
   without replicating the check.
3. **4-bit quantization can break the raw-input code path while
   leaving the chat pathway intact.** Gemma 4 is the first model in
   this sweep where the chat-template fix alone was not a complete
   characterization. Always triangulate across throughput, perplexity,
   and decode-stability — any single subcommand would have missed the
   Gemma 4 asymmetry.
4. **Sanity-check the harness with a known-good model before
   declaring an anomaly real.** Llama 3.2 3B-Instruct 4-bit returning
   6.484 perplexity (matching the original 6.48 to three decimals) is
   what pivoted the Gemma 4 investigation from "harness regression" to
   "real characterization finding." Build this sanity check into the
   workflow for any anomaly larger than ~2× the nearest reference.

## Caveats and heuristic notes

- **Perplexity window is 1024 tokens** of a single neutral-prose reference.
  A domain-specific perplexity (code, math, dialogue) could look quite
  different, especially for instruction-tuned models. These numbers are
  the "English-prose floor."
- **Decode-stability prompt is a single long-form essay prompt.** Other
  prompt shapes (e.g. structured JSON generation) might degrade
  differently.
- **Strip-thinking is a heuristic.** It matches preamble openers at the
  start of generation (after lstrip and an optional parenthetical) and
  splits on the first `\n\n`. It will miss cases where the preamble uses
  a different structural marker or a different opener, and it may
  false-positive on prose that legitimately begins with "Okay,". The
  haiku result here is a known false negative.
- **Baseline is stock mlx-community 4-bit Llama 3.2.** Other quantization
  schemes (e.g. AWQ, 8-bit, 3-bit) were not tested. The comparison point
  is "the default quantized model you would reach for on Apple Silicon
  today."
- **All runs are single-process, single-model.** Memory numbers are peak
  GPU memory as reported by MLX and do not include Python-process
  overhead.

## Reproducing

```bash
cd scripts/
# one-time: fetch the reference text (public domain)
curl -L https://www.gutenberg.org/cache/epub/64317/pg64317.txt \
     -o reference.txt
# run the full sweep
python3 litmus.py --cmd throughput
python3 litmus.py --cmd throughput --strip-thinking --sizes 8B --max-tokens 256
python3 litmus.py --cmd perplexity
python3 litmus.py --cmd prefill-scaling
python3 litmus.py --cmd decode-stability --max-tokens 1024
python3 litmus.py --cmd baseline
python3 litmus.py --cmd cold-start
```

All runs completed in under ~30 minutes total on a Mac Mini M4 Pro 24 GB.

---

## Addendum 2026-06-04/05: Gemma 4 12B Unified vs Qwen3.6-35B-A3B — one-shot coding accuracy

**Motivation:** Reddit anecdote claimed Gemma 4 12B Unified one-shots coding tasks better than Qwen3.6-35B-A3B despite the latter's decode-speed advantage. Tested directly on the local stack.

### Test setup

Both models served via mlx_vlm at :7979 on M4 Pro 24GB. Identical prompts to both, single-shot, no iteration. Output scored on:
- Did the code run as-shipped without manual edits?
- How many edits would be required to make it work?
- Output completeness and correctness

Tests run direct via curl (litmus can't load Gemma 4 Unified yet — mlx_lm 0.31.3 lacks `gemma4_unified` support).

### Test 1: HTML Snake game (single-file, embedded CSS+JS)

| Model | Output size | Edits to ship | Notable bugs |
|-------|-------------|---------------|--------------|
| `gemma-4-12B-it-6bit` | 6 KB | 1 (strip markdown fences) | Clean implementation. Snake grows correctly on food, collision detection works, restart works. |
| `Qwen3.6-35B-A3B-4bit` | 9 KB | 1-4 (fences + acknowledged bug patch + visible artifact) | `moveSnake()` pops tail unconditionally. Model literally wrote a meta-comment in the output code acknowledging this bug and adding a compensation hack in `checkFood()` — `snake.push(snake[snake.length-1])`. Functions but with visible stacking artifact on eat frames. |

### Test 2: Animated TUI Tower of Hanoi (Python, standard library only)

| Model | Wall-clock | Output size | Edits to ship | Notable bugs |
|-------|-----------|-------------|---------------|--------------|
| `gemma-4-12B-it-6bit` | 77 sec | 4 KB | 2 (fences + one constant) | Class-based, recursive solve correct, terminal-width-aware centering, KeyboardInterrupt cleanup. **One real bug:** `total_moves = (3**num_disks - 1)` — Tower of Hanoi is 2^n - 1, not 3^n - 1. Animation runs correctly; only the displayed expected-total is wrong. |
| `Qwen3.6-35B-A3B-4bit` | 48 sec | 2 KB | 3-5 (won't run as-shipped) | **Catastrophic NameError:** `move_disk()` calls `render(pegss, ...)` but `pegss` is undefined in that scope (it's a local variable in `main()`). Code crashes on first move. Additionally: `total_moves` uses current peg-A length not initial count, and the render loop puts largest disks at top of peg (inverted from standard display). |

### Finding

**Gemma 4 12B Unified at 6-bit consistently produces more correct first-pass code than Qwen3.6-35B-A3B at 4-bit on these tasks.** Pattern holds across both tests. Qwen3.6's decode-speed advantage (~4.4x faster) is erased — and then some — by the time spent fixing its broken output. **End-to-end time-to-working-code favors Gemma despite slower decode.**

Reddit user's anecdote validated for this stack.

### Implications

- Daily-driver model choice is not as decisively Qwen3.6 as the decode-speed-only A/B implied.
- For coding-heavy workflows where one-shot accuracy matters, Gemma 4 12B Unified at 6-bit is a credible alternative. Worth more tests across other problem types (CLI tools with arg parsing, Flask handlers, etc.) before committing.
- Qwen3.6 likely still wins for: long-running generation where edit count is amortized; non-code text generation; high-throughput batch use.
- The 4-bit quantization quality gap may explain part of this (Reddit user reported same Q4→Q5 quality jump on syntax accuracy). QAT-trained Q4 variants from Google (Unsloth has GGUF; no MLX conversion yet) might close the gap if/when available in MLX format.

---

## Addendum 2026-06-06: CUDA cross-validation — quantization attribution RETRACTED, finding upgraded

**Motivation:** The "Gemma 4 4-bit finding" above attributed the raw-input
pathway collapse to 4-bit quantization ("quantization preserves the chat
pathway at the cost of the base-LM raw-continuation pathway"). That
hypothesis was formed without an unquantized control — bf16 Gemma 4 does
not fit on a 24GB Mac. Rented a Lambda 1x A100-SXM4-40GB ($1.99/hr,
~2 hours total) and ran `scripts/litmus_cuda.py`, a transformers-
backend port of the three relevant subcommands (same prompts, same Gatsby
window, same 128-token windows and distinct-trigram metric, greedy
decoding to match mlx_lm).

### Harness sanity gate

`unsloth/Llama-3.2-3B-Instruct` @ bnb-nf4: raw Gatsby perplexity **6.191**
vs the MLX sweep's 6.484 (different backend, different 4-bit scheme,
within 5%), and clean raw-mode prose continuation. Harness validated.

### Results matrix

| model | precision/scheme | raw decode-stability | raw Gatsby ppl | chat decode-stability |
|---|---|---|---|---|
| gemma-4-12B-it | **bf16 (unquantized)** | DEAD — "1"-loop, trigram 0.002 | **8847.6** | clean, 0.971 |
| gemma-4-12B-it | bnb-nf4 | DEAD — "."-loop, EOS ~45 tok | — | clean |
| gemma-4-12B-it | AWQ-INT4 (QDQ¹) | DEAD — "$"-loop, 0.027, EOS ~150 tok | 7317.5 | clean, 0.946 |
| gemma-4-31B-it | unsloth bnb-4bit | DEAD — "CTCT"-loop, 0.003 | 3488.3 | clean, 0.968 |
| gemma-4-26B-A4B-it | (unreachable²) | — | — | — |

¹ transformers loads compressed-tensors AWQ checkpoints by decompressing
to bf16 (12B AWQ peak memory = 24.4GB ≈ the bf16 run). The 4-bit rounding
damage is baked into the weights, so this is a valid quantize-dequantize
evaluation, executed in bf16 arithmetic.

² Both routes OOM on 40GB: the AWQ checkpoint decompresses to ~52GB, and
bitsandbytes on-the-fly nf4 cannot quantize the MoE's fused expert
tensors (only nn.Linear), leaving ~52GB of experts in bf16. The MoE data
point needs a 96GB-class GPU (GH200 at $2.29/hr), where it can be run
unquantized — folded into the planned GH200 session.

### Finding (supersedes "The Gemma 4 4-bit finding" hypothesis above)

**The raw-input pathway is dead in the unquantized bf16 model.** Same
weights, same harness, same greedy decoding: raw-mode collapses to a
single-token loop with catastrophic perplexity (8847 vs 6.2 for a 3B
Llama on identical text/hardware), while the chat-templated pathway
produces clean prose (trigram 0.97). Quantization was never the culprit
— the prior attribution is **retracted**. "Chat-template-only" is a
property of the Gemma 4 instruct models themselves, presumably the
instruction tuning overwriting raw-continuation behavior entirely.

The evidence stack for the upgraded finding: 2 backends (MLX, CUDA), 3
quant schemes (MLX 4-bit, AWQ QDQ, bnb-nf4), 2 independent quantizers
(mlx-community, unsloth/cyankiwi), plus the bf16 control. Failure is
invariant; only the attractor token varies by variant ("1", ".", "$",
"CTCT" — and on MLX: 額, ST, English phrases). Chat-mode output is
near-bit-faithful across precisions under greedy decoding (bf16 and AWQ
12B produced essays with identical titles, diverging only mid-body).

**Open question carried to the GH200 session:** the MLX sweep observed
the MoE failing into *coherent English phrase loops* while dense models
fail into token stutters — hypothesized as expert routing dodging
quantization damage. With quantization exonerated, does the MoE still
fail distinctively at bf16? (Also pending there: gemma4_assistant MTP
drafter acceptance-rate measurements — transformers 5.10.2 registers the
arch natively.)

### Methodology lesson #5 (extends the four above)

**An anomaly characterization without an unquantized control is an
attribution waiting to be retracted.** The original finding correctly
characterized the asymmetry but mis-attributed its cause; the control
run that settled it cost ~$0.70 of rented GPU time. When a finding's
mechanism story can't be tested on local hardware, rent the control
before publishing the mechanism.

### Operational notes (rental workflow, for next time)

- Lambda Stack 22.04 image warts, all of the half-pip/half-system kind:
  pillow <9.1, stale jinja2 (Gemma 4 chat template needs newer), system
  torchvision built against system torch (breaks gemma4_unified import),
  missing compressed-tensors. All pinned in `scripts/a6000_bootstrap.sh`.
- A100 40GB rents when A6000s are dry; the bf16 12B control needs >24GB
  either way.
- HF Xet transfer on Lambda's peering: >1 GB/s bursts; 52GB MoE download
  ≈ 1 minute. Cattle-not-pets confirmed economical.
- Total phase cost: ≈ $4 of a $150 credit.

---

## Addendum 2026-06-07: Nemotron 3 Nano 30B-A3B — first MLX 4-bit quant + head-to-head vs Qwen3.6-35B-A3B

**Motivation:** NVIDIA's Nemotron 3 family (hybrid Mamba-2 + attention +
MoE, permissive OpenMDW-1.1 license) had no published MLX quant of the
Nano 30B-A3B — the one size that fits consumer Macs. mlx-community
covered Super (120B) and Ultra (550B) but skipped Nano; the only Nano
MLX artifact on HF was an experimental 2-bit of the multimodal Omni
variant. Converted it ourselves and benchmarked against the daily
driver. Plan: `scripts/nemotron3_nano_mlx_plan.md`.

### Conversion

`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (61GB) →
`mlx_lm.convert -q --q-bits 4 --q-group-size 64` → 17GB, 4 shards.
Conversion took ~2 minutes on the M4 Pro (read from external SSD, wrote
internal). `model_type: nemotron_h`, supported by released mlx-lm
out of the box (same arch path as the working mlx-community Super
conversion). 52 layers (23 Mamba-2/MoE + 6 attention per the family
paper), 128 experts, 6 active.

### Results (M4 Pro 24GB, same harness/metrics as all prior sweeps)

| metric | Qwen3.6-35B-A3B-4bit | Nemotron3-Nano-30B-A3B-4bit | verdict |
|---|---|---|---|
| decode t/s (sustained, 1024 tok) | 92.0 | ~95 (96.6 peak smoke) | Nano, by a nose |
| Gatsby perplexity (raw, 1024 win) | 1.445 | 2.984 | **Qwen3.6, decisively** |
| chat trigram diversity | 0.913 | 0.962 | Nano |
| raw-mode decode-stability | robust | robust (0.922) | tie |
| peak memory | 18.7 GB | **17.1 GB, FLAT across windows** | Nano |
| prefill @1K tokens | ~686 t/s | 619 t/s | Qwen3.6 |
| TTFT @1K | 1.46 s | 1.61 s | Qwen3.6 |
| load time | 23.3s cold / fast warm | 4.3-4.7s (warm) | n/c |

### Observations

1. **The Mamba memory signature is real and visible.** Per-window peak
   memory was *flat* (17130/17077 MB constant across all 8 windows) in
   both chat and raw decode-stability — every transformer MoE in prior
   sweeps grew window-over-window as KV accumulated. Only 6 of 52
   layers carry KV here; the Mamba layers hold constant-size recurrent
   state. This is the property that compounds at long context.
2. **Raw mode is robust — the Gemma 4 counter-example, same weekend.**
   One day after the 2026-06-06 retraction established chat-template-
   only as a Gemma 4 model property, Nemotron 3 Nano (also a heavily
   instruct/reasoning-tuned 2026 model) handled bare prompts fine
   (0.922 trigram, coherent structured essay). Instruct tuning *can*
   kill raw continuation; it demonstrably doesn't have to. Gemma's
   pathology is a training choice, not an industry trend.
3. **Prose perplexity lands exactly on the old Qwen3-30B-A3B number**
   (2.984 vs 2.98) — same total/active class, same Gatsby score, almost
   to the decimal. Qwen3.6 broke through that apparent size-class
   ceiling (1.445); Nemotron didn't. Caveat from the 2026-06-04 coding
   addendum: Gatsby prose ppl does not predict one-shot code quality —
   snake/hanoi one-shots are the fair next test before any daily-driver
   verdict. Nemotron's training mix is reasoning/agent-heavy.
4. **Thinking mode on by default** via chat template (`</think>`-closed
   reasoning preamble; ships `nano_v3_reasoning_parser.py`). At small
   token budgets the model spends the whole allowance thinking — the
   `--strip-thinking` heuristic misfires in both directions on it at
   `--max-tokens 128` (haiku row scored useful 0.0 t/s after hitting
   the cap mid-think; "The answer..." opener dodged the preamble regex
   and overstated). Judge useful throughput at ≥512-token budgets.
5. Decode at ~95 t/s with -2% drift over 1024 tokens; prefill plateaus
   ~620 t/s @1K (vs Qwen3.6's ~686) — the recurrent layers' tradeoff:
   slightly slower prefill, flat memory at decode.

### Verdict

**Qwen3.6-35B-A3B keeps the daily-driver slot** on raw LM quality.
Nemotron 3 Nano immediately earns two specialist slots: long-context
work (flat memory curve) and a reasoning-mode alternative, pending the
coding one-shots. It's also now published — first public MLX 4-bit of
the model (see below).

**Toolchain notes for the next conversion:** huggingface_hub 1.x ships
the `hf` CLI in the base package ([cli] extra is gone); zsh's command
index doesn't see binaries installed mid-session into already-indexed
PATH dirs (`rehash`); mlx_lm.convert CLI quant defaults are `None`
falling through to mlx core's 4/64 — pin `--q-bits/--q-group-size`
explicitly for published artifacts.

---

## Addendum 2026-06-07 (evening): GH200 session — failure taxonomy completed, drafter speedups, 122B taste test

Lambda 1x GH200 (96GB HBM3, Grace/aarch64, $2.29/hr, ~2.5 hours). ARM
was a non-event: the full bootstrap, bitsandbytes included, worked
unmodified. Raw outputs: `scripts/gh200_results_2026-06-07/`. Plan:
`scripts/gh200_phase_plan.md`. Sanity gate: Llama-3.2-3B nf4 Gatsby ppl
6.197 (vs 6.191 A100, 6.484 MLX — third backend, same harness truth).

### 1-2. The Gemma 4 raw-pathway question, fully mapped

bf16 unquantized runs on the two remaining models:

| model | raw generation | raw ppl | chat |
|---|---|---|---|
| 26B-A4B MoE bf16 | **instant EOS** (1 newline, then stop) | **457** | clean 0.964 |
| 31B dense bf16 | "la la la la…" → LlLl token loop, 0.011 | 3238 | clean 0.964 |
| (12B Unified bf16, rerun) | "_____"→"1"-loop, 0.007 | 9231 | clean 0.956 |

**Final taxonomy:**
1. **Deadness is precision-independent.** Raw ppl barely moves between
   4-bit and bf16 (26B: 457 bf16 vs 536 MLX-4bit; 31B: 3238 bf16 vs
   3488 bnb-4bit).
2. **Failure expression is architecture-determined.** Dense models loop
   at every precision — only the attractor token changes ("la la"/LlLl
   at bf16, CTCT at bnb, 額 at MLX-4bit). The MoE at bf16 simply emits
   EOS immediately; quantization noise perturbs that collapsed
   distribution into the English-phrase loops the MLX sweep observed.
   The "expert routing dodges damage" hypothesis dies in its original
   form but survives in a weaker one:
3. **Architecture sets degradation depth.** MoE raw ppl ~460-540; dense
   31B ~1400-3500; dense 12B *Unified* (multimodal) ~7300-9200 — the
   raw pathway is least damaged in the MoE and most damaged in the
   multimodal Unified arch, consistently across precisions/backends.

Bonus specimen: the broken cyankiwi 26B AWQ load (per-expert
compressed-tensors keys vs fused-expert arch = experts randomly
initialized — transformers prints UNEXPECTED/MISSING for every expert
tensor) generates distinctive multilingual word-salad
(`gh200_results .../gemma-4-26B-A4B-it-AWQ-4bit_prequant_raw.txt`) —
useful reference for "what random experts actually look like," and a
warning: that checkpoint is incompatible with transformers ≥5.10's
fused-expert Gemma 4 implementation regardless of VRAM.

### 3. Nemotron Nano bf16 reference — a cross-implementation surprise

bf16 Gatsby ppl on CUDA/transformers: **5.673**. fp32 (CPU-spilled,
same harness): **5.773** — dtype exonerated. The MLX-4bit measurement
was 2.984. Since the Llama sanity model agrees across backends within
5%, the ~2x gap is **implementation-level divergence between
transformers' and mlx-lm's `nemotron_h`** (likely in hybrid-Mamba
teacher-forced scoring), not quantization, not dtype. Cross-backend ppl
for this arch is non-comparable; the HF model card now needs the
"within-backend comparisons only" footnote. Possibly worth an upstream
issue. (Chat decode-stability bf16: clean 0.948, peak memory FLAT at
61GB — the Mamba signature confirmed at full precision.)

### 4. Gemma 4 MTP drafter speedups (transformers assisted generation)

| target | essay | code | qa | mean |
|---|---|---|---|---|
| 12B-it + 12B drafter | 1.09x | 1.59x | 1.19x | **1.29x** |
| 26B-A4B + 26B drafter | 1.61x | 2.07x | 1.61x | **1.76x** |

Spread is the classic acceptance-rate signature (predictable code ≫
free prose). The MoE gains MORE — likely because eager-mode MoE pays
heavy per-forward overhead (routing/gathers/launches) that speculative
verification amortizes across drafted tokens; i.e. the number partly
flatters the framework. The model card's "up to 3x" needs a tuned
stack; stock transformers delivers 1.3-1.8x mean, 2x on code. Caveat:
default candidate-length heuristic, untuned.

### 5. Qwen3.5-122B-A10B Q4_K_M taste test (llama.cpp, GGUF)

llama-bench: **pp512 2421 t/s, tg128 121.6 t/s** — a 122B decoding
faster than the daily-driver 35B does on the Mac (92). One-shot coding
(reconstructed snake/hanoi prompts, now on record in this repo):

| task | Gemma-12B-6bit (6/04) | Qwen3.6-35B-4bit (6/04) | Qwen3.5-122B-Q4 (today) |
|---|---|---|---|
| HTML snake | 1 edit | 1-4 edits | **0 edits** (static review) |
| TUI hanoi | 2 edits (3^n bug) | 3-5, won't run | **0 edits** (static review) |

All three 6/04 failure modes individually checked and absent: 2^n−1
correct, render orientation correct, no scope NameError. Conditional
tail-pop correct (Qwen3.6's snake bug). Pending: runtime confirmation
on the Mac; cosmetic nit candidate: hardcoded hanoi label centering.

**Hardware translation:** at Q4, A10B active ≈ 6GB reads/token.
Bandwidth-scaled: GB10-class (273GB/s) → ~10-20 t/s for this exact
model; M5-Ultra-class (~1.1TB/s) → ~30-40. The "nobody trains for this
box" complaint is now doubly answered: Nemotron 3 ladder + Qwen3.5
122B-A10B are architecture-matched to capacity-rich/bandwidth-poor
hardware, and the 122B is the first model to post 0-edit one-shots in
this table. The box class runs the models worth wanting, at the speed
of patience.

### Session ops notes

- aarch64: full bootstrap incl. bitsandbytes worked unmodified; nvcc
  deprecation warnings for pre-sm_75 targets are noise; llama.cpp
  CUDA build on 72 Grace cores ~5 min.
- transformers' Mamba "fast path" kernels (mamba-ssm/causal-conv1d)
  absent → naive fallback: speed-only effect, reference math.
- Two parallel `hf download` queues contend politely on file locks —
  `tail -F` (not `-f`) when watching a log that doesn't exist yet.
- Session cost ≈ $6; running credit spend ≈ $11 of $150.

---

## Addendum 2026-06-08 (afternoon): B300 / NVFP4 — the bleeding-edge tax, and a confounded headliner

RunPod 1x **B300** (Blackwell Ultra, ~268 GiB usable, ~$7.39/hr). Goal:
NVFP4-vs-bf16 122B format shootout (`scripts/b300_phase_plan.md`).
**Outcome: partial — NVFP4 quality result is clean; the speed number is
confounded; bf16 leg cut for budget.** Net spend ~$12.40 (of a $20
RunPod preload), the great majority on first-run friction, not compute.
Raw outputs: `scripts/b300_results_2026-06-08/`.

### The result that's actually clean: NVFP4 preserves coding quality

`nvidia/Qwen3.5-122B-A10B-NVFP4` (ModelOpt FP4), served on vLLM. One-shot
coding, same prompts as the 6/04 + GH200 table:

| task | NVFP4-122B (B300) | GGUF-Q4-122B (GH200) | notes |
|---|---|---|---|
| HTML snake | ~1 edit (fence+preamble) | 0 edit | NVFP4 has the reversal guards Qwen3.6 lacked |
| TUI hanoi | ~1 edit (fence+preamble) | 0 edit | NVFP4 has correct `2**n-1` (Gemma-12B's bug avoided) |

**NVFP4-quantized 122B holds one-shot coding quality** — same tier as the
GGUF-Q4 of the same model, dodging both prior models' signature bugs.
4-bit NVFP4 does not degrade the coding pathway. (This model "thinks out
loud" in a numbered-analysis preamble before the code, not in `<think>`
tags — strip it + the markdown fence to ship.)

### The speed number, with its asterisks

Single-stream decode (prefill-isolated, vLLM `/v1/completions`):
**84.0 ± 1.5 tok/s.** Lower than the GH200 GGUF's 121.6 — but the
comparison is **confounded on two axes and not a verdict on NVFP4**:
1. **Runtime:** vLLM (NVFP4) vs llama.cpp (GGUF). vLLM optimizes batched
   throughput; at batch=1 it underperforms llama.cpp's single-stream
   tuning. NVFP4's FP4-tensor-core advantage shows in *batched* serving
   (the regime of the "8.2x perf/dollar" claims), which this single-
   stream test does not measure.
2. **Hardware:** B300 (Blackwell) vs GH200 (Hopper).
So 84 t/s is "vLLM-NVFP4 single-stream on B300," honest but narrow; it is
NOT "NVFP4 is slower than GGUF." A batched-throughput test would be the
fair NVFP4 measurement; budget ran out first. Logged as a gap.

### The real lesson: the bleeding-edge first-run tax is the cost driver

The B300 cost ~3-4x a GH200 run for *less* data, and almost none of it
was compute:
- **CUDA toolkit too old.** B300 is compute capability **sm_103a**
  (Blackwell Ultra, newer than B200's sm_100). The common cu12.8 images
  fail FP4-kernel JIT: `nvcc fatal: Unsupported gpu architecture
  'compute_103a'`. **Needs CUDA >= 12.9**; used a cu13.0/torch2.9
  community template. (Doc's S2 now gates on `nvcc --version`.)
- **Uncached FP4 CUTLASS compile.** First serve JIT-compiles FlashInfer's
  FP4 kernels: multiple cicc->ptxas cycles, each single-threaded (172
  cores idle), ptxas alone ran 6+ min on one kernel. Minutes of $7.39/hr
  to watch a compiler. Caches after — but the first run pays it.
- **Region roulette:** drew a **Croatia (HR)** host; HF's Xet CAS
  endpoint was unreachable (<1MB/s) → `HF_HUB_DISABLE_XET=1` forced the
  regular CDN (~100MB/s).
- **Community-image gotchas:** hardcoded root password (`runpod`);
  auto-started JupyterLab; a killed-worker GPU-memory leak (~36 GiB) that
  `pkill -f vllm` missed and a blanket `pkill -f python` over-killed
  (dropped SSH) — sidestepped by lowering `--gpu-memory-utilization` to
  fit the free memory rather than chasing the ghost.

**Takeaway for future rentals:** for "just get the number" work, prefer
mature/available hardware (H200/GH200) — cost-predictable, kernels
cached, toolchains ready. Reserve the B300 for when you specifically
need *it* (288GB, or a genuine batched-NVFP4 throughput study where FP4
earns its tax). Bleeding-edge silicon bills you to be the one who
compiles the kernels first. See `feedback_rental_compute_discipline.md`.

---

## Addendum 2026-06-17: LFM2.5-8B-A1B — Liquid AI's hybrid conv+attention edge MoE

**Motivation:** First architecture in the sweep that is *not* attention-dominant.
Despite Liquid AI's continuous-time-net heritage, LFM2 is a hybrid of gated
short-**convolution** blocks plus a small number of grouped-query-attention (GQA)
blocks, with the mix chosen by hardware-in-the-loop architecture search and
co-designed for on-device latency/peak-memory. The 8B-A1B is the MoE variant:
24 layers (18 double-gated LIV short-conv + 6 GQA), 32 experts, top-4 routing,
~1B active of 8.3B total. Question: does a conv-dominant stack behave differently
from the attention-heavy MoEs (Qwen3, Gemma 4, Nemotron) — especially on prefill?

### Setup

`LiquidAI/LFM2.5-8B-A1B-MLX-8bit` (official Liquid MLX build; mlx-community also
ships 3/6/8-bit). **8-bit only tested.** Requires a current mlx-lm for the `lfm2`
model class.

**Install gotcha (cost ~15 min):** a Homebrew `libmlx 0.31.1` shadowed the pip
`mlx 0.31.2` wheel, so `import mlx.core` threw
`ImportError: dlopen ... Symbol not found: ...as_strided...`. The pip
`mlx`/`mlx-lm` pair (0.31.2 / 0.31.3) was fine — they version independently; the
stale third copy from Homebrew was the culprit. Fix: `brew upgrade mlx` to align
the versions (or decouple the venv from Homebrew's lib).

### Results (M4 Pro 24 GB, same harness/metrics as all prior sweeps)

| metric | LFM2.5-8B-A1B-8bit | notes |
|---|---|---|
| decode t/s (raw, `--chat`, 1024 win) | ~112 | fast-for-an-8B = the ~1B-active MoE payoff |
| decode drift (1024 / 4096 win) | −2.0% / −4.7% | gentle linear KV decay, no degeneration |
| trigram diversity (1024 / 4096) | 0.924 / 0.913 | healthy, no repetition loop |
| useful t/s (post-`</think>`, 4096 budget) | 89.3 | see thinking-tax below |
| Gatsby perplexity (raw, 1024 win) | 87.0 | anomalous — raw-input pathway, see below |
| peak memory | 8.6 GB, ~flat | ~15 GB headroom on the 24 GB box |
| prefill @1K tokens | 1463.5 t/s | still rising, no O(n²) wall |
| TTFT @1K | 683.3 ms | |
| cold start | 6.65 s load; 499.6 ms cold TTFT → 59.1 ms warm | 440 ms one-time JIT warmup |

### The fixed-overhead thinking tax (the headline finding)

LFM2.5 is a reasoning model — every response opens with a `<think>` block. The
novel result vs. prior reasoners: **the think block is a fixed ~775 tokens
regardless of token budget**, not proportional to it.

| max-tokens | scratchpad tok | useful tok | useful % | useful t/s | answer state |
|---:|---:|---:|---:|---:|---|
| 1024 | 775 | 248 | 24% | 27.2 | truncated mid-sentence |
| 4096 | 775 | 3278 | 81% | 89.3 | completed (real conclusion) |

Because the reasoning cost is *fixed*, the useful fraction is purely a function of
how much room you give it. The 27.2 useful-t/s at 1024 was a **truncation
artifact** — the model spent 76% of the budget thinking and ran out before
finishing. At 4096 it lands a real conclusion and useful throughput rises to
~89 t/s (the 112 raw → 89 useful gap is just those 775 think-tokens amortized
across the run).

**Deployment implication:** amortizes beautifully on long-form generation;
punishing on terse Q&A (you pay ~775 think-tokens for a one-line answer). Suppress
reasoning for short/latency-sensitive tasks; give generous budgets for long-form.

*Harness note:* `--strip-thinking` was only wired into `throughput` before this
sweep, and its old heuristic was tag-blind — it silently no-op'd on the `<think>`
tag (the first LFM2.5 run reported the flag as having no effect). Commit `0ab6fb7`
made `_strip_thinking` prefer the literal `</think>` tag and wired the flag into
`decode-stability`; the table above is from the fixed version.

### Perplexity: the raw-input-pathway anomaly (second family after Gemma 4)

Gatsby raw perplexity is **87.0** — wildly out of line with healthy instruct/
reasoning models on this harness (QwQ-32B 1.705, Qwen3-4B 19.5, Llama-3.2-3B
anchor 6.48), but nowhere near Gemma 4 4-bit's broken-raw-path 536/1408. Same
dissociation signature as Gemma: **junk raw-input perplexity, clean chat-mode
generation** (the 4096-token chat essay was coherent and complete).

- **Not a reasoning-tuning tax.** QwQ-32B is reasoning-first and scored the *best*
  of the whole sweep (1.705); the 2026-04-08 doc already retired that framing.
- **Cause not isolated.** Only 8-bit was tested. Per the 2026-06-06 Gemma
  retraction, raw-pathway death can be a model/instruct-tuning property
  independent of quant — but here it could equally be new-arch immaturity in
  mlx-lm's `lfm2` implementation. Untested.
- **Operational rule:** treat LFM2.5's raw-input perplexity as unreliable; the
  chat-mode output is the quality signal. This is now the **second** family
  (after Gemma 4) where new-arch + raw-input = inflated perplexity — see
  Methodology lessons.

### Prefill scaling: conv-friendly, but the ceiling is too low to prove it

| n_tokens | prefill t/s | TTFT ms |
|---:|---:|---:|
| 10 | 50.0 | 200.1 |
| 50 | 287.0 | 174.2 |
| 200 | 899.7 | 222.3 |
| 500 | 1279.1 | 390.9 |
| 1000 | 1463.5 | 683.3 |

Prefill rises monotonically and is **still climbing at 1000 tokens** (1463 t/s)
with no O(n²) attention wall — the conv-dominant shape predicted by 18 conv (O(n))
layers vs. only 6 GQA. **But the harness caps at 1000 tokens, and at 1K context
every model prefills fine** — the attention penalty only bites hard at 8K-32K. So
this curve is *consistent with* the conv advantage but does not *prove* it. To get
the figure that shows conv-flat vs. attention-bend, extend the `lengths` list and
reference text out to 4K-32K. (The n=10 row is fixed-overhead noise; real signal
starts at n=200.)

### Verdict

A fast, RAM-light, on-device **specialist** — not a daily-driver challenger to
Qwen3.6-35B-A3B. At 8.6 GB it leaves ~15 GB free on the 24 GB box, decodes
~112 t/s (~89 useful), and warms to a 59 ms TTFT — built for resident, long-lived,
agentic/structured-output work where its fixed think-block amortizes. **Keep it
resident** (cold-invoke-per-request re-pays the ~7 s load + ~0.5 s warmup each
time). The conv-arch's theoretical long-context prefill advantage is plausible
from the 1K curve but unproven — the extended-context prefill run (4K-32K) is the
open item before any "conv beats attention at scale" claim.

## Open items (reconciled 2026-07-23)

Consolidated from a full read of this doc, reconciling every "planned / next /
pending / carried to" line against the later dated addendums. Everything tied to
the **GH200 evening session** is DONE despite earlier forward-references — the
Gemma 4 MoE bf16 control (line 714), the "does the MoE fail distinctively at
bf16?" question (726–728), and the MTP drafter speedups (756–768). The
reasoning-tuning-tax-on-prose caution was replicated across families and retired.
What genuinely remains:

**Coding evaluation (under-covered):**
- Coding runs never moved past snake + hanoi; the CLI-arg-parsing / Flask-handler
  task types called for at line 522 were never run.
- Nemotron 3 Nano coding one-shots — promised as the fair next test (668–671) and
  as the gate on its specialist slots (687–688), never run.
- Mac-runtime confirmation of the Qwen3.5-122B one-shot code — later passes were
  static review only (784–785).

**Format / throughput studies (budget-truncated):**
- NVFP4 batched-throughput fair test + the bf16 leg of the 122B shootout — cut for
  budget, logged as a gap (846–847). B300/NVFP4 phase is author-labeled "partial."

**LFM2.5 (newest arch, two unresolved gates):**
- Extended-context 4K-32K prefill run — the explicit gate before any "conv beats
  attention at scale" claim; harness still caps at 1000 tokens (977–982, 991–993).
- Raw-input perplexity anomaly (87.0) — model property vs. mlx-lm `lfm2` new-arch
  immaturity, untested (958–959).

**Housekeeping:**
- Gemma QAT-trained Q4 in MLX format — "if/when available" (524); depends on a
  conversion that hasn't appeared.
- nemotron_h cross-backend perplexity footnote / upstream issue — flagged but never
  filed (751–752).
