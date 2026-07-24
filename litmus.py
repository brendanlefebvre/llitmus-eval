"""Litmus - a local-LLM eval suite (perf: throughput, perplexity, decode
stability, prefill scaling, baseline, cold-start).

Backend-agnostic: defaults to MLX on Apple Silicon; pass --backend cuda for
the torch path (see litmus_cuda.py for the historical CUDA CLI). See
litmus_core for the shared algorithms and litmus_common for the Backend seam.
"""
from __future__ import annotations

import argparse

from litmus_common import get_backend
from litmus_core import COMMANDS, REFERENCE_TEXT_PATH


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backend", choices=["mlx", "cuda", "auto"], default="mlx",
                    help="model runtime (default: mlx)")
    ap.add_argument("--cmd", choices=list(COMMANDS), default="throughput",
                    help="which benchmark to run (default: throughput)")
    ap.add_argument("--sizes", default="1.7B,4B,8B",
                    help="comma-separated subset of 1.7B,4B,8B")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--strip-thinking", action="store_true",
                    help="(throughput) also report 'useful' tok/s after "
                         "stripping reasoning preamble")
    ap.add_argument("--ppl-window", type=int, default=1024,
                    help="(perplexity) max tokens of reference text to score")
    ap.add_argument("--reference-text", default=REFERENCE_TEXT_PATH,
                    help=f"path to reference text (default: {REFERENCE_TEXT_PATH})")
    ap.add_argument("--repo", default=None,
                    help="HF repo id to benchmark instead of a Bonsai size")
    ap.add_argument("--label", default=None,
                    help="display label for --repo")
    ap.add_argument("--chat", action="store_true",
                    help="(decode-stability) wrap prompt in the chat template")
    args = ap.parse_args()

    backend = get_backend(args.backend)
    COMMANDS[args.cmd](backend, args)


if __name__ == "__main__":
    main()
