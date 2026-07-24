"""CUDA compat shim.

Historical CLI for the A6000/H100/GH200 rental phases. The perf algorithms now
live in litmus_core; this file only preserves the --repo/--quant/--cmd surface
that a6000_bootstrap.sh and results-bonsai-1bit.md reference. throughput /
perplexity / decode-stability route to the shared core through TorchBackend;
`assisted` stays torch-native (speculative decoding has no MLX equivalent).

Setup + usage: unchanged — see a6000_bootstrap.sh.
"""
from __future__ import annotations

import argparse

from litmus_core import COMMANDS as CORE_COMMANDS
from litmus_torch import TorchBackend, cmd_assisted


class _QuantBackend:
    """Wraps TorchBackend so the core's backend.load(repo) picks up --quant."""
    def __init__(self, inner, quant: str):
        self._inner = inner
        self._quant = quant
        self.name = inner.name

    def load(self, repo, **opts):
        opts.setdefault("quant", self._quant)
        return self._inner.load(repo, **opts)

    def __getattr__(self, item):
        return getattr(self._inner, item)


# Historical command set (subset of core + the torch-native assisted).
CUDA_COMMANDS = {
    "throughput": CORE_COMMANDS["throughput"],
    "perplexity": CORE_COMMANDS["perplexity"],
    "decode-stability": CORE_COMMANDS["decode-stability"],
    "assisted": cmd_assisted,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cmd", choices=list(CUDA_COMMANDS), default="throughput")
    ap.add_argument("--repo", required=True, help="HF repo id to benchmark")
    ap.add_argument("--label", default=None)
    ap.add_argument("--quant", choices=["bf16", "fp32", "nf4", "prequant"],
                    default="bf16")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--ppl-window", type=int, default=1024)
    ap.add_argument("--reference-text", default=None)
    ap.add_argument("--sizes", default=None)   # unused; --repo is required here
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--assistant-repo", default=None)
    args = ap.parse_args()
    if args.reference_text is None:
        from litmus_core import REFERENCE_TEXT_PATH
        args.reference_text = REFERENCE_TEXT_PATH

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — this is the rental-phase harness.")

    backend = _QuantBackend(TorchBackend(), quant=args.quant)
    CUDA_COMMANDS[args.cmd](backend, args)


if __name__ == "__main__":
    main()
