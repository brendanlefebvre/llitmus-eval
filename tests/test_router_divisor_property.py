"""Property: the router's estimate_prompt_tokens never underestimates the
reference-tokenizer count on the calibration corpus. Pins the divisor to
the corpus it was calibrated on; new captures may require recalibration
(scripts/calibrate_router_divisor.py)."""
import json
import os
import pathlib

import pytest

from litmus_spec import count_ref_tokens, load_cases
from loxo_llm_router import estimate_prompt_tokens

CASES = "cases/main_replay.jsonl"


@pytest.fixture(scope="module")
def ref_tokenizer():
    meta_path = pathlib.Path("cases/main_replay.meta.json")
    repo = "mlx-community/Qwen3-14B-4bit"
    if meta_path.exists():
        repo = json.loads(meta_path.read_text()).get("tokenizer", repo)
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            repo, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"reference tokenizer {repo} not in local HF cache "
                    f"({type(e).__name__})")


def test_estimator_never_underestimates_on_corpus(ref_tokenizer):
    if not os.path.exists(CASES):
        pytest.skip(f"{CASES} missing")
    cases = load_cases(CASES, "main-replay")
    undercounts, skipped = [], []
    for case in cases:
        if not os.path.exists(case.capture_path):
            skipped.append(case.id)
            continue
        with open(case.capture_path, encoding="utf-8") as f:
            body = json.load(f)
        est = estimate_prompt_tokens(body)
        real = count_ref_tokens(ref_tokenizer, body)
        if est < real:
            undercounts.append(f"  {case.id}: est={est} < real={real}")
    if skipped:
        print(f"  (skipped: {', '.join(skipped)})")
    assert not undercounts, (
        "estimator underestimates — recalibrate the divisor "
        "(scripts/calibrate_router_divisor.py):\n" + "\n".join(undercounts))
