---
description: Check the safety invariants of the repo
---
1. Run `pytest -q tests/test_validation.py tests/test_adversarial_qwen_outputs.py`.
2. Search the codebase: confirm `corrected_text` is only assembled in `pipeline.py`'s `apply_accepted()`.
3. Confirm `apply_accepted()` accepts only a `ValidationResult` with `accepted=True`.
4. Run `python scripts/run_safety_audit.py` if logs exist.
5. Report PASS/FAIL per item. Do not modify code in this workflow.
