# Rule: Coding Standards (Always On)
- Python 3.11, type hints everywhere, pydantic v2 models for every cross-module object (`stage3/schemas.py`).
- Pure functions where possible; `correct_page(ocr_json) -> Stage3Result` has no hidden global mutation.
- Config only from `config/*.yaml`; no magic numbers in code.
- Model/dictionaries load once (module-level cached loader), never per call.
- Use `logging`, not `print`. Deterministic: temperature 0, fixed seed.
- Docstrings state inputs, outputs, and failure behaviour.
- Keep files under ~300 lines; split otherwise.
