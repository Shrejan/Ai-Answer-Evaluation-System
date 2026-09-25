# Rule: Testing (Always On)
- Every module ships with pytest tests in `tests/`.
- The LLM is **mocked** in all unit tests via the `LLMClient` protocol. Real Ollama is used only in `tests/integration/` marked `@pytest.mark.ollama`.
- Worked examples that must have tests: `beat`, `attack`, `pregnue`, `local lacking`, `volontrar-y`, `denital`, `nubation`, `transisttor`.
- Adversarial validation tests are mandatory (see `docs/06_TESTING.md`).
- Never weaken or delete a failing test to make it pass; report it instead.
