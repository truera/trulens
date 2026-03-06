Thanks for the thoughtful review — this was very helpful. We addressed all
three points:

1. Groundedness dual-prompt consolidation

- Agreed this is a functional change for direct template consumers.
- We updated the PR description to explicitly call this out as intentional:
  - `llm_provider` groundedness judge behavior remains aligned with prior
    runtime usage.
  - Direct consumers of `v2.Groundedness.user_prompt` now intentionally receive
    the JSON-format prompt instead of the old block-format prompt.

2. Missing `__all__` on domain files

- Added explicit `__all__` declarations to:
  - `src/feedback/trulens/feedback/templates/rag.py`
  - `src/feedback/trulens/feedback/templates/safety.py`
  - `src/feedback/trulens/feedback/templates/quality.py`
  - `src/feedback/trulens/feedback/templates/agent.py`
  - `src/feedback/trulens/feedback/templates/base.py`
- Updated `src/feedback/trulens/feedback/templates/__init__.py` to aggregate
  exports from each module's explicit `__all__`.
- Added targeted tests in
  `tests/unit/test_feedback_templates_exports.py` to lock the intended export
  surface and prevent accidental leakage/collisions.

3. OTEL test isolation split

- Split the OTEL tempfile isolation change into a dedicated branch:
  `josh/isolate_otel_tests`.
- That branch contains only:
  - `tests/util/otel_test_case.py`
  - `tests/unit/test_otel_distributed.py`
- Branch is pushed to origin and ready to review independently.

Appreciate the detailed feedback — these changes make the API surface and
history much clearer.
