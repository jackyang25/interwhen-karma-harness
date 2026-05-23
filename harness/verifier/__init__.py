"""Verifier modules.

Two related but distinct verifiers in this package:

- `posthoc` (Condition D'): LLM-based post-hoc inspection of the primary
  model's final answer. A separate model (Sonnet 4.6, different from the
  primary Qwen3) reads the case + candidate answer and flags inconsistencies.
  Used to test "did a cheap end-of-output review help?"

- `semantic` (Condition E): deterministic comparison between a planned tool
  call's inputs and the structured patient facts produced by the fact
  extractor (in `harness.extraction`). The split between LLM-based fact
  extraction and deterministic field comparison is the defense against
  "AI verifying AI" circularity (TESTING.md §5).
"""
from harness.verifier.posthoc import PostHocVerifierAdapter, VerifiedResponse
from harness.verifier.semantic import Violation, format_feedback, verify

__all__ = [
    "PostHocVerifierAdapter",
    "VerifiedResponse",
    "Violation",
    "format_feedback",
    "verify",
]
