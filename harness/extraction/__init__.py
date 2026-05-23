"""Fact extractor (Condition E).

LLM-based component that reads a clinical vignette and returns a structured
patient object (age, sex, weight, height, labs, lifestyle factors, etc.) via
the `FactExtractor` class.

Uses a different model from the one being evaluated (Sonnet 4.6 by default)
to avoid contamination. Validated against a hand-annotated held-out set
operationally before production use — TESTING.md §8 sets ≥95% field-level
accuracy as the gate; that check is a manual step, not enforced in code.
"""
from harness.extraction.extractor import FactExtractor, PatientFacts

__all__ = ["FactExtractor", "PatientFacts"]
