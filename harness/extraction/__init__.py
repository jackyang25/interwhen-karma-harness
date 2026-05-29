"""Fact extractor (B'+E conditions).

LLM-based component that reads a clinical vignette and returns a flat JSON
object of patient facts keyed by MCP calculator field names. The schema is
generated at run time by `harness.extraction.prompt_builder` from MCP's
per-calculator input schemas (preflight cell in `notebooks/02_run_all.py`).

Uses a different model from the one being evaluated (Sonnet 4.6 by default)
to avoid AI-verifying-AI circularity. Validated against a hand-annotated
held-out set operationally before production use — the paper recommends ≥95%
field-level accuracy as the gate; that check is a manual step, not enforced
in code.
"""
from harness.extraction.extractor import FactExtractor, PatientFacts
from harness.extraction.citations import validate_substring, coerce_citations_to_bare
from harness.extraction.voting import majority_vote

__all__ = [
    "FactExtractor",
    "PatientFacts",
    "validate_substring",
    "coerce_citations_to_bare",
    "majority_vote",
]
