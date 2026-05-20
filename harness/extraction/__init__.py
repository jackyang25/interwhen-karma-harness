"""Fact extractor.

LLM-based component that reads a clinical vignette and returns a structured
patient object (age, sex, weight, height, labs, lifestyle factors, etc.).

Uses a different model from the one being evaluated to avoid contamination.
Validated against a hand-annotated held-out set; field-level accuracy threshold
locked in pre-registration before production use.
"""
