"""Semantic input verifier.

Deterministic comparison between a planned tool call's inputs and the
structured patient facts produced by the extractor. Checks values, enums, and
units; returns validity plus per-input feedback for the feedback formatter.

No LLM calls live here — verification logic is purely deterministic. The
LLM-noise / determinism split is the defense against "AI verifying AI"
circularity (see TESTING.md Section 5).
"""
