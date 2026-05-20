"""Unit tests for harness components.

Each module under harness/ has a corresponding test file here:
- test_extraction.py    — fact extractor JSON schema, field-level accuracy
- test_verifier.py      — semantic comparison logic (deterministic, fast)
- test_monitors.py      — step_extractor / verify / fix on synthetic streams
- test_karma_adapter.py — end-to-end smoke test against a mocked model
- test_conditions.py    — condition configurations build correctly
"""
