"""interwhen Monitor subclasses for clinical tool-call verification.

`ClinicalInputMonitor` is the literal interwhen `VerifyMonitor` subclass
used in Condition E. It wraps:
- `harness.extraction.FactExtractor` (LLM-based) for extracting structured
  patient facts from the vignette
- `harness.verifier.semantic.verify` (deterministic) for comparing planned
  tool-call arguments against those facts
- `conf/prompts/condition_e_feedback.txt` for the feedback template

The actual streaming + monitor dispatch + retry logic lives in interwhen
(microsoft/interwhen). This module just provides the domain plug-in
(step_extractor, verify, fix) — same shape as interwhen's published examples
(thinkingPhaseVerifierMaze, verina_code_verifier, etc.).
"""
from harness.monitors.clinical_input import ClinicalInputMonitor, MonitorMetrics

__all__ = ["ClinicalInputMonitor", "MonitorMetrics"]
