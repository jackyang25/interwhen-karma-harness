"""interwhen-style Monitor for clinical tool-call verification.

`ClinicalInputMonitor` subclasses `interwhen.monitors.base.VerifyMonitor` to
inherit the (step_extractor, verify, fix) contract, used in the B'+E
conditions. It wraps:
- `harness.extraction.FactExtractor` (LLM-based) for extracting structured
  patient facts from the vignette
- `harness.verifier.semantic.verify` (deterministic) for comparing planned
  tool-call arguments against those facts
- `prompts/condition_e_feedback_query.txt` (query-style intervention) for the
  feedback template

The loop driver is the inline `_VerifiedAdapter` in `notebooks/02_run_all.py`,
not interwhen's `stream_completion`. The monitor's interface is reused; the
streaming + dispatch + retry logic lives in the adapter.
"""
from harness.monitors.clinical_input import ClinicalInputMonitor, MonitorMetrics

__all__ = ["ClinicalInputMonitor", "MonitorMetrics"]
