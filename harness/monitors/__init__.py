"""interwhen Monitor subclasses for clinical tool-call verification.

Implements the framework's three-method API:
- step_extractor(chunk, generated_text): detect a complete tool call has been
  emitted in the visible reasoning stream (e.g., closing </tool_call> tag).
- verify(...): invoke the semantic verifier on the extracted call.
- fix(...): construct corrective feedback for injection into the stream.

Tool calls are emitted as text in the reasoning trace (not via structured
function-calling) so step boundaries are detectable. See TESTING.md Section
4.2 for the tool-calling mechanism rationale.
"""
