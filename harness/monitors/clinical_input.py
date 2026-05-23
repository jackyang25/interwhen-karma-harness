"""ClinicalInputMonitor — literal interwhen VerifyMonitor for clinical tool-call
input verification.

This module subclasses `interwhen.monitors.base.VerifyMonitor` so the
underlying interwhen library drives the model loop, step extraction, and
feedback injection. We provide the three abstract methods:

- `step_extractor`: returns (True, step_text) when a complete
  `<tool_call>...</tool_call>` block has been emitted, signalling interwhen
  to fire verification at that commit point.
- `verify`: parses the tool_call JSON, runs `harness.verifier.semantic.verify`
  against the patient facts produced by the fact extractor, sets the
  asyncio.Event if violations are found.
- `fix`: rebuilds the prompt with the formatted feedback message replacing
  the bad tool_call, so interwhen's `stream_completion` can recurse and
  re-generate from that point.

The verifier itself is deterministic (no LLM); the fact extractor is an LLM
call but is bounded to structured extraction. This split is the methodological
defense against AI-verifying-AI circularity (TESTING.md §5).
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from interwhen.monitors.base import VerifyMonitor

from harness.extraction.extractor import PatientFacts
from harness.karma_adapter.mcp_tools import call_tool
from harness.verifier.semantic import Violation, format_feedback, verify

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_OPEN_TAG = "<tool_call>"
_CLOSE_TAG = "</tool_call>"


@dataclass
class MonitorMetrics:
    """Accumulated characterization metrics for a single vignette run."""
    n_steps_seen: int = 0           # number of <tool_call> blocks the monitor observed
    n_verifier_fires: int = 0       # number of those that produced violations
    n_fixes_applied: int = 0        # number of times `fix` rewrote the prompt
    violations_history: list[dict[str, Any]] = field(default_factory=list)


class ClinicalInputMonitor(VerifyMonitor):
    """interwhen VerifyMonitor that runs the deterministic semantic verifier
    on each `<tool_call>` Qwen3 emits.

    Instantiate one per vignette (the patient facts are vignette-specific).
    """

    def __init__(
        self,
        patient_facts: PatientFacts,
        feedback_template: str,
        name: str = "clinical_input",
        priority: int = 0,
    ):
        super().__init__(name=name, priority=priority)
        self.patient_facts = patient_facts
        self.feedback_template = feedback_template
        self.metrics = MonitorMetrics()
        # Track the last seen position in generated_text so step_extractor
        # only fires on each new <tool_call> block, not on every chunk that
        # contains an earlier one.
        self._last_step_end = 0

    # ── step extractor ────────────────────────────────────────────────────
    def step_extractor(self, chunk: str, generated_text: str) -> tuple[bool, str]:
        """Detect when a complete <tool_call>...</tool_call> has appeared in
        `generated_text` past our last-seen position. interwhen calls this on
        every streamed chunk; we return (True, step_text) only when a new
        commit boundary is reached."""
        # Search only the tail since the last step we processed.
        tail = generated_text[self._last_step_end:]
        match = _TOOL_CALL_RE.search(tail)
        if match is None:
            return (False, "")
        # Advance the cursor past this match so we don't re-fire on it.
        self._last_step_end += match.end()
        self.metrics.n_steps_seen += 1
        return (True, match.group(0))   # group(0) is the full <tool_call>...</tool_call>

    # ── verify ────────────────────────────────────────────────────────────
    async def verify(self, chunk, token_index, event, event_info):
        """Inspect every complete <tool_call> block.

        We ALWAYS set the event because interwhen's stream_completion does
        not dispatch tools itself — it just streams text. So whenever a
        tool_call appears we must interrupt the stream so `fix` can either
        (a) execute the tool and inject its real result, or (b) inject
        feedback if the call's inputs were inconsistent with the case.

        `event_info` carries the parsed call, the verifier's findings, and
        the literal block text so `fix` can splice the prompt cleanly.
        """
        match = _TOOL_CALL_RE.search(chunk)
        if match is None:
            return

        try:
            call_obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            # Malformed JSON in the tool_call — record the block, let `fix`
            # decide. Set event so we can recover.
            event_info["malformed"] = True
            event_info["tool_call_block"] = chunk
            event_info["call_obj"] = None
            event_info["violations"] = []
            event_info["token_index"] = token_index
            event.set()
            return

        if not isinstance(call_obj, dict):
            event_info["malformed"] = True
            event_info["tool_call_block"] = chunk
            event_info["call_obj"] = None
            event_info["violations"] = []
            event_info["token_index"] = token_index
            event.set()
            return

        arguments = call_obj.get("arguments", {}) or {}
        violations: list[Violation] = verify(arguments, self.patient_facts)

        if violations:
            # Violation path: record and signal.
            self.metrics.n_verifier_fires += 1
            self.metrics.violations_history.append(
                {
                    "tool": call_obj.get("name"),
                    "arguments": arguments,
                    "violations": [
                        {"field": v.field, "planned": v.planned, "expected": v.expected, "note": v.note}
                        for v in violations
                    ],
                }
            )

        # Always interrupt — even on no-violation calls, because `fix` must
        # dispatch the tool and inject its result. Without this, the model
        # continues generating as if the tool returned nothing.
        event_info["malformed"] = False
        event_info["tool_call_block"] = chunk
        event_info["call_obj"] = call_obj
        event_info["violations"] = violations
        event_info["token_index"] = token_index
        event.set()
        return

    # ── fix ───────────────────────────────────────────────────────────────
    async def fix(self, generated_text: str, event_info: dict, fix_method=None) -> str:
        """Rebuild the prefix the model will continue from.

        Three paths:
        1. Violations present → splice in formatted feedback in place of the
           tool_call. The model re-plans from that point.
        2. No violations → dispatch the tool to MedAI MCP, splice in the real
           tool_response. The model continues with the actual result.
        3. Malformed tool_call → splice in a tool_response asking the model
           to emit a valid JSON tool_call.
        """
        block: str = event_info.get("tool_call_block", "")
        violations: list[Violation] = event_info.get("violations", [])
        call_obj = event_info.get("call_obj")
        malformed: bool = bool(event_info.get("malformed", False))

        cut = generated_text.rfind(block) if block else -1
        if cut < 0:
            prefix = generated_text
        else:
            prefix = generated_text[:cut]

        if malformed or call_obj is None:
            payload = (
                "<|im_end|>\n"
                "<|im_start|>tool\n"
                "<tool_response>\n"
                "The previous tool_call was not valid JSON. Re-emit a single "
                "<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call> block.\n"
                "</tool_response>\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            self.metrics.n_fixes_applied += 1
            return prefix + payload

        if violations:
            # Feedback path — the model retries with corrected understanding.
            feedback = format_feedback(self.feedback_template, violations)
            payload = (
                "<|im_end|>\n"
                "<|im_start|>tool\n"
                f"<tool_response>\n{feedback}\n</tool_response>\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            self.metrics.n_fixes_applied += 1
            return prefix + payload

        # No violations: dispatch the tool and inject the real response.
        try:
            tool_name = call_obj.get("name", "")
            tool_args = call_obj.get("arguments", {}) or {}
            result = call_tool(tool_name, tool_args)
        except Exception as e:
            result = f"Tool error: {type(e).__name__}: {e}"

        payload = (
            f"{block}"          # keep the original tool_call in the transcript
            "<|im_end|>\n"
            "<|im_start|>tool\n"
            f"<tool_response>\n{result}\n</tool_response>\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.metrics.n_fixes_applied += 1
        return prefix + payload
