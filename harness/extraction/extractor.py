"""Sonnet-based fact extractor for Condition E.

Reads a vignette, returns a structured patient JSON object. The schema is
generated at run time by the preflight cell in `notebooks/02_run_all.py`
from MCP's per-calculator input schemas (see
`harness.extraction.prompt_builder`). The runtime prompt lives at
`/dbfs/results/_runtime/extractor_prompt.txt`. The extractor model is
Sonnet 4.6, deliberately different from the Qwen3 model being evaluated
so the verifier isn't comparing the model against itself.

The output is consumed by `harness.verifier.semantic` to check tool-call
inputs against the case before they are dispatched.

§8 of TESTING.md requires the extractor to demonstrate ≥95% field-level
accuracy on a hand-annotated held-out set before production runs. That gate
is enforced operationally (manual review at pilot time), not in code — this
module provides the runtime extractor; the validation set construction is a
separate manual step.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic

DEFAULT_EXTRACTOR_MODEL = "claude-sonnet-4-6"


@dataclass
class PatientFacts:
    raw: dict[str, Any] = field(default_factory=dict)
    extractor_ok: bool = True
    extractor_error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


class FactExtractor:
    """Stateless extractor — one instance can serve the whole eval."""

    def __init__(
        self,
        prompt_path: str,
        model: str = DEFAULT_EXTRACTOR_MODEL,
        api_key: str | None = None,
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ):
        self.system_prompt = open(prompt_path).read().strip()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def extract(self, vignette: str) -> PatientFacts:
        """Extract using the system prompt set at construction time. Used by the
        upfront-extraction architecture in E and B_prime_E (one call per vignette)."""
        return self._call_sonnet(vignette, self.system_prompt)

    def extract_with_prompt(self, vignette: str, system_prompt: str) -> PatientFacts:
        """Extract using a per-call system prompt. Used by the reactive extraction
        architecture in B_prime_E_reactive: each tool call gets a focused prompt
        covering only the fields it actually requested, avoiding long-schema
        attention dilution."""
        return self._call_sonnet(vignette, system_prompt)

    def _call_sonnet(self, vignette: str, system_prompt: str) -> PatientFacts:
        """Single Sonnet API call with the provided system prompt. Shared by
        extract() and extract_with_prompt() — keeps token-counting and error
        handling identical across the two architectures."""
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": vignette}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            in_tok = (getattr(resp.usage, "input_tokens", 0) or 0) + (
                getattr(resp.usage, "cache_read_input_tokens", 0) or 0
            )
            out_tok = getattr(resp.usage, "output_tokens", 0) or 0
        except Exception as e:
            return PatientFacts(extractor_ok=False, extractor_error=f"{type(e).__name__}: {e}")

        parsed = _parse_json(text)
        if parsed is None:
            return PatientFacts(
                extractor_ok=False,
                extractor_error=f"json_parse_failed: {text[:200]}",
                prompt_tokens=in_tok,
                completion_tokens=out_tok,
            )
        return PatientFacts(
            raw=parsed,
            extractor_ok=True,
            prompt_tokens=in_tok,
            completion_tokens=out_tok,
        )


_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict[str, Any] | None:
    """Pull the outermost {...} blob out and parse it as JSON."""
    m = _OBJ_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
