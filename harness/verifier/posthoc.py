"""Post-hoc verification (Condition D).

After the primary model produces an answer, a separate verifier model (Sonnet
4.6 — different from Qwen3 to avoid self-agreement bias) inspects the case +
candidate answer and flags inconsistencies. On flag, the primary model is
asked to revise once.

Design decisions (locked before production runs):
- Verifier model: Claude Sonnet 4.6
- Verifier prompt: `prompts/condition_d.txt`
- Revision policy: one revision attempt on flag
- Trigger: every primary-model answer (no skip-on-confidence heuristic)

This module wraps an arbitrary primary `Adapter` and exposes the same `run`
interface; `run_eval` doesn't need to change.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic

DEFAULT_VERIFIER_MODEL = "claude-sonnet-4-6"


class _PrimaryAdapter(Protocol):
    def run(self, prompt: str, system: str | None = None) -> Any: ...


@dataclass
class VerifiedResponse:
    text: str
    n_tool_calls: int
    raw_completion: str   # rolling prompt tail from the Qwen3 primary (matches Qwen3Response)
    stop_reason: str
    # Honest cross-condition totals (primary Qwen3 + Sonnet verifier + any
    # revision). Use the *_primary_* and *_verifier_* breakouts below to
    # separate on-GPU cost from Anthropic API cost.
    prompt_tokens: int
    completion_tokens: int
    n_model_calls: int
    # D'-specific fields:
    verifier_consistent: bool   # True if verifier did not flag (or wasn't called)
    verifier_issue: str          # Description of flagged issue, empty if consistent
    revised: bool                # True if the primary was asked to revise
    # Cost / latency breakouts for deployment-honest interpretation.
    # Primary = Qwen3 inference (vLLM on H100), summed across initial + revision.
    # Verifier = Sonnet API call (billed by Anthropic). Always exactly one
    # verifier call per row regardless of revision.
    primary_prompt_tokens: int = 0
    primary_completion_tokens: int = 0
    primary_elapsed_s: float = 0.0
    verifier_prompt_tokens: int = 0
    verifier_completion_tokens: int = 0
    verifier_elapsed_s: float = 0.0


class PostHocVerifierAdapter:
    """Wraps a primary adapter with a Sonnet-based post-hoc verifier.

    For each vignette:
      1. Run the primary adapter to get an initial answer.
      2. Send (case, instruction, candidate answer) to the verifier model
         along with the locked verifier prompt.
      3. Parse the verifier's JSON {consistent, issue}.
      4. If consistent=false, re-run the primary adapter once with the issue
         appended to the user prompt as feedback. Return the revised answer.
      5. Otherwise, return the initial answer.

    Token counts and model-call counts include verifier calls + retries so the
    cost reflects the full D' workflow.
    """

    def __init__(
        self,
        primary: _PrimaryAdapter,
        verifier_prompt_path: str,
        verifier_model: str = DEFAULT_VERIFIER_MODEL,
        api_key: str | None = None,
        max_tokens: int = 256,
    ):
        self.primary = primary
        self.verifier_prompt = open(verifier_prompt_path).read().strip()
        self.verifier_model = verifier_model
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    def run(self, prompt: str, system: str | None = None) -> VerifiedResponse:
        # Step 1: primary answer (Qwen3 on local GPU)
        t_primary = time.time()
        first = self.primary.run(prompt, system=system)
        primary_elapsed_s = time.time() - t_primary
        primary_prompt_tokens = getattr(first, "prompt_tokens", 0)
        primary_completion_tokens = getattr(first, "completion_tokens", 0)
        n_model_calls = getattr(first, "n_model_calls", 0)
        n_tool_calls = getattr(first, "n_tool_calls", 0)

        # Step 2-3: verifier inspection (Sonnet API)
        t_verifier = time.time()
        verifier_consistent, verifier_issue, v_in, v_out = self._verify(prompt, first.text)
        verifier_elapsed_s = time.time() - t_verifier
        verifier_prompt_tokens = v_in
        verifier_completion_tokens = v_out
        n_model_calls += 1

        # Step 4: revision on flag (Qwen3 again, on local GPU)
        revised = False
        final_text = first.text
        final_completion = getattr(first, "raw_completion", "")
        final_stop = first.stop_reason
        if not verifier_consistent and verifier_issue:
            revised = True
            revision_prompt = (
                f"{prompt}\n\n"
                f"A reviewer flagged your previous answer: \"{verifier_issue}\". "
                f"Please reconsider the case and provide a corrected JSON answer."
            )
            t_revision = time.time()
            second = self.primary.run(revision_prompt, system=system)
            primary_elapsed_s += time.time() - t_revision
            final_text = second.text
            final_completion = getattr(second, "raw_completion", "")
            final_stop = second.stop_reason
            primary_prompt_tokens += getattr(second, "prompt_tokens", 0)
            primary_completion_tokens += getattr(second, "completion_tokens", 0)
            n_model_calls += getattr(second, "n_model_calls", 0)
            n_tool_calls += getattr(second, "n_tool_calls", 0)

        return VerifiedResponse(
            text=final_text,
            n_tool_calls=n_tool_calls,
            raw_completion=final_completion,
            stop_reason=final_stop,
            # Honest totals: primary + verifier summed.
            prompt_tokens=primary_prompt_tokens + verifier_prompt_tokens,
            completion_tokens=primary_completion_tokens + verifier_completion_tokens,
            n_model_calls=n_model_calls,
            verifier_consistent=verifier_consistent,
            verifier_issue=verifier_issue,
            revised=revised,
            primary_prompt_tokens=primary_prompt_tokens,
            primary_completion_tokens=primary_completion_tokens,
            primary_elapsed_s=primary_elapsed_s,
            verifier_prompt_tokens=verifier_prompt_tokens,
            verifier_completion_tokens=verifier_completion_tokens,
            verifier_elapsed_s=verifier_elapsed_s,
        )

    def _verify(self, case_prompt: str, candidate_answer: str) -> tuple[bool, str, int, int]:
        """Call the verifier model; return (consistent, issue, in_tokens, out_tokens)."""
        verifier_user_message = (
            f"PATIENT CASE AND QUESTION:\n{case_prompt}\n\n"
            f"CANDIDATE JSON ANSWER:\n{candidate_answer}"
        )
        try:
            resp = self.client.messages.create(
                model=self.verifier_model,
                max_tokens=self.max_tokens,
                temperature=0.0,
                system=self.verifier_prompt,
                messages=[{"role": "user", "content": verifier_user_message}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            in_tok = getattr(resp.usage, "input_tokens", 0) or 0
            in_tok += getattr(resp.usage, "cache_read_input_tokens", 0) or 0
            out_tok = getattr(resp.usage, "output_tokens", 0) or 0
        except Exception as e:
            # On verifier failure, fail open (assume consistent) so the run doesn't crash.
            return True, f"verifier_error: {type(e).__name__}: {e}", 0, 0

        parsed = _parse_verifier_json(text)
        if parsed is None:
            return True, f"verifier_parse_failed: {text[:200]}", in_tok, out_tok
        return bool(parsed.get("consistent", True)), str(parsed.get("issue", "")), in_tok, out_tok


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_verifier_json(text: str) -> dict[str, Any] | None:
    """Extract the verifier's JSON object from its response."""
    for m in reversed(list(_JSON_OBJECT_RE.finditer(text))):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "consistent" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None
