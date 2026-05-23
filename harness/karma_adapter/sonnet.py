"""Anthropic Sonnet adapter with MedAI MCP tool-use loop.

Used for apparatus validation (Condition B reproduced on Sonnet 4.6). After the
gate passes, this adapter is also the reference implementation for Conditions
A/B/C on Qwen3 — same loop, different backend.

Prompt caching is enabled on the system message + tool schemas so the 1,066
vignettes share cache for the static prefix.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import anthropic

from harness.karma_adapter.mcp_tools import call_tool, fetch_tool_schemas


@dataclass
class SonnetResponse:
    text: str
    n_tool_calls: int
    raw_messages: list[dict[str, Any]]
    stop_reason: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_model_calls: int = 0


class SonnetAdapter:
    """Run a single vignette through Sonnet with MedAI tools.

    Stateless across vignettes; one instance can be reused for the whole eval.
    Tool schemas are fetched once at construction.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        use_tools: bool = True,
        max_tokens: int = 4096,
        max_tool_turns: int = 10,
        temperature: float = 0.0,
        api_key: str | None = None,
    ):
        self.model = model
        self.use_tools = use_tools
        self.max_tokens = max_tokens
        self.max_tool_turns = max_tool_turns
        self.temperature = temperature
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.tools = fetch_tool_schemas() if use_tools else None

    def run(self, prompt: str, system: str | None = None) -> SonnetResponse:
        """Run the tool-use loop until the model emits a final text answer."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        n_tool_calls = 0

        sys_blocks = None
        if system:
            # Cache the static system prompt; tool schemas are auto-cached separately
            # via the tools=[...] argument.
            sys_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

        prompt_tokens = 0
        completion_tokens = 0
        n_model_calls = 0
        for _ in range(self.max_tool_turns + 1):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": messages,
            }
            if sys_blocks is not None:
                kwargs["system"] = sys_blocks
            if self.tools:
                kwargs["tools"] = self.tools

            resp = self.client.messages.create(**kwargs)
            n_model_calls += 1
            if getattr(resp, "usage", None) is not None:
                # Anthropic uses input_tokens / output_tokens (with cache fields).
                # Normalize to prompt_tokens / completion_tokens for cross-adapter parity.
                prompt_tokens += getattr(resp.usage, "input_tokens", 0) or 0
                prompt_tokens += getattr(resp.usage, "cache_read_input_tokens", 0) or 0
                completion_tokens += getattr(resp.usage, "output_tokens", 0) or 0
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason != "tool_use":
                text = _extract_text(resp.content)
                return SonnetResponse(
                    text=text,
                    n_tool_calls=n_tool_calls,
                    raw_messages=messages,
                    stop_reason=resp.stop_reason,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    n_model_calls=n_model_calls,
                )

            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    n_tool_calls += 1
                    try:
                        result = call_tool(block.name, dict(block.input))
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )
                    except Exception as e:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"Tool error: {type(e).__name__}: {e}",
                                "is_error": True,
                            }
                        )
            messages.append({"role": "user", "content": tool_results})

        # Hit max_tool_turns without a stop. Return whatever the last assistant text was.
        return SonnetResponse(
            text=_extract_text(messages[-1]["content"]),
            n_tool_calls=n_tool_calls,
            raw_messages=messages,
            stop_reason="max_tool_turns",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            n_model_calls=n_model_calls,
        )


def _extract_text(content: list[Any]) -> str:
    parts = []
    for block in content:
        text = getattr(block, "text", None) if not isinstance(block, dict) else block.get("text")
        if text:
            parts.append(text)
    return "\n".join(parts)
