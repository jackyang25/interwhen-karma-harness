"""Qwen3-30B-A3B-Thinking-2507 adapter with MedAI MCP tool-calling.

vLLM-served, offline (in-process) inference. Tool calls are emitted by the
model as `<tool_call>{...}</tool_call>` text blocks (Qwen's native format,
rendered by the chat template). The adapter parses those out, dispatches via
MCP, and injects results back as `<tool_response>` blocks before regenerating.

This adapter is what Conditions A (no tools) and B (with tools) run on. The
interwhen-instrumented version for Condition E will subclass this and add
streaming + mid-generation verifier hooks; that lands later.

Threading: the adapter is NOT thread-safe (vLLM's LLM is meant for one
generation at a time per process). run_eval should use max_workers=1.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from harness.karma_adapter.mcp_tools import call_tool, fetch_tool_schemas

QWEN3_MODEL_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507"

# Qwen's tool-call wire format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass
class Qwen3Response:
    text: str
    n_tool_calls: int
    raw_messages: list[dict[str, Any]]
    stop_reason: str


class Qwen3Adapter:
    """Run a single vignette through Qwen3 with optional MedAI tool access.

    Heavy at construction time: loads the model into GPU memory. Reuse one
    instance for the whole eval.
    """

    def __init__(
        self,
        model_id: str = QWEN3_MODEL_ID,
        use_tools: bool = True,
        max_tokens: int = 4096,
        max_tool_turns: int = 10,
        temperature: float = 0.0,
        top_p: float = 1.0,
        gpu_memory_utilization: float = 0.80,
        max_model_len: int = 16384,
        tensor_parallel_size: int = 1,
    ):
        # Imported lazily so non-GPU notebooks can import this module without vllm.
        from vllm import LLM, SamplingParams  # type: ignore[import-untyped]
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.use_tools = use_tools
        self.max_tool_turns = max_tool_turns
        self.tools = fetch_tool_schemas() if use_tools else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=["</tool_call>"],   # stop right after a tool call so we can dispatch
            include_stop_str_in_output=True,
        )
        self._final_sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    def run(self, prompt: str, system: str | None = None) -> Qwen3Response:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        n_tool_calls = 0
        for turn in range(self.max_tool_turns + 1):
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tools=self.tools if self.use_tools else None,
                add_generation_prompt=True,
                tokenize=False,
            )
            params = self.sampling_params if self.use_tools else self._final_sampling_params
            outputs = self.llm.generate([rendered], params, use_tqdm=False)
            text = outputs[0].outputs[0].text

            tool_match = _TOOL_CALL_RE.search(text) if self.use_tools else None
            if tool_match is None:
                messages.append({"role": "assistant", "content": text})
                return Qwen3Response(
                    text=_strip_thinking(text),
                    n_tool_calls=n_tool_calls,
                    raw_messages=messages,
                    stop_reason="stop",
                )

            n_tool_calls += 1
            messages.append({"role": "assistant", "content": text})
            try:
                call = json.loads(tool_match.group(1))
                result = call_tool(call["name"], call.get("arguments", {}))
            except Exception as e:
                result = f"Tool error: {type(e).__name__}: {e}"
            messages.append(
                {
                    "role": "tool",
                    "content": result,
                }
            )

        return Qwen3Response(
            text=_strip_thinking(text),
            n_tool_calls=n_tool_calls,
            raw_messages=messages,
            stop_reason="max_tool_turns",
        )


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove Qwen's <think>...</think> reasoning block from the visible answer."""
    return _THINK_RE.sub("", text).strip()
