"""Qwen3-30B-A3B-Thinking-2507 adapter (text-mode via vLLM /v1/completions).

Why text-mode (and not chat-completions + structured tools)? Two reasons:

1. **Single channel across conditions.** The B'+E conditions use the inline
   `_VerifiedAdapter` in `notebooks/02_run_all.py`, which watches the raw
   text stream for `<tool_call>` blocks and dispatches via the
   ClinicalInputMonitor. Using a different endpoint for them than for
   A/B/C/B'/D would mix channels and confound comparisons. All Qwen3
   conditions therefore go through `/v1/completions` text-mode.

2. **Qwen3's native trained channel.** Qwen3 was fine-tuned to emit
   `<tool_call>{...}</tool_call>` tags as raw text. The structured tools API
   we used earlier was just vLLM post-processing those same tags. Same model
   behavior, different parsing layer. Text-mode exposes the raw stream
   directly, which is what we want.

Architecture (this file, used by A/B/C/B' and as the primary in D):
- `apply_chat_template(messages, tools=...)` from the HF tokenizer builds the
  prompt in Qwen3's native format (no need to hand-write the chatml).
- Call `/v1/completions` (text endpoint), stop at `</tool_call>` so we can
  dispatch immediately.
- Parse the tool call, dispatch to MedAI MCP, append result as a
  `<tool_response>` block to the prompt, and continue generation from there.
- Loop until the model produces output without a tool call.

The B'+E conditions use a separate adapter (inline in 02_run_all) that
inserts the ClinicalInputMonitor between detection and dispatch, but the
underlying text-mode channel is identical.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from harness.karma_adapter.mcp_tools import call_tool, fetch_tool_schemas

QWEN3_MODEL_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507"

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


@dataclass
class Qwen3Response:
    text: str
    n_tool_calls: int
    raw_completion: str   # full rolling prompt tail (tool_call + tool_response blocks); used by 02_run_all smoke test
    stop_reason: str
    # Honest totals (summed across all model calls in this row).
    # For A/B/C/B' these are pure Qwen3 vLLM usage.
    # For the inline E adapter in 02_run_all these are extractor + Qwen3.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_model_calls: int = 0
    # Condition-E verifier instrumentation. Zero for all other conditions.
    n_verifier_fires: int = 0   # tool calls where ClinicalInputMonitor found violations
    n_fixes_applied: int = 0    # prompt rewrites (feedback injections + malformed recoveries)
    # ── E-shape deployment-honest breakouts (zero for A/B/C/B') ───────────
    # Extractor = Sonnet API call (billed by Anthropic).
    # qwen3_*   = on-GPU vLLM inference (single condition E session, summed
    #             across the tool-calling loop's vLLM calls).
    extractor_prompt_tokens: int = 0
    extractor_completion_tokens: int = 0
    extractor_elapsed_s: float = 0.0
    qwen3_prompt_tokens: int = 0
    qwen3_completion_tokens: int = 0
    qwen3_elapsed_s: float = 0.0
    # Per-row structured record of every verifier intervention (E only).
    violations_history: list[dict[str, Any]] = field(default_factory=list)
    # Per-row Sonnet-extracted patient facts (E and B'+E only). Lets you see
    # exactly what the verifier had to compare against — distinguishes
    # "model was faithful, verifier correctly silent" from "extractor was
    # sparse, verifier had nothing to compare". Empty dict for non-E rows.
    extracted_facts: dict[str, Any] = field(default_factory=dict)
    # Per-row mechanism diagnostics for the two new primary conditions.
    # citation_reports: list of per-tool-call dicts produced by the
    #   citations adapter. Each entry: {"wanted_fields": [...], "report":
    #   {field: {"value", "source_span", "valid", "reason"}}}. Empty list
    #   for every condition other than B_prime_E_reactive_citations.
    # voting_reports: list of per-tool-call dicts produced by the k-shot
    #   adapter. Each entry: {"wanted_fields": [...], "report":
    #   {field: {"samples", "winner", "count", "accepted", "reason"}}}.
    #   Empty list for every condition other than B_prime_E_reactive_kshot.
    # Both are serialized as JSON strings into the per-row parquet via
    # harness/runner.py, mirroring the violations_history shape.
    citation_reports: list[dict[str, Any]] = field(default_factory=list)
    voting_reports: list[dict[str, Any]] = field(default_factory=list)


class Qwen3Adapter:
    """Talks to vLLM's /v1/completions endpoint in text mode.

    Tool descriptions are inlined in the prompt via the tokenizer's
    `apply_chat_template(..., tools=...)`. The model is expected to emit
    `<tool_call>{...}</tool_call>` tags in the output stream; the adapter
    detects them, dispatches via MedAI MCP, and appends a `<tool_response>`
    block before continuing.

    Threading: thread-safe at this granularity (httpx-backed OpenAI client,
    one MCP call per tool dispatch). run_eval can use max_workers > 1.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str = QWEN3_MODEL_ID,
        use_tools: bool = True,
        max_tokens: int = 4096,
        max_tool_turns: int = 10,
        temperature: float = 0.0,
        top_p: float = 1.0,
        api_key: str = "EMPTY",
        tokenizer_id: str | None = None,
    ):
        from openai import OpenAI  # lazy import
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.use_tools = use_tools
        self.max_tokens = max_tokens
        self.max_tool_turns = max_tool_turns
        self.temperature = temperature
        self.top_p = top_p
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        # The tokenizer builds the chatml-format prompt that Qwen3 was trained
        # on. We bypass /v1/chat/completions and hit /v1/completions with the
        # rendered string directly so we own the raw text stream interwhen
        # needs to monitor.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or model_id)
        self._tools_schema: list[dict[str, Any]] | None = (
            fetch_tool_schemas() if use_tools else None
        )

    def run(self, prompt: str, system: str | None = None) -> Qwen3Response:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        rendered = self.tokenizer.apply_chat_template(
            messages,
            tools=self._tools_schema if self.use_tools else None,
            add_generation_prompt=True,
            tokenize=False,
        )

        n_tool_calls = 0
        n_model_calls = 0
        prompt_tokens = 0
        completion_tokens = 0
        rolling_prompt = rendered

        for _ in range(self.max_tool_turns + 1):
            # /v1/completions — raw text in, raw text out.
            resp = self.client.completions.create(
                model=self.model_id,
                prompt=rolling_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=(["</tool_call>"] if self.use_tools else None),
            )
            n_model_calls += 1
            if resp.usage is not None:
                prompt_tokens += getattr(resp.usage, "prompt_tokens", 0) or 0
                completion_tokens += getattr(resp.usage, "completion_tokens", 0) or 0

            choice = resp.choices[0]
            generated = choice.text or ""

            # If the model hit the stop string, vLLM strips it. Re-attach so the
            # regex match below sees a complete <tool_call>...</tool_call>.
            tool_call_open = "<tool_call>" in generated
            stopped_at_tool = (
                self.use_tools
                and choice.finish_reason == "stop"
                and tool_call_open
            )
            if stopped_at_tool:
                generated = generated + "</tool_call>"

            rolling_prompt += generated

            if not self.use_tools or not tool_call_open:
                return Qwen3Response(
                    text=_strip_thinking(_extract_assistant_tail(generated)),
                    n_tool_calls=n_tool_calls,
                    raw_completion=rolling_prompt[len(rendered):],
                    stop_reason=choice.finish_reason or "stop",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    n_model_calls=n_model_calls,
                )

            tool_match = _TOOL_CALL_RE.search(generated)
            if tool_match is None:
                # Model said "<tool_call>" but never closed it. Treat as final answer.
                return Qwen3Response(
                    text=_strip_thinking(_extract_assistant_tail(generated)),
                    n_tool_calls=n_tool_calls,
                    raw_completion=rolling_prompt[len(rendered):],
                    stop_reason="malformed_tool_call",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    n_model_calls=n_model_calls,
                )

            n_tool_calls += 1
            try:
                call = json.loads(tool_match.group(1))
                result = call_tool(call["name"], call.get("arguments", {}))
            except Exception as e:
                result = f"Tool error: {type(e).__name__}: {e}"

            # Close the current assistant turn (the model emitted <tool_call>
            # then stopped at </tool_call>, so the turn is still open with no
            # <|im_end|>), then add a tool turn with the response, then re-open
            # the assistant turn for continuation. This matches Qwen3's
            # training-time tool-use format.
            rolling_prompt += (
                "<|im_end|>\n"
                "<|im_start|>tool\n"
                f"<tool_response>\n{result}\n</tool_response>\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        # Max tool turns exhausted.
        return Qwen3Response(
            text=_strip_thinking(_extract_assistant_tail(rolling_prompt[len(rendered):])),
            n_tool_calls=n_tool_calls,
            raw_completion=rolling_prompt[len(rendered):],
            stop_reason="max_tool_turns",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            n_model_calls=n_model_calls,
        )


# ──────────────────────────────────────────────────────────────────────────────
# vLLM server manager (unchanged from prior text-mode design; preserved here so
# notebooks that import VLLMServer don't break).
# ──────────────────────────────────────────────────────────────────────────────

import os
import signal
import subprocess
import time
from pathlib import Path


class VLLMServer:
    """Launch and manage a `vllm serve` subprocess.

    Exposes /v1/completions and /v1/chat/completions endpoints. We use the
    former for all Qwen3 work. The default launch args intentionally do NOT
    include `--enable-auto-tool-choice` or `--tool-call-parser`; those flags
    only affect /v1/chat/completions and would be no-ops on our text-mode
    path.
    """

    def __init__(
        self,
        model_id: str = QWEN3_MODEL_ID,
        host: str = "127.0.0.1",
        port: int = 8000,
        gpu_memory_utilization: float = 0.80,
        max_model_len: int = 16384,
        tensor_parallel_size: int = 1,
        dtype: str = "bfloat16",
        extra_args: list[str] | None = None,
        log_path: str | Path = "/tmp/vllm_server.log",
        env_overrides: dict[str, str] | None = None,
        vllm_bin: str = "vllm",
    ):
        self.model_id = model_id
        self.host = host
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.extra_args = list(extra_args or [])
        self.log_path = Path(log_path)
        self.env_overrides = env_overrides or {}
        self.vllm_bin = vllm_bin
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            raise RuntimeError("Server already running")

        cmd = [
            self.vllm_bin, "serve", self.model_id,
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--dtype", self.dtype,
            "--trust-remote-code",
            *self.extra_args,
        ]

        env = os.environ.copy()
        env.update(self.env_overrides)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = self.log_path.open("w")
        print(f"Launching: {' '.join(cmd)}")
        print(f"Logs: {self.log_path}")
        self.proc = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )

    def wait_ready(self, timeout: int = 600, poll_interval: float = 2.0) -> None:
        import httpx

        if self.proc is None:
            raise RuntimeError("Server not started")
        deadline = time.time() + timeout
        url = f"{self.base_url}/models"
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = self._log_tail(60)
                raise RuntimeError(
                    f"vllm serve exited with code {self.proc.returncode} before becoming ready.\n"
                    f"Last log lines:\n{tail}"
                )
            try:
                r = httpx.get(url, timeout=2.0)
                if r.status_code == 200:
                    print(f"Server ready at {self.base_url}")
                    return
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
                pass
            time.sleep(poll_interval)
        raise TimeoutError(f"vllm serve did not become ready within {timeout}s")

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, timeout: int = 30) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)

    def _log_tail(self, n_lines: int) -> str:
        try:
            text = self.log_path.read_text()
        except FileNotFoundError:
            return "(no log file written yet)"
        return "\n".join(text.splitlines()[-n_lines:])


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _extract_assistant_tail(text: str) -> str:
    """Pull the model's final visible answer out of a chatml-style completion.

    Qwen3's chat template uses `<|im_start|>assistant ... <|im_end|>` markers.
    For the final-answer return we want the assistant's most recent content,
    after any tool_call/tool_response cycles and after thinking.
    """
    # Strip the chatml turn markers — keep the content of the last assistant
    # turn or, if those markers aren't present, the whole tail.
    last = text
    if "<|im_start|>assistant" in text:
        last = text.rsplit("<|im_start|>assistant", 1)[-1]
    last = last.split("<|im_end|>", 1)[0]
    return last.strip()
