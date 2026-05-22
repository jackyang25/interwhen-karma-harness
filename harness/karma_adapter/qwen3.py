"""Qwen3-30B-A3B-Thinking-2507 adapter via subprocess vLLM server.

Why subprocess rather than in-process vLLM? On Databricks runtimes newer than
the open-source ML ecosystem (CUDA 13 / torch 2.11 era), vLLM crashes during
model load and takes the Python kernel down with it — no traceback, no
debuggable signal. Running vLLM as a separate process via `vllm serve` gives
us three things:

1. **Process isolation**: vLLM crashes don't kill the notebook kernel.
2. **Real error messages**: stderr is captured to a log file.
3. **Production-shaped**: `vllm serve` is the recommended deployment pattern,
   not a workaround.

The adapter talks to the local server over the OpenAI-compatible HTTP API.
Tool calls are parsed from response text (Qwen3 emits `<tool_call>{...}</tool_call>`
blocks natively), dispatched to MedAI MCP, and injected back as user messages.

Threading: HTTP client is thread-safe; vLLM server batches across concurrent
requests internally. run_eval can use max_workers > 1.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.karma_adapter.mcp_tools import call_tool, fetch_tool_schemas

QWEN3_MODEL_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507"

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess server manager
# ──────────────────────────────────────────────────────────────────────────────


class VLLMServer:
    """Launch and manage a `vllm serve` subprocess.

    Typical lifecycle in a notebook:
        server = VLLMServer(model_id="Qwen/Qwen3-30B-A3B-Thinking-2507")
        server.start()
        server.wait_ready(timeout=600)   # raises with stderr tail if it dies
        # ... use the server via Qwen3Adapter ...
        server.stop()                     # at notebook end / kernel detach
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
    ):
        self.model_id = model_id
        self.host = host
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.extra_args = extra_args or []
        self.log_path = Path(log_path)
        self.env_overrides = env_overrides or {}
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            raise RuntimeError("Server already running")

        cmd = [
            "vllm",
            "serve",
            self.model_id,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-model-len",
            str(self.max_model_len),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--dtype",
            self.dtype,
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
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,   # so SIGINT to notebook doesn't kill us mid-startup
        )

    def wait_ready(self, timeout: int = 600, poll_interval: float = 2.0) -> None:
        """Poll /v1/models until server responds or it dies.

        If the subprocess exits before becoming ready, raise with the tail of
        the log so we see the real error instead of "kernel unresponsive".
        """
        import httpx

        if self.proc is None:
            raise RuntimeError("Server not started")

        deadline = time.time() + timeout
        url = f"{self.base_url}/models"
        while time.time() < deadline:
            # subprocess died?
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
# Adapter (HTTP client over the vLLM server)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Qwen3Response:
    text: str
    n_tool_calls: int
    raw_messages: list[dict[str, Any]]
    stop_reason: str


class Qwen3Adapter:
    """Talks to a `vllm serve` instance over OpenAI-compatible HTTP.

    Tool-call format: Qwen3 emits `<tool_call>{...}</tool_call>` in its
    assistant content. We parse that out, dispatch via MCP, and append the
    result as a follow-up user message wrapped in `<tool_response>` tags.

    We do NOT use vLLM's --enable-auto-tool-choice because that requires a
    tool-call parser that maps to the OpenAI tools API, and Qwen3-Thinking's
    behavior with auto-tool-choice is less well-tested than the native format.
    Text-mode parsing is what Qwen3 was trained on.
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
    ):
        from openai import OpenAI  # lazy import

        self.model_id = model_id
        self.use_tools = use_tools
        self.max_tokens = max_tokens
        self.max_tool_turns = max_tool_turns
        self.temperature = temperature
        self.top_p = top_p
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        if use_tools:
            tools = fetch_tool_schemas()
            self._tool_instructions = _format_tool_instructions(tools)
        else:
            self._tool_instructions = ""

    def run(self, prompt: str, system: str | None = None) -> Qwen3Response:
        sys_text = system or ""
        if self.use_tools and self._tool_instructions:
            sys_text = (sys_text + "\n\n" + self._tool_instructions).strip()

        messages: list[dict[str, Any]] = []
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        messages.append({"role": "user", "content": prompt})

        n_tool_calls = 0
        last_text = ""
        for _ in range(self.max_tool_turns + 1):
            stop_strings = ["</tool_call>"] if self.use_tools else None
            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=stop_strings,
            )
            text = resp.choices[0].message.content or ""
            # vLLM strips the stop string from output by default; re-attach
            # to keep regex parsing uniform.
            if stop_strings and resp.choices[0].finish_reason == "stop" and "<tool_call>" in text:
                text = text + "</tool_call>"
            last_text = text

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
                {"role": "user", "content": f"<tool_response>\n{result}\n</tool_response>"}
            )

        return Qwen3Response(
            text=_strip_thinking(last_text),
            n_tool_calls=n_tool_calls,
            raw_messages=messages,
            stop_reason="max_tool_turns",
        )


def _format_tool_instructions(tools: list[dict[str, Any]]) -> str:
    """Render MedAI tool schemas as a Qwen-style tool-use instruction block."""
    parts = [
        "You have access to the following tools. To call a tool, write a JSON",
        "object inside <tool_call>...</tool_call> tags with keys 'name' and",
        "'arguments'. The tool result will appear as a user message inside",
        "<tool_response>...</tool_response>. Make at most one tool call per turn.",
        "",
        "Available tools:",
    ]
    for t in tools:
        parts.append(f"- {t['name']}: {t.get('description', '').strip()}")
        parts.append(f"  schema: {json.dumps(t.get('input_schema', {}))}")
    return "\n".join(parts)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()
