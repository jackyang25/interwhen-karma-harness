"""Qwen3-30B-A3B-Thinking-2507 adapter via subprocess vLLM server.

Why subprocess rather than in-process vLLM? On Databricks runtimes newer than
the open-source ML ecosystem (CUDA 13 / torch 2.11 era), vLLM crashes during
model load and takes the Python kernel down with it. Running vLLM as a
separate process gives us crash isolation and real stderr we can read.

**Tool-calling methodology.** We use vLLM's native structured tool API
(`--enable-auto-tool-choice --tool-call-parser hermes`) and talk to it with
OpenAI's `tools=[...]` argument. This mirrors how Sonnet got tools (via
Anthropic's structured Tools API) and matches what each model's native
tool-use channel looks like — the same comparison structure as EkaCare's
published table. Hermes-style parsing fits Qwen3, which was trained on this
format.

Threading: the HTTP client is thread-safe; vLLM batches across concurrent
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

# Default vLLM args to enable native structured tool-calling. The hermes parser
# matches Qwen3's training format. `--enable-auto-tool-choice` lets the model
# choose to invoke tools through the OpenAI tools API channel.
DEFAULT_VLLM_TOOL_ARGS = ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]

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
        vllm_bin: str = "vllm",
    ):
        """
        vllm_bin: path to the `vllm` executable. Defaults to PATH lookup. Set
            to /tmp/vllm_env/bin/vllm (or wherever) to run vllm from an isolated
            venv that doesn't see Databricks's preinstalled TensorFlow / FIPS
            libs.
        """
        self.model_id = model_id
        self.host = host
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        # Default to enabling structured tool-calling. Pass extra_args=[] to opt out.
        self.extra_args = (
            list(DEFAULT_VLLM_TOOL_ARGS) if extra_args is None else list(extra_args)
        )
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
            self.vllm_bin,
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

    Tool-calling uses the OpenAI structured tools API. vLLM is launched with
    `--enable-auto-tool-choice --tool-call-parser hermes`, which lets Qwen3
    use its native trained tool-call format while exposing it through the
    standard OpenAI `tools=[...]` / `tool_calls` interface.

    This mirrors how the Sonnet adapter works (Anthropic's structured Tools
    API): each model gets its native, fine-tuned tool-use channel, so the
    A/B/C/D'/E comparisons stand on the same methodological footing as
    EkaCare's published numbers.
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
        self.tools = _to_openai_tools(fetch_tool_schemas()) if use_tools else None

    def run(self, prompt: str, system: str | None = None) -> Qwen3Response:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        n_tool_calls = 0
        last_text = ""
        for _ in range(self.max_tool_turns + 1):
            kwargs: dict[str, Any] = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            if self.tools:
                kwargs["tools"] = self.tools
                kwargs["tool_choice"] = "auto"

            resp = self.client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            msg = choice.message
            text = msg.content or ""
            last_text = text

            tool_calls = msg.tool_calls or []
            # Append the assistant message — include tool_calls if present so
            # the model can refer back to its own request.
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": text}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)

            if not tool_calls:
                return Qwen3Response(
                    text=_strip_thinking(text),
                    n_tool_calls=n_tool_calls,
                    raw_messages=messages,
                    stop_reason=choice.finish_reason or "stop",
                )

            for tc in tool_calls:
                n_tool_calls += 1
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    result = call_tool(tc.function.name, args)
                except Exception as e:
                    result = f"Tool error: {type(e).__name__}: {e}"
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

        return Qwen3Response(
            text=_strip_thinking(last_text),
            n_tool_calls=n_tool_calls,
            raw_messages=messages,
            stop_reason="max_tool_turns",
        )


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic-style MCP tool schemas to OpenAI tools format.

    fetch_tool_schemas() returns dicts shaped as
        {"name": ..., "description": ..., "input_schema": {...}}
    OpenAI's chat completions API expects
        {"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}
    """
    out = []
    for t in tools:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", "") or "",
                    "parameters": t.get("input_schema", {}) or {"type": "object", "properties": {}},
                },
            }
        )
    return out


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()
