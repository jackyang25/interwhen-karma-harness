# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Qwen3 Condition E (literal interwhen)
# MAGIC
# MAGIC Runs Qwen3-30B-A3B-Thinking-2507 + MedAI tools with **the actual
# MAGIC interwhen library** driving the streaming generation, step extraction,
# MAGIC and feedback injection. Our deterministic clinical input verifier is
# MAGIC plugged in as a `VerifyMonitor` subclass.
# MAGIC
# MAGIC Locked design (TESTING.md §4.2/§5/§6, pre_registration.md):
# MAGIC - **Inference loop driver:** `interwhen.stream_completion` (literal
# MAGIC   library, no reimplementation)
# MAGIC - **Monitor:** `harness.monitors.ClinicalInputMonitor` — subclass of
# MAGIC   `interwhen.monitors.base.VerifyMonitor`
# MAGIC - **Fact extractor:** Sonnet 4.6 (`harness.extraction.FactExtractor`)
# MAGIC - **Semantic verifier:** deterministic field comparison
# MAGIC   (`harness.verifier.semantic`)
# MAGIC - **Tool channel:** raw text via vLLM's `/v1/completions`, Qwen3's
# MAGIC   native `<tool_call>` tags as commit boundaries
# MAGIC - **Feedback template:** `conf/prompts/condition_e_feedback.txt`
# MAGIC - **Calculator subset:** all (start broad)
# MAGIC
# MAGIC **Pre-registration gate (TESTING.md §8):** spot-check ~20 extractor
# MAGIC outputs from the pilot below before kicking off the full run. If
# MAGIC field-level accuracy is <95% on the spot-check, refine
# MAGIC `conf/prompts/extractor.txt` and re-pilot before proceeding.

# COMMAND ----------

# MAGIC %md ## 1. Install deps (incl. interwhen, --no-deps to skip its vllm pin)

# COMMAND ----------

# MAGIC %pip install -q \
# MAGIC   "transformers==4.52.4" \
# MAGIC   "openai>=1.0" "httpx" "anthropic" \
# MAGIC   "fastmcp-slim[client]>=2.0" "datasets>=2.0" "huggingface-hub" \
# MAGIC   "pandas" "numpy" "scipy" "statsmodels" "nest-asyncio" \
# MAGIC   "nvidia-cusparselt-cu12" \
# MAGIC   "pydantic>=2.0" "regex" "tiktoken" "z3-solver" "sympy"
# MAGIC %pip install -q --no-deps \
# MAGIC   "interwhen @ git+https://github.com/microsoft/interwhen.git@2d041c2f3ed2a6f0a4b063463b3aef844e7dba5e"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## 2. Paste secrets (Qwen3 via vLLM, Sonnet for extractor)

# COMMAND ----------

import os

os.environ["EKA_API_TOKEN"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""

# COMMAND ----------

# MAGIC %md ## 3. Pre-flight

# COMMAND ----------

# MAGIC %sh nvidia-smi
# MAGIC echo "---"
# MAGIC find /databricks /local_disk0 -name "libcusparseLt.so.0" 2>/dev/null | head -3

# COMMAND ----------

import os

CUSPARSELT_DIR = "/databricks/python3/lib/python3.12/site-packages/cusparselt/lib"
os.environ["LD_LIBRARY_PATH"] = CUSPARSELT_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")
print("LD_LIBRARY_PATH:", os.environ["LD_LIBRARY_PATH"])

# COMMAND ----------

# MAGIC %md ## 4. Launch vLLM (same venv-based pattern as other conditions)

# COMMAND ----------

# MAGIC %sh
# MAGIC set -e
# MAGIC rm -rf /tmp/vllm_env
# MAGIC apt-get update && apt-get install -y python3-venv
# MAGIC python3 -m venv /tmp/vllm_env
# MAGIC /tmp/vllm_env/bin/python -m ensurepip --upgrade
# MAGIC /tmp/vllm_env/bin/pip install --quiet --upgrade pip
# MAGIC /tmp/vllm_env/bin/pip install --quiet \
# MAGIC   vllm==0.9.2 transformers==4.52.4 openai httpx
# MAGIC ls /tmp/vllm_env/bin/vllm && echo "venv ready"

# COMMAND ----------

try:
    server.stop()
except NameError:
    pass

import os, subprocess, textwrap, pathlib

_ossl_conf = pathlib.Path("/tmp/openssl_nofips.cnf")
_ossl_conf.write_text(textwrap.dedent("""\
    openssl_conf = openssl_init

    [openssl_init]
    providers = provider_sect

    [provider_sect]
    default = default_sect

    [default_sect]
    activate = 1
"""))

subprocess.run(
    ["/tmp/vllm_env/bin/pip", "uninstall", "-y", "opencv-python-headless"],
    capture_output=True, timeout=60,
)
print("opencv-python-headless removed from venv")

os.environ.pop("_PIP_USE_IMPORTLIB_METADATA", None)
os.chdir("/tmp")

from harness.karma_adapter.qwen3 import VLLMServer

server = VLLMServer(
    model_id="Qwen/Qwen3-30B-A3B-Thinking-2507",
    gpu_memory_utilization=0.80,
    max_model_len=16384,
    vllm_bin="/tmp/vllm_env/bin/vllm",
    env_overrides={
        "VLLM_USE_V1": "0",
        "LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"],
        "OPENSSL_CONF": str(_ossl_conf),
        "OPENSSL_MODULES": "",
        "OPENSSL_FORCE_FIPS_MODE": "0",
    },
)
server.start()

# COMMAND ----------

server.wait_ready(timeout=900)

# COMMAND ----------

# MAGIC %md ## 5. Smoke test (text endpoint + Anthropic)

# COMMAND ----------

import httpx
import anthropic

# /v1/completions check — this is the endpoint interwhen will drive.
r = httpx.post(
    f"{server.base_url}/completions",
    json={
        "model": "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "prompt": "Reply with exactly: pong",
        "max_tokens": 20,
        "temperature": 0.0,
    },
    timeout=60,
)
print("vLLM /v1/completions OK:", r.status_code, r.json()["choices"][0]["text"].strip()[:60])

ac = anthropic.Anthropic()
r2 = ac.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=20,
    messages=[{"role": "user", "content": "Reply with exactly: pong"}],
)
print("Anthropic OK:", r2.content[0].text)

# COMMAND ----------

# MAGIC %md ## 6. Confirm interwhen import works

# COMMAND ----------

import interwhen
from interwhen.monitors.base import VerifyMonitor
from interwhen import stream_completion

print(f"interwhen loaded: {interwhen.__file__}")
print(f"VerifyMonitor:    {VerifyMonitor}")
print(f"stream_completion: {stream_completion}")

# COMMAND ----------

# MAGIC %md ## 7. Wire fact extractor + ClinicalInputMonitor

# COMMAND ----------

import pathlib

_repo_root = pathlib.Path("/Workspace/Users/jack.yang@gatesfoundation.org/interwhen-karma-harness")

from harness.extraction import FactExtractor
from harness.monitors import ClinicalInputMonitor
from harness.karma_adapter.mcp_tools import fetch_tool_schemas

extractor = FactExtractor(
    prompt_path=str(_repo_root / "conf/prompts/extractor.txt"),
    model="claude-sonnet-4-6",
)
feedback_template = (_repo_root / "conf/prompts/condition_e_feedback.txt").read_text()
TOOL_SCHEMAS = fetch_tool_schemas()
print(f"Extractor: {extractor.model}")
print(f"Tools: {len(TOOL_SCHEMAS)} MedAI tools loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Adapter that drives interwhen + the monitor
# MAGIC
# MAGIC This adapter exposes the same `.run(prompt, system=...)` interface as
# MAGIC the other conditions so the runner doesn't change. Internally it:
# MAGIC 1. Extracts patient facts (Sonnet)
# MAGIC 2. Builds a per-vignette `ClinicalInputMonitor` with those facts
# MAGIC 3. Renders the chatml prompt via the Qwen3 tokenizer
# MAGIC 4. Calls `interwhen.stream_completion` with the monitor — that drives
# MAGIC    the streaming generation, fires `verify` at each tool-call commit,
# MAGIC    invokes `fix` to inject feedback on violations, and recurses
# MAGIC 5. Returns the final text + monitor metrics

# COMMAND ----------

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import nest_asyncio
nest_asyncio.apply()

from transformers import AutoTokenizer

QWEN3_MODEL_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507"
_TOKENIZER = AutoTokenizer.from_pretrained(QWEN3_MODEL_ID)


@dataclass
class CondEResponse:
    text: str
    n_tool_calls: int          # reflected from monitor.metrics.n_steps_seen
    raw_messages: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "stop"
    prompt_tokens: int = 0     # not tracked at fine granularity in interwhen streaming
    completion_tokens: int = 0
    n_model_calls: int = 0
    # E-specific
    n_verifier_fires: int = 0
    n_fixes_applied: int = 0
    extractor_ok: bool = True
    extractor_error: str = ""
    violations_history: list[dict[str, Any]] = field(default_factory=list)


class Qwen3InterwhenAdapter:
    def __init__(self, base_url, extractor, feedback_template, tool_schemas, model_id=QWEN3_MODEL_ID,
                 max_tokens=4096, temperature=0.0):
        self.base_url = base_url.rstrip("/")
        self.completions_url = f"{self.base_url}/completions"
        self.extractor = extractor
        self.feedback_template = feedback_template
        self.tool_schemas = tool_schemas
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    def run(self, prompt: str, system: str | None = None) -> CondEResponse:
        # 1) Extract patient facts (one Sonnet call per vignette)
        facts = self.extractor.extract(prompt)
        prompt_tokens = facts.prompt_tokens
        completion_tokens = facts.completion_tokens

        # 2) Build the rendered chatml prompt for Qwen3
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        rendered = _TOKENIZER.apply_chat_template(
            messages,
            tools=self.tool_schemas,
            add_generation_prompt=True,
            tokenize=False,
        )

        # 3) Build per-vignette monitor with the patient facts
        monitor = ClinicalInputMonitor(
            patient_facts=facts,
            feedback_template=self.feedback_template,
        )

        # 4) Configure llm_server for interwhen (it POSTs to /v1/completions with streaming)
        llm_server = {
            "url": self.completions_url,
            "headers": {"Content-Type": "application/json"},
            "payload": {
                "model": self.model_id,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": True,
                # interwhen's stream_completion sets payload["prompt"] itself.
            },
        }

        # 5) Drive interwhen (async) from our sync run() — nest_asyncio lets us
        # do this inside a notebook's running event loop.
        loop = asyncio.get_event_loop()
        generated = loop.run_until_complete(
            stream_completion(
                prompt=rendered,
                prev_text="",
                llm_server=llm_server,
                monitors=[monitor],
                async_execution=True,
            )
        )

        # 6) Pull the assistant's final text out of the generated continuation
        text = _extract_final_answer(generated)

        return CondEResponse(
            text=text,
            n_tool_calls=monitor.metrics.n_steps_seen,
            stop_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            n_model_calls=1,   # extractor; interwhen streams so we don't recount Qwen3
            n_verifier_fires=monitor.metrics.n_verifier_fires,
            n_fixes_applied=monitor.metrics.n_fixes_applied,
            extractor_ok=facts.extractor_ok,
            extractor_error=facts.extractor_error,
            violations_history=monitor.metrics.violations_history,
        )


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _extract_final_answer(generated: str) -> str:
    """Strip <think> blocks and chatml turn markers, return the visible answer."""
    text = generated
    # Drop everything up to the last <|im_start|>assistant boundary if present
    if "<|im_start|>assistant" in text:
        text = text.rsplit("<|im_start|>assistant", 1)[-1]
    text = text.split("<|im_end|>", 1)[0]
    text = _THINK_RE.sub("", text)
    return text.strip()


adapter = Qwen3InterwhenAdapter(
    base_url=server.base_url,
    extractor=extractor,
    feedback_template=feedback_template,
    tool_schemas=TOOL_SCHEMAS,
)
print("Condition E adapter wired (literal interwhen + ClinicalInputMonitor).")

# COMMAND ----------

# MAGIC %md ## 9. Pilot — 10 vignettes through the full E pipeline
# MAGIC
# MAGIC Sanity checks before the full run:
# MAGIC 1. `extractor_ok = True` on most/all rows
# MAGIC 2. Some `n_verifier_fires > 0` rows (else the verifier never engages)
# MAGIC 3. interwhen handles the stream correctly (no asyncio errors)
# MAGIC 4. Spot-check 3-5 rows: extractor output vs vignette by hand → ≥95% field-level accuracy gate (§8)

# COMMAND ----------

from harness.runner import run_eval
from harness.analysis import wilson_ci

pilot = run_eval(adapter, n=10, max_workers=2, out_dir="/dbfs/results/qwen3_condition_E_pilot/")
print(f"Pilot accuracy: {pilot.accuracy:.1%} ({pilot.n_correct}/{pilot.n})")
print(f"Parse failures: {pilot.n_parse_failures}")
print(f"Tool calls per vignette (mean): {pilot.rows['n_tool_calls'].mean():.1f}")

# COMMAND ----------

import pandas as pd
pilot_df = pd.read_parquet("/dbfs/results/qwen3_condition_E_pilot/rows.parquet")
display(pilot_df[["id", "primary_field", "expected", "predicted", "correct", "n_tool_calls", "parse_failed"]])  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 10. Full 1,066-row Condition E
# MAGIC
# MAGIC Only run after spot-checking extractor outputs. ~2-3 hours expected
# MAGIC (slowest condition — extractor per vignette + interwhen streaming +
# MAGIC potential retries on violations).

# COMMAND ----------

full = run_eval(adapter, n=None, max_workers=16, out_dir="/dbfs/results/qwen3_condition_E_full/")
ci = wilson_ci(full.n_correct, full.n)
print(f"Condition E (interwhen + semantic verifier) accuracy: {ci}")
print()
print("--- Cost / time ---")
print(f"Total wall-clock for full run: {full.total_run_seconds/60:.1f} min")
print(f"Median latency per vignette:   {full.median_latency_seconds:.2f} s")
print(f"Mean prompt tokens/vignette:   {full.mean_prompt_tokens:.0f}  (extractor only — interwhen streaming not counted at fine grain)")
print(f"Mean output tokens/vignette:   {full.mean_completion_tokens:.0f}")
print(f"Est. USD for 1,066 vignettes (@ $3/hr H100 + Sonnet API extractor): ${full.estimated_cost_usd(3.0):.2f} GPU only")

# COMMAND ----------

print("By category:")
print(full.rows.groupby("category")["correct"].agg(["mean", "count"]).sort_values("mean"))
print("\nTool-call distribution:")
print(full.rows["n_tool_calls"].describe())

# COMMAND ----------

# MAGIC %md ## 11. Cleanup

# COMMAND ----------

server.stop()
print("server stopped:", not server.is_alive())
