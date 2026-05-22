# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Qwen3 baseline (Condition B)
# MAGIC
# MAGIC Runs Qwen3-30B-A3B-Thinking-2507 + MedAI tools on `medical_calculator_eval`.
# MAGIC This is the **baseline** for the actual study. Every other Qwen3 condition
# MAGIC (A no-tools, C prompt-instruction, D' post-hoc verify, E interwhen) is
# MAGIC measured against this.
# MAGIC
# MAGIC **Why subprocess vLLM (not in-process):** on Databricks runtimes newer
# MAGIC than the open-source ML ecosystem (DBR 17.x has CUDA 13 / torch 2.11),
# MAGIC in-process vLLM crashes during model load and takes the kernel down. The
# MAGIC subprocess pattern isolates failures and surfaces real error messages.
# MAGIC
# MAGIC **Run cells one at a time, top to bottom. Don't Run All.** Each cell
# MAGIC has a purpose; you want to see what each one does.

# COMMAND ----------
# MAGIC %md ## 1. Install deps

# COMMAND ----------
# MAGIC %pip install -q \
# MAGIC   "vllm==0.9.2" "transformers==4.52.4" \
# MAGIC   "openai>=1.0" "httpx" \
# MAGIC   "fastmcp>=2.0" "datasets>=2.0" "huggingface-hub" \
# MAGIC   "pandas" "numpy" "scipy" "statsmodels" "nest-asyncio" \
# MAGIC   "nvidia-cusparselt-cu12"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md ## 2. Paste secrets (locally only — clear before pushing)

# COMMAND ----------
import os

os.environ["EKA_API_TOKEN"] = ""

# COMMAND ----------
# MAGIC %md ## 3. Pre-flight: GPU visible, libraries findable

# COMMAND ----------
# MAGIC %sh nvidia-smi
# MAGIC echo "---"
# MAGIC ls /databricks/python3/lib/python3.12/site-packages/cusparselt/lib/ 2>/dev/null || echo "cusparselt path not here — check the find below"
# MAGIC find /databricks /local_disk0 -name "libcusparseLt.so.0" 2>/dev/null | head -3

# COMMAND ----------
# MAGIC %md
# MAGIC Make the CUDA libs findable by vLLM. Setting LD_LIBRARY_PATH here applies
# MAGIC to subprocesses we launch (like vllm serve) — that's what we need.

# COMMAND ----------
import os

CUSPARSELT_DIR = "/databricks/python3/lib/python3.12/site-packages/cusparselt/lib"
os.environ["LD_LIBRARY_PATH"] = CUSPARSELT_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")
print("LD_LIBRARY_PATH:", os.environ["LD_LIBRARY_PATH"])

# COMMAND ----------
# MAGIC %md ## 4. Launch vLLM as a background subprocess
# MAGIC
# MAGIC This runs `vllm serve` in its own process. If it crashes, we'll see the
# MAGIC actual error in `/tmp/vllm_server.log` instead of "kernel unresponsive."
# MAGIC
# MAGIC First load downloads Qwen3-30B weights (~60 GB) — expect 10-15 min.

# COMMAND ----------
import harness  # noqa: F401  — also runs _patches for MCP auth
from harness.karma_adapter.qwen3 import VLLMServer

server = VLLMServer(
    model_id="Qwen/Qwen3-30B-A3B-Thinking-2507",
    gpu_memory_utilization=0.80,
    max_model_len=16384,
    env_overrides={
        "VLLM_USE_V1": "0",  # use the more-stable engine on this runtime
        "LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"],
    },
)
server.start()

# COMMAND ----------
# MAGIC %md
# MAGIC Wait for the server to come up. If vllm crashes during startup, this
# MAGIC cell raises with the tail of the log — we finally see the real error.

# COMMAND ----------
server.wait_ready(timeout=900)   # 15 min ceiling for first-time model download

# COMMAND ----------
# MAGIC %md
# MAGIC While we wait, you can tail the log from a separate cell:
# MAGIC ```
# MAGIC %sh tail -f /tmp/vllm_server.log
# MAGIC ```

# COMMAND ----------
# MAGIC %md ## 5. Smoke-test the server with a tiny request

# COMMAND ----------
from openai import OpenAI

client = OpenAI(base_url=server.base_url, api_key="EMPTY")
r = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Thinking-2507",
    messages=[{"role": "user", "content": "Reply with exactly: pong"}],
    max_tokens=20,
)
print("Server OK:", r.choices[0].message.content)

# COMMAND ----------
# MAGIC %md ## 6. Pilot — 10 vignettes with tools

# COMMAND ----------
from harness.karma_adapter.qwen3 import Qwen3Adapter
from harness.runner import run_eval
from harness.analysis import wilson_ci

adapter = Qwen3Adapter(base_url=server.base_url, use_tools=True)
pilot = run_eval(adapter, n=10, max_workers=4, out_dir="/dbfs/results/qwen3_baseline_pilot/")
print(f"Pilot accuracy: {pilot.accuracy:.1%} ({pilot.n_correct}/{pilot.n})")
print(f"Parse failures: {pilot.n_parse_failures}")
print(f"Tool calls per vignette (mean): {pilot.rows['n_tool_calls'].mean():.1f}")

# COMMAND ----------
display(pilot.rows[["id", "primary_field", "expected", "predicted", "correct", "n_tool_calls", "parse_failed", "stop_reason"]])  # noqa: F821

# COMMAND ----------
# MAGIC %md ## 7. Full 1,066-row baseline
# MAGIC
# MAGIC Only run after the pilot looks healthy (tool calls > 0, parse failures
# MAGIC low). Expected runtime ~1-2 hours with max_workers=16 against the server.

# COMMAND ----------
full = run_eval(adapter, n=None, max_workers=16, out_dir="/dbfs/results/qwen3_baseline_full/")
ci = wilson_ci(full.n_correct, full.n)
print(f"Baseline (Condition B) accuracy: {ci}")

# COMMAND ----------
print("By category:")
print(full.rows.groupby("category")["correct"].agg(["mean", "count"]).sort_values("mean"))
print("\nParse-failure rate by category (top 10):")
print(full.rows.groupby("category")["parse_failed"].mean().sort_values(ascending=False).head(10))
print("\nTool-call distribution:")
print(full.rows["n_tool_calls"].describe())

# COMMAND ----------
# MAGIC %md ## 8. Cleanup
# MAGIC
# MAGIC Stop the vLLM subprocess when you're done. Otherwise it keeps holding
# MAGIC the GPU until the cluster idle-terminates.

# COMMAND ----------
server.stop()
print("server stopped:", not server.is_alive())
