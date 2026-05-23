# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Qwen3 Condition B (with tools)
# MAGIC
# MAGIC Runs Qwen3-30B-A3B-Thinking-2507 + MedAI tools on `medical_calculator_eval`.
# MAGIC The reference condition for the study — every other Qwen3 condition
# MAGIC (A no-tools, C prompt-instruction, D' post-hoc, E interwhen) is
# MAGIC compared against this.
# MAGIC
# MAGIC **vLLM is launched as a subprocess** in an isolated venv so its crashes
# MAGIC show real errors instead of taking the notebook kernel down. The venv
# MAGIC also avoids the FIPS-enabled OpenSSL bundled with opencv on DBR 17.x
# MAGIC Azure images.
# MAGIC
# MAGIC **Run All should work top-to-bottom.**

# COMMAND ----------

# MAGIC %md ## 1. Install deps

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

# MAGIC %md ## 2. Paste secrets

# COMMAND ----------

import os

os.environ["EKA_API_TOKEN"] = ""

# COMMAND ----------

# MAGIC %md ## 3. Pre-flight (GPU + cusparseLt path)

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

# MAGIC %md
# MAGIC ## 4. Create vLLM venv & launch server
# MAGIC
# MAGIC Builds a fresh virtualenv, removes opencv (which bundles FIPS-enabled OpenSSL
# MAGIC 1.1.1k that aborts vLLM), then launches `vllm serve` as a subprocess.

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

# Stop any zombie server from a previous run.
try:
    server.stop()
except NameError:
    pass

import os, subprocess, textwrap, pathlib

# --- Write a minimal openssl.cnf that loads ONLY the default provider ---
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

# --- Uninstall opencv-python-headless from the venv ---
# opencv bundles a FIPS-enabled OpenSSL 1.1.1k that triggers the FIPS self-test
# failure. vLLM doesn't need cv2 for text inference. Removing the package avoids
# both the FIPS abort and any missing-library errors.
subprocess.run(
    ["/tmp/vllm_env/bin/pip", "uninstall", "-y", "opencv-python-headless"],
    capture_output=True, timeout=60,
)
print("opencv-python-headless removed from venv")

os.environ.pop("_PIP_USE_IMPORTLIB_METADATA", None)

# Change cwd to /tmp so vLLM's stat() on relative model paths doesn't hit
# Databricks's restricted working directory.
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

# MAGIC %md ## 5. Server smoke test

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
pilot = run_eval(adapter, n=10, max_workers=4, out_dir="/dbfs/results/qwen3_condition_B_pilot/")
print(f"Pilot accuracy: {pilot.accuracy:.1%} ({pilot.n_correct}/{pilot.n})")
print(f"Parse failures: {pilot.n_parse_failures}")
print(f"Tool calls per vignette (mean): {pilot.rows['n_tool_calls'].mean():.1f}")

# COMMAND ----------

display(pilot.rows[["id", "primary_field", "expected", "predicted", "correct", "n_tool_calls", "parse_failed", "stop_reason"]])  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 7. Full 1,066-row Condition B

# COMMAND ----------

# 64 workers: matches what a single H100 with Qwen3-30B-A3B-Thinking can
# realistically batch via vLLM's continuous-batching scheduler. Higher just
# queues HTTP requests without GPU speedup; lower under-utilizes the GPU.
full = run_eval(adapter, n=None, max_workers=64, out_dir="/dbfs/results/qwen3_condition_B_full/")
ci = wilson_ci(full.n_correct, full.n)
print(f"Condition B (with tools) accuracy: {ci}")

# COMMAND ----------

print("By category:")
print(full.rows.groupby("category")["correct"].agg(["mean", "count"]).sort_values("mean"))
print("\nParse-failure rate by category (top 10):")
print(full.rows.groupby("category")["parse_failed"].mean().sort_values(ascending=False).head(10))
print("\nTool-call distribution:")
print(full.rows["n_tool_calls"].describe())

# COMMAND ----------

# MAGIC %md ## 8. Cleanup

# COMMAND ----------

server.stop()
print("server stopped:", not server.is_alive())
