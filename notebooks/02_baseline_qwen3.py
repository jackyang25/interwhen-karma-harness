# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Qwen3 baseline (Condition B)
# MAGIC
# MAGIC Runs Qwen3-30B-A3B-Thinking-2507 + MedAI tools on `medical_calculator_eval`.
# MAGIC The reference baseline for the actual study.
# MAGIC
# MAGIC **vLLM is launched as a subprocess** so its crashes show real errors
# MAGIC instead of taking the notebook kernel down. DBR 17.x's FIPS-enabled
# MAGIC OpenSSL conflicts with vLLM's bundled deps; this notebook has two
# MAGIC attempts at working around it (Path A and Path B). Try A first; if it
# MAGIC dies with the FIPS error, switch to Path B.
# MAGIC
# MAGIC **Run cells one at a time, top to bottom. Don't Run All.**

# COMMAND ----------
# MAGIC %md ## 1. Install deps (system Python — used by Path A)

# COMMAND ----------
# MAGIC %pip install -q \
# MAGIC   "vllm==0.9.2" "transformers==4.52.4" \
# MAGIC   "openai>=1.0" "httpx" \
# MAGIC   "fastmcp>=2.0" "datasets>=2.0" "huggingface-hub" \
# MAGIC   "pandas" "numpy" "scipy" "statsmodels" "nest-asyncio" \
# MAGIC   "nvidia-cusparselt-cu12"
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
# MAGIC # Path A — system Python with TF + OpenSSL bypass
# MAGIC
# MAGIC Try this first. We:
# MAGIC - Tell vLLM to skip platform auto-detection so it doesn't import TensorFlow
# MAGIC - Strip TF-related env so anything that does import TF is quiet
# MAGIC - Tell OpenSSL not to load provider modules (disables FIPS provider entirely)
# MAGIC
# MAGIC If this works → great, no venv needed. If it dies with the same FIPS
# MAGIC error → skip to Path B below.

# COMMAND ----------
import harness  # noqa: F401
from harness.karma_adapter.qwen3 import VLLMServer

server = VLLMServer(
    model_id="Qwen/Qwen3-30B-A3B-Thinking-2507",
    gpu_memory_utilization=0.80,
    max_model_len=16384,
    env_overrides={
        "VLLM_USE_V1": "0",
        "LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"],
        # Skip vLLM's platform auto-detection — it imports TF and other
        # frameworks, which on Databricks ML pulls in FIPS-conflicting libs.
        "VLLM_PLATFORM": "cuda",
        # Quiet TF if anything still loads it
        "TF_CPP_MIN_LOG_LEVEL": "3",
        # OpenSSL: prevent provider modules (incl. the FIPS provider) from
        # loading at all. The FIPS self-test failure happens during provider
        # init; if no provider loads, no self-test runs.
        "OPENSSL_MODULES": "",
        "OPENSSL_NO_DEFAULT_CONFIG": "1",
        "OPENSSL_CONF": "",
        "OPENSSL_FORCE_FIPS_MODE": "0",
    },
)
server.start()

# COMMAND ----------
server.wait_ready(timeout=900)

# COMMAND ----------
# MAGIC %md
# MAGIC If `wait_ready` raised with the FIPS error again, stop here and jump
# MAGIC down to Path B. If it raised with a DIFFERENT error, paste it — we
# MAGIC can iterate. If it succeeded, skip Path B and go to section 5.

# COMMAND ----------
# MAGIC %md
# MAGIC # Path B — isolated venv (only run if Path A failed)
# MAGIC
# MAGIC Create a fresh Python virtualenv with just vllm + deps. The venv won't
# MAGIC have Databricks's preinstalled TensorFlow or other FIPS-conflicting
# MAGIC libraries, so vLLM's startup doesn't trigger FIPS provider loading.

# COMMAND ----------
# MAGIC %sh
# MAGIC set -e
# MAGIC rm -rf /tmp/vllm_env
# MAGIC python3 -m venv /tmp/vllm_env
# MAGIC /tmp/vllm_env/bin/pip install --quiet --upgrade pip
# MAGIC /tmp/vllm_env/bin/pip install --quiet \
# MAGIC   vllm==0.9.2 transformers==4.52.4 openai httpx
# MAGIC ls /tmp/vllm_env/bin/vllm && echo "venv ready"

# COMMAND ----------
# Stop any zombie server from Path A.
try:
    server.stop()
except NameError:
    pass

server = VLLMServer(
    model_id="Qwen/Qwen3-30B-A3B-Thinking-2507",
    gpu_memory_utilization=0.80,
    max_model_len=16384,
    vllm_bin="/tmp/vllm_env/bin/vllm",
    env_overrides={
        "VLLM_USE_V1": "0",
        "LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"],
        "OPENSSL_MODULES": "",
        "OPENSSL_NO_DEFAULT_CONFIG": "1",
        "OPENSSL_CONF": "",
    },
)
server.start()

# COMMAND ----------
server.wait_ready(timeout=900)

# COMMAND ----------
# MAGIC %md ## 5. Server smoke test (run after EITHER path succeeds)

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

# COMMAND ----------
server.stop()
print("server stopped:", not server.is_alive())
