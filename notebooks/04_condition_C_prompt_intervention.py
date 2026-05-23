# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Qwen3 Condition C (best-effort prompt intervention)
# MAGIC
# MAGIC Runs Qwen3-30B-A3B-Thinking-2507 + MedAI tools on
# MAGIC `medical_calculator_eval` with a **best-effort prompt instruction**
# MAGIC asking the model to verify tool-call inputs (units, enums, numerics)
# MAGIC against the case before calling any calculator. Locked prompt lives at
# MAGIC `conf/prompts/condition_c.txt` (see pre-registration commit).
# MAGIC
# MAGIC **Sections 1–5 are identical to notebooks 02 and 03** so the only
# MAGIC difference between this run and Conditions A/B is the `system=` prompt
# MAGIC passed to `run_eval`. Same model, same server, same adapter, same scorer,
# MAGIC same parallelism. The C−B accuracy delta is therefore a clean measurement
# MAGIC of the prompt-instruction effect.
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
# MAGIC Same setup as notebooks 02 and 03. Server config identical so the
# MAGIC C vs B (and C vs A) comparisons are unconfounded.

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

# MAGIC %md ## 6. Load locked Condition C prompt
# MAGIC
# MAGIC The system prompt for Condition C is the locked text in
# MAGIC `conf/prompts/condition_c.txt`. Reading it from disk (rather than
# MAGIC inlining) means the git-committed prompt is the single source of truth;
# MAGIC any change shows up as a diff on that file.

# COMMAND ----------

import pathlib

# Locate the repo root from the Databricks Git Folder mount.
_repo_root = pathlib.Path("/Workspace/Users/jack.yang@gatesfoundation.org/interwhen-karma-harness")
CONDITION_C_SYSTEM = (_repo_root / "conf/prompts/condition_c.txt").read_text().strip()
print("Loaded Condition C prompt:")
print("-" * 60)
print(CONDITION_C_SYSTEM)
print("-" * 60)

# COMMAND ----------

# MAGIC %md ## 7. Pilot — 10 vignettes with C prompt
# MAGIC
# MAGIC Sanity check: tool calls per vignette should be at least as high as B
# MAGIC (mean 0.38 in the B baseline). If C is dramatically lower, the prompt
# MAGIC may be suppressing tool use — fix before launching the full run.

# COMMAND ----------

from harness.karma_adapter.qwen3 import Qwen3Adapter
from harness.runner import run_eval
from harness.analysis import wilson_ci

adapter = Qwen3Adapter(base_url=server.base_url, use_tools=True)
pilot = run_eval(
    adapter,
    n=10,
    max_workers=4,
    system=CONDITION_C_SYSTEM,
    out_dir="/dbfs/results/qwen3_condition_C_pilot/",
)
print(f"Pilot accuracy: {pilot.accuracy:.1%} ({pilot.n_correct}/{pilot.n})")
print(f"Parse failures: {pilot.n_parse_failures}")
print(f"Tool calls per vignette (mean): {pilot.rows['n_tool_calls'].mean():.1f}  (B baseline: 0.38)")

# COMMAND ----------

display(pilot.rows[["id", "primary_field", "expected", "predicted", "correct", "n_tool_calls", "parse_failed", "stop_reason"]])  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 8. Full 1,066-row Condition C

# COMMAND ----------

# 64 workers: same as Condition B since C also has the tool-call loop. Keeping
# parallelism identical to B ensures the C vs B comparison is unconfounded by
# any throughput-related effects.
full = run_eval(
    adapter,
    n=None,
    max_workers=64,
    system=CONDITION_C_SYSTEM,
    out_dir="/dbfs/results/qwen3_condition_C_full/",
)
ci = wilson_ci(full.n_correct, full.n)
print(f"Condition C (best-effort prompt) accuracy: {ci}")
print()
print("--- Cost / time (LMIC deployment proxies) ---")
print(f"Total wall-clock for full run: {full.total_run_seconds/60:.1f} min")
print(f"Median latency per vignette:   {full.median_latency_seconds:.2f} s")
print(f"Mean prompt tokens/vignette:   {full.mean_prompt_tokens:.0f}")
print(f"Mean output tokens/vignette:   {full.mean_completion_tokens:.0f}")
print(f"Mean total tokens/vignette:    {full.mean_total_tokens:.0f}")
print(f"Est. USD for 1,066 vignettes (@ $3/hr H100): ${full.estimated_cost_usd(3.0):.2f}")
print(f"Est. USD per 1,000 vignettes:                ${full.estimated_cost_usd(3.0) * 1000 / full.n:.2f}")

# COMMAND ----------

print("By category:")
print(full.rows.groupby("category")["correct"].agg(["mean", "count"]).sort_values("mean"))
print("\nParse-failure rate by category (top 10):")
print(full.rows.groupby("category")["parse_failed"].mean().sort_values(ascending=False).head(10))
print("\nTool-call distribution:")
print(full.rows["n_tool_calls"].describe())

# COMMAND ----------

# MAGIC %md ## 9. Cleanup

# COMMAND ----------

server.stop()
print("server stopped:", not server.is_alive())
