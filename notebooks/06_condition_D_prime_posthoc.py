# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Qwen3 Condition D' (post-hoc verifier)
# MAGIC
# MAGIC Runs Qwen3-30B-A3B-Thinking-2507 + MedAI tools with a **post-hoc
# MAGIC verifier**: after Qwen3 produces an answer, a separate Sonnet-4.6
# MAGIC verifier inspects (case, answer) and flags inconsistencies; on flag,
# MAGIC Qwen3 revises once.
# MAGIC
# MAGIC Locked design (TESTING.md §6, pre_registration.md):
# MAGIC - Verifier model: **Claude Sonnet 4.6** (different from Qwen3 → no
# MAGIC   self-agreement bias)
# MAGIC - Verifier prompt: `conf/prompts/condition_d_prime.txt`
# MAGIC - Revision policy: **one revision attempt** on flag
# MAGIC - Trigger: every primary answer (no skip)
# MAGIC
# MAGIC E vs D' is one of the primary confirmatory comparisons — "did
# MAGIC mid-stream verification beat the cheapest verification pattern?" D'
# MAGIC is that cheapest pattern: one extra API call after the answer.

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
# MAGIC
# MAGIC ANTHROPIC_API_KEY is required for the Sonnet verifier.

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

# MAGIC %md ## 4. Create vLLM venv & launch server

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

# MAGIC %md ## 5. Smoke test (vLLM + Anthropic)

# COMMAND ----------

from openai import OpenAI

client = OpenAI(base_url=server.base_url, api_key="EMPTY")
r = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Thinking-2507",
    messages=[{"role": "user", "content": "Reply with exactly: pong"}],
    max_tokens=20,
)
print("vLLM OK:", r.choices[0].message.content)

import anthropic
ac = anthropic.Anthropic()
r2 = ac.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=20,
    messages=[{"role": "user", "content": "Reply with exactly: pong"}],
)
print("Anthropic OK:", r2.content[0].text)

# COMMAND ----------

# MAGIC %md ## 6. Wire primary adapter + post-hoc verifier

# COMMAND ----------

import pathlib

_repo_root = pathlib.Path("/Workspace/Users/jack.yang@gatesfoundation.org/interwhen-karma-harness")

from harness.karma_adapter.qwen3 import Qwen3Adapter
from harness.verifier import PostHocVerifierAdapter

primary = Qwen3Adapter(base_url=server.base_url, use_tools=True)
adapter = PostHocVerifierAdapter(
    primary=primary,
    verifier_prompt_path=str(_repo_root / "conf/prompts/condition_d_prime.txt"),
    verifier_model="claude-sonnet-4-6",
)
print("D' adapter wired (Qwen3 primary + Sonnet post-hoc verifier).")

# COMMAND ----------

# MAGIC %md ## 7. Pilot — 10 vignettes with D' workflow
# MAGIC
# MAGIC Sanity check: each vignette runs Qwen3 → Sonnet verifier → (revise if
# MAGIC flagged). Expect at least a few verifier flags on n=10 if the verifier
# MAGIC is working. If 0 flags, verifier may be too lenient — review the
# MAGIC prompt before full run.

# COMMAND ----------

from harness.runner import run_eval
from harness.analysis import wilson_ci

pilot = run_eval(
    adapter,
    n=10,
    max_workers=4,
    # D' uses DEFAULT_SYSTEM (same as B) — the intervention is the verifier
    # wrapping, not a different prompt.
    out_dir="/dbfs/results/qwen3_condition_D_prime_pilot/",
)
print(f"Pilot accuracy: {pilot.accuracy:.1%} ({pilot.n_correct}/{pilot.n})")
print(f"Parse failures: {pilot.n_parse_failures}")
print(f"Tool calls per vignette (mean): {pilot.rows['n_tool_calls'].mean():.1f}")
print(f"Model calls per vignette (mean): {pilot.rows['n_model_calls'].mean():.1f}  (includes verifier + revisions)")

# COMMAND ----------

display(pilot.rows[["id", "primary_field", "expected", "predicted", "correct", "n_tool_calls", "n_model_calls", "parse_failed"]])  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 8. Full 1,066-row Condition D'

# COMMAND ----------

# max_workers=32: D' adds an extra Sonnet API call per vignette (and maybe a
# revision = another Qwen3 call). Anthropic enterprise rate limits are
# generous but contention with Qwen3 GPU + tool MCP at 64 workers gets noisy.
# 32 keeps the pipeline balanced.
full = run_eval(
    adapter,
    n=None,
    max_workers=32,
    out_dir="/dbfs/results/qwen3_condition_D_prime_full/",
)
ci = wilson_ci(full.n_correct, full.n)
print(f"Condition D' (post-hoc verifier) accuracy: {ci}")
print()
print("--- Cost / time (LMIC deployment proxies) ---")
print(f"Total wall-clock for full run: {full.total_run_seconds/60:.1f} min")
print(f"Median latency per vignette:   {full.median_latency_seconds:.2f} s")
print(f"Mean prompt tokens/vignette:   {full.mean_prompt_tokens:.0f}  (includes verifier + revisions)")
print(f"Mean output tokens/vignette:   {full.mean_completion_tokens:.0f}")
print(f"Mean model calls/vignette:     {full.rows['n_model_calls'].mean():.2f}")
print(f"Est. USD for 1,066 vignettes (@ $3/hr H100 + Sonnet API): ${full.estimated_cost_usd(3.0):.2f} GPU only")
print("  (Add Anthropic API cost separately — see Anthropic billing for verifier + revision calls.)")

# COMMAND ----------

print("By category:")
print(full.rows.groupby("category")["correct"].agg(["mean", "count"]).sort_values("mean"))
print("\nModel calls distribution (1 = primary only, 2 = primary + verifier, 3+ = with revision):")
print(full.rows["n_model_calls"].describe())

# COMMAND ----------

# MAGIC %md ## 9. Cleanup

# COMMAND ----------

server.stop()
print("server stopped:", not server.is_alive())
