# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Smoke test
# MAGIC
# MAGIC Validates the bridge between this repo (cloned via Git Folder) and the
# MAGIC Databricks cluster:
# MAGIC 1. `pip install -e .` succeeds inside the notebook
# MAGIC 2. KARMA, interwhen, and `harness` all import
# MAGIC 3. Anthropic API responds
# MAGIC 4. MedAI MCP responds and lists tools
# MAGIC 5. GPU is visible (if attached to an H100 cluster)
# MAGIC
# MAGIC Pass this entire notebook before running `01_apparatus_validation.py`.

# COMMAND ----------
# MAGIC %md
# MAGIC Install external dependencies. The `harness` package itself is on `sys.path`
# MAGIC automatically because this notebook lives inside the Git Folder — no
# MAGIC `pip install -e` needed (and editable installs fail on Workspace paths
# MAGIC because pip can't write build artifacts there).

# COMMAND ----------
# MAGIC %md
# MAGIC `interwhen` is intentionally omitted here — it has `vllm` as a hard dep
# MAGIC and won't install cleanly on CPU/Serverless. It gets installed in the
# MAGIC GPU notebooks (Condition E onward) once we're on an H100.
# MAGIC
# MAGIC `karma-medeval` is similarly heavy (pulls torch, transformers); it's
# MAGIC needed once we plug into KARMA's CLI but not for apparatus validation.
# MAGIC Skip it here too — the apparatus validation only needs Anthropic + MCP.

# COMMAND ----------
# MAGIC %pip install -q \
# MAGIC   "anthropic>=0.40" "fastmcp>=2.0" "datasets>=2.0" "huggingface-hub" \
# MAGIC   "pandas" "numpy" "scipy" "statsmodels" "nest-asyncio"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# Paste keys locally in Databricks before running this cell.
# Do NOT commit this cell with keys filled in — leave the values empty when
# you push back to git so future pulls don't merge-conflict with your local
# edits.
import os

os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["EKA_API_TOKEN"] = ""

# COMMAND ----------
# MAGIC %md ### 1. Import packages

# COMMAND ----------
import harness

print("harness:", harness.__file__)
# karma and interwhen are not installed at this stage — see install cell above.

# COMMAND ----------
# MAGIC %md ### 2. Anthropic API ping

# COMMAND ----------
import anthropic

client = anthropic.Anthropic()
r = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=50,
    messages=[{"role": "user", "content": "Reply with exactly: pong"}],
)
print("Anthropic OK:", r.content[0].text)

# COMMAND ----------
# MAGIC %md ### 3. MedAI MCP — list tools
# MAGIC
# MAGIC `harness._patches` patches FastMCPClient to inject EKA_API_TOKEN. If this
# MAGIC fails with 401/403, the token is wrong or expired.

# COMMAND ----------
from harness.karma_adapter.mcp_tools import fetch_tool_schemas

tools = fetch_tool_schemas()
print(f"MedAI OK: {len(tools)} tools available")
for t in tools[:5]:
    print(f"  - {t['name']}: {t['description'][:80]}")

# COMMAND ----------
# MAGIC %md ### 4. GPU visibility (only meaningful on H100 cluster)

# COMMAND ----------
# MAGIC %sh nvidia-smi || echo "no GPU on this cluster (fine for apparatus validation)"

# COMMAND ----------
# MAGIC %md
# MAGIC If every cell above succeeded, the apparatus is wired. Proceed to
# MAGIC `01_apparatus_validation.py` next.
