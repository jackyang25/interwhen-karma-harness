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
# MAGIC %pip install -q \
# MAGIC   "karma-medeval @ git+https://github.com/eka-care/KARMA-OpenMedEvalKit.git@d3fb194acba00aa014a89d48671b402c4cff8e85" \
# MAGIC   "interwhen @ git+https://github.com/microsoft/interwhen.git@2d041c2f3ed2a6f0a4b063463b3aef844e7dba5e" \
# MAGIC   "anthropic>=0.40" "fastmcp>=2.0" "datasets>=2.0" "huggingface-hub" \
# MAGIC   "pandas" "numpy" "scipy" "statsmodels"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import os

# Pull secrets from Databricks if available, else expect env vars to be set.
# Replace "apikeys" with whatever scope name you create via:
#   databricks secrets create-scope apikeys
try:
    os.environ["ANTHROPIC_API_KEY"] = dbutils.secrets.get("apikeys", "anthropic")  # noqa: F821
    os.environ["EKA_API_TOKEN"] = dbutils.secrets.get("apikeys", "eka")  # noqa: F821
    print("Secrets loaded from Databricks scope 'apikeys'")
except Exception as e:
    print(f"Could not load Databricks secrets ({e}). Falling back to os.environ.")
    assert "ANTHROPIC_API_KEY" in os.environ, "ANTHROPIC_API_KEY missing"
    assert "EKA_API_TOKEN" in os.environ, "EKA_API_TOKEN missing"

# COMMAND ----------
# MAGIC %md ### 1. Import packages

# COMMAND ----------
import karma
import interwhen
import harness

print("karma:", karma.__file__)
print("interwhen:", interwhen.__file__)
print("harness:", harness.__file__)

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
