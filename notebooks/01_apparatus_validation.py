# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Apparatus validation
# MAGIC
# MAGIC Reproduce EkaCare's Claude Sonnet 4.6 + tools = **81.9%** on
# MAGIC `medical_calculator_eval` (within ±3pp) through this harness. This is
# MAGIC the gate before any experimental runs (§8 of TESTING.md).
# MAGIC
# MAGIC The notebook runs in two stages:
# MAGIC 1. **Pilot (n=10)** — proves the wiring works end-to-end. Costs cents.
# MAGIC 2. **Full (n=1066)** — the real apparatus check. Only run after the pilot
# MAGIC    looks sane.

# COMMAND ----------
# MAGIC %pip install -q \
# MAGIC   "karma-medeval @ git+https://github.com/eka-care/KARMA-OpenMedEvalKit.git@d3fb194acba00aa014a89d48671b402c4cff8e85" \
# MAGIC   "interwhen @ git+https://github.com/microsoft/interwhen.git@2d041c2f3ed2a6f0a4b063463b3aef844e7dba5e" \
# MAGIC   "anthropic>=0.40" "fastmcp>=2.0" "datasets>=2.0" "huggingface-hub" \
# MAGIC   "pandas" "numpy" "scipy" "statsmodels"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
import os

try:
    os.environ["ANTHROPIC_API_KEY"] = dbutils.secrets.get("apikeys", "anthropic")  # noqa: F821
    os.environ["EKA_API_TOKEN"] = dbutils.secrets.get("apikeys", "eka")  # noqa: F821
except Exception:
    assert "ANTHROPIC_API_KEY" in os.environ and "EKA_API_TOKEN" in os.environ

# COMMAND ----------
import harness  # noqa: F401 — applies _patches on import
from harness.karma_adapter import SonnetAdapter
from harness.runner import run_eval
from harness.analysis import wilson_ci

# COMMAND ----------
# MAGIC %md ### Stage 1 — 10-vignette pilot

# COMMAND ----------
adapter = SonnetAdapter(model="claude-sonnet-4-6", use_tools=True)
pilot = run_eval(adapter, n=10, out_dir="/dbfs/results/apparatus_pilot/")
print(f"Pilot accuracy: {pilot.accuracy:.1%} ({pilot.n_correct}/{pilot.n})")
print(f"Parse failures: {pilot.n_parse_failures}")
print(f"Tool calls per vignette (mean): {pilot.rows['n_tool_calls'].mean():.1f}")

# COMMAND ----------
# MAGIC %md
# MAGIC **Sanity checks before going full-scale:**
# MAGIC - Pilot accuracy in a plausible range (this is n=10, so noisy — but if it's
# MAGIC   0% something is fundamentally broken).
# MAGIC - `parse_failures` should be 0 or very low. High parse-failure rate means
# MAGIC   the model isn't returning JSON correctly → fix prompt/system before
# MAGIC   spending on the full run.
# MAGIC - `n_tool_calls > 0` — confirms MCP tool-use is actually happening.

# COMMAND ----------
display(pilot.rows[["id", "primary_field", "expected", "predicted", "correct", "n_tool_calls", "parse_failed"]])  # noqa: F821

# COMMAND ----------
# MAGIC %md ### Stage 2 — Full 1,066-vignette run
# MAGIC
# MAGIC Only run after pilot looks healthy. Expected duration: ~2–4 hours at
# MAGIC ~10s/vignette with tool calls. Keep the cluster alive — long-running cell
# MAGIC counts as activity for the 60-min idle timeout.

# COMMAND ----------
full = run_eval(adapter, n=None, out_dir="/dbfs/results/apparatus_full/")
ci = wilson_ci(full.n_correct, full.n)
print(f"Full accuracy: {ci}")
print(f"Target: 81.9% ± 3pp (i.e., 78.9% – 84.9%)")

# COMMAND ----------
# MAGIC %md ### Gate decision

# COMMAND ----------
TARGET_LO, TARGET_HI = 0.789, 0.849
if TARGET_LO <= full.accuracy <= TARGET_HI:
    print(f"APPARATUS VALIDATED ({full.accuracy:.1%}). Safe to proceed to baseline.")
else:
    print(f"APPARATUS FAILED ({full.accuracy:.1%} outside [{TARGET_LO:.1%}, {TARGET_HI:.1%}]).")
    print("Do NOT proceed to Qwen3 runs until this is reconciled. Likely causes:")
    print("  - prompt format diverges from EkaCare's setup")
    print("  - MedAI tools returning errors / wrong calculator subset")
    print("  - scorer parsing JSON incorrectly")
    print("Inspect rows.parquet and adapter_error / parse_failed columns.")

# COMMAND ----------
# MAGIC %md
# MAGIC Stratified breakdowns to help debug if the gate fails.

# COMMAND ----------
print("By category:")
print(full.rows.groupby("category")["correct"].agg(["mean", "count"]).sort_values("mean"))
print("\nParse-failure rate by category:")
print(full.rows.groupby("category")["parse_failed"].mean().sort_values(ascending=False).head(10))
