# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Qwen3 baseline (Condition B)
# MAGIC
# MAGIC Runs Qwen3-30B-A3B-Thinking-2507 + MedAI tools on `medical_calculator_eval`.
# MAGIC This is the **reference baseline** for the actual study — every other
# MAGIC Qwen3 condition (A no-tools, C prompt-instruction, D' post-hoc verify,
# MAGIC E interwhen) is measured against this.
# MAGIC
# MAGIC Requires an H100 80GB cluster. vLLM loads Qwen3-30B in bf16 (~60 GB
# MAGIC weights), leaves ~16 GB headroom for KV cache + shared neighbors.
# MAGIC
# MAGIC Two stages, same pattern as notebook 01:
# MAGIC 1. **Pilot (n=10)** — validates text-mode tool parsing end-to-end.
# MAGIC 2. **Full (n=1066)** — the baseline accuracy.

# COMMAND ----------
# MAGIC %pip install -q \
# MAGIC   "vllm>=0.6.0" "transformers>=4.45" \
# MAGIC   "fastmcp>=2.0" "datasets>=2.0" "huggingface-hub" \
# MAGIC   "pandas" "numpy" "scipy" "statsmodels" "nest-asyncio"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# Paste your MedAI token locally before running. ANTHROPIC_API_KEY is not
# needed for Qwen3. Do NOT commit values back.
import os

os.environ["EKA_API_TOKEN"] = ""

# COMMAND ----------
# MAGIC %sh nvidia-smi
# MAGIC echo "---"
# MAGIC echo "Check VRAM is mostly free before loading Qwen3-30B (~60 GB needed)."

# COMMAND ----------
import harness  # noqa: F401
from harness.karma_adapter import Qwen3Adapter
from harness.runner import run_eval
from harness.analysis import wilson_ci

# COMMAND ----------
# MAGIC %md ### Load Qwen3 into GPU memory (~5-10 min first time)
# MAGIC
# MAGIC HuggingFace downloads the weights to the cluster's local cache. Subsequent
# MAGIC runs in the same cluster session reuse them. If the cluster terminates,
# MAGIC the weights are re-downloaded next session.

# COMMAND ----------
adapter = Qwen3Adapter(use_tools=True)

# COMMAND ----------
# MAGIC %md ### Stage 1 — 10-vignette pilot
# MAGIC
# MAGIC Goal: confirm Qwen3 actually emits `<tool_call>` blocks the parser can
# MAGIC catch, and that the tool result injection produces sensible final answers.
# MAGIC If tool calls are 0, parsing is broken or the chat template isn't
# MAGIC rendering tools. If many parse failures, the model isn't following the
# MAGIC confinement_instruction JSON format.

# COMMAND ----------
pilot = run_eval(adapter, n=10, max_workers=1, out_dir="/dbfs/results/qwen3_baseline_pilot/")
print(f"Pilot accuracy: {pilot.accuracy:.1%} ({pilot.n_correct}/{pilot.n})")
print(f"Parse failures: {pilot.n_parse_failures}")
print(f"Tool calls per vignette (mean): {pilot.rows['n_tool_calls'].mean():.1f}")

# COMMAND ----------
display(pilot.rows[["id", "primary_field", "expected", "predicted", "correct", "n_tool_calls", "parse_failed", "stop_reason"]])  # noqa: F821

# COMMAND ----------
# MAGIC %md ### Stage 2 — full 1,066-row baseline
# MAGIC
# MAGIC Expected: ~3-5 hours sequential at ~10-20s per vignette. vLLM batches
# MAGIC internally; cross-vignette parallelism via max_workers > 1 doesn't help
# MAGIC (and is not thread-safe with vLLM's offline LLM). Keep the cell running
# MAGIC to prevent the 60-min idle timeout.

# COMMAND ----------
full = run_eval(adapter, n=None, max_workers=1, out_dir="/dbfs/results/qwen3_baseline_full/")
ci = wilson_ci(full.n_correct, full.n)
print(f"Baseline (Condition B) accuracy: {ci}")
print(f"EkaCare published Qwen-class with-tools numbers are mid-70s — this is")
print(f"our reference, not a target. Whatever this is, it's what we improve on.")

# COMMAND ----------
# MAGIC %md ### Stratified breakdowns

# COMMAND ----------
print("By category:")
print(full.rows.groupby("category")["correct"].agg(["mean", "count"]).sort_values("mean"))
print("\nParse-failure rate by category (top 10):")
print(full.rows.groupby("category")["parse_failed"].mean().sort_values(ascending=False).head(10))
print("\nTool-call distribution:")
print(full.rows["n_tool_calls"].describe())
