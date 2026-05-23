# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Cross-condition analysis
# MAGIC
# MAGIC Loads the per-row parquets from each condition (A, B, C, D', E, B') and
# MAGIC runs the §7 statistical plan:
# MAGIC
# MAGIC **Primary (Bonferroni family, α = 0.05 / 4 = 0.0125):**
# MAGIC - B vs A — Did tools help?
# MAGIC - E vs B — Did verification beat tool access alone? (foundational)
# MAGIC - E vs C — Did verification beat best-effort prompt?
# MAGIC - E vs D' — Did mid-stream beat post-hoc verifier?
# MAGIC
# MAGIC **Secondary (exploratory, uncorrected α = 0.05):**
# MAGIC - B' vs B — Did forced tool use close the underuse gap?
# MAGIC
# MAGIC Plus: cost/accuracy Pareto plot, per-category breakdowns, verifier
# MAGIC characterization for E, stratified analysis on the tool-using subset.

# COMMAND ----------

# MAGIC %pip install -q pandas numpy scipy statsmodels matplotlib
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
from pathlib import Path
from harness.analysis import wilson_ci, mcnemar, bonferroni

# COMMAND ----------

# MAGIC %md ## 1. Load all condition results

# COMMAND ----------

RESULTS_DIR = Path("/dbfs/results")

CONDITIONS = {
    "A":      "qwen3_condition_A_full",
    "B":      "qwen3_condition_B_full",     # if you re-ran with instrumentation
    "C":      "qwen3_condition_C_full",
    "D_prime": "qwen3_condition_D_prime_full",
    "E":      "qwen3_condition_E_full",
    "B_prime": "qwen3_condition_B_prime_full",
}

# Fallback: if B wasn't re-run with instrumentation, fall back to the legacy
# path. Cost/time columns will be missing for B in that case.
LEGACY_FALLBACKS = {
    "B": "qwen3_baseline_full",
}

dfs: dict[str, pd.DataFrame] = {}
for cond, path in CONDITIONS.items():
    full_path = RESULTS_DIR / path / "rows.parquet"
    if not full_path.exists():
        fallback = LEGACY_FALLBACKS.get(cond)
        if fallback:
            full_path = RESULTS_DIR / fallback / "rows.parquet"
    if full_path.exists():
        dfs[cond] = pd.read_parquet(full_path)
        print(f"{cond}: loaded {len(dfs[cond])} rows from {full_path}")
    else:
        print(f"{cond}: missing parquet at {full_path}")

# COMMAND ----------

# MAGIC %md ## 2. Per-condition accuracy table (Wilson CIs)

# COMMAND ----------

rows = []
for cond, df in dfs.items():
    ci = wilson_ci(int(df["correct"].sum()), len(df))
    rows.append(
        {
            "condition": cond,
            "n": ci.n,
            "n_correct": int(df["correct"].sum()),
            "accuracy": ci.accuracy,
            "ci_low": ci.lo,
            "ci_high": ci.hi,
            "parse_failures": int(df["parse_failed"].sum()),
            "mean_tool_calls": df["n_tool_calls"].mean(),
        }
    )
accuracy_table = pd.DataFrame(rows)
display(accuracy_table)  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 3. Primary comparisons (McNemar paired + Bonferroni)
# MAGIC
# MAGIC McNemar's test requires the same questions in both arms (paired by `id`).

# COMMAND ----------

def paired_correctness(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Align two condition DataFrames on `id` and return paired correctness arrays."""
    merged = df_a[["id", "correct"]].rename(columns={"correct": "a"}).merge(
        df_b[["id", "correct"]].rename(columns={"correct": "b"}),
        on="id",
        how="inner",
    )
    return merged["a"].to_numpy(), merged["b"].to_numpy()


primary_results: dict[str, dict] = {}

for label, (left, right) in {
    "B_vs_A": ("B", "A"),
    "E_vs_B": ("E", "B"),
    "E_vs_C": ("E", "C"),
    "E_vs_Dprime": ("E", "D_prime"),
}.items():
    if left not in dfs or right not in dfs:
        primary_results[label] = {"status": "missing condition data"}
        continue
    x, y = paired_correctness(dfs[left], dfs[right])
    res = mcnemar(x, y)
    primary_results[label] = {
        "b_x_correct_y_wrong": res.b,
        "c_x_wrong_y_correct": res.c,
        "uncorrected_p": res.pvalue,
        "n_paired": len(x),
        "accuracy_left": float(x.mean()),
        "accuracy_right": float(y.mean()),
        "delta_pp": float((x.mean() - y.mean()) * 100),
    }

# Bonferroni on the family of 4 confirmatory comparisons.
uncorrected_pvals = {
    k: v["uncorrected_p"]
    for k, v in primary_results.items()
    if "uncorrected_p" in v
}
adjusted = bonferroni(uncorrected_pvals, n_tests=4)
for k, p_adj in adjusted.items():
    primary_results[k]["bonferroni_p"] = p_adj
    primary_results[k]["significant_at_0.05"] = p_adj < 0.05

primary_df = pd.DataFrame(primary_results).T
display(primary_df)  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 4. Secondary comparison (B' vs B, exploratory)

# COMMAND ----------

if "B_prime" in dfs and "B" in dfs:
    x, y = paired_correctness(dfs["B_prime"], dfs["B"])
    res = mcnemar(x, y)
    print(f"B' vs B (exploratory, uncorrected α = 0.05):")
    print(f"  n paired: {len(x)}")
    print(f"  B' accuracy: {x.mean():.3f}")
    print(f"  B  accuracy: {y.mean():.3f}")
    print(f"  Delta:       {(x.mean() - y.mean()) * 100:+.2f} pp")
    print(f"  McNemar:     b={res.b}, c={res.c}, p={res.pvalue:.4g}")
    print(f"  Significant at α=0.05 (uncorrected): {res.pvalue < 0.05}")
else:
    print("B' or B missing — cannot compute secondary comparison")

# COMMAND ----------

# MAGIC %md ## 5. Stratified analysis — effect on the tool-using subset
# MAGIC
# MAGIC The primary input-verification interventions (C, D', E) can only act on
# MAGIC rows where the baseline (B) used at least one tool. Reporting verifier
# MAGIC effect on this subset is the honest reading of the hypothesis.

# COMMAND ----------

if "B" in dfs:
    b_tool_using_ids = set(dfs["B"][dfs["B"]["n_tool_calls"] > 0]["id"])
    print(f"Tool-using subset (B): {len(b_tool_using_ids)} / {len(dfs['B'])} rows")

    sub_rows = []
    for cond in ["A", "B", "C", "D_prime", "E"]:
        if cond not in dfs:
            continue
        sub = dfs[cond][dfs[cond]["id"].isin(b_tool_using_ids)]
        ci = wilson_ci(int(sub["correct"].sum()), len(sub))
        sub_rows.append(
            {
                "condition": cond,
                "n_subset": ci.n,
                "accuracy_on_subset": ci.accuracy,
                "ci_low": ci.lo,
                "ci_high": ci.hi,
            }
        )
    display(pd.DataFrame(sub_rows))  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 6. Cost / latency Pareto (LMIC deployment view)

# COMMAND ----------

cost_rows = []
for cond, df in dfs.items():
    if "elapsed_seconds" not in df.columns:
        cost_rows.append({"condition": cond, "note": "instrumentation missing (legacy parquet)"})
        continue
    cost_rows.append(
        {
            "condition": cond,
            "accuracy": df["correct"].mean(),
            "mean_total_tokens": df["total_tokens"].mean() if "total_tokens" in df.columns else None,
            "median_latency_s": df["elapsed_seconds"].median(),
            "mean_model_calls": df["n_model_calls"].mean() if "n_model_calls" in df.columns else None,
        }
    )
cost_df = pd.DataFrame(cost_rows)
display(cost_df)  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC Quick Pareto plot — accuracy vs cost-proxy (total tokens). Conditions in
# MAGIC the upper-left are the deployment sweet spots.

# COMMAND ----------

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
plot_df = cost_df.dropna(subset=["accuracy", "mean_total_tokens"])
ax.scatter(plot_df["mean_total_tokens"], plot_df["accuracy"], s=80)
for _, r in plot_df.iterrows():
    ax.annotate(r["condition"], (r["mean_total_tokens"], r["accuracy"]),
                xytext=(5, 5), textcoords="offset points")
ax.set_xlabel("Mean total tokens per vignette (cost proxy)")
ax.set_ylabel("Accuracy")
ax.set_title("Cost / accuracy Pareto frontier — LMIC deployment view")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ## 7. Per-category accuracy across conditions

# COMMAND ----------

cat_df = pd.DataFrame()
for cond, df in dfs.items():
    cat_acc = df.groupby("category")["correct"].mean().rename(cond)
    cat_df = pd.concat([cat_df, cat_acc], axis=1)
display(cat_df.round(3))  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 8. Verifier characterization (Condition E)
# MAGIC
# MAGIC When did the verifier fire? Did its firings correlate with corrections?
# MAGIC The full violations history lives in the per-row response object, not
# MAGIC the parquet — refer to the saved raw_output column for individual rows.

# COMMAND ----------

if "E" in dfs:
    e = dfs["E"]
    print(f"Total rows: {len(e)}")
    print(f"E accuracy:        {e['correct'].mean():.1%}")
    if "B" in dfs:
        b_correct_by_id = dict(zip(dfs["B"]["id"], dfs["B"]["correct"]))
        merged = e[["id", "correct"]].copy()
        merged["b_correct"] = merged["id"].map(b_correct_by_id)
        flips_to_correct = merged[(~merged["b_correct"]) & (merged["correct"])]
        flips_to_wrong = merged[(merged["b_correct"]) & (~merged["correct"])]
        print(f"E flipped wrong→right: {len(flips_to_correct)} rows")
        print(f"E flipped right→wrong: {len(flips_to_wrong)} rows (verifier false positives that derailed correct answers)")
        print(f"Net: {len(flips_to_correct) - len(flips_to_wrong):+d}")
