# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Cross-condition analysis
# MAGIC
# MAGIC Loads the per-row parquets from each condition (A, B, C, D, E, B') and
# MAGIC runs the §7 statistical plan:
# MAGIC
# MAGIC **Primary (Bonferroni family, α = 0.05 / 4 = 0.0125):**
# MAGIC - B vs A — Did tools help?
# MAGIC - E vs B — Did verification beat tool access alone? (foundational)
# MAGIC - E vs C — Did verification beat best-effort prompt?
# MAGIC - E vs D — Did mid-stream beat post-hoc verifier?
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

# DBTITLE 1,Load all condition results
RESULTS_DIR = Path("/dbfs/results")

CONDITIONS = {
    "A":         "qwen3_condition_A_full",
    "B":         "qwen3_condition_B_full",
    "C":         "qwen3_condition_C_full",
    "D":         "qwen3_condition_D_full",
    "E":         "qwen3_condition_E_full",
    "B_prime":   "qwen3_condition_B_prime_full",
    # Exploratory (post-hoc, like B'). Tests whether interwhen helps once
    # the tool-use ceiling is removed. Kept out of the pre-registered
    # Bonferroni family; compared in a separate exploratory block below.
    "B_prime_E":          "qwen3_condition_B_prime_E_full",
    # Exploratory follow-up: same as B_prime_E but with reactive per-tool-call
    # focused fact extraction instead of upfront 500-field extraction. Tests
    # whether extractor schema size was the bottleneck in B_prime_E's harm.
    "B_prime_E_reactive": "qwen3_condition_B_prime_E_reactive_full",
}

# One-time migration: rename legacy D_prime dirs to D
for _old, _new in [
    ("qwen3_condition_D_prime_full",  "qwen3_condition_D_full"),
    ("qwen3_condition_D_prime_pilot", "qwen3_condition_D_pilot"),
]:
    if (RESULTS_DIR / _old).exists() and not (RESULTS_DIR / _new).exists():
        (RESULTS_DIR / _old).rename(RESULTS_DIR / _new)
        print(f"Migrated {_old} \u2192 {_new}")

dfs: dict[str, pd.DataFrame] = {}
for cond, path in CONDITIONS.items():
    full_path = RESULTS_DIR / path / "rows.parquet"
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
    "E_vs_D": ("E", "D"),
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

# MAGIC %md ## 4b. Exploratory comparisons (B_prime_E, post-hoc)
# MAGIC
# MAGIC B_prime_E is a second post-hoc condition (interwhen verifier on top of
# MAGIC forced tool use). Reported as exploratory — NOT included in the
# MAGIC pre-registered Bonferroni family. Headline: B_prime_E vs B_prime
# MAGIC ("does interwhen help once the tool-use ceiling is removed?").

# COMMAND ----------

exploratory_results: dict[str, dict] = {}

for label, (left, right) in {
    "B_prime_E_vs_B_prime":                 ("B_prime_E", "B_prime"),  # headline: does interwhen add value once tools are forced?
    "B_prime_E_vs_E":                       ("B_prime_E", "E"),         # secondary: does forcing tools rescue interwhen?
    # Reactive-extraction follow-up comparisons (post-hoc):
    "B_prime_E_reactive_vs_B_prime_E":      ("B_prime_E_reactive", "B_prime_E"),  # does reactive extraction fix the B_prime_E harm?
    "B_prime_E_reactive_vs_B_prime":        ("B_prime_E_reactive", "B_prime"),    # does the verifier-with-reactive-extraction help vs no verifier?
}.items():
    if left not in dfs or right not in dfs:
        exploratory_results[label] = {"status": f"missing condition data ({left} or {right} not loaded)"}
        print(f"{label}: missing condition data ({left} or {right} not loaded)")
        continue
    x, y = paired_correctness(dfs[left], dfs[right])
    _res = mcnemar(x, y)
    exploratory_results[label] = {
        "n_paired":                       int(len(x)),
        "accuracy_left":                  float(x.mean()),
        "accuracy_right":                 float(y.mean()),
        "delta_pp":                       float((x.mean() - y.mean()) * 100),
        "mcnemar_b":                      int(_res.b),
        "mcnemar_c":                      int(_res.c),
        "p_uncorrected":                  float(_res.pvalue),
        "significant_at_0.05_uncorrected": bool(_res.pvalue < 0.05),
    }
    r = exploratory_results[label]
    print(f"{label} (exploratory, uncorrected α = 0.05):")
    print(f"  n paired:  {r['n_paired']}")
    print(f"  {left} accuracy:  {r['accuracy_left']:.3f}")
    print(f"  {right} accuracy: {r['accuracy_right']:.3f}")
    print(f"  Delta:     {r['delta_pp']:+.2f} pp")
    print(f"  McNemar:   b={r['mcnemar_b']}, c={r['mcnemar_c']}, p={r['p_uncorrected']:.4g}")
    print(f"  Significant at α=0.05 (uncorrected): {r['significant_at_0.05_uncorrected']}")

# COMMAND ----------

# MAGIC %md ## 5. Stratified analysis — effect on the tool-using subset
# MAGIC
# MAGIC The primary input-verification interventions (C, D, E) can only act on
# MAGIC rows where the baseline (B) used at least one tool. Reporting verifier
# MAGIC effect on this subset is the honest reading of the hypothesis.

# COMMAND ----------

if "B" in dfs:
    b_tool_using_ids = set(dfs["B"][dfs["B"]["n_tool_calls"] > 0]["id"])
    print(f"Tool-using subset (B): {len(b_tool_using_ids)} / {len(dfs['B'])} rows")

    sub_rows = []
    for cond in ["A", "B", "C", "D", "E", "B_prime_E", "B_prime_E_reactive"]:
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

def _col_or_zero(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] if present, else a zero-filled Series of the same length.
    Lets us derive api_/on_gpu_ aggregates uniformly across legacy (no breakout
    columns) and new (extractor_/qwen3_/primary_/verifier_) schemas."""
    if col in df.columns:
        return df[col]
    return pd.Series([0] * len(df), index=df.index)


cost_rows = []
for cond, df in dfs.items():
    if "elapsed_seconds" not in df.columns:
        cost_rows.append({"condition": cond, "note": "instrumentation missing (legacy parquet)"})
        continue
    # API-side (Anthropic billing): extractor (E, B_prime_E) + verifier (D).
    # Both go to 0 for A/B/C/B' where neither column exists.
    api_prompt     = _col_or_zero(df, "extractor_prompt_tokens")     + _col_or_zero(df, "verifier_prompt_tokens")
    api_completion = _col_or_zero(df, "extractor_completion_tokens") + _col_or_zero(df, "verifier_completion_tokens")
    api_elapsed    = _col_or_zero(df, "extractor_elapsed_s")         + _col_or_zero(df, "verifier_elapsed_s")
    # On-GPU (vLLM): everything in `prompt_tokens` / `completion_tokens` that
    # wasn't API-side. Works for both legacy schemas (api_*=0 → on_gpu_*=total)
    # and new schemas where breakouts add up to the total.
    on_gpu_prompt     = _col_or_zero(df, "prompt_tokens")     - api_prompt
    on_gpu_completion = _col_or_zero(df, "completion_tokens") - api_completion

    cost_rows.append(
        {
            "condition": cond,
            "accuracy": df["correct"].mean(),
            "mean_total_tokens": df["total_tokens"].mean() if "total_tokens" in df.columns else None,
            "median_latency_s": df["elapsed_seconds"].median(),
            "mean_model_calls": df["n_model_calls"].mean() if "n_model_calls" in df.columns else None,
            # Deployment-honest cost split: what's billed by Anthropic vs what
            # ran on the local GPU. Zero for A/B/C/B' (pure Qwen3), populated
            # for D (verifier), E and B_prime_E (extractor).
            "mean_api_prompt_tokens":      float(api_prompt.mean()),
            "mean_api_completion_tokens":  float(api_completion.mean()),
            "mean_api_elapsed_s":          float(api_elapsed.mean()),
            "mean_on_gpu_prompt_tokens":     float(on_gpu_prompt.mean()),
            "mean_on_gpu_completion_tokens": float(on_gpu_completion.mean()),
        }
    )
cost_df = pd.DataFrame(cost_rows)
display(cost_df)  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 6.1 Pareto plot
# MAGIC
# MAGIC Accuracy vs cost-proxy (total tokens). Conditions in the upper-left are
# MAGIC the deployment sweet spots.

# COMMAND ----------

# DBTITLE 1,Pareto plot (accuracy vs total tokens)
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

# MAGIC %md ## 8. Verifier characterization (E and B_prime_E)
# MAGIC
# MAGIC When did the verifier fire? Did its firings correlate with corrections?
# MAGIC As of the post-2026-05 patch, the full violations history is exported
# MAGIC as a JSON-encoded column in the per-row parquet (`violations_history`).
# MAGIC Legacy E parquets that pre-date the patch will lack the column; the
# MAGIC block below handles both cases.

# COMMAND ----------

import json as _vh_json

verifier_summaries: dict[str, dict] = {}

for cond in ["E", "B_prime_E", "B_prime_E_reactive"]:
    if cond not in dfs:
        continue
    df = dfs[cond]
    print(f"\n── {cond} ──")
    print(f"Total rows: {len(df)}")
    print(f"{cond} accuracy: {df['correct'].mean():.1%}")

    # B as baseline for flip counts (same as the original E reconciliation logic)
    if "B" in dfs:
        b_correct_by_id = dict(zip(dfs["B"]["id"], dfs["B"]["correct"]))
        merged = df[["id", "correct"]].copy()
        merged["b_correct"] = merged["id"].map(b_correct_by_id)
        flips_to_correct = merged[(~merged["b_correct"]) & ( merged["correct"])]
        flips_to_wrong   = merged[( merged["b_correct"]) & (~merged["correct"])]
        print(f"{cond} flipped wrong→right vs B: {len(flips_to_correct)} rows")
        print(f"{cond} flipped right→wrong vs B: {len(flips_to_wrong)} rows")
        print(f"Net vs B: {len(flips_to_correct) - len(flips_to_wrong):+d}")

    # Structured verifier-event analysis from the violations_history column.
    if "violations_history" not in df.columns:
        verifier_summaries[cond] = {"note": "violations_history not in parquet (pre-patch run)"}
        print("  (violations_history column not present — legacy parquet; skipping structured analysis)")
        continue

    # Parse the JSON-encoded list of intervention events per row.
    parsed = df["violations_history"].apply(
        lambda s: _vh_json.loads(s) if isinstance(s, str) else (s or [])
    )
    intervention_mask = parsed.apply(lambda lst: len(lst) > 0)
    n_intervention_rows = int(intervention_mask.sum())

    # Count total violations and tally by flagged field.
    total_violations = int(
        parsed.apply(lambda lst: sum(len(e.get("violations", [])) for e in lst)).sum()
    )
    field_counts: dict[str, int] = {}
    for entries in parsed:
        for entry in entries:
            for v in entry.get("violations", []):
                f = v.get("field", "unknown")
                field_counts[f] = field_counts.get(f, 0) + 1

    # Of rows where the verifier intervened, how many ended correct?
    if n_intervention_rows > 0:
        n_correct_on_intervention = int(df.loc[intervention_mask, "correct"].sum())
        pct_correct = n_correct_on_intervention / n_intervention_rows * 100
    else:
        n_correct_on_intervention = 0
        pct_correct = 0.0

    summary = {
        "n_rows":                              int(len(df)),
        "n_intervention_rows":                 n_intervention_rows,
        "pct_intervention_rows":               float(n_intervention_rows / max(len(df), 1) * 100),
        "total_violations":                    total_violations,
        "mean_violations_per_intervention_row": float(total_violations / max(n_intervention_rows, 1)),
        "n_correct_on_intervention_rows":      n_correct_on_intervention,
        "pct_correct_on_intervention_rows":    float(pct_correct),
        "violations_by_field":                 dict(sorted(field_counts.items(), key=lambda kv: -kv[1])),
    }
    verifier_summaries[cond] = summary
    print(f"  Verifier intervention rows: {n_intervention_rows} ({summary['pct_intervention_rows']:.1f}%)")
    print(f"  Total violations:           {total_violations}")
    print(f"  Of intervened rows, % correct: {summary['pct_correct_on_intervention_rows']:.1f}%")
    if field_counts:
        top = list(summary["violations_by_field"].items())[:5]
        print(f"  Top flagged fields: {', '.join(f'{f}={c}' for f, c in top)}")

# COMMAND ----------

# DBTITLE 1,Export — bundle all results + analysis into one zip
import hashlib, json, shutil, subprocess, zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO    = Path("/Workspace/Users/jack.yang@gatesfoundation.org/interwhen-karma-harness")
OUT     = Path("/tmp/karma_export")
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def git_sha(repo):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unavailable"

# COMMAND ----------

# DBTITLE 1,Export §1 — provenance
# ── 1. Provenance ─────────────────────────────────────────────────────────────
prov_dir = OUT / "provenance"
prov_dir.mkdir()

print("[1/9] provenance...")

# pip freeze
(prov_dir / "pip_freeze.txt").write_text(
    subprocess.check_output(["pip", "freeze"]).decode()
)

# run records from orchestrator
agg = json.loads(Path("/dbfs/results/_AGGREGATED_RESULTS.json").read_text())
run_records = agg.get("run_records", [])

# cluster / runtime info
try:
    dbr   = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion", "unknown")  # noqa: F821
    cname = spark.conf.get("spark.databricks.clusterUsageTags.clusterName", "unknown")   # noqa: F821
except Exception:
    dbr, cname = "unknown", "unknown"

provenance = {
    "exported_at":           datetime.now(timezone.utc).isoformat(),
    "harness_git_sha":       git_sha(REPO),
    "interwhen_git_sha":     "2d041c2f3ed2a6f0a4b063463b3aef844e7dba5e",  # pinned in 02_run_all Cell 3
    "dbr_version":           dbr,
    "cluster_name":          cname,
    "gpu_type":              "Standard_NC40ads_H100_v5 (H100 80GB, Azure)",
    "model_id":              "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "extractor_model_E":     "claude-sonnet-4-6",
    "dataset":               "ekacare/medical_calculator_eval",
    "dataset_split":         "test",
    "n_vignettes":           1066,
    "run_records":           run_records,
}
(prov_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
print("  provenance.json")

# MCP calculator schemas snapshot — the preflight in 02_run_all dumps this to
# /dbfs/results/provenance/mcp_calculator_schemas.json. It's the source of truth
# for the verifier's vocabulary on this run; without it the bundle can't be
# replayed exactly (EkaCare may have added calculators since).
_mcp_schemas_src = Path("/dbfs/results/provenance/mcp_calculator_schemas.json")
if _mcp_schemas_src.exists():
    shutil.copy(_mcp_schemas_src, prov_dir / "mcp_calculator_schemas.json")
    print("  mcp_calculator_schemas.json")
else:
    print("  WARNING: mcp_calculator_schemas.json not found — preflight didn't run?")

# COMMAND ----------

# DBTITLE 1,Export §2 — locked prompts (verbatim)
# ── 2. Prompts (verbatim) ─────────────────────────────────────────────────────
print("[2/9] prompts...")
prompts_dir = OUT / "prompts"
prompts_dir.mkdir()

# (a) Static condition prompts committed in prompts/
conf_prompts = REPO / "prompts"
if conf_prompts.exists():
    for f in sorted(conf_prompts.rglob("*")):
        if f.is_file():
            shutil.copy(f, prompts_dir / f.name)
            print(f"  {f.name}")
else:
    (prompts_dir / "_note.txt").write_text(f"prompts/ not found at {conf_prompts}")
    print("  WARNING: prompts/ not found")

# (b) Runtime-generated extractor prompt (regenerated by 02_run_all's preflight
# from the MCP schema dump). This is the actual prompt Sonnet saw — the
# committed prompts above don't include the extractor since it's derived
# from MCP at run time.
_runtime_extractor = Path("/dbfs/results/_runtime/extractor_prompt.txt")
if _runtime_extractor.exists():
    shutil.copy(_runtime_extractor, prompts_dir / "extractor_prompt.txt")
    print("  extractor_prompt.txt (runtime-generated)")
else:
    print("  WARNING: runtime extractor_prompt.txt not found — preflight didn't run?")

# COMMAND ----------

# DBTITLE 1,Export §3 — raw per-vignette CSVs
# ── 3. Raw per-vignette CSVs ──────────────────────────────────────────────────
print("[3/9] raw per-vignette rows...")
raw_dir = OUT / "raw"
raw_dir.mkdir()
for cond, df in dfs.items():
    df.to_csv(raw_dir / f"condition_{cond}.csv", index=False)
    print(f"  condition_{cond}.csv  ({len(df)} rows, {df.shape[1]} cols)")

# COMMAND ----------

# DBTITLE 1,Export §4 — analysis tables, comparisons, characterization
# ── 4. Analysis outputs ───────────────────────────────────────────────────────
print("[4/9] analysis tables...")
anal_dir = OUT / "analysis"
anal_dir.mkdir()

# accuracy + Wilson CIs
accuracy_table.to_csv(anal_dir / "accuracy_table.csv", index=False)

# primary McNemar + Bonferroni
primary_df.to_csv(anal_dir / "primary_comparisons.csv")

# secondary B' vs B
if "B_prime" in dfs and "B" in dfs:
    x, y = paired_correctness(dfs["B_prime"], dfs["B"])
    _res = mcnemar(x, y)
    (anal_dir / "secondary_Bprime_vs_B.json").write_text(json.dumps({
        "comparison":           "B_prime_vs_B",
        "n_paired":             int(len(x)),
        "accuracy_B_prime":     float(x.mean()),
        "accuracy_B":           float(y.mean()),
        "delta_pp":             float((x.mean() - y.mean()) * 100),
        "mcnemar_b":            int(_res.b),
        "mcnemar_c":            int(_res.c),
        "p_uncorrected":        float(_res.pvalue),
        "significant_at_0.05": bool(_res.pvalue < 0.05),
    }, indent=2))

# per-category accuracy
cat_df.round(3).to_csv(anal_dir / "per_category_accuracy.csv")

# stratified analysis (tool-using subset)
if "B" in dfs:
    b_tool_ids = set(dfs["B"][dfs["B"]["n_tool_calls"] > 0]["id"])
    strat_rows = []
    for cond in ["A", "B", "C", "D", "E", "B_prime_E", "B_prime_E_reactive"]:
        if cond not in dfs:
            continue
        sub = dfs[cond][dfs[cond]["id"].isin(b_tool_ids)]
        ci = wilson_ci(int(sub["correct"].sum()), len(sub))
        strat_rows.append({"condition": cond, "n_subset": ci.n,
                           "accuracy_on_subset": ci.accuracy,
                           "ci_low": ci.lo, "ci_high": ci.hi})
    pd.DataFrame(strat_rows).to_csv(anal_dir / "stratified_tool_subset.csv", index=False)

# exploratory B_prime_E comparisons (post-hoc)
for label, results in exploratory_results.items():
    if "status" in results:
        continue
    (anal_dir / f"exploratory_{label}.json").write_text(json.dumps({
        "comparison": label,
        **results,
    }, indent=2))

# verifier characterization (E and B_prime_E)
if verifier_summaries:
    (anal_dir / "verifier_characterization.json").write_text(json.dumps(verifier_summaries, indent=2))

# cost / latency
cost_df.to_csv(anal_dir / "cost_latency_table.csv", index=False)

# tool-use distribution per condition
tool_rows = []
for cond, df in dfs.items():
    if "n_tool_calls" in df.columns:
        for n_calls, cnt in df["n_tool_calls"].value_counts().sort_index().items():
            tool_rows.append({"condition": cond, "n_tool_calls": int(n_calls), "count": int(cnt)})
pd.DataFrame(tool_rows).to_csv(anal_dir / "tool_use_distribution.csv", index=False)

# paired rows joined across all conditions on id
base = list(dfs.values())[0][["id"]].copy()
for cond, df in dfs.items():
    cols = {"correct": f"{cond}_correct", "predicted": f"{cond}_predicted"}
    if "n_tool_calls" in df.columns:
        cols["n_tool_calls"] = f"{cond}_n_tool_calls"
    base = base.merge(df[["id"] + list(cols.keys())].rename(columns=cols), on="id", how="left")
base.to_csv(anal_dir / "paired_rows.csv", index=False)

# Per-row flip reconciliation: each interwhen variant against its natural
# baseline. E uses B (the no-verifier tools baseline); B_prime_E uses B_prime
# (the forced-tool-use baseline). The flip_type tells you whether the
# verifier helped, hurt, or did nothing for that vignette.
for cond, baseline in [("E", "B"), ("B_prime_E", "B_prime"), ("B_prime_E_reactive", "B_prime")]:
    if cond not in dfs or baseline not in dfs:
        continue
    sub = dfs[cond].copy()
    base_correct = dict(zip(dfs[baseline]["id"], dfs[baseline]["correct"]))
    sub["baseline_correct"] = sub["id"].map(base_correct)
    sub["flip_type"] = "no_change"
    sub.loc[(~sub["baseline_correct"]) & ( sub["correct"]), "flip_type"] = "wrong_to_right"
    sub.loc[( sub["baseline_correct"]) & (~sub["correct"]), "flip_type"] = "right_to_wrong"
    out_cols = ["id", "correct", "baseline_correct", "flip_type", "n_tool_calls"]
    sub[out_cols].to_csv(anal_dir / f"condition_{cond}_reconciliation.csv", index=False)
    (anal_dir / f"condition_{cond}_verifier_summary.json").write_text(
        json.dumps({
            "condition":    cond,
            "baseline":     baseline,
            **sub["flip_type"].value_counts().to_dict(),
        }, indent=2)
    )

print("  accuracy_table, primary_comparisons, secondary_Bprime_vs_B,")
print("  exploratory_BprimeE_vs_Bprime, exploratory_BprimeE_vs_E,")
print("  per_category, cost_latency, tool_use_distribution, paired_rows,")
print("  E_reconciliation, B_prime_E_reconciliation, verifier_characterization,")
print("  stratified_tool_subset")

# COMMAND ----------

# DBTITLE 1,Export §5 — Pareto + per-category heatmap
# ── 5. Plots ──────────────────────────────────────────────────────────────────
print("[5/9] plots...")
plots_dir = OUT / "plots"
plots_dir.mkdir()

# Pareto — reuse figure from Cell 17 (still in scope, plt.show() doesn't close it)
fig.savefig(plots_dir / "pareto.png", dpi=150)
plt.close(fig)

# Per-category accuracy heatmap
cat_vals = cat_df.values.astype(float)
fig2, ax2 = plt.subplots(figsize=(max(6, len(cat_df.columns) * 1.2), max(4, len(cat_df) * 0.35)))
im = ax2.imshow(cat_vals, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax2.set_xticks(range(len(cat_df.columns)))
ax2.set_xticklabels(cat_df.columns, fontsize=9)
ax2.set_yticks(range(len(cat_df.index)))
ax2.set_yticklabels(cat_df.index, fontsize=7)
plt.colorbar(im, ax=ax2, label="Accuracy")
ax2.set_title("Per-category accuracy by condition")
plt.tight_layout()
fig2.savefig(plots_dir / "per_category_heatmap.png", dpi=150)
plt.close(fig2)

print("  pareto.png, per_category_heatmap.png")

# COMMAND ----------

# DBTITLE 1,Export §6 — vLLM server log
# ── 6. vLLM server log ───────────────────────────────────────────────────────
print("[6/9] vLLM server log...")
vllm_log = Path("/tmp/vllm_server.log")
if vllm_log.exists():
    shutil.copy(vllm_log, OUT / "vllm_server.log")
    print(f"  vllm_server.log ({vllm_log.stat().st_size / 1024 / 1024:.1f} MB)")
else:
    (OUT / "vllm_server.log").write_text("[log not found — cluster may have restarted]")
    print("  WARNING: vllm_server.log not found")

# COMMAND ----------

# DBTITLE 1,Export §7 — executed analysis notebook
# ── 7. Executed analysis notebook ────────────────────────────────────────────
print("[7/9] analysis notebook...")
nb_src = REPO / "notebooks" / "03_analysis.ipynb"
if nb_src.exists():
    shutil.copy(nb_src, OUT / "03_analysis_executed.ipynb")
    print(f"  03_analysis_executed.ipynb")
else:
    print(f"  WARNING: {nb_src} not found (export notebook as .ipynb manually)")

# COMMAND ----------

# DBTITLE 1,Export §8 — apparatus gate
# ── 8. Study gates ────────────────────────────────────────────────────────────
print("[8/9] study gates...")
gates_dir = OUT / "study_gates"
gates_dir.mkdir()
apparatus_p = RESULTS_DIR / "apparatus_full/rows.parquet"
if apparatus_p.exists():
    app_df = pd.read_parquet(apparatus_p)
    app_df.to_csv(gates_dir / "apparatus_results.csv", index=False)
    app_acc = app_df["correct"].mean()
    # Pre-registered gate is 81.9% ± 3pp (see TESTING.md §1 and
    # notebooks/01_apparatus_validation.py:76-77). Use the two-sided band, not
    # a one-sided >= at the target.
    GATE_LO, GATE_HI = 0.789, 0.849
    gate_passed = bool(GATE_LO <= app_acc <= GATE_HI)
    (gates_dir / "apparatus_gate.json").write_text(json.dumps({
        "apparatus_accuracy": float(app_acc),
        "apparatus_n":        int(len(app_df)),
        "target_accuracy":    0.819,
        "tolerance_pp":       3.0,
        "gate_lo":            GATE_LO,
        "gate_hi":            GATE_HI,
        "delta_pp":           round((app_acc - 0.819) * 100, 2),
        "gate_passed":        gate_passed,
    }, indent=2))
    print(f"  apparatus gate: {app_acc:.1%} vs target 81.9% ± 3pp  (passed: {gate_passed})")
else:
    (gates_dir / "apparatus_gate.json").write_text(json.dumps({"note": "apparatus_full not found"}))
    print("  WARNING: apparatus_full/rows.parquet not found")

# COMMAND ----------

# DBTITLE 1,Export §9 — MANIFEST.json (sha256 per file)
# ── 9. Manifest (sha256 per file) ─────────────────────────────────────────────
print("[9/9] manifest...")
manifest = []
for f in sorted(OUT.rglob("*")):
    if f.is_file():
        rel = str(f.relative_to(OUT))
        manifest.append({"path": rel, "size_bytes": f.stat().st_size, "sha256": sha256(f)})
(OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print(f"  {len(manifest)} files indexed")

# ── Zip to /tmp (next cell copies to UC Volume) ───────────────────────────────────────────────────────
zip_path = Path("/tmp/karma_export.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in OUT.rglob("*"):
        if f.is_file():
            zf.write(f, f.relative_to(OUT.parent))

size_mb = zip_path.stat().st_size / 1024 / 1024
print(f"\nDone. Zip: {size_mb:.1f} MB — run next cell to copy to UC Volume for download.")

# COMMAND ----------

# DBTITLE 1,Download links (click to save to your computer)
# Create volume if needed, copy zip there.
# idm_main.default is the only schema writable without admin grants.
spark.sql("CREATE VOLUME IF NOT EXISTS idm_main.default.karma_results")  # noqa: F821

src = Path("/tmp/karma_export.zip")
dst = Path("/Volumes/idm_main/default/karma_results/karma_export.zip")
shutil.copy(src, dst)

size_mb = dst.stat().st_size / 1024 / 1024
print(f"Done! {size_mb:.1f} MB")
print("\nCatalog Explorer → idm_main → default → karma_results → karma_export.zip → right-click → Download")
