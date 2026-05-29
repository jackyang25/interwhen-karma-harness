# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Cross-condition analysis (9-condition pre-registered design)
# MAGIC
# MAGIC Loads per-row parquets from each condition and runs the pre-registered
# MAGIC statistical plan against the **6-contrast Bonferroni family**
# MAGIC (α = 0.05 / 6 = 0.00833).
# MAGIC
# MAGIC ## Condition table
# MAGIC
# MAGIC | # | ID | Group | Description |
# MAGIC |---|----|-------|-------------|
# MAGIC | 1 | A | anchor | No tools — capability floor |
# MAGIC | 2 | B | anchor | Tools + no system prompt — apparatus baseline |
# MAGIC | 3 | B_prime | anchor | Tools + force-tool-use prompt |
# MAGIC | 4 | B_prime_E | primary | Upfront full-schema extractor + hygiene |
# MAGIC | 5 | B_prime_E_reactive | primary | Reactive per-call extractor + hygiene |
# MAGIC | 6 | B_prime_E_reactive_citations | primary | Reactive + (value, source_span) |
# MAGIC | 7 | B_prime_E_reactive_kshot | primary | Reactive + k=3 majority vote |
# MAGIC | 8 | C | exploratory | Force-tool + prompt-only input self-verify (B' footing, no extractor) |
# MAGIC | 9 | D | exploratory | Force-tool + post-hoc Sonnet output verifier (B' footing) |
# MAGIC
# MAGIC ## Pre-registered primary contrasts (Bonferroni n=6, α=0.00833)
# MAGIC
# MAGIC | # | Comparison | Question |
# MAGIC |---|------------|----------|
# MAGIC | 1 | B vs A | Does tool access help? |
# MAGIC | 2 | B_prime vs B | Does forced tool use help (no verifier)? |
# MAGIC | 3 | B_prime_E vs B_prime | Does upfront verifier-guided extraction help? |
# MAGIC | 4 | B_prime_E_reactive vs B_prime | Does reactive verifier-guided extraction help? |
# MAGIC | 5 | B_prime_E_reactive_citations vs B_prime_E_reactive | Does citation grounding improve reactive? |
# MAGIC | 6 | B_prime_E_reactive_kshot vs B_prime_E_reactive | Does k-shot voting improve reactive? |
# MAGIC
# MAGIC Plus exploratory contrasts: B_prime_E (upfront) vs B_prime_E_reactive
# MAGIC (placement); C vs B_prime (prompt self-check vs no-check, matched
# MAGIC footing); B_prime_E_reactive vs C (external verifier vs prompt
# MAGIC self-check); D vs B_prime (post-hoc output check, matched footing).

# COMMAND ----------

# MAGIC %pip install -q pandas numpy scipy statsmodels matplotlib
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
from pathlib import Path
from harness.analysis import wilson_ci, mcnemar, bonferroni

# COMMAND ----------

# MAGIC %md ## 1. Canonical condition table

# COMMAND ----------

# DBTITLE 1,Canonical 9-condition table (mirror of 02_run_all)
# This dict is the analysis-side mirror of ALL_CONDITIONS in 02_run_all.py.
# Keys must match exactly so per-vignette parquets line up by condition_id.
ALL_CONDITIONS: dict[str, dict] = {
    "A":                              {"group": "anchor",      "label": "A"},
    "B":                              {"group": "anchor",      "label": "B"},
    "B_prime":                        {"group": "anchor",      "label": "B'"},
    "B_prime_E":                      {"group": "primary",     "label": "B'+E (upfront)"},
    "B_prime_E_reactive":             {"group": "primary",     "label": "B'+E (reactive)"},
    "B_prime_E_reactive_citations":   {"group": "primary",     "label": "B'+E (reactive + citations)"},
    "B_prime_E_reactive_kshot":       {"group": "primary",     "label": "B'+E (reactive + k-shot)"},
    "C":                              {"group": "exploratory", "label": "C"},
    "D":                              {"group": "exploratory", "label": "D"},
}

PRIMARY_CONDITIONS = [k for k, v in ALL_CONDITIONS.items() if v["group"] == "primary"]
ANCHOR_CONDITIONS  = [k for k, v in ALL_CONDITIONS.items() if v["group"] == "anchor"]
EXPLORATORY_CONDITIONS = [k for k, v in ALL_CONDITIONS.items() if v["group"] == "exploratory"]

# Ordered for the headline forest plot (anchors → primary → exploratory).
PLOT_ORDER = ANCHOR_CONDITIONS + PRIMARY_CONDITIONS + EXPLORATORY_CONDITIONS

# COMMAND ----------

# MAGIC %md ## 2. Load all condition results (uniform)

# COMMAND ----------

# DBTITLE 1,Load per-condition parquets
RESULTS_DIR = Path("/dbfs/results")


def _result_path(cond: str) -> Path:
    return RESULTS_DIR / f"qwen3_condition_{cond}_full" / "rows.parquet"


# One-time migration: rename legacy D_prime dirs to D.
for _old, _new in [
    ("qwen3_condition_D_prime_full",  "qwen3_condition_D_full"),
    ("qwen3_condition_D_prime_pilot", "qwen3_condition_D_pilot"),
]:
    if (RESULTS_DIR / _old).exists() and not (RESULTS_DIR / _new).exists():
        (RESULTS_DIR / _old).rename(RESULTS_DIR / _new)
        print(f"Migrated {_old} → {_new}")

dfs: dict[str, pd.DataFrame] = {}
for cond in ALL_CONDITIONS:
    p = _result_path(cond)
    if p.exists():
        dfs[cond] = pd.read_parquet(p)
        print(f"{cond}: loaded {len(dfs[cond])} rows from {p}")
    else:
        print(f"{cond}: MISSING parquet at {p}")

print(f"\nLoaded {len(dfs)}/{len(ALL_CONDITIONS)} conditions.")
_missing = [c for c in ALL_CONDITIONS if c not in dfs]
if _missing:
    print(f"Missing: {_missing}")
    print("(Analysis below proceeds with what's available; contrasts that need")
    print("missing conditions will be skipped and logged.)")

# COMMAND ----------

# MAGIC %md ## 3. Per-condition accuracy table (Wilson 95% CIs)

# COMMAND ----------

# DBTITLE 1,Accuracy table — all 9 conditions symmetric
rows = []
for cond in PLOT_ORDER:
    if cond not in dfs:
        rows.append({
            "condition": cond, "label": ALL_CONDITIONS[cond]["label"],
            "group": ALL_CONDITIONS[cond]["group"],
            "n": None, "n_correct": None, "accuracy": None,
            "ci_low": None, "ci_high": None,
            "parse_failures": None, "mean_tool_calls": None,
            "status": "missing",
        })
        continue
    df = dfs[cond]
    ci = wilson_ci(int(df["correct"].sum()), len(df))
    rows.append({
        "condition":       cond,
        "label":           ALL_CONDITIONS[cond]["label"],
        "group":           ALL_CONDITIONS[cond]["group"],
        "n":               ci.n,
        "n_correct":       int(df["correct"].sum()),
        "accuracy":        ci.accuracy,
        "ci_low":          ci.lo,
        "ci_high":         ci.hi,
        "parse_failures":  int(df["parse_failed"].sum()) if "parse_failed" in df.columns else None,
        "mean_tool_calls": df["n_tool_calls"].mean() if "n_tool_calls" in df.columns else None,
        "status":          "ok",
    })
accuracy_table = pd.DataFrame(rows)
display(accuracy_table)  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 4. Pre-registered McNemar contrasts (Bonferroni n=6, α=0.00833)

# COMMAND ----------

# DBTITLE 1,Paired correctness helper
def paired_correctness(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Align two condition DataFrames on `id` and return paired correctness arrays."""
    merged = df_a[["id", "correct"]].rename(columns={"correct": "a"}).merge(
        df_b[["id", "correct"]].rename(columns={"correct": "b"}),
        on="id", how="inner",
    )
    return merged["a"].to_numpy(), merged["b"].to_numpy()


def run_contrast(label: str, left: str, right: str) -> dict:
    if left not in dfs or right not in dfs:
        return {"label": label, "left": left, "right": right, "status": "missing"}
    x, y = paired_correctness(dfs[left], dfs[right])
    res = mcnemar(x, y)
    return {
        "label":              label,
        "left":               left,
        "right":              right,
        "n_paired":           int(len(x)),
        "accuracy_left":      float(x.mean()),
        "accuracy_right":     float(y.mean()),
        "delta_pp":           float((x.mean() - y.mean()) * 100),
        "mcnemar_b":          int(res.b),
        "mcnemar_c":          int(res.c),
        "p_uncorrected":      float(res.pvalue),
        "status":             "ok",
    }

# COMMAND ----------

# DBTITLE 1,Primary contrasts — 6 pre-registered, Bonferroni α=0.00833
PRIMARY_CONTRASTS = [
    ("B_vs_A",                                              "B",                            "A"),
    ("B_prime_vs_B",                                        "B_prime",                      "B"),
    ("B_prime_E_vs_B_prime",                                "B_prime_E",                    "B_prime"),
    ("B_prime_E_reactive_vs_B_prime",                       "B_prime_E_reactive",           "B_prime"),
    ("B_prime_E_reactive_citations_vs_B_prime_E_reactive",  "B_prime_E_reactive_citations", "B_prime_E_reactive"),
    ("B_prime_E_reactive_kshot_vs_B_prime_E_reactive",      "B_prime_E_reactive_kshot",     "B_prime_E_reactive"),
]
BONFERRONI_N = len(PRIMARY_CONTRASTS)  # 6 → per-test α = 0.05 / 6 = 0.00833

primary_results: dict[str, dict] = {}
for label, left, right in PRIMARY_CONTRASTS:
    primary_results[label] = run_contrast(label, left, right)

# Bonferroni correction over the family of 6 confirmatory comparisons.
uncorrected_pvals = {
    k: v["p_uncorrected"]
    for k, v in primary_results.items()
    if v.get("status") == "ok"
}
adjusted = bonferroni(uncorrected_pvals, n_tests=BONFERRONI_N)
PRIMARY_ALPHA = 0.05 / BONFERRONI_N

for k, p_adj in adjusted.items():
    primary_results[k]["p_bonferroni"]                    = p_adj
    primary_results[k]["significant_at_family_0.05"]      = p_adj < 0.05
    primary_results[k]["significant_at_per_test_alpha"]   = primary_results[k]["p_uncorrected"] < PRIMARY_ALPHA

primary_df = pd.DataFrame(primary_results).T
display(primary_df)  # noqa: F821
print(f"\nFamily α = 0.05; n_tests = {BONFERRONI_N}; per-test α = {PRIMARY_ALPHA:.5f}")

# COMMAND ----------

# MAGIC %md ## 5. Exploratory contrasts (uncorrected)
# MAGIC
# MAGIC - **Placement** (upfront vs reactive): a within-architecture contrast
# MAGIC   isolating the effect of extraction placement at fixed format and sampling.
# MAGIC - **Other verifier mechanisms (C, D)**: tested against B as historical
# MAGIC   comparators — they are not in the primary family.

# COMMAND ----------

# DBTITLE 1,Exploratory contrasts
EXPLORATORY_CONTRASTS = [
    ("B_prime_E_upfront_vs_reactive", "B_prime_E", "B_prime_E_reactive"),
    # C now runs on the B' forced-tool footing, so it is compared against B'
    # (matched baseline) and head-to-head against the best extractor arm.
    ("C_vs_B_prime",                  "C",         "B_prime"),
    ("B_prime_E_reactive_vs_C",       "B_prime_E_reactive", "C"),
    # D now runs on the B' forced-tool footing, so it is compared against B'.
    ("D_vs_B_prime",                  "D",         "B_prime"),
]

exploratory_results: dict[str, dict] = {}
for label, left, right in EXPLORATORY_CONTRASTS:
    exploratory_results[label] = run_contrast(label, left, right)
    r = exploratory_results[label]
    if r.get("status") != "ok":
        print(f"{label}: missing condition data ({r['left']} or {r['right']} not loaded)")
        continue
    print(f"{label} (exploratory, uncorrected α = 0.05):")
    print(f"  n paired:        {r['n_paired']}")
    print(f"  {r['left']:<35s} accuracy: {r['accuracy_left']:.3f}")
    print(f"  {r['right']:<35s} accuracy: {r['accuracy_right']:.3f}")
    print(f"  Delta:           {r['delta_pp']:+.2f} pp")
    print(f"  McNemar:         b={r['mcnemar_b']}, c={r['mcnemar_c']}, p={r['p_uncorrected']:.4g}")

# COMMAND ----------

# MAGIC %md ## 6. Axis-effect view — placement, output format, sampling
# MAGIC
# MAGIC Each row isolates ONE architectural axis (with other axes held fixed).
# MAGIC This is the cleanest reading of "which axis matters" for the paper.

# COMMAND ----------

# DBTITLE 1,Architectural axis-effect table
AXIS_CONTRASTS = [
    ("placement",     "upfront vs reactive (bare, k=1)",        "B_prime_E",                    "B_prime_E_reactive"),
    ("output_format", "bare vs citation (reactive, k=1)",       "B_prime_E_reactive",           "B_prime_E_reactive_citations"),
    ("sampling",      "k=1 vs k=3 voting (reactive, bare)",     "B_prime_E_reactive",           "B_prime_E_reactive_kshot"),
]

axis_rows = []
for axis, desc, left, right in AXIS_CONTRASTS:
    r = run_contrast(f"axis_{axis}", left, right)
    if r.get("status") != "ok":
        axis_rows.append({"axis": axis, "description": desc, "status": "missing"})
        continue
    axis_rows.append({
        "axis":           axis,
        "description":    desc,
        "left":           left,
        "right":          right,
        "accuracy_left":  r["accuracy_left"],
        "accuracy_right": r["accuracy_right"],
        "delta_pp":       r["delta_pp"],
        "mcnemar_b":      r["mcnemar_b"],
        "mcnemar_c":      r["mcnemar_c"],
        "p_uncorrected":  r["p_uncorrected"],
    })
axis_df = pd.DataFrame(axis_rows)
display(axis_df)  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 7. Stratified analysis — effect on the tool-using subset

# COMMAND ----------

# DBTITLE 1,Stratified subset where B used tools
if "B" in dfs:
    b_tool_using_ids = set(dfs["B"][dfs["B"]["n_tool_calls"] > 0]["id"])
    print(f"Tool-using subset (B): {len(b_tool_using_ids)} / {len(dfs['B'])} rows")

    sub_rows = []
    for cond in PLOT_ORDER:
        if cond not in dfs:
            continue
        sub = dfs[cond][dfs[cond]["id"].isin(b_tool_using_ids)]
        if len(sub) == 0:
            continue
        ci = wilson_ci(int(sub["correct"].sum()), len(sub))
        sub_rows.append({
            "condition":         cond,
            "label":             ALL_CONDITIONS[cond]["label"],
            "group":             ALL_CONDITIONS[cond]["group"],
            "n_subset":          ci.n,
            "accuracy_on_subset": ci.accuracy,
            "ci_low":            ci.lo,
            "ci_high":           ci.hi,
        })
    stratified_df = pd.DataFrame(sub_rows)
    display(stratified_df)  # noqa: F821
else:
    stratified_df = pd.DataFrame()
    print("B missing — cannot compute stratified subset.")

# COMMAND ----------

# MAGIC %md ## 8. Cost and latency (per condition)

# COMMAND ----------

# DBTITLE 1,Cost/latency table (symmetric across 9 conditions)
def _col_or_zero(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] if present, else a zero-filled Series. Lets us derive
    api_/on_gpu_ aggregates uniformly across schemas with or without the
    extractor/verifier breakout columns."""
    if col in df.columns:
        return df[col]
    return pd.Series([0] * len(df), index=df.index)


# Latency is read from the fixed-concurrency timing pass (02_run_all §9),
# NOT from the full accuracy run. The full run uses a different worker count
# per condition (16–128) for throughput, which makes its per-vignette
# wall-clock incomparable across conditions. Tokens and accuracy are
# load-independent and come from the full run.
def _timing_df(cond: str):
    """Per-condition fixed-concurrency latency rows, or None if the timing
    pass has not been run for this condition."""
    p = RESULTS_DIR / f"qwen3_condition_{cond}_timing" / "rows.parquet"
    return pd.read_parquet(p) if p.exists() else None


cost_rows = []
for cond in PLOT_ORDER:
    if cond not in dfs:
        cost_rows.append({"condition": cond, "label": ALL_CONDITIONS[cond]["label"], "status": "missing"})
        continue
    df = dfs[cond]
    if "elapsed_seconds" not in df.columns:
        cost_rows.append({"condition": cond, "label": ALL_CONDITIONS[cond]["label"], "status": "legacy parquet (no instrumentation)"})
        continue

    # API-side (Anthropic billing): extractor (any B'+E variant) + verifier (D).
    # Token counts are load-independent → taken from the full run.
    api_prompt     = _col_or_zero(df, "extractor_prompt_tokens")     + _col_or_zero(df, "verifier_prompt_tokens")
    api_completion = _col_or_zero(df, "extractor_completion_tokens") + _col_or_zero(df, "verifier_completion_tokens")
    # On-GPU (vLLM): everything in `prompt_tokens` / `completion_tokens` not
    # accounted for by API-side.
    on_gpu_prompt     = _col_or_zero(df, "prompt_tokens")     - api_prompt
    on_gpu_completion = _col_or_zero(df, "completion_tokens") - api_completion

    # Latency: from the fixed-concurrency timing pass only.
    tdf = _timing_df(cond)
    if tdf is not None and "elapsed_seconds" in tdf.columns:
        median_latency_s   = float(tdf["elapsed_seconds"].median())
        mean_api_elapsed_s = float((_col_or_zero(tdf, "extractor_elapsed_s")
                                    + _col_or_zero(tdf, "verifier_elapsed_s")).mean())
        latency_n          = int(len(tdf))
    else:
        # No timing pass on disk → do NOT fall back to the confounded full-run
        # wall-clock. Report latency as unavailable so it cannot be misread.
        median_latency_s   = None
        mean_api_elapsed_s = None
        latency_n          = 0

    cost_rows.append({
        "condition":                     cond,
        "label":                         ALL_CONDITIONS[cond]["label"],
        "group":                         ALL_CONDITIONS[cond]["group"],
        "status":                        "ok",
        "accuracy":                      df["correct"].mean(),
        "mean_total_tokens":             df["total_tokens"].mean() if "total_tokens" in df.columns else None,
        "median_latency_s":              median_latency_s,
        "latency_source":                "fixed-concurrency timing pass (1 worker)" if latency_n else "MISSING — run 02 §9 latency pass",
        "latency_n":                     latency_n,
        "mean_model_calls":              df["n_model_calls"].mean() if "n_model_calls" in df.columns else None,
        "mean_api_prompt_tokens":        float(api_prompt.mean()),
        "mean_api_completion_tokens":    float(api_completion.mean()),
        "mean_api_elapsed_s":            mean_api_elapsed_s,
        "mean_on_gpu_prompt_tokens":     float(on_gpu_prompt.mean()),
        "mean_on_gpu_completion_tokens": float(on_gpu_completion.mean()),
    })
cost_df = pd.DataFrame(cost_rows)
display(cost_df)  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 9. Figures
# MAGIC
# MAGIC Symmetric per-condition treatment. Each figure works for any subset
# MAGIC of `dfs` that loaded successfully.

# COMMAND ----------

# DBTITLE 1,Forest plot — per-condition accuracy with Wilson CIs
import matplotlib.pyplot as plt

_GROUP_COLOR = {"anchor": "#2b8cbe", "primary": "#e34a33", "exploratory": "#7f7f7f"}

fig_forest, ax_f = plt.subplots(figsize=(9, 5.5))
plot_data = accuracy_table[accuracy_table["status"] == "ok"].copy()
plot_data["yidx"] = range(len(plot_data))[::-1]  # top-down visual order
ax_f.errorbar(
    plot_data["accuracy"], plot_data["yidx"],
    xerr=[plot_data["accuracy"] - plot_data["ci_low"], plot_data["ci_high"] - plot_data["accuracy"]],
    fmt="o", capsize=4,
    c="black", ecolor="black", markerfacecolor="black",
)
for _, r in plot_data.iterrows():
    ax_f.scatter(r["accuracy"], r["yidx"], s=80, c=_GROUP_COLOR.get(r["group"], "k"), zorder=3)
ax_f.set_yticks(plot_data["yidx"])
ax_f.set_yticklabels(plot_data["label"])
ax_f.set_xlim(0, 1)
ax_f.set_xlabel("Accuracy (Wilson 95% CI)")
ax_f.set_title("Per-condition accuracy")
ax_f.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Axis-effect plot — placement / format / sampling
fig_axis, ax_axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
for ax, (axis, desc, left, right) in zip(ax_axes, AXIS_CONTRASTS):
    r = run_contrast(f"axis_{axis}", left, right)
    if r.get("status") != "ok":
        ax.set_title(f"{axis} (missing)")
        ax.axis("off")
        continue
    bars = ax.bar([ALL_CONDITIONS[left]["label"], ALL_CONDITIONS[right]["label"]],
                   [r["accuracy_left"], r["accuracy_right"]],
                   color=["#7f7f7f", "#e34a33"])
    ax.set_ylim(0, 1)
    ax.set_title(f"{axis}\nΔ={r['delta_pp']:+.1f} pp, p={r['p_uncorrected']:.3g}")
    ax.tick_params(axis="x", labelrotation=20)
ax_axes[0].set_ylabel("Accuracy")
plt.suptitle("Architectural axis-effects (paired McNemar)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Pareto plot — accuracy vs total tokens
fig_pareto, ax_p = plt.subplots(figsize=(8, 5))
# Robust to cost_df missing the status/accuracy columns entirely (e.g. when no
# condition has instrumented data yet): default status to "ok" and skip the
# plot cleanly rather than raising.
if {"accuracy", "mean_total_tokens"}.issubset(cost_df.columns):
    _status = cost_df.get("status", pd.Series("ok", index=cost_df.index)).fillna("").astype(str)
    plot_df = cost_df[_status != "missing"].dropna(subset=["accuracy", "mean_total_tokens"])
else:
    plot_df = cost_df.iloc[0:0]
for _, r in plot_df.iterrows():
    ax_p.scatter(r["mean_total_tokens"], r["accuracy"], s=80, c=_GROUP_COLOR.get(r["group"], "k"))
    ax_p.annotate(r["label"], (r["mean_total_tokens"], r["accuracy"]),
                  xytext=(5, 5), textcoords="offset points", fontsize=8)
ax_p.set_xlabel("Mean total tokens per vignette")
ax_p.set_ylabel("Accuracy")
ax_p.set_title("Cost / accuracy frontier")
ax_p.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md ## 10. Per-category accuracy heatmap (primary + B')

# COMMAND ----------

# DBTITLE 1,Per-category heatmap (focus on primary + B' for readability)
HEATMAP_CONDS = ["B_prime"] + PRIMARY_CONDITIONS  # 5 conditions max for readability
cat_df = pd.DataFrame()
for cond in HEATMAP_CONDS:
    if cond not in dfs or "category" not in dfs[cond].columns:
        continue
    cat_acc = dfs[cond].groupby("category")["correct"].mean().rename(ALL_CONDITIONS[cond]["label"])
    cat_df = pd.concat([cat_df, cat_acc], axis=1)
display(cat_df.round(3))  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 11. Verifier characterization (per primary-study condition)
# MAGIC
# MAGIC Flag composition per condition. Under schema-gating, non-required-field
# MAGIC flags are zero by construction — the distribution shown here is over
# MAGIC required fields only.

# COMMAND ----------

# DBTITLE 1,Verifier characterization (symmetric across primary conditions)
import json as _vh_json

verifier_summaries: dict[str, dict] = {}

for cond in PRIMARY_CONDITIONS:
    if cond not in dfs:
        verifier_summaries[cond] = {"status": "missing"}
        continue
    df = dfs[cond]
    print(f"\n── {cond} ──")
    print(f"Total rows: {len(df)}")
    print(f"Accuracy:   {df['correct'].mean():.1%}")

    # Vs B_prime as the natural baseline for all primary conditions.
    if "B_prime" in dfs:
        bp_correct = dict(zip(dfs["B_prime"]["id"], dfs["B_prime"]["correct"]))
        merged = df[["id", "correct"]].copy()
        merged["bp_correct"] = merged["id"].map(bp_correct)
        flips_w2r = merged[(~merged["bp_correct"]) & ( merged["correct"])]
        flips_r2w = merged[( merged["bp_correct"]) & (~merged["correct"])]
        print(f"Flipped wrong→right vs B': {len(flips_w2r)} rows")
        print(f"Flipped right→wrong vs B': {len(flips_r2w)} rows")
        print(f"Net vs B': {len(flips_w2r) - len(flips_r2w):+d}")

    if "violations_history" not in df.columns:
        verifier_summaries[cond] = {"status": "no violations_history column (legacy parquet)"}
        print("  (violations_history not in parquet — skipping flag composition)")
        continue

    parsed = df["violations_history"].apply(
        lambda s: _vh_json.loads(s) if isinstance(s, str) else (s or [])
    )
    intervention_mask = parsed.apply(lambda lst: len(lst) > 0)
    n_intervention_rows = int(intervention_mask.sum())

    total_violations = int(
        parsed.apply(lambda lst: sum(len(e.get("violations", [])) for e in lst)).sum()
    )
    field_counts: dict[str, int] = {}
    for entries in parsed:
        for entry in entries:
            for v in entry.get("violations", []):
                f = v.get("field", "unknown")
                field_counts[f] = field_counts.get(f, 0) + 1

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
    print(f"  Intervention rows: {n_intervention_rows} ({summary['pct_intervention_rows']:.1f}%)")
    print(f"  Total violations:  {total_violations}")
    print(f"  Of intervened rows, % correct: {summary['pct_correct_on_intervention_rows']:.1f}%")
    if field_counts:
        top = list(summary["violations_by_field"].items())[:5]
        print(f"  Top flagged fields: {', '.join(f'{f}={c}' for f, c in top)}")

# COMMAND ----------

# MAGIC %md ## 11.5 Mechanism diagnostics — citation acceptance + k-shot agreement
# MAGIC
# MAGIC Per-field acceptance / agreement rates from the new primary conditions'
# MAGIC parquet columns:
# MAGIC
# MAGIC - **citation_reports** (from B_prime_E_reactive_citations) — per-field
# MAGIC   {value, source_span, valid, reason}. Computes the fraction of spans
# MAGIC   that passed substring validation, per field and overall.
# MAGIC - **voting_reports** (from B_prime_E_reactive_kshot) — per-field
# MAGIC   {samples, winner, count, accepted, reason}. Computes per-field
# MAGIC   agreement distribution (3/3, 2/3, no-majority) and abstention rates.
# MAGIC
# MAGIC These power the discussion's "why did the mechanism arm work / not work"
# MAGIC narrative. Empty for any condition that didn't produce these reports.

# COMMAND ----------

# DBTITLE 1,Mechanism diagnostics
mechanism_diagnostics: dict[str, dict] = {}

# Citation diagnostics ───────────────────────────────────────────────────────
if "B_prime_E_reactive_citations" in dfs:
    cit_df = dfs["B_prime_E_reactive_citations"]
    if "citation_reports" in cit_df.columns:
        parsed = cit_df["citation_reports"].apply(
            lambda s: _vh_json.loads(s) if isinstance(s, str) else (s or [])
        )
        per_field_total:    dict[str, int] = {}
        per_field_valid:    dict[str, int] = {}
        per_reason_counts:  dict[str, int] = {}
        n_total_spans = 0
        n_valid_spans = 0
        for vignette_reports in parsed:
            for tool_call_entry in vignette_reports:
                for field, rep in (tool_call_entry.get("report") or {}).items():
                    n_total_spans += 1
                    per_field_total[field] = per_field_total.get(field, 0) + 1
                    if rep.get("valid"):
                        n_valid_spans += 1
                        per_field_valid[field] = per_field_valid.get(field, 0) + 1
                    reason = rep.get("reason", "unknown")
                    per_reason_counts[reason] = per_reason_counts.get(reason, 0) + 1

        per_field_acceptance = {
            f: {
                "n_total":          per_field_total[f],
                "n_valid":          per_field_valid.get(f, 0),
                "acceptance_rate":  per_field_valid.get(f, 0) / max(per_field_total[f], 1),
            }
            for f in per_field_total
        }
        mechanism_diagnostics["citations"] = {
            "n_total_spans":             n_total_spans,
            "n_valid_spans":             n_valid_spans,
            "overall_acceptance_rate":   n_valid_spans / max(n_total_spans, 1),
            "by_reason":                 per_reason_counts,
            "by_field":                  dict(sorted(
                per_field_acceptance.items(),
                key=lambda kv: -kv[1]["n_total"],
            )),
        }
        print(f"\n── Citation acceptance ──")
        print(f"  Spans seen:            {n_total_spans}")
        print(f"  Validated (substring): {n_valid_spans} ({100*n_valid_spans/max(n_total_spans,1):.1f}%)")
        print(f"  Rejection reasons:     {per_reason_counts}")
        if per_field_acceptance:
            top5 = sorted(per_field_acceptance.items(), key=lambda kv: -kv[1]["n_total"])[:5]
            print(f"  Top fields by volume:")
            for f, s in top5:
                print(f"    {f:<25s} {s['n_valid']:>3d}/{s['n_total']:<3d} ({100*s['acceptance_rate']:.0f}%)")
    else:
        mechanism_diagnostics["citations"] = {"status": "citation_reports column missing"}
        print("  citations: column missing (legacy parquet or stub run)")

# k-shot diagnostics ─────────────────────────────────────────────────────────
if "B_prime_E_reactive_kshot" in dfs:
    ks_df = dfs["B_prime_E_reactive_kshot"]
    if "voting_reports" in ks_df.columns:
        parsed = ks_df["voting_reports"].apply(
            lambda s: _vh_json.loads(s) if isinstance(s, str) else (s or [])
        )
        per_field_total:        dict[str, int] = {}
        per_field_full_agree:   dict[str, int] = {}   # 3/3
        per_field_partial:      dict[str, int] = {}   # 2/3
        per_field_no_majority:  dict[str, int] = {}   # abstain
        n_total = 0
        n_full = 0
        n_partial = 0
        n_no_maj = 0
        for vignette_reports in parsed:
            for tool_call_entry in vignette_reports:
                for field, rep in (tool_call_entry.get("report") or {}).items():
                    n_total += 1
                    per_field_total[field] = per_field_total.get(field, 0) + 1
                    count = rep.get("count", 0)
                    accepted = rep.get("accepted", False)
                    if accepted and count >= 3:
                        n_full += 1
                        per_field_full_agree[field] = per_field_full_agree.get(field, 0) + 1
                    elif accepted:
                        n_partial += 1
                        per_field_partial[field] = per_field_partial.get(field, 0) + 1
                    else:
                        n_no_maj += 1
                        per_field_no_majority[field] = per_field_no_majority.get(field, 0) + 1

        per_field_agreement = {
            f: {
                "n_total":         per_field_total[f],
                "n_full_agree":    per_field_full_agree.get(f, 0),
                "n_partial":       per_field_partial.get(f, 0),
                "n_no_majority":   per_field_no_majority.get(f, 0),
                "agreement_rate":  (per_field_full_agree.get(f, 0) + per_field_partial.get(f, 0))
                                    / max(per_field_total[f], 1),
            }
            for f in per_field_total
        }
        mechanism_diagnostics["kshot"] = {
            "n_total_field_votes":  n_total,
            "n_full_agreement":     n_full,
            "n_partial_agreement":  n_partial,
            "n_no_majority":        n_no_maj,
            "full_agreement_rate":  n_full / max(n_total, 1),
            "partial_agreement_rate": n_partial / max(n_total, 1),
            "no_majority_rate":     n_no_maj / max(n_total, 1),
            "by_field":             dict(sorted(
                per_field_agreement.items(),
                key=lambda kv: -kv[1]["n_total"],
            )),
        }
        print(f"\n── k-shot agreement ──")
        print(f"  Field-votes seen:    {n_total}")
        print(f"  3/3 agreement:       {n_full} ({100*n_full/max(n_total,1):.1f}%)")
        print(f"  2/3 agreement:       {n_partial} ({100*n_partial/max(n_total,1):.1f}%)")
        print(f"  No majority (null):  {n_no_maj} ({100*n_no_maj/max(n_total,1):.1f}%)")
        if per_field_agreement:
            top5 = sorted(per_field_agreement.items(), key=lambda kv: -kv[1]["n_total"])[:5]
            print(f"  Top fields by volume:")
            for f, s in top5:
                print(f"    {f:<25s} 3/3={s['n_full_agree']:>3d}  2/3={s['n_partial']:>3d}  "
                      f"none={s['n_no_majority']:>3d}  (n={s['n_total']})")
    else:
        mechanism_diagnostics["kshot"] = {"status": "voting_reports column missing"}
        print("  kshot: column missing (legacy parquet or stub run)")

# COMMAND ----------

# MAGIC %md ## 12. Export bundle (all results + analysis)

# COMMAND ----------

# DBTITLE 1,Export §1 — provenance + manifest setup
import hashlib, json, shutil, subprocess, zipfile
from datetime import datetime, timezone

REPO    = Path("/Workspace/Users/jack.yang@gatesfoundation.org/interwhen-karma-harness")
OUT     = Path("/tmp/karma_export")
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)


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
prov_dir = OUT / "provenance"
prov_dir.mkdir()
print("[1/9] provenance...")

(prov_dir / "pip_freeze.txt").write_text(
    subprocess.check_output(["pip", "freeze"]).decode()
)

agg = json.loads(Path("/dbfs/results/_AGGREGATED_RESULTS.json").read_text()) if Path("/dbfs/results/_AGGREGATED_RESULTS.json").exists() else {}
run_records = agg.get("run_records", [])

try:
    dbr   = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion", "unknown")  # noqa: F821
    cname = spark.conf.get("spark.databricks.clusterUsageTags.clusterName", "unknown")   # noqa: F821
except Exception:
    dbr, cname = "unknown", "unknown"

# Pre-registered configuration snapshot (mirrors PREREG_CONFIG in 02_run_all.py).
# Recorded here for the export bundle's provenance so reviewers can verify the
# locked hyperparameters used during the rerun without opening the notebook.
PREREG_CONFIG_SNAPSHOT = {
    "schema_gated_verifier":          True,
    "intervention_style":             "query",
    "abstention":                     "prompt-only",
    "sonnet_extraction_temperature":  0.7,
    "sonnet_seed_strategy":           "fresh-per-call",
    "qwen3_temperature":              0.0,
    "tool_call_loop_turn_cap":        10,
    "verifier_reprompt_cap":          2,
    "citation_validity":              "verbatim-substring",
    "citation_failure":               "null",
    "kshot_voting_rule":              "majority-of-3",
    "kshot_no_majority":              "null",
    "kshot_k":                        3,
    "failure_handling":               "drop-and-report",
}

provenance = {
    "exported_at":           datetime.now(timezone.utc).isoformat(),
    "harness_git_sha":       git_sha(REPO),
    "interwhen_git_sha":     "2d041c2f3ed2a6f0a4b063463b3aef844e7dba5e",
    "dbr_version":           dbr,
    "cluster_name":          cname,
    "gpu_type":              "Standard_NC40ads_H100_v5 (H100 80GB, Azure)",
    "model_id":              "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "extractor_model":       "claude-sonnet-4-6",
    "dataset":               "ekacare/medical_calculator_eval",
    "dataset_split":         "test",
    "n_vignettes":           1066,
    "study_design":          "9-condition pre-registered (paper §methods_conditions)",
    "bonferroni_n":          BONFERRONI_N,
    "primary_alpha":         PRIMARY_ALPHA,
    "prereg_config":         PREREG_CONFIG_SNAPSHOT,
    "all_conditions":        {k: {"group": v["group"], "label": v["label"]} for k, v in ALL_CONDITIONS.items()},
    "primary_contrasts":     [{"label": l, "left": lt, "right": rt} for l, lt, rt in PRIMARY_CONTRASTS],
    "axis_contrasts":        [{"axis": a, "left": lt, "right": rt, "description": d} for a, d, lt, rt in AXIS_CONTRASTS],
    "run_records":           run_records,
}
(prov_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
print("  provenance.json")

_mcp_schemas_src = Path("/dbfs/results/provenance/mcp_calculator_schemas.json")
if _mcp_schemas_src.exists():
    shutil.copy(_mcp_schemas_src, prov_dir / "mcp_calculator_schemas.json")
    print("  mcp_calculator_schemas.json")
else:
    print("  WARNING: mcp_calculator_schemas.json not found")

# COMMAND ----------

# DBTITLE 1,Export §2 — locked prompts (verbatim)
print("[2/9] prompts...")
prompts_dir = OUT / "prompts"
prompts_dir.mkdir()

conf_prompts = REPO / "prompts"
if conf_prompts.exists():
    for f in sorted(conf_prompts.rglob("*")):
        if f.is_file():
            shutil.copy(f, prompts_dir / f.name)
            print(f"  {f.name}")

_runtime_extractor = Path("/dbfs/results/_runtime/extractor_prompt.txt")
if _runtime_extractor.exists():
    shutil.copy(_runtime_extractor, prompts_dir / "extractor_prompt.txt")
    print("  extractor_prompt.txt (runtime-generated)")

# COMMAND ----------

# DBTITLE 1,Export §3 — raw per-vignette CSVs (one per condition, all 9)
print("[3/9] raw per-vignette rows...")
raw_dir = OUT / "raw"
raw_dir.mkdir()
for cond in PLOT_ORDER:
    if cond not in dfs:
        continue
    dfs[cond].to_csv(raw_dir / f"condition_{cond}.csv", index=False)
    print(f"  condition_{cond}.csv  ({len(dfs[cond])} rows, {dfs[cond].shape[1]} cols)")

# Fixed-concurrency latency-pass rows (source of all latency numbers).
for cond in PLOT_ORDER:
    tdf = _timing_df(cond)
    if tdf is None:
        continue
    tdf.to_csv(raw_dir / f"condition_{cond}_timing.csv", index=False)
    print(f"  condition_{cond}_timing.csv  ({len(tdf)} rows, single-worker latency pass)")

# COMMAND ----------

# DBTITLE 1,Export §4 — analysis tables
print("[4/9] analysis tables...")
anal_dir = OUT / "analysis"
anal_dir.mkdir()

accuracy_table.to_csv(anal_dir / "accuracy_table.csv", index=False)
primary_df.to_csv(anal_dir / "primary_comparisons.csv")
pd.DataFrame(exploratory_results).T.to_csv(anal_dir / "exploratory_comparisons.csv")
axis_df.to_csv(anal_dir / "axis_effects.csv", index=False)
if not stratified_df.empty:
    stratified_df.to_csv(anal_dir / "stratified_tool_subset.csv", index=False)
cost_df.to_csv(anal_dir / "cost_latency_table.csv", index=False)
if not cat_df.empty:
    cat_df.round(3).to_csv(anal_dir / "per_category_accuracy.csv")

# Paired rows joined across all loaded conditions on id (for downstream analysis).
if dfs:
    base = next(iter(dfs.values()))[["id"]].copy()
    for cond, df in dfs.items():
        cols = {"correct": f"{cond}_correct", "predicted": f"{cond}_predicted"}
        if "n_tool_calls" in df.columns:
            cols["n_tool_calls"] = f"{cond}_n_tool_calls"
        base = base.merge(df[["id"] + list(cols.keys())].rename(columns=cols), on="id", how="left")
    base.to_csv(anal_dir / "paired_rows.csv", index=False)

# Verifier characterization for primary conditions
if verifier_summaries:
    (anal_dir / "verifier_characterization.json").write_text(json.dumps(verifier_summaries, indent=2))

# Mechanism diagnostics — per-field citation acceptance + k-shot agreement
# rates. Empty/absent when those conditions weren't run or used a legacy
# parquet without the new columns.
if mechanism_diagnostics:
    (anal_dir / "mechanism_diagnostics.json").write_text(
        json.dumps(mechanism_diagnostics, indent=2, default=str)
    )

# Per-row reconciliation against B_prime (the shared baseline). Covers the four
# primary conditions AND the two redefined forced-tool comparators (C, D), which
# now sit on the B_prime footing — so their per-row flips vs B_prime are captured
# too, not silently dropped.
for cond in PRIMARY_CONDITIONS + ["C", "D"]:
    if cond not in dfs or "B_prime" not in dfs:
        continue
    sub = dfs[cond].copy()
    bp_correct = dict(zip(dfs["B_prime"]["id"], dfs["B_prime"]["correct"]))
    sub["baseline_correct"] = sub["id"].map(bp_correct)
    sub["flip_type"] = "no_change"
    sub.loc[(~sub["baseline_correct"]) & ( sub["correct"]), "flip_type"] = "wrong_to_right"
    sub.loc[( sub["baseline_correct"]) & (~sub["correct"]), "flip_type"] = "right_to_wrong"
    out_cols = ["id", "correct", "baseline_correct", "flip_type"]
    if "n_tool_calls" in sub.columns:
        out_cols.append("n_tool_calls")
    sub[out_cols].to_csv(anal_dir / f"condition_{cond}_reconciliation.csv", index=False)

print("  accuracy_table, primary_comparisons, exploratory_comparisons,")
print("  axis_effects, stratified_tool_subset, cost_latency_table,")
print("  per_category_accuracy, paired_rows, verifier_characterization,")
print("  per-condition reconciliation files")

# COMMAND ----------

# DBTITLE 1,Export §5 — figures
print("[5/9] figures...")
plots_dir = OUT / "plots"
plots_dir.mkdir()
fig_forest.savefig(plots_dir / "forest_accuracy.png", dpi=150)
fig_axis.savefig(plots_dir / "axis_effects.png", dpi=150)
fig_pareto.savefig(plots_dir / "pareto.png", dpi=150)
plt.close("all")
print("  forest_accuracy.png, axis_effects.png, pareto.png")

# COMMAND ----------

# DBTITLE 1,Export §6 — vLLM server log
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
print("[7/9] analysis notebook...")
nb_src = REPO / "notebooks" / "03_analysis.ipynb"
if nb_src.exists():
    shutil.copy(nb_src, OUT / "03_analysis_executed.ipynb")
    print("  03_analysis_executed.ipynb")
else:
    print(f"  WARNING: {nb_src} not found (export notebook as .ipynb manually)")

# COMMAND ----------

# DBTITLE 1,Export §8 — apparatus gate
print("[8/9] apparatus gate...")
gates_dir = OUT / "study_gates"
gates_dir.mkdir()
apparatus_p = RESULTS_DIR / "apparatus_full/rows.parquet"
if apparatus_p.exists():
    app_df = pd.read_parquet(apparatus_p)
    app_df.to_csv(gates_dir / "apparatus_results.csv", index=False)
    app_acc = app_df["correct"].mean()
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

# DBTITLE 1,Export §9 — MANIFEST.json + zip
print("[9/9] manifest...")
manifest = []
for f in sorted(OUT.rglob("*")):
    if f.is_file():
        rel = str(f.relative_to(OUT))
        manifest.append({"path": rel, "size_bytes": f.stat().st_size, "sha256": sha256(f)})
(OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print(f"  {len(manifest)} files indexed")

zip_path = Path("/tmp/karma_export.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in OUT.rglob("*"):
        if f.is_file():
            zf.write(f, f.relative_to(OUT.parent))

size_mb = zip_path.stat().st_size / 1024 / 1024
print(f"\nDone. Zip: {size_mb:.1f} MB — next cell copies to UC Volume.")

# COMMAND ----------

# DBTITLE 1,Copy zip to UC Volume for download
spark.sql("CREATE VOLUME IF NOT EXISTS idm_main.default.karma_results")  # noqa: F821
src = Path("/tmp/karma_export.zip")
dst = Path("/Volumes/idm_main/default/karma_results/karma_export.zip")
shutil.copy(src, dst)
print(f"Done! {dst.stat().st_size / 1024 / 1024:.1f} MB at {dst}")
