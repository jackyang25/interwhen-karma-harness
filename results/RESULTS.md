# Results

Findings from the run bundle in this directory. Every number below is taken
verbatim from the CSV/JSON artifacts listed in `MANIFEST.json`; no values are
re-derived or rounded except where explicitly stated.

If a claim is not made here, it is not supported by these artifacts. If a
number disagrees with what you remember from a notebook, trust the artifact
and update the notebook — the artifacts are the system of record.

## 1. What was run

- **Model under test (all conditions A, B, C, D, E, B′):** `Qwen/Qwen3-30B-A3B-Thinking-2507`
- **Verifier extractor for Condition E:** `claude-sonnet-4-6`
- **Apparatus reproduction model:** Claude Sonnet 4.6 on Condition B
  (inferred from `apparatus_results.csv`: mean 4.60 tool calls per
  non-parse-failed vignette, consistent with Sonnet's behavior described in
  `TESTING.md` §6 and inconsistent with Qwen3's mean of 0.60 in Condition B)
- **Dataset:** `ekacare/medical_calculator_eval`, `test` split, **n = 1066** vignettes
- **interwhen commit:** `2d041c2f3ed2a6f0a4b063463b3aef844e7dba5e`
- **Harness git SHA:** not recorded (`provenance.json.harness_git_sha = "unavailable"`)
- **Hardware:** Azure `Standard_NC40ads_H100_v5` (H100 80GB), DBR 17.3.x-gpu-ml-scala2.13
- **Export timestamp:** 2026-05-24T07:22:18Z

Condition definitions are in `TESTING.md` §6. Restated briefly:

| Condition | Description |
|---|---|
| A  | No tools, no intervention |
| B  | Tools, no intervention (baseline) |
| C  | Tools + best-effort upfront prompt to verify inputs |
| D  | Tools + post-hoc verification call on the produced answer |
| E  | Tools + interwhen with semantic verifier (headline) |
| B′ | Tools + prompt requiring the model to use the calculator tool (secondary) |

## 2. Headline numbers (from `analysis/accuracy_table.csv`)

All accuracies are computed over the same n = 1066 paired vignettes.
Confidence intervals are as reported in the artifact (column names
`ci_low`/`ci_high`; method not recorded in the file itself).

| Condition | Accuracy | 95% CI | Parse failures | Mean tool calls |
|---|---:|---|---:|---:|
| A       | 0.4156 | [0.3863, 0.4454] |  1 | 0.000 |
| B       | 0.5038 | [0.4738, 0.5337] | 10 | 0.600 |
| C       | 0.4944 | [0.4644, 0.5244] |  7 | 0.502 |
| D       | 0.5094 | [0.4794, 0.5393] | 14 | 0.593 |
| E       | 0.5113 | [0.4813, 0.5412] | 15 | 0.604 |
| B′      | 0.7270 | [0.6995, 0.7529] | 21 | 3.724 |

## 3. Primary comparisons (McNemar paired, from `primary_comparisons.csv`)

`b` = rows where the left condition is correct and the right is wrong; `c`
= the reverse. `delta_pp` = (acc_left − acc_right) × 100. `bonferroni_p`
applies Bonferroni correction across the four primary comparisons listed
in the file. "Significant at 0.05" uses the **Bonferroni-corrected** p,
which is the column the artifact uses.

| Comparison | b | c | Δ (pp) | p (uncorrected) | p (Bonferroni) | Significant at α=0.05? |
|---|---:|---:|---:|---:|---:|:--:|
| B vs A | 145 | 51 | +8.82 | 3.08e-11 | 1.23e-10 | **Yes** |
| E vs B |  33 | 25 | +0.75 | 0.358    | 1.000    | No |
| E vs C |  89 | 71 | +1.69 | 0.179    | 0.716    | No |
| E vs D |  36 | 34 | +0.19 | 0.905    | 1.000    | No |

### What this means, said carefully

1. **Tool access matters.** Going from no-tools (A) to tools-only (B) lifts
   accuracy by **+8.82 pp** with p ≈ 3.1e-11 (Bonferroni p ≈ 1.2e-10).
   This is the only primary comparison that is statistically significant
   after correction.

2. **The headline intervention (E = interwhen + semantic verifier) does
   *not* show a detectable improvement over any of B, C, or D on this
   dataset and model.** All three pairwise comparisons against E are
   non-significant both uncorrected and after Bonferroni correction
   (uncorrected p ≥ 0.179 in every case). The point estimates of the
   effect are small: **+0.75 pp vs B**, **+1.69 pp vs C**, **+0.19 pp vs D**.

3. **"Non-significant" ≠ "no effect."** The McNemar discordant cells
   (e.g., E vs B: 33 wrong→right vs 25 right→wrong) show the verifier is
   actively flipping rows in both directions. The data are consistent
   with either a small positive effect or a small negative effect; we
   cannot tell which on n = 1066.

## 4. Secondary comparison: B′ vs B (from `secondary_Bprime_vs_B.json`)

Reported as **exploratory** because B′ was specified after observing
baseline behavior (see `TESTING.md` §3 and §6).

- n_paired: 1066
- Accuracy: B′ = **0.7270**, B = 0.5038
- Δ = **+22.33 pp**
- McNemar: b = 270 (B′ correct, B wrong), c = 32 (B correct, B′ wrong)
- p (uncorrected) = **2.39e-42**
- significant_at_0.05 (per artifact) = True

Mean tool calls per vignette went from **0.60 (B) → 3.72 (B′)** — a 6.2×
increase. Mean total tokens went from **8,152 (B) → 29,908 (B′)** — a 3.7×
increase. See §6 for the cost/latency cross-cut.

**Interpretation, said carefully.** B′ is the largest accuracy movement
in the entire study. It targets a different mechanism than C/D/E (forcing
tool *use*, not verifying tool *inputs*) and was specified post-hoc; it is
not a substitute for the primary input-verification claim and should not
be reported as such.

## 5. Tool-use distribution (from `tool_use_distribution.csv`)

Fraction of vignettes with zero tool calls (the cap on what
input-verification interventions can affect):

| Condition | Zero-tool-call rows | Fraction |
|---|---:|---:|
| A       | 1066 | 1.000 |
| B       |  901 | 0.845 |
| C       |  931 | 0.873 |
| D       |  903 | 0.847 |
| E       |  901 | 0.845 |
| B′      |  101 | 0.095 |

Restated: under B, **84.5%** of vignettes are answered with zero tool
calls. C, D, and E do not meaningfully change that distribution
(zero-call fractions within ±3 pp of B). B′ drives the zero-call fraction
down to **9.5%**.

This is the structural ceiling referenced in `TESTING.md` §9: any
intervention that only acts when a tool call occurs can only differ from
B on the ~15% of vignettes where a tool call actually happens.

## 6. Stratified subset: vignettes where tools were actually used (from `stratified_tool_subset.csv`)

This subset is the natural target of input-verification interventions
(C/D/E), since they can only act on rows with a tool call. The artifact
reports n = 165 per condition; the file does not document how the subset
was selected (e.g., union vs intersection of tool-using rows across
conditions), so the absolute level is interpretable but the *exact*
comparison rule should be confirmed against the notebook before quoting
externally.

| Condition | n   | Accuracy on subset | 95% CI |
|---|---:|---:|---|
| A | 165 | 0.158 | [0.110, 0.221] |
| B | 165 | 0.655 | [0.579, 0.723] |
| C | 165 | 0.418 | [0.346, 0.494] |
| D | 165 | 0.570 | [0.493, 0.643] |
| E | 165 | 0.624 | [0.548, 0.695] |

**What stands out (descriptively only — no significance test is
provided in the artifact):**

- On the tool-using subset, **B (0.655) numerically beats C (0.418), D
  (0.570), and E (0.624)**. This is the opposite ordering from §2 on the
  full set. CIs for B and E overlap substantially; CIs for B and C do
  not.
- Treat this descriptively. The subset selection rule is not
  documented in the artifact, and no paired test is reported here.

## 7. Cost and latency (from `cost_latency_table.csv`)

| Condition | Accuracy | Mean total tokens | Median latency (s) | Mean model calls |
|---|---:|---:|---:|---:|
| A  | 0.4156 |  2,976 | 138.28 | 1.00 |
| B  | 0.5038 |  8,152 | 123.42 | 1.59 |
| C  | 0.4944 |  8,004 | 134.94 | 1.50 |
| D  | 0.5094 | 10,684 |  87.06 | 2.61 |
| E  | 0.5113 |  9,351 |  90.51 | 1.59 |
| B′ | 0.7270 | 29,908 | 408.44 | 4.71 |

**Read this section with the caveats below — the E row is not directly
comparable to the others in this bundle's token column.**

### Soundness of these numbers

**Latency (`median_latency_s`)** — sound and comparable across conditions.
For every row, this is the wall-clock around `adapter.run()` in
`harness/runner.py:_score_one`. Same measurement for every condition.

**Tokens for A, B, C, D, B′** — sound. These are Qwen3 inference tokens
reported by the vLLM `/v1/completions` `usage` field, summed across the
condition's per-turn calls.

**Tokens for E in *this bundle* are extractor-only, not the full
interwhen cost.** Inspection of the E adapter at the time of this run
(`notebooks/07_condition_E_interwhen.py`, original version) shows
`prompt_tokens` / `completion_tokens` were populated only from the
Sonnet fact-extractor call:

```
prompt_tokens = facts.prompt_tokens       # Sonnet extractor only
completion_tokens = facts.completion_tokens  # Sonnet extractor only
```

The Qwen3 streaming tokens (and any retry overhead) under interwhen
were **not counted** at the time these CSVs were produced. The
notebook's print line at the end of §10 acknowledges this explicitly:
*"Mean prompt tokens/vignette: … (extractor only — interwhen streaming
not counted at fine grain)"*.

What that means concretely:
- E's "9,351 mean total tokens" reflects Sonnet API usage on the
  extractor, **not** the Qwen3 GPU inference cost. The real total
  (extractor + Qwen3) is higher.
- The B vs E delta in this column is meaningless. Do not quote
  "E uses only ~15% more tokens than B" — that's an artifact, not a
  finding.

**Latency comparison is still honest** because `elapsed_seconds` wraps
the entire `adapter.run()`, including both the Sonnet extractor call and
the interwhen streaming session. So E's median 90.5s vs B's 123.4s is
real wall-clock time, comparable end-to-end. The puzzle of "E is faster
than B despite doing more work" most likely reflects architectural
differences (E's interwhen streams via one open HTTP connection; B's
adapter does turn-by-turn round-trips on tool calls), not a measurement
bug.

**`mean_model_calls` for E (1.59)** — this is the number of vLLM
streaming sessions per vignette (1 initial + 1 per `fix` callback,
which fires on every tool call). It does **not** include the Sonnet
extractor call. If you want "total billable model calls including
Sonnet," add 1 to E's value.

### What's fixed for future re-runs

The E adapter at `notebooks/07_condition_E_interwhen.py` has been
patched to emit deployment-honest cost accounting:

- `prompt_tokens` / `completion_tokens` / `total_tokens` are now the
  **honest cross-condition totals** (extractor + Qwen3), so a future
  `cost_latency_table.csv` row for E is directly comparable to B/C/D.
- Six new breakout columns are written per row by `harness/runner.py`:
  `extractor_prompt_tokens`, `extractor_completion_tokens`,
  `extractor_elapsed_s`, `qwen3_prompt_tokens`,
  `qwen3_completion_tokens`, `qwen3_elapsed_s`. These let downstream
  analysis separate Sonnet API cost (billed by Anthropic) from on-GPU
  Qwen3 inference cost, which matters for deployment economics.
- Non-E conditions write zero values for those columns, so the schema
  is consistent.

The fix is not retroactive — the existing E rows in this bundle keep
the original extractor-only accounting. Treat this §7 table as a
*latency*-honest, *token*-incomplete artifact for E until the next run
with the patched adapter.

## 8. Condition E verifier behavior (from `condition_E_verifier_summary.json`)

Counting rows by what the verifier did to the model's answer relative to
the unverified (Condition B) answer:

| Outcome | Count | Fraction of 1066 |
|---|---:|---:|
| No change       | 1008 | 0.946 |
| Wrong → Right   |   33 | 0.031 |
| Right → Wrong   |   25 | 0.023 |
| **Net** | **+8** | **+0.75 pp** |

The verifier flips **5.4%** of rows; the net effect is **+8 rows correct
out of 1066** (+0.75 pp), matching the E vs B delta in §3. The verifier
is doing work — it is not silent — but the work cancels nearly out.

A row-by-row reconciliation is in `analysis/condition_E_reconciliation.csv`.

## 9. Per-category accuracy (from `per_category_accuracy.csv`)

Full table is in the CSV; a per-category heatmap is in
`plots/per_category_heatmap.png`. Quick read:

- **Largest B → E movements:** none exceed roughly ±5 pp in either
  direction (e.g., digestive +3.8, gynecology −2.1). E does not show a
  consistent per-category lift over B.
- **Largest B → B′ movements (illustrative, not exhaustive):**
  intensive/emergency care +43.8 pp, radiology +58.3 pp, percentile
  +43.8 pp, urology/nephrology +17.5 pp.
- Two categories where every condition stays low (≤ 0.10 in places):
  `radiology_calculators` and `geriatric_medicine_calculators`. Worth
  investigating qualitatively, but not addressed by these artifacts.

## 10. Apparatus reproduction gate (from `study_gates/apparatus_gate.json`)

| Field | Value |
|---|---|
| Apparatus model (inferred) | Claude Sonnet 4.6 on Condition B |
| Apparatus n                | 1066 |
| Apparatus accuracy         | **0.8039** (857/1066) |
| Target accuracy            | 0.8190 (EkaCare published) |
| Pre-registered band (±3 pp) | [0.789, 0.849] |
| Δ                          | −1.51 pp |
| `gate_passed`              | **true** |

**Status: PASSED per pre-registration.** 0.8039 falls inside the
pre-registered ±3 pp band of [0.789, 0.849]; this is the same criterion
checked by `notebooks/01_apparatus_validation.py:76-77`, which prints
"APPARATUS VALIDATED" at run time.

**Note on a prior export bug (fixed).** An earlier version of
`apparatus_gate.json` recorded `gate_passed: false`. The cause was a
one-sided check (`app_acc >= 0.819`) in the export at
`notebooks/09_analysis.py:512` rather than the two-sided ±3 pp band
specified in the pre-registration. The export was corrected and the
artifact re-emitted; the underlying accuracy (0.8039) is unchanged and
no rerun was needed. `MANIFEST.json` was updated with the new hash.

## 11. What can and cannot be claimed from this bundle

### Claims supported by these artifacts

- Tool access (B vs A) significantly improves accuracy on Qwen3-30B for
  this dataset (n = 1066, Δ = +8.82 pp, Bonferroni p ≈ 1.2e-10).
- On Qwen3-30B with the locked prompts in `prompts/`, **none of the
  three input-verification interventions (C, D, E) produced a
  statistically detectable improvement over B**, and pairwise
  comparisons among them are likewise non-significant. Point estimates
  for all primary comparisons against E are between +0.19 and +1.69 pp.
- B′ (forcing tool use) raises accuracy on this model to 0.727 vs B at
  0.504 (Δ = +22.33 pp, uncorrected p ≈ 2.4e-42). This is exploratory
  and addresses a different mechanism (tool *use* rate, not tool *input*
  verification).
- The verifier in Condition E is active (5.4% of rows flipped) but its
  positive and negative flips nearly cancel (net +8 rows / +0.75 pp).
- 84.5% of B rows have zero tool calls, capping the maximum effect any
  input-verification intervention can produce on this model.

### Claims **not** supported by these artifacts

- Anything about closed-API models (Sonnet, GPT, etc.) on the primary
  conditions. The apparatus run is the only Sonnet number in this
  bundle and it is a B-only reproduction, not a head-to-head.
- Generalization to other datasets, calculators, or languages.
- Any causal claim about *why* C/D/E fail to differ from B; the
  failure-analysis writeup that would address this is not in this
  bundle.
- Per-category significance. The per-category table is point estimates
  only.

## 12. Files referenced

| File | What it contains |
|---|---|
| `analysis/accuracy_table.csv` | Per-condition accuracy, CI, parse-failure count, mean tool calls |
| `analysis/primary_comparisons.csv` | McNemar B vs A and E vs B/C/D, with Bonferroni p |
| `analysis/secondary_Bprime_vs_B.json` | McNemar B′ vs B |
| `analysis/condition_E_verifier_summary.json` | Verifier flip counts |
| `analysis/condition_E_reconciliation.csv` | Per-row reconciliation for E |
| `analysis/tool_use_distribution.csv` | Histogram of tool-call counts per condition |
| `analysis/stratified_tool_subset.csv` | Accuracy on the tool-using subset |
| `analysis/per_category_accuracy.csv` | Accuracy by calculator category |
| `analysis/cost_latency_table.csv` | Tokens, latency, model-call counts |
| `study_gates/apparatus_gate.json` | Apparatus reproduction summary |
| `study_gates/apparatus_results.csv` | Per-row apparatus data |
| `provenance/provenance.json` | Models, dataset, hardware, commit SHAs |
| `plots/pareto.png` | Accuracy-vs-cost Pareto plot |
| `plots/per_category_heatmap.png` | Per-condition per-category accuracy heatmap |
| `raw/condition_*.csv` | Per-vignette raw rows per condition |
