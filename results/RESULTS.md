# Results

> **Bundle date:** 2026-05-24. **Scope of this writeup:** describes findings
> from the bundle exported on that date — *before* the verifier-vocabulary
> rebuild (schema-aligned `prompts/extractor_prompt.txt`, MCP-derived field
> coverage in `harness/verifier/semantic.py`). The verifier-related findings
> below (in particular §5 on the verifier firing zero times) reflect the
> *pre-rebuild* coverage. A fresh writeup will replace this one after the
> next run of `02_run_all` and `03_analysis` produces a new bundle.

Every number below comes verbatim from the CSV/JSON files in this bundle.
Section headers cite the artifact they read from.

## 1. What was run

- **Model under test:** `Qwen/Qwen3-30B-A3B-Thinking-2507` (all 7 conditions)
- **Fact extractor (E, B'+E):** `claude-sonnet-4-6`
- **Post-hoc verifier (D):** `claude-sonnet-4-6`
- **Apparatus reproduction model:** `claude-sonnet-4-6` on Condition B
- **Dataset:** `ekacare/medical_calculator_eval`, `test` split, **n = 1066**
- **interwhen commit:** `2d041c2f3ed2a6f0a4b063463b3aef844e7dba5e`
- **Hardware:** Azure H100 80GB (Standard_NC40ads_H100_v5)

Conditions (full definitions in `TESTING.md` §6):

| | Track | Description |
|---|---|---|
| A      | Primary       | No tools, no intervention |
| B      | Primary       | Tools, no intervention (baseline) |
| C      | Primary       | Tools + upfront prompt to verify inputs |
| D      | Primary       | Tools + post-hoc Sonnet verifier |
| E      | Primary       | Tools + interwhen semantic verifier (headline) |
| B'     | Exploratory   | Tools + prompt forcing tool use (post-hoc) |
| **B'+E** | Exploratory | **B' system prompt + interwhen verifier (post-hoc, new)** |

## 2. Accuracy (`analysis/accuracy_table.csv`)

n = 1066 paired vignettes. CIs are Wilson.

| Condition | Accuracy | 95% CI            | Parse failures | Mean tool calls |
|-----------|---------:|:-----------------:|---------------:|----------------:|
| A         | 0.4156   | [0.3863, 0.4454]  | 1              | 0.00            |
| B         | 0.5038   | [0.4738, 0.5337]  | 10             | 0.60            |
| C         | 0.4944   | [0.4644, 0.5244]  | 7              | 0.50            |
| D         | 0.5009   | [0.4710, 0.5309]  | 14             | 0.56            |
| E         | 0.5019   | [0.4719, 0.5318]  | 13             | 0.59            |
| B'        | 0.7270   | [0.6995, 0.7529]  | 21             | 3.72            |
| **B'+E**  | 0.7233   | [0.6956, 0.7493]  | 24             | 3.77            |

## 3. Primary comparisons — pre-registered (`analysis/primary_comparisons.csv`)

McNemar paired test. `b` = left correct & right wrong; `c` = reverse.
`bonferroni_p` corrected across the 4 confirmatory comparisons.

| Comparison | b   | c   | Δ (pp) | p (uncorrected) | p (Bonferroni) | Sig at α=0.05 |
|------------|----:|----:|-------:|----------------:|---------------:|:-------------:|
| B vs A     | 145 | 51  | +8.82  | 3.08e-11        | **1.23e-10**   | **Yes**       |
| E vs B     | 25  | 27  | -0.19  | 0.890           | 1.000          | No            |
| E vs C     | 78  | 70  | +0.75  | 0.565           | 1.000          | No            |
| E vs D     | 27  | 26  | +0.09  | 1.000           | 1.000          | No            |

**Reading**

- Tool access (B vs A) is the only intervention with a detectable effect: **+8.82 pp**, robustly significant.
- The headline intervention (E = interwhen semantic verifier) shows **no detectable improvement** over B, C, or D. Point estimates are at or below zero (E vs B is slightly negative).

## 4. Exploratory comparisons — post-hoc (`analysis/secondary_Bprime_vs_B.json`, `analysis/exploratory_*.json`)

Not part of the pre-registered family. Reported with uncorrected p only.

| Comparison         | n_paired | acc_left | acc_right | Δ (pp)  | b   | c   | p (uncorrected) | Sig at α=0.05 (uncorrected) |
|--------------------|---------:|---------:|----------:|--------:|----:|----:|----------------:|:---------------------------:|
| B' vs B            | 1066     | 0.7270   | 0.5038    | **+22.33** | 270 | 32  | 2.39e-42        | **Yes**                     |
| **B'+E vs B'**     | 1066     | 0.7233   | 0.7270    | **-0.38**  | 35  | 39  | 0.727           | **No**                      |
| B'+E vs E          | 1066     | 0.7233   | 0.5019    | +22.14  | 276 | 40  | 6.75e-40        | Yes                         |

**Reading**

- **Forcing tool use is the only intervention that moves accuracy materially.** B' gains +22.33 pp over B; B'+E gains +22.14 pp over E. Both effects are essentially the B' system prompt's effect.
- **Adding interwhen on top of B' did nothing measurable.** B'+E vs B' is −0.38 pp with p = 0.73. The verifier added cost without adding accuracy.

## 5. Verifier behavior (`analysis/verifier_characterization.json`, `analysis/condition_*_verifier_summary.json`)

**The semantic verifier did not fire on a single row in either E or B'+E.**

```json
"E":         { "n_intervention_rows": 0, "total_violations": 0 }
"B_prime_E": { "n_intervention_rows": 0, "total_violations": 0 }
```

What this means precisely: the deterministic comparison between (a) Sonnet-extracted patient facts and (b) the model's planned tool-call arguments found zero discrepancies across 1066 × 2 = 2132 vignette-runs. The interwhen verifier's `fix()` path was never invoked with a violation in either condition.

The per-row "flip" tables (`condition_E_reconciliation.csv`, `condition_B_prime_E_reconciliation.csv`) still show non-zero `wrong_to_right` and `right_to_wrong` counts (25 / 27 for E vs B; 35 / 39 for B'+E vs B'). These flips are run-to-run noise between independent vLLM runs of the same model at temperature 0 — **they are not evidence of verifier action**, because the verifier never acted.

**Implication.** On this model + dataset, the interwhen mechanism as implemented (Sonnet fact extraction + deterministic field comparison + feedback re-prompt) is a no-op. The mechanism is not failing silently — it is structurally not triggering, because the extracted facts and the planned tool arguments agreed on every tool call the model made.

**Possible reasons (not adjudicated by these artifacts):**

1. The model's tool inputs are copied closely from the case text, so verifier-detectable mismatches (wrong units, wrong enum, wrong number) rarely occur.
2. The fact extractor returns a sparse set of facts; fields the model uses may simply not be in the extractor output, so there is no comparison to perform.
3. The comparison logic in `harness/verifier/semantic.py` is too strict-or-too-lenient in a way that always returns "consistent."

Distinguishing these requires inspecting the extracted facts against tool-call arguments per row — not done in this bundle.

## 6. Tool-use distribution (`analysis/tool_use_distribution.csv`)

Fraction of vignettes with zero tool calls per condition:

| Condition | Zero-tool-call rows | Fraction |
|-----------|--------------------:|---------:|
| A         | 1066                | 1.000    |
| B         | 901                 | 0.845    |
| C         | 931                 | 0.873    |
| D         | 909                 | 0.853    |
| E         | 901                 | 0.845    |
| B'        | 101                 | 0.095    |
| **B'+E**  | **94**              | **0.088**|

**Reading.** Without the B' prompt, the model elects to use tools on only ~15% of vignettes. With it, ~91%. B' is the only intervention that changes this distribution.

## 7. Stratified subset — tool-using rows only (`analysis/stratified_tool_subset.csv`)

Restricted to the n = 165 vignettes where B used at least one tool. Descriptive only (no significance test reported).

| Condition | n   | Accuracy on subset | 95% CI            |
|-----------|----:|-------------------:|:-----------------:|
| A         | 165 | 0.158              | [0.110, 0.221]    |
| B         | 165 | 0.655              | [0.579, 0.723]    |
| C         | 165 | 0.418              | [0.346, 0.494]    |
| D         | 165 | 0.558              | [0.481, 0.631]    |
| E         | 165 | 0.570              | [0.493, 0.643]    |
| **B'+E**  | 165 | **0.745**          | [0.674, 0.806]    |

**Reading.** On the rows where the baseline used tools, B' + E reaches **0.745** — its highest accuracy stratification. But B alone is 0.655 on this subset, and adding the interwhen verifier (E) yields 0.570, *worse* than B. The +9 pp on B'+E vs B on this subset is consistent with B'+E getting the model to use tools on rows where B wouldn't have, not from verifier-driven correction.

## 8. Cost and latency (`analysis/cost_latency_table.csv`)

Deployment-honest split: `api_*` columns are Sonnet API (billed by Anthropic); `on_gpu_*` columns are local vLLM inference (your H100). Tokens and elapsed times are means per vignette unless noted.

| Condition | Total tokens | API prompt | API completion | API elapsed (s) | On-GPU prompt | On-GPU completion | Median total latency (s) | Mean model calls |
|-----------|------------:|----------:|---------------:|----------------:|--------------:|------------------:|-------------------------:|-----------------:|
| A         | 2,976       | 0         | 0              | 0.00            | 229           | 2,747             | 138.28                   | 1.00             |
| B         | 8,152       | 0         | 0              | 0.00            | 5,673         | 2,479             | 123.42                   | 1.59             |
| C         | 8,004       | 0         | 0              | 0.00            | 5,415         | 2,589             | 134.94                   | 1.50             |
| D         | 10,426      | 2,360     | 81             | 2.53            | 5,476         | 2,510             | 83.15                    | 2.57             |
| E         | 9,174       | 811       | 321            | 5.80            | 5,593         | 2,449             | 87.65                    | 1.58             |
| B'        | 29,908      | 0         | 0              | 0.00            | 25,623        | 4,285             | 408.44                   | 4.71             |
| **B'+E**  | **31,595**  | **807**   | **320**        | **5.70**        | **26,158**    | **4,311**         | **207.11**               | **4.78**         |

**Reading**

- B'+E adds ~1,127 Sonnet tokens (extractor) per vignette over B'. At public Sonnet 4.6 pricing this is ~$0.005/vignette; over the 1066-vignette set, ~$5 in extractor API cost for **no accuracy gain** (§4).
- B'+E's median latency (207 s) is roughly half B''s (408 s). This is not necessarily a real efficiency win — the conditions ran at different `max_workers` (B' at 64, B'+E at 32) and on different request paths (turn-loop vs streaming-with-monitor). Cross-condition latency comparison has a concurrency confounder; the recorded numbers are honest wall-clock per row, not contention-adjusted.
- D's lower latency than B (83 s vs 123 s) carries the same caveat.

## 9. Per-category accuracy (`analysis/per_category_accuracy.csv`)

Full table in the CSV; heatmap at `plots/per_category_heatmap.png`. Selected observations:

- **B' and B'+E are nearly identical across categories** (within ±5 pp on most). Where they differ, B'+E is sometimes better (pediatric_calculators +13 pp, psychiatry +13 pp) and sometimes worse (sleep_calculators −22 pp, percentile −13 pp). No coherent direction.
- **Radiology** stays at 0% for all primary conditions and jumps to 58–67% for B' / B'+E — consistent with the structural-ceiling story (tools available but unused without forcing).
- **Geriatric medicine** stays low across all conditions (≤ 46%); not addressed by any intervention here.

## 10. Apparatus reproduction (`study_gates/apparatus_gate.json`)

| Field                       | Value                  |
|-----------------------------|-----------------------|
| Apparatus model             | Claude Sonnet 4.6 on Condition B |
| Apparatus n                 | 1066                  |
| Apparatus accuracy          | **0.8039** (857/1066) |
| Target accuracy             | 0.8190 (EkaCare published) |
| Pre-registered band (±3 pp) | [0.789, 0.849]        |
| Δ                           | −1.51 pp              |
| `gate_passed`               | **true**              |

Passed per the locked ±3 pp band. The accuracy is computed each export from `apparatus_full/rows.parquet` (the original 2026-05-23 Sonnet run); the gate uses the two-sided band that matches `notebooks/01_apparatus_validation.py:76-77`.

## 11. What this bundle does and does not show

### Claims supported

- On Qwen3-30B-A3B-Thinking-2507 with the locked prompts and tools:
  - Tool access (B vs A) significantly improves accuracy (+8.82 pp; Bonferroni p ≈ 1.2e-10).
  - **None of the input-verification interventions (C, D, E) produced a statistically detectable improvement over B.** Point estimates for all primary comparisons against E are within ±1 pp.
  - **The interwhen semantic verifier did not fire on a single vignette in either E or B'+E**, so any accuracy difference between E/B'+E and their respective baselines on this run is run-to-run vLLM variability, not verifier action.
  - **B' (forcing tool use) is the only intervention that materially moves accuracy on this model** (+22.33 pp over B, exploratory).
  - Adding interwhen on top of B' (B'+E) added Sonnet API cost without changing accuracy (−0.38 pp, p = 0.73).
- Apparatus reproduction is within the pre-registered ±3 pp band of the EkaCare-published Sonnet+B accuracy.

### Claims not supported

- **Why the verifier did not fire.** The bundle does not contain extracted-facts-vs-planned-arguments diagnostics; investigating requires a per-row inspection step not run here.
- **Generalization.** No claim about other models, datasets, languages, or calculator subsets.
- **Per-category significance.** Per-category numbers are point estimates only.
- **Latency efficiency claims.** Cross-condition latency differs partly because of `max_workers` and request-path differences, not just intrinsic mechanism efficiency.

## 12. Files

| File | Contents |
|------|----------|
| `analysis/accuracy_table.csv` | Per-condition accuracy + Wilson CI + parse-failure count + mean tool calls |
| `analysis/primary_comparisons.csv` | McNemar B vs A and E vs B/C/D, Bonferroni-corrected |
| `analysis/secondary_Bprime_vs_B.json` | McNemar B' vs B (exploratory) |
| `analysis/exploratory_B_prime_E_vs_B_prime.json` | Headline new comparison (post-hoc) |
| `analysis/exploratory_B_prime_E_vs_E.json` | B'+E vs E (post-hoc, sanity) |
| `analysis/cost_latency_table.csv` | Tokens, latency, model-call counts; **plus `api_*` (Anthropic billing) and `on_gpu_*` (local vLLM) breakouts** |
| `analysis/tool_use_distribution.csv` | Histogram of tool-call counts per condition |
| `analysis/stratified_tool_subset.csv` | Accuracy on rows where B used a tool |
| `analysis/per_category_accuracy.csv` | Accuracy by calculator category × condition |
| `analysis/paired_rows.csv` | All conditions joined on vignette id |
| `analysis/condition_E_reconciliation.csv` | Per-row flip table E vs B |
| `analysis/condition_E_verifier_summary.json` | Aggregated flip counts E vs B |
| `analysis/condition_B_prime_E_reconciliation.csv` | Per-row flip table B'+E vs B' |
| `analysis/condition_B_prime_E_verifier_summary.json` | Aggregated flip counts B'+E vs B' |
| `analysis/verifier_characterization.json` | Verifier intervention counts + violation field tallies (E and B'+E) |
| `study_gates/apparatus_gate.json` | Apparatus reproduction summary (passed) |
| `study_gates/apparatus_results.csv` | Per-row apparatus data (Sonnet B run) |
| `provenance/provenance.json` | Models, dataset, hardware, commit SHAs, run records |
| `provenance/pip_freeze.txt` | Pinned dependency list |
| `plots/pareto.png` | Accuracy-vs-cost Pareto |
| `plots/per_category_heatmap.png` | Per-condition × per-category heatmap |
| `raw/condition_*.csv` | Per-vignette raw rows per condition (7 files) |
| `MANIFEST.json` | sha256 + size for every file in the bundle |
