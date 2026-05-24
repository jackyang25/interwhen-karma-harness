# Results manifest

Bundles in this directory are gitignored (only this README is committed). The bundles themselves are produced by the EDP packager and dropped here for local analysis.

**Findings writeup:** see [`RESULTS.md`](./RESULTS.md) for the analysis of the currently-present bundle.

## Bundle types

- **Per-run bundle** — one per condition execution. Contains config, environment, per-vignette rows, raw completions, monitor metrics (E only).
- **Analysis bundle** — produced by `notebooks/08_analysis.py`. Contains paired-row joins, McNemar results, CIs, plots, executed notebook.
- **Study-gates bundle** — one for the whole study. Apparatus reproduction, extractor accuracy validation, pre-registration deviations.

## Index

| Bundle | Type | Condition | Date | n | Accuracy | Notes |
|---|---|---|---|---|---|---|
| _(populate as bundles arrive)_ | | | | | | |

## Conventions

- Bundle directory naming: `<type>_<condition>_<YYYY-MM-DDTHH-MM-SSZ>/` (e.g. `run_qwen3_E_2026-05-23T14-22Z/`)
- Each bundle contains a top-level `MANIFEST.json` listing every artifact with sha256
- Closed-API model "revisions" (e.g. claude-sonnet-4-6) are recorded as model string + run timestamp; Anthropic does not expose commit SHAs
- Analysis notebook "version" is the sha256 of the exported `.ipynb`, not a git SHA (notebooks live in the Databricks workspace, not in git)

## Where the full data lives

Large artifacts (raw completions, full parquets, vLLM logs) are kept on DBFS at the path recorded in each bundle's `MANIFEST.json`. The local bundles in this directory are working copies for analysis; treat DBFS as the system of record.
