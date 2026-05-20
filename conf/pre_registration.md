# Pre-registration

This document is committed to git **before any production experimental runs proceed**.
After commit, the locked values become inputs to the experiment; changes are
treated as deviations and documented separately.

See TESTING.md Section 6 ("Open methodological decisions") and Section 7
("Pre-registration") for the methodological context.

---

## Hypothesis

(Primary hypothesis and the four confirmatory comparisons from TESTING.md Section 3.)

## Locked configuration

| Item | Value | Source |
|---|---|---|
| KARMA version | `<commit SHA>` | github.com/<fork>/KARMA-OpenMedEvalKit |
| interwhen version | `<commit SHA>` | github.com/microsoft/interwhen |
| Benchmark version | `<HF dataset revision>` | ekacare/medical_calculator_eval |
| Model | `Qwen3-30B-A3B-Thinking-2507` | TBD pinned revision |
| Fact extractor model | TBD | — |
| Sampling: temperature | TBD | — |
| Sampling: top-p | TBD | — |
| Condition C prompt | `conf/prompts/condition_c.txt` | — |
| Condition D' verifier prompt | `conf/prompts/condition_d_prime.txt` | — |
| Condition D' revision policy | TBD (max rounds, when to invoke) | — |
| Condition E feedback format | TBD (first-person reflection vs user-message) | — |
| Condition E max retry rounds | TBD (default candidate: 2) | — |
| Calculator subset covered by verifier | TBD (list in `conf/calculator_subset.json`) | — |
| Extractor field-level accuracy threshold | TBD (default candidate: 95%) | — |
| Effect-size threshold (minimum effect of interest) | TBD (default candidate: 3pp absolute) | — |
| Random seeds for non-deterministic runs | TBD | — |
| Number of seeds for variance characterization | TBD | — |

## Confirmatory comparison family

Bonferroni-corrected at α = 0.05 / 4 = 0.0125 per comparison:

- B vs A — McNemar's paired test
- E vs B — McNemar's paired test
- E vs C — McNemar's paired test
- E vs D' — McNemar's paired test

## Exploratory analyses (reported separately, not in confirmatory family)

(List any pairwise comparisons outside the four above that will be reported,
e.g., C vs D', condition-level error-type decomposition.)

## Apparatus validation gate

Before any Qwen3 condition runs proceed: Claude Sonnet 4.6 + tools (Condition B
equivalent) on the full `medical_calculator_eval` must reproduce
EkaCare's published 81.9% (±3pp). If not, investigate before continuing.

Result of apparatus run: TBD (commit result here once obtained).
