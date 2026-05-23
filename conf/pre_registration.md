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
| Condition C prompt (input verification, upfront) | `conf/prompts/condition_c.txt` | — |
| Condition B' prompt (force tool use, upfront) | `conf/prompts/condition_b_prime.txt` | — |
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

- **B' vs B** — McNemar's paired test at uncorrected α = 0.05. Tests whether prompt-level tool-use enforcement closes the tool-underuse gap observed in the Qwen3 baseline (median 0 calls/vignette in B). Designated exploratory because the condition was specified after observing baseline behavior; this commit pre-dates running C, D', E, and B'. Findings reported with the exploratory designation explicit.
- **Cost / latency secondary endpoints.** Per-vignette: mean prompt tokens, mean completion tokens, median wall-clock latency, number of model invocations, number of tool calls. Per-run: total wall-clock, estimated USD at $3/hr H100 (stated in any reported figure). Token counts compared across conditions; wall-clock compared only at matched `max_workers`. Reported alongside accuracy, not in the Bonferroni family. Motivated by the LMIC-deployment framing of the study (TESTING.md §4.5).
- **Stratified analysis of E vs B on the tool-using subset** — restricted to rows where Condition B made ≥1 tool call. Direct test of whether input verification helps where it can act.
- **Condition-level error-type decomposition** — wrong-calculator vs wrong-input failure rates per condition, using EkaCare's taxonomy where applicable.
- **Per-category breakdown** — accuracy by `category` field of the dataset.
- (List any other pairwise comparisons added before run-time here.)

## Apparatus validation gate

Before any Qwen3 condition runs proceed: Claude Sonnet 4.6 + tools (Condition B
equivalent) on the full `medical_calculator_eval` must reproduce
EkaCare's published 81.9% (±3pp). If not, investigate before continuing.

**Result (2026-05-20):** Sonnet 4.6 + tools = **80.4%** (Wilson CI [77.9%, 82.7%], n = 1,066). Inside the ±3pp band. **Gate passed.**

## Baseline observation that motivated B'

**Condition B on Qwen3-30B-A3B-Thinking-2507 (2026-05-22):** 49.1% accuracy (Wilson CI [46.1%, 52.1%], n = 1,066). Tool calls per vignette: mean 0.38, median 0, 75th percentile 0. The model used tools on ~25% of vignettes — substantially less than Sonnet under the same setup (mean 4.3 calls/vignette).

**Condition A on Qwen3 (2026-05-23):** 41.5% accuracy (Wilson CI [38.5%, 44.4%], n = 1,066). Tool calls: 0.0 across all rows (sanity check confirmed). B − A = +7.6 pp.

The tool-underuse pattern observed in B prompted the addition of Condition B' (force tool use via prompt) as a secondary, exploratory condition prior to running C, D', or E. This commit is the timestamped record of that decision.

## Methodological refactor: channel choice for Qwen3 (2026-05-23)

The earlier B/A baseline runs used vLLM's structured tools API
(`--enable-auto-tool-choice --tool-call-parser hermes` on `/v1/chat/completions`).
After review it became clear this channel is incompatible with interwhen's
streaming step-extractor pattern, which requires raw text on `/v1/completions`.
To preserve interwhen integration for Condition E and keep a single channel
across all Qwen3 conditions, the Qwen3 adapter is refactored to text-mode:
- vLLM `/v1/completions` endpoint
- Prompt rendered via `tokenizer.apply_chat_template(..., tools=...)`
- Qwen3 emits `<tool_call>{...}</tool_call>` tags as raw text
- interwhen `VerifyMonitor` watches the stream and fires at `</tool_call>`

Conditions A, B, B', C, D' need to be re-run on the new channel for clean
comparisons. The earlier (structured-tools) B = 49.1% and A = 41.5% numbers
are recorded for historical reference but are not part of the final analysis
under this refactored design. The new channel is the same model behaviour
(Qwen3 emits the same `<tool_call>` text either way); only the layer that
parses the tags differs.
