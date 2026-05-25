# Clinical AI Verification

**Testing whether intermediate verification reduces wrong-input failures in tool-using clinical AI.**

---

## 1. Summary

**What:** Test whether interwhen-style intermediate verification — checking that a clinical calculator's inputs match what a patient case actually says — reduces wrong-input failures beyond what tool access alone achieves.

**On what:** EkaCare's `medical_calculator_eval` benchmark (1,066 Indian clinical vignettes), through the KARMA evaluation framework, against EkaCare's MedAI MCP calculator tools.

**Model:** Qwen3-30B-A3B-Thinking-2507, served via vLLM. See [Section 4.5](#45-model-choice) for rationale.

**Apparatus validation:** One Claude Sonnet 4.6 run on Condition B reproduces EkaCare's published 81.9% (± 3pp) before any experimental runs proceed (see [Section 8](#8-pre-study-grounding)).

**Contribution:** The clinical semantic verifier design and the empirical answer to whether intermediate verification adds value beyond tool access in a clinical reasoning context. interwhen, KARMA, MedAI, and the benchmark already exist; this project assembles them and tests a specific claim.

---

## 2. Motivation

EkaCare's published benchmarking work (April 2026) showed that tool access produces large accuracy gains on Indian clinical reasoning tasks. For `medical_calculator_eval`:

| Model | No Tools | With Tools | Lift |
|---|---|---|---|
| Claude Sonnet 4.6 | 43.6% | 81.9% | +38.3 pp |
| GPT-5.2 | 56.3% | 79.3% | +23.0 pp |
| GPT-5-mini | 53.4% | 70.5% | +17.1 pp |

Despite these gains, EkaCare documented a **12–13% residual failure rate** where models pick the *right* calculator but feed it *wrong inputs* — e.g., the case states "sedentary lifestyle" but the model passes `activity_level=very_active`. This failure mode is not addressed by adding more tool access; it requires some intervention between the model's reasoning and its tool calls.

This project tests whether interwhen-style intermediate verification is that intervention.

---

## 3. Research Question

**Primary:** Does deterministic intermediate verification of tool-call inputs reduce wrong-input failures beyond what tool access alone provides — and beyond what a cheaper prompt-level instruction provides?

**Confirmatory comparisons (primary, Bonferroni family):**
- **B vs A** — Did tools help at all? (Replication check.)
- **E vs B** — Did verification add value beyond tools? (Foundational claim.)
- **E vs C** — Did verification beat a best-effort upfront prompt intervention? (Deployment-critical.)
- **E vs D** — Did mid-stream verification beat the cheapest verification pattern? (Deployment-critical.)

**Secondary comparison (exploratory, not in confirmatory family):**
- **B' vs B** — Does forcing tool use via a prompt close the tool-underuse gap observed in the open-weights baseline?

The secondary comparison was added after Condition B was run on Qwen3 and revealed that the model invokes tools on a small minority of vignettes (median 0 calls/vignette). Because input-verification interventions (C/D/E) can only act on rows where a tool call occurs, the underuse rate caps the effect size of the primary track. B' characterizes whether prompt-level tool-use enforcement narrows that gap; it is reported as exploratory because the condition was specified after observing baseline behavior. The timestamped commits in `prompts/` document when each condition's prompt was locked.

---

## 4. Frameworks Used

### 4.1 interwhen

Microsoft Research's test-time intermediate verification framework. Watches a model's streaming reasoning trace, runs verifiers on intermediate steps, and injects corrective feedback into the stream when verification fails. The model revises without restarting from scratch.

**Key constraint:** interwhen requires direct control of the inference loop (vLLM-served open-weights models). Closed APIs like Anthropic and OpenAI do not expose mid-generation token injection.

**What interwhen does not provide:** the clinical knowledge, the verifier logic, or the fact extractor. These are built by this project.

### 4.2 KARMA

EkaCare's open-source medical AI evaluation framework. Provides dataset loading, model adapters (pluggable, registry-based), metric computation, and caching.

**Integration approach:** a custom KARMA model adapter wraps the vLLM-served Qwen3 model with interwhen monitors and the semantic verifier. KARMA sees the adapter through its standard model interface; verification logic is encapsulated inside.

**Tool-calling mechanism.** Each model uses its native trained tool-use channel, but those channels differ by model role:

- **Sonnet (apparatus validation only):** Anthropic's structured Tools API. Sonnet's natively trained tool-call mechanism, what EkaCare's published table presumably used.
- **Qwen3 (all study conditions A/B/C/D'/E/B'):** raw text via vLLM's `/v1/completions` endpoint. The adapter places MedAI tool descriptions in the prompt via the model's chat template (`apply_chat_template(..., tools=...)`) and Qwen3 emits tool calls as `<tool_call>{...}</tool_call>` text blocks. This makes each tool call a detectable step in the visible stream, so interwhen's `step_extractor` can fire verification at the commit boundary — matching the framework's published step definitions (a move in Maze, an op in Game24, a code block in Verina). The structured tool-choice API path (vLLM's `--enable-auto-tool-choice`) is *not* used for Qwen3 because it bypasses the raw text stream interwhen needs to monitor.

The Sonnet-vs-Qwen3 channel asymmetry is by-role rather than within-comparison: Sonnet is only used for apparatus validation, never as an experimental arm. All six Qwen3 conditions (A, B, C, D, E, B') run on the same text-mode channel, so the comparisons stand on the same methodological footing.

### 4.3 MedAI MCP Tools

EkaCare's hosted MCP server at `medai-tools.eka.care/mcp`. Exposes 403 clinical calculators via three hierarchical meta-tools (discover, schema, compute). Access gated by Eka account. OIDC or direct API token auth.

### 4.4 medical_calculator_eval

EkaCare's published benchmark (`ekacare/medical_calculator_eval`, HuggingFace). 1,066 clinical vignettes asking for specific computed values. Ground truth is calculator output. Per-question numeric tolerance. Vignette language reflects Indian clinical conventions.

### 4.5 Model choice

Qwen3-30B-A3B-Thinking-2507, served via vLLM. Three reasons:

1. **Methodological:** interwhen requires direct control of the inference loop for mid-generation token injection. Closed APIs (Claude, GPT) don't expose this; open-weights models served via vLLM do. Qwen3 is also the model class interwhen's paper itself evaluates on.
2. **Deployment relevance:** mid-sized open models are the realistic option for clinical AI in resource-constrained LMIC settings — frontier closed APIs are cost- and infrastructure-prohibitive at scale. Improvements measured here transfer to the deployment context EkaCare's work targets.
3. **Headroom:** Qwen3 has more documented failure modes than frontier models, giving verification a meaningful surface to act on. Frontier models with already-strong tool-use leave narrow room for any intervention to register.

---

## 5. The Verifier

The technical contribution of this project: a **semantic input verifier** for clinical calculator tool calls.

### What it verifies

When the model is about to invoke a calculator with specific inputs, the verifier checks whether those inputs accurately represent what the patient case describes — **not just whether they are schema-valid.**

- *Schema validation* (does the input fit the calculator's format) is already handled by model providers, MCP, and the calculator implementations. Redundant.
- *Semantic validation* — does the value match what the vignette states — is the gap. Enum misselection is schema-valid but semantically wrong. This is where the documented 12–13% failures occur.

### Components

1. **Fact extractor.** LLM call with strict JSON output schema. Reads the patient vignette, returns a structured patient object (age, sex, weight, height, labs, lifestyle factors). Different model from the one being evaluated to avoid contamination.

2. **Semantic verifier.** Takes a planned tool call (calculator name + inputs) and the extracted patient object. For each input, checks whether the value, enum, or unit matches the case description. Returns validity + per-input feedback.

3. **Feedback formatter.** When verification fails, constructs feedback identifying the discrepancy: "The case describes a sedentary lifestyle but you are passing `activity_level=very_active`."

### Avoiding "AI verifying AI" circularity

The fact extractor uses an LLM (bounded extraction task). The verifier itself is deterministic (compare values, check enums, look up units). LLM translates text into structured data; deterministic logic does the comparison. The split is the defense against circularity.

### Why we extend interwhen's published pattern (and why this is justified)

interwhen's published monitors (Maze, Game24, Verina, ZebraLogic, SpatialMap) share a structural property: **the ground truth they verify against is symbolic and independent of any model.** A maze path either reaches the exit or doesn't; an arithmetic expression either evaluates to 24 or doesn't; Lean code either type-checks or doesn't. The verifier compares the model's emitted artifact against fixed external rules. interwhen's "extraction" step in those monitors is *self-elicitation*: inject a prompt that asks the test model to emit its current answer in a structured form (e.g., `</think>\nThe expression I found is {…`), then parse + verify.

Clinical input verification has a different structure: **the ground truth is the patient case itself — unstructured natural-language text.** To compare a planned tool-call argument against "what the case says," the verifier needs a structured representation of the case. Having the test model produce that representation would be circular: if the model misreads the case while planning the tool call (the exact failure mode we are trying to catch), it will misread it identically when asked to summarize the case. Self-elicitation does not break the circularity for input-vs-case verification the way it does for arithmetic / code / logic verification.

So we make one deliberate extension to interwhen's pattern: an LLM-based **fact extractor that operates on the case text**, run by a *different* model from the one under test. The extractor's output (a structured `PatientFacts` object) becomes the reference against which the test model's tool-call arguments are compared.

Three design choices contain the risk this introduces:

1. **Bounded translation, not judgment.** The extractor produces a fixed-schema JSON object capturing only what the case explicitly states. Anything not stated is `null`; the extractor does not infer values. The judgment step — *"is this argument consistent with the case?"* — remains deterministic code, not an LLM.

2. **Independent of the test model.** The extractor model (Sonnet 4.6) is different from the model under test (Qwen3-30B-A3B-Thinking-2507). The two readings of the case are produced by different model weights, breaking the circularity that single-model self-elicitation would introduce.

3. **Auditable accuracy gate (§8).** The extractor is validated against a hand-annotated held-out set; field-level accuracy must hit ≥95% before production runs. This is the audit point. If Sonnet's extraction is unreliable, the verifier's reference is unreliable, and the study's verifier characterization is meaningless — the gate prevents us from running with a broken extractor.

The LLM-extract + deterministic-compare split is a published pattern in the verification literature (code agents with compiler/type-check back-ends, function-calling agents with JSON-schema validation, math agents with symbolic backends). We apply it here because the task's ground truth lives in natural-language text rather than in symbols, and because the test model's self-reading of the case cannot serve as ground truth for the failure mode we're targeting.

---

## 6. Experimental Design

### Conditions

Five primary conditions plus one secondary condition on Qwen3-30B-A3B-Thinking-2507:

| Condition | Track | Description |
|---|---|---|
| **A** | Primary | No tools, no intervention |
| **B** | Primary | Tools, no intervention (replicates EkaCare's setup) |
| **C** | Primary | Tools + best-effort prompt instruction asking the model to verify tool-call inputs (units, enums, numerics) against the case before computing (upfront) |
| **D** | Primary | Tools + post-hoc verification call on the produced answer (cheapest verification pattern) |
| **E** | Primary | Tools + interwhen with semantic verifier (headline) |
| **B'** | Secondary | Tools + prompt instruction requiring the model to use the calculator tool for every computation (tool-use enforcement) |

### Why these conditions and not more

The original plan considered 8 conditions including self-consistency (F), few-shot examples (C'), mid-reasoning self-verification, and a schema-only verifier ablation (G). These were dropped for leanness.

**Cost of dropping them:** reviewers can ask "couldn't variance reduction (F), worked examples (C'), or model self-checking (D) have achieved the same effect?" None of these are tested. The claim is scoped narrowly to the conditions run — not all possible cheaper alternatives.

**Framing of Condition C as best-effort prompting (not minimal prompting).** Earlier drafts described C as "the cheapest upfront intervention" — a one-sentence nudge. After internal review, C was reframed as a **best-effort prompt-only intervention**: 2-3 sentences, concretely specifying what to verify (units, enum choices, numerics), with a fallback instruction for missing evidence. This raises the bar that E must clear. The point of C is no longer "what does the laziest possible prompt achieve" but "what does prompting alone, done well, achieve" — the comparison that actually matters for deployment decisions. The locked text lives at `prompts/condition_c.txt`.

**Why D is kept:** it is structurally distinct from C — post-hoc on the produced output, not an upfront instruction — and it is the cheapest verification pattern (one extra call vs interwhen's mid-stream forking). Without D, "E beats C" cannot rule out that a simple end-of-output check would have done equally well, which is the most important deployment question this study can answer.

**Why B' was added (secondary).** Condition B on Qwen3 revealed that the model elects to call tools on ~25% of vignettes (median 0 calls/vignette) — substantially less than closed-API models like Sonnet (mean 4.3 calls/vignette under the same setup). Because the primary input-verification track (C/D'/E) only acts on rows where a tool call occurs, the underuse rate constrains the effect-size ceiling of those conditions. B' adds a prompt-level intervention targeting the tool-use channel itself, structurally parallel to D' (both are B-variants intervening on different stages of the tool-use pipeline). B' vs B answers a distinct question — "does forcing tool use help open-weights models close the tool-access gap" — and is reported as exploratory because the condition was specified after observing baseline behavior. The primary input-verification claim is unchanged.

### Open methodological decisions (locked before production)

- **Exact prompt text for Condition C.** Drafted and pilot-validated; locked before production.
- **Verifier model and prompt for Condition D.** Verifier model is a different instance from the model being evaluated to avoid self-agreement bias. Verifier prompt drafted and pilot-validated. Revision policy (whether and how many times the original model is asked to revise on flagged inconsistency) pilot-determined.
- **Fact extractor model.** Different from the model being evaluated. Validation against a hand-annotated held-out set must show ≥95% field-level accuracy before production use.
- **Feedback format for Condition E.** First-person reflection vs user-message style. Pilot-determined.
- **Max retry rounds per fork point in E.** Default candidate: 2. Pilot-validated.
- **Sampling parameters** (temperature, top-p) for A/B/C/D/E. Match EkaCare's setup once verified in KARMA codebase, otherwise pilot-determined.
- **Calculator subset the verifier handles.** Specific calculators from EkaCare's 403 covered by the verifier. Affects which vignettes are eligible for E.
- **KARMA, model, and benchmark version pinning.** Specific git commits / model revisions / dataset revisions are pinned in `pyproject.toml` and recorded per-run in each bundle's `provenance/provenance.json`.

---

## 7. Statistical Approach

### Primary analysis

For each condition: overall accuracy with confidence intervals (Wilson). Accuracy decomposed by error type (wrong calculator vs wrong input, with sub-types from EkaCare's taxonomy).

For paired comparisons (same questions, different conditions): **McNemar's test**.

### Multiple comparisons

Four **confirmatory** comparisons (B vs A, E vs B, E vs C, E vs D) → Bonferroni correction applied to the family at α = 0.05 / 4 = 0.0125 per comparison.

The secondary comparison **B' vs B** is reported as **exploratory** and is *not* included in the Bonferroni family. It is McNemar-tested at uncorrected α = 0.05 with the exploratory designation made explicit in any reported result. This preserves statistical power on the primary input-verification claim and reflects that B' was added in response to a baseline observation rather than as a pre-specified confirmatory hypothesis.

### Pre-registration

Before running production experiments, commit an analysis plan to git documenting hypothesis, metrics, statistical tests, multiple-comparison strategy, effect-size threshold, confirmatory vs exploratory analyses. Separates pre-planned findings from post-hoc analyses.

### Verifier characterization

Beyond accuracy: firing rate, precision (when fired, was the issue real?), recall (when issues existed, did it catch them?), correction success rate, false-positive cost (did it derail correct answers?).

### Cost and latency (secondary endpoints)

Per the study's LMIC-deployment motivation (§4.5), each condition is also evaluated on resource cost. Per-vignette: mean prompt tokens, mean completion tokens, median wall-clock latency, number of model invocations, number of tool calls. Per-run: total wall-clock, estimated USD at a stated H100 hourly rate (default $3/hr).

These are **secondary endpoints**, reported alongside accuracy but not in the Bonferroni confirmatory family. The relevant deployment question is the cost–accuracy Pareto frontier: a condition that improves accuracy by 2 pp at 3× the cost may be unfit for the deployment context this study targets. Token counts are the most defensible cost metric (deterministic given prompt); wall-clock is condition-comparable only when `max_workers` is held constant.

---

## 8. Pre-Study Grounding

### Apparatus reproduction

Reproduce EkaCare's Claude Sonnet 4.6 + tools = 81.9% on `medical_calculator_eval` (within ±3pp) through the KARMA + MedAI wiring. Validates the apparatus before any experimental runs.

### Manual failure analysis

Examine 30–50 baseline (Condition B) failures from the Qwen3 reproduction by hand. Categorize using EkaCare's taxonomy (wrong calculator, wrong input — with enum / unit sub-types). Document any patterns EkaCare did not.

This grounds the verifier design in failures independently characterized on this study's actual baseline run, not just inherited from EkaCare's analysis.

---

## 9. Scope and Limitations

### What this study tests

Input-semantic verification for clinical calculator tool calls, on `medical_calculator_eval`, with Qwen3-30B-A3B-Thinking, through EkaCare's MedAI MCP (hierarchical meta-tool architecture), via KARMA.

### What this study does not test

- **Other clinical tasks** (drug ID, pharmacology, protocol retrieval). Out of scope.
- **Calculator-selection verification.** Verifier focuses on inputs only; wrong-calculator failures are out of scope. (Condition B' addresses *tool-use rate*, not tool *selection*; the latter remains out of scope.)
- **Other models, other LMIC contexts.** Single-model study; India-specific tools.
- ~~Cost and latency tradeoffs.~~ **Now in scope as secondary endpoints** (see §7). Token counts and wall-clock latency are recorded per vignette; total run cost is estimated in USD at a stated H100 hourly rate. The cost–accuracy Pareto frontier is reported alongside the primary accuracy results.
- **Cheaper alternatives beyond prompt instruction (C), post-hoc verification (D'), and tool-use enforcement (B').** Specifically: self-consistency, few-shot examples (C'), mid-reasoning self-verification (D), and schema-only verification (G). All dropped for leanness. Conclusions are scoped to the alternatives tested.

### Assumptions about deployment context

The architecture assumes the patient case is available **upfront as a single text input** at the start of the AI's session — corresponding to EHR notes, intake summaries, structured records summarized as text, or other static documents. The extractor reads this text once per case; the resulting patient facts are stable for the duration of the model's tool-use session within that case.

This maps cleanly to several real-world deployment patterns (EHR-summary-driven assistants, single-document clinical reasoning, intake-form review). It does **not** cover dynamic information-gathering scenarios where new patient information arrives incrementally during dialog (AI asks follow-up questions, patient or clinician provides more data over time). Those settings would require re-extraction or fact-merging logic; they are out of scope for this study.

### Effect-size ceiling on this model

Condition B on Qwen3 revealed substantial tool underuse (median 0 calls/vignette, mean 0.38; ~75% of vignettes solved with zero tool calls). Because the input-verification interventions (C/D/E) only act on rows where a tool call occurs, the absolute effect they can produce is bounded by the tool-use rate. Reported effect sizes for the primary track must be interpreted against this ceiling. Condition B' is included specifically to characterize this gap; its effect is therefore expected to be larger than the primary-track effects, but addresses a different question (tool-use rate) rather than the wrong-input residual the primary claim targets.

### Method-related notes

- **interwhen applied to a tool-use task.** Published interwhen evaluations cover pure-reasoning benchmarks (Maze, Game24, ZebraLogic, SpatialMap, Verina). Clinical tool-use is an extension to a task class the paper has not tested. Justifiable extension; not direct replication.
- **Verifier with LLM-based fact extractor.** Published interwhen verifiers are fully symbolic (Z3, compilers, arithmetic checks). This verifier has an LLM-based extraction step in front of deterministic comparison. Extractor accuracy is validated on a held-out set before production use.
- **Single benchmark, single model.** Findings generalize within this scope; cross-benchmark and cross-model claims would require additional work.

### What the result claims and does not claim

**The result claims:** On `medical_calculator_eval`, using EkaCare's MedAI tools through KARMA, adding interwhen-style intermediate verification of input semantics to a Qwen3-30B-A3B-Thinking model reduces wrong-input failure rates beyond what tool access alone or a prompt-level instruction provides.

**The result does not claim:** That intermediate verification solves all clinical AI failure modes; that it generalizes to non-calculator tasks, non-Indian contexts, or other models; that it removes the need for clinician review; that it beats cheaper alternatives not tested here (self-consistency, few-shot, schema-only verification).

---

## 10. Project Structure

```
interwhen-karma-harness/
├── pyproject.toml            # Pinned dependencies (KARMA, interwhen via git@sha)
├── README.md                 # Quickstart pointer
├── TESTING.md                # This document
├── harness/                  # Python package (this project's code)
│   ├── extraction/           # Fact extractor (LLM with structured output)
│   ├── verifier/             # Semantic verifier (deterministic comparison)
│   ├── monitors/             # interwhen Monitor subclass (clinical_input)
│   ├── karma_adapter/        # Sonnet adapter (apparatus) + Qwen3 adapter
│   ├── analysis/             # Statistical primitives (Wilson, McNemar, Bonferroni)
│   └── _patches.py           # Runtime patch: adds EKA_API_TOKEN Bearer auth to KARMA
├── prompts/                  # Locked prompt texts per condition (committed)
├── notebooks/                # Databricks notebooks: 00_smoke, 01_apparatus, 02_run_all, 03_analysis
└── results/                  # Gitignored — outputs to DBFS / object storage
```

KARMA and interwhen are **external pinned dependencies** (`pip install` from
pinned git commits), not vendored source. The custom adapter extends KARMA via
its registry interface; the custom monitors subclass interwhen's `VerifyMonitor`.
Neither framework is modified in this repo.

---

## 11. Replication

All code, model versions, dataset revisions, MCP calculator schemas (snapshotted per run in `provenance/mcp_calculator_schemas.json`), and locked prompts released open-source. Raw logs (model traces, verifier firings, feedback messages) released where licensing permits.

**Compute requirements:** vLLM-served Qwen3-30B-A3B-Thinking-2507 requires a single-node GPU with ≥80GB VRAM (1× H100 80GB or 2× A100 40GB with tensor parallelism). Any environment that supports this works — local workstation, cloud GPU rental, or managed clusters (e.g., Databricks single-node GPU compute).

**MedAI auth on headless compute.** Browser-based OIDC is not usable on headless cluster nodes. Use an Eka API token via direct `Authorization` header (request from `ekaconnect@eka.care`), or perform the OAuth flow once on a workstation and copy the persisted token store (`~/.fastmcp/oauth-tokens/`) to the cluster.

**Non-replicable element:** MedAI access (gated by Eka account). Request via `ekaconnect@eka.care`.

---

## 12. References

- **EkaCare benchmarking:** "Beyond Frontier Models: Grounding AI in Indian Clinical Reality" (April 2026). Motivating findings; benchmark, tools, and KARMA framework.
- **interwhen:** Bhat et al. (2026), arXiv:2602.11202. Verifier-guided reasoning framework. Code: `github.com/microsoft/interwhen`.
- **KARMA:** `github.com/eka-care/KARMA-OpenMedEvalKit`. Docs: `karma.eka.care`.
- **MedAI Tools:** `medai-tools.eka.care/mcp`. Docs: `developer.eka.care`.
