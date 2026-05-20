# interwhen-karma-harness

Evaluation harness for testing intermediate verification of clinical AI tool calls.

Combines:
- **interwhen** (Microsoft Research) — verifier-guided reasoning at test time
- **KARMA** (EkaCare) — medical AI evaluation framework
- **MedAI MCP** (EkaCare) — hosted clinical calculator tools
- **medical_calculator_eval** — benchmark of Indian clinical vignettes

The full methodology, experimental design, and statistical plan live in [`TESTING.md`](./TESTING.md).

## Setup

```bash
cp .env.example .env       # then fill in real values
pip install -e ".[serving,tracking,dev]"
```

All required keys and env vars live in [`.env.example`](./.env.example) — that is the single source of truth for what you need to provide. On Databricks, store the same keys in a secrets scope and load them at the top of each notebook (see `notebooks/`).

See [`TESTING.md` Section 11](./TESTING.md#11-replication) for compute requirements and auth notes for headless cluster runs.

## Layout

```
harness/              # this project's code (Python package)
conf/                 # pre-registered prompts, sampling params
data/                 # fact-extractor validation set, calculator subset spec
notebooks/            # Databricks notebooks for runs and analysis
docs/                 # failure analysis and other write-ups
results/              # gitignored — outputs go to DBFS / object storage
```
