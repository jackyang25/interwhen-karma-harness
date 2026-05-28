# Databricks notebook source
# DBTITLE 1,02 — Orchestrator
# MAGIC %md
# MAGIC # 02 — Orchestrator: run all conditions, aggregate, report
# MAGIC
# MAGIC Single notebook to click before going to bed. Runs every Qwen3 condition
# MAGIC against one vLLM server, recovers from per-condition errors, and at the
# MAGIC end produces an aggregated results table.
# MAGIC
# MAGIC **Nine-condition pre-registered design** (see paper §methods_conditions):
# MAGIC
# MAGIC | # | Condition | Group | Description |
# MAGIC |---|-----------|-------|-------------|
# MAGIC | 1 | A | anchor | No tools, no system prompt — capability floor |
# MAGIC | 2 | B | anchor | Tools enabled, no system prompt — apparatus baseline |
# MAGIC | 3 | B' | anchor | Tools + force-tool-use system prompt |
# MAGIC | 4 | B'+E (upfront) | primary | Upfront full-schema extraction, schema-gated verifier, query intervention, abstention |
# MAGIC | 5 | B'+E (reactive) | primary | Reactive per-tool-call scoped extraction, same hygiene |
# MAGIC | 6 | B'+E (reactive + citations) | primary | Reactive + (value, source_span) extraction; substring-validated |
# MAGIC | 7 | B'+E (reactive + k-shot) | primary | Reactive + k=3 majority-vote extraction |
# MAGIC | 8 | C | exploratory | Prompt-only self-verify — no extractor |
# MAGIC | 9 | D | exploratory | Post-hoc Sonnet verifier — no extractor |
# MAGIC
# MAGIC **Pre-registered hygiene (baked into all primary conditions 4-7):**
# MAGIC - schema-gated verifier comparison (only fields the active calculator requires)
# MAGIC - query-style intervention prompts (no assertion of model error)
# MAGIC - prompt-only calibrated abstention (extractor omits low-confidence fields)
# MAGIC - Sonnet temperature 0.7 with fresh seeds per call
# MAGIC - re-prompt cap = 2 per vignette
# MAGIC
# MAGIC **Design properties:**
# MAGIC - **Idempotent:** each condition checks for an existing
# MAGIC   `summary.json` and skips if found. Re-running picks up where it left off.
# MAGIC - **Error-tolerant:** if a condition crashes, log + continue to the next.
# MAGIC - **One vLLM server:** launched once, used for all conditions, stopped at end.
# MAGIC - **Pilot-gated:** each condition runs an n=10 pilot first; warnings on 0%
# MAGIC   accuracy or 0 tool calls; proceeds either way (user inspects).
# MAGIC - **Symmetric logging:** every condition writes the same per-vignette
# MAGIC   record shape (see paper §app:logging) so downstream analysis is uniform.
# MAGIC
# MAGIC **Before clicking Run All:**
# MAGIC - Smoke-test once via 00_smoke_test before launching
# MAGIC - Paste API keys in section 2
# MAGIC - Confirm `IDM-H100GPU-Compute_*` is attached
# MAGIC - Set `CONDITIONS_TO_RUN` in §⚙️ below to the conditions you want fresh data for

# COMMAND ----------

# MAGIC %md ## 1. Install deps

# COMMAND ----------

# MAGIC %pip install -q \
# MAGIC   "transformers==4.52.4" \
# MAGIC   "openai>=1.0" "httpx" "anthropic" \
# MAGIC   "fastmcp-slim[client]>=2.0" "datasets>=2.0" "huggingface-hub" \
# MAGIC   "pandas" "numpy" "scipy" "statsmodels" "nest-asyncio" \
# MAGIC   "nvidia-cusparselt-cu12" \
# MAGIC   "pydantic>=2.0" "regex" "tiktoken" "matplotlib" "z3-solver" "sympy"
# MAGIC %pip install -q --no-deps \
# MAGIC   "interwhen @ git+https://github.com/microsoft/interwhen.git@2d041c2f3ed2a6f0a4b063463b3aef844e7dba5e"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Patch interwhen EAT_helper (upstream vllm top-level import)
# interwhen@2d041c2f has `from vllm import SamplingParams` at the top of
# EAT_helper.py, which is pulled in by interwhen/monitors/__init__.py via
# earlyStopping.py — even though condition E never uses EATMonitor/DEERMonitor.
# vllm lives in /tmp/vllm_env (the isolated venv), not the main kernel, so
# the import fails at module load time. Patch it to a try/except on every boot.
import pathlib, sys

for sp in sys.path:
    p = pathlib.Path(sp) / "interwhen/utils/EAT_helper.py"
    if p.exists():
        content = p.read_text()
        old = "from vllm import SamplingParams"
        new = (
            "try:\n"
            "    from vllm import SamplingParams\n"
            "except ModuleNotFoundError:\n"
            "    SamplingParams = None  # vllm not in main kernel"
        )
        if old in content and new not in content:
            p.write_text(content.replace(old, new, 1))
            print(f"Patched interwhen EAT_helper.py at {p}")
        else:
            print(f"EAT_helper.py already patched at {p}")
        break
else:
    print("WARNING: interwhen/utils/EAT_helper.py not found on sys.path")

# COMMAND ----------

# MAGIC %md ## 2. Paste secrets

# COMMAND ----------

import os

os.environ["EKA_API_TOKEN"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""

# COMMAND ----------

# MAGIC %md ## 3. Pre-flight + LD_LIBRARY_PATH

# COMMAND ----------

# MAGIC %sh nvidia-smi
# MAGIC echo "---"
# MAGIC find /databricks /local_disk0 -name "libcusparseLt.so.0" 2>/dev/null | head -3

# COMMAND ----------

import os

CUSPARSELT_DIR = "/databricks/python3/lib/python3.12/site-packages/cusparselt/lib"
os.environ["LD_LIBRARY_PATH"] = CUSPARSELT_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")
print("LD_LIBRARY_PATH:", os.environ["LD_LIBRARY_PATH"])

# COMMAND ----------

# MAGIC %md ## 4. Build vLLM venv + launch server (ONCE for all conditions)

# COMMAND ----------

# MAGIC %sh
# MAGIC set -e
# MAGIC rm -rf /tmp/vllm_env
# MAGIC apt-get update && apt-get install -y python3-venv
# MAGIC python3 -m venv /tmp/vllm_env
# MAGIC /tmp/vllm_env/bin/python -m ensurepip --upgrade
# MAGIC /tmp/vllm_env/bin/pip install --quiet --upgrade pip
# MAGIC /tmp/vllm_env/bin/pip install --quiet \
# MAGIC   vllm==0.9.2 transformers==4.52.4 openai httpx
# MAGIC ls /tmp/vllm_env/bin/vllm && echo "venv ready"

# COMMAND ----------

try:
    server.stop()
except NameError:
    pass

import os, subprocess, textwrap, pathlib

_ossl_conf = pathlib.Path("/tmp/openssl_nofips.cnf")
_ossl_conf.write_text(textwrap.dedent("""\
    openssl_conf = openssl_init

    [openssl_init]
    providers = provider_sect

    [provider_sect]
    default = default_sect

    [default_sect]
    activate = 1
"""))

subprocess.run(
    ["/tmp/vllm_env/bin/pip", "uninstall", "-y", "opencv-python-headless"],
    capture_output=True, timeout=60,
)
print("opencv-python-headless removed from venv")

os.environ.pop("_PIP_USE_IMPORTLIB_METADATA", None)
os.chdir("/tmp")

from harness.karma_adapter.qwen3 import VLLMServer

server = VLLMServer(
    model_id="Qwen/Qwen3-30B-A3B-Thinking-2507",
    gpu_memory_utilization=0.80,
    max_model_len=16384,
    vllm_bin="/tmp/vllm_env/bin/vllm",
    env_overrides={
        "VLLM_USE_V1": "0",
        "LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"],
        "OPENSSL_CONF": str(_ossl_conf),
        "OPENSSL_MODULES": "",
        "OPENSSL_FORCE_FIPS_MODE": "0",
    },
)
server.start()
server.wait_ready(timeout=900)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.5 Schema-aligned extractor vocabulary (preflight)
# MAGIC
# MAGIC Walk every MCP calculator's input schema, dump to provenance, and
# MAGIC regenerate the Sonnet extractor prompt from the field union. This
# MAGIC keeps the extractor's vocabulary a deterministic function of MCP's
# MAGIC current state — no hand-curated aliases, no synonym dictionary, no
# MAGIC bridging logic between MCP and the verifier.
# MAGIC
# MAGIC Cached: if `provenance/mcp_calculator_schemas.json` already exists,
# MAGIC the dump is skipped. Delete that file to force a rebuild (e.g.,
# MAGIC after EkaCare adds a calculator).

# COMMAND ----------

# DBTITLE 1,Preflight: fetch MCP schemas + regenerate extractor prompt
import asyncio
import json as _pf_json
import nest_asyncio
nest_asyncio.apply()

from fastmcp import Client
from harness.karma_adapter.mcp_tools import MEDAI_MCP_URL, _run_async
from harness.extraction.prompt_builder import regenerate_extractor_prompt

_PROVENANCE = pathlib.Path("/dbfs/results/provenance")
_RUNTIME    = pathlib.Path("/dbfs/results/_runtime")
_PROVENANCE.mkdir(parents=True, exist_ok=True)
_RUNTIME.mkdir(parents=True, exist_ok=True)

SCHEMA_DUMP_PATH      = _PROVENANCE / "mcp_calculator_schemas.json"
RUNTIME_EXTRACTOR_PATH = _RUNTIME    / "extractor_prompt.txt"


def _as_text(call_result):
    return "".join(b.text for b in call_result.content if hasattr(b, "text"))


async def _dump_all_calculator_schemas():
    """Walk MCP: list categories → list calculators per category → get each
    calculator's input schema. Returns the full dump dict."""
    async with Client(MEDAI_MCP_URL) as c:
        # 1. Categories
        r = await c.call_tool("medical_calculator_list", {"intent": "categories"})
        categories = _pf_json.loads(_as_text(r))

        # 2. Calculators per category
        all_calcs = []
        for cat_obj in categories:
            cat = cat_obj["category"]
            r2 = await c.call_tool("medical_calculator_list",
                                    {"intent": "calculators", "category": cat})
            for calc in _pf_json.loads(_as_text(r2)):
                all_calcs.append({
                    "category":        cat,
                    "name":            calc["name"],
                    "normalized_name": calc["normalized_name"],
                })

        # 3. Per-calculator input schemas
        per_calc, failures = {}, []
        for i, calc in enumerate(all_calcs, 1):
            try:
                r3 = await c.call_tool("medical_calculator_input",
                                        {"calculator_name": calc["normalized_name"]})
                per_calc[calc["normalized_name"]] = {
                    "category": calc["category"],
                    "name":     calc["name"],
                    "schema":   _pf_json.loads(_as_text(r3)),
                }
            except Exception as e:
                failures.append({"calc": calc["normalized_name"],
                                  "error": f"{type(e).__name__}: {e}"})
            if i % 50 == 0:
                print(f"  ...{i}/{len(all_calcs)} schemas fetched")

        # 4. Field union for quick reference (also recoverable from per_calc)
        all_fields = set()
        for cd in per_calc.values():
            all_fields.update((cd["schema"].get("properties") or {}).keys())

        return {
            "n_categories":         len(categories),
            "n_calculators_listed": len(all_calcs),
            "n_schemas_fetched":    len(per_calc),
            "n_failures":           len(failures),
            "field_union":          sorted(all_fields),
            "per_calc_schemas":     per_calc,
            "failures":             failures,
        }


# Run the preflight (cached unless the dump file is missing).
if SCHEMA_DUMP_PATH.exists():
    print(f"[preflight] schema dump cached at {SCHEMA_DUMP_PATH} — skipping fetch")
    print(f"[preflight]   (delete it to force a rebuild)")
else:
    print(f"[preflight] dumping MCP calculator schemas to {SCHEMA_DUMP_PATH}...")
    dump_data = _run_async(_dump_all_calculator_schemas())
    SCHEMA_DUMP_PATH.write_text(_pf_json.dumps(dump_data, indent=2))
    print(f"[preflight]   {dump_data['n_schemas_fetched']}/{dump_data['n_calculators_listed']} "
          f"calculator schemas fetched ({dump_data['n_failures']} failures)")
    print(f"[preflight]   {len(dump_data['field_union'])} unique clinical field names")

# Always regenerate the extractor prompt from the (now-current) dump.
# The runtime prompt path is what FactExtractor reads (see _build_adapter('E')).
meta = regenerate_extractor_prompt(SCHEMA_DUMP_PATH, RUNTIME_EXTRACTOR_PATH)
print(f"[preflight] extractor prompt regenerated → {meta['output_path']}")
print(f"[preflight]   {meta['n_fields']} fields, {meta['prompt_chars']} chars")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ Orchestration knobs — edit before clicking Run All
# MAGIC
# MAGIC - `ALL_CONDITIONS`: canonical 9-condition list with group + rerun-status
# MAGIC   metadata. **Do not edit this dict** — it's the source of truth for what
# MAGIC   the orchestrator can run. The paper's results tables join on these
# MAGIC   condition IDs.
# MAGIC - `CONDITIONS_TO_RUN`: subset of `ALL_CONDITIONS` keys that this
# MAGIC   notebook attempts. Use one of the preset lists below or define your
# MAGIC   own. Idempotency means already-complete conditions are still skipped
# MAGIC   even if you leave them in (unless `FORCE_RERUN=True`).
# MAGIC - `FORCE_RERUN`: if True, ignores existing `summary.json` and re-runs
# MAGIC   each selected condition from scratch. Old per-row parquets are backed
# MAGIC   up first (see BACKUP_LEGACY).
# MAGIC - `BACKUP_LEGACY`: if True (default), any existing `rows.parquet` in a
# MAGIC   condition's result dir is moved to a timestamped backup directory
# MAGIC   before the new run writes. Safe-by-default — no historical data is destroyed.

# COMMAND ----------

# DBTITLE 1,Canonical 9-condition table — DO NOT EDIT
# ──────────────────────────────────────────────────────────────────────────────
# This dict is the canonical source of truth for the 9-condition pre-registered
# design. Each entry records:
#   - group:    "anchor" | "primary" | "exploratory"
#   - rerun:    True if this condition is in the rerun set for the new design,
#               False if it inherits existing results from the prior run.
#   - placement / output_format / sampling: orthogonal axes (primary only)
#   - description: one-liner
#
# DO NOT add ad-hoc conditions here. The paper's contrast family is built
# against this table.
# ──────────────────────────────────────────────────────────────────────────────
ALL_CONDITIONS: dict[str, dict] = {
    "A": {
        "group": "anchor", "rerun": False,
        "description": "No tools, no system prompt — capability floor",
    },
    "B": {
        "group": "anchor", "rerun": True,
        "description": "Tools enabled, no system prompt — apparatus baseline",
    },
    "B_prime": {
        "group": "anchor", "rerun": True,
        "description": "Tools + force-tool-use system prompt",
    },
    "B_prime_E": {
        "group": "primary", "rerun": True,
        "placement": "upfront", "output_format": "bare", "sampling": "single",
        "description": "Upfront full-schema extraction + best-practice hygiene",
    },
    "B_prime_E_reactive": {
        "group": "primary", "rerun": True,
        "placement": "reactive", "output_format": "bare", "sampling": "single",
        "description": "Reactive per-call extraction + best-practice hygiene",
    },
    "B_prime_E_reactive_citations": {
        "group": "primary", "rerun": True,
        "placement": "reactive", "output_format": "citation", "sampling": "single",
        "description": "Reactive + (value, source_span); substring-validated",
    },
    "B_prime_E_reactive_kshot": {
        "group": "primary", "rerun": True,
        "placement": "reactive", "output_format": "bare", "sampling": "kshot_3",
        "description": "Reactive + k=3 majority-vote extraction",
    },
    "C": {
        "group": "exploratory", "rerun": False,
        "description": "Prompt-only self-verify — inherits existing results",
    },
    "D": {
        "group": "exploratory", "rerun": False,
        "description": "Post-hoc Sonnet verifier — inherits existing results",
    },
}

# Convenience presets — pick one for CONDITIONS_TO_RUN.
# ────────────────────────────────────────────────────────────────────────────
RERUN_ALL_NEW    = [k for k, v in ALL_CONDITIONS.items() if v["rerun"]]                # 6 conditions
RERUN_PRIMARY    = [k for k, v in ALL_CONDITIONS.items() if v["group"] == "primary"]   # 4 primary
RERUN_ANCHORS    = [k for k, v in ALL_CONDITIONS.items() if v["group"] == "anchor" and v["rerun"]]  # B, B_prime
RERUN_NEW_ARMS   = ["B_prime_E_reactive_citations", "B_prime_E_reactive_kshot"]        # the 2 new mechanism arms
INHERIT_ONLY     = [k for k, v in ALL_CONDITIONS.items() if not v["rerun"]]            # A, C, D

# COMMAND ----------

# DBTITLE 1,Orchestration knobs (edit before clicking Run All)
# ──────────────────────────────────────────────────────────────────────────────
# Edit the line below to choose which conditions this notebook will rerun.
# Use one of the presets above or list condition IDs explicitly.
#
# Examples:
#   CONDITIONS_TO_RUN = RERUN_ALL_NEW   # all 6 conditions needing fresh data (default)
#   CONDITIONS_TO_RUN = RERUN_PRIMARY   # the 4 primary study conditions
#   CONDITIONS_TO_RUN = RERUN_ANCHORS   # just B and B_prime
#   CONDITIONS_TO_RUN = RERUN_NEW_ARMS  # just the 2 new mechanism arms
#                                       # (citations + k-shot)
#   CONDITIONS_TO_RUN = ["B_prime_E"]   # a single condition
# ──────────────────────────────────────────────────────────────────────────────
CONDITIONS_TO_RUN: list[str] = RERUN_ALL_NEW

FORCE_RERUN   = True        # ignore existing summary.json; rerun selected conditions
BACKUP_LEGACY = True        # move existing rows.parquet to _backup_<ts>/ before write

# Validate selection up-front so a typo doesn't surface mid-overnight-run.
_unknown = [c for c in CONDITIONS_TO_RUN if c not in ALL_CONDITIONS]
assert not _unknown, f"Unknown condition(s) in CONDITIONS_TO_RUN: {_unknown}"
print(f"Will run {len(CONDITIONS_TO_RUN)} conditions:")
for c in CONDITIONS_TO_RUN:
    print(f"  - {c}  [{ALL_CONDITIONS[c]['group']}]  {ALL_CONDITIONS[c]['description']}")

# COMMAND ----------

# MAGIC %md ## 5. Infrastructure

# COMMAND ----------

# DBTITLE 1,Imports, paths, hygiene config, WORKERS, system-prompt routing
import json
import time
import traceback
from pathlib import Path

from harness.karma_adapter.qwen3 import Qwen3Adapter
from harness.runner import run_eval
from harness.analysis import wilson_ci

REPO_ROOT    = Path("/Workspace/Users/jack.yang@gatesfoundation.org/interwhen-karma-harness")
RESULTS_ROOT = Path("/dbfs/results")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Pre-registered hygiene configuration (paper §methods_prereg).
# These values are identical across all primary conditions (4-7). They are NOT
# experimental factors — they are design choices documented in methods.
# DO NOT modify these between runs without updating the pre-registration.
# ──────────────────────────────────────────────────────────────────────────────
PREREG_CONFIG = {
    "schema_gated_verifier":          True,    # compare only fields the active calculator declares required
    "intervention_style":             "query", # "query" template (not "correction")
    "abstention":                     "prompt-only",  # extractor returns null on low confidence
    "sonnet_extraction_temperature":  0.7,
    "sonnet_seed_strategy":           "fresh-per-call",  # no prompt caching
    "qwen3_temperature":              0.0,
    "tool_call_loop_turn_cap":        10,
    "verifier_reprompt_cap":          2,        # per-vignette total
    "citation_validity":              "verbatim-substring",
    "citation_failure":               "null",   # treat as abstention
    "kshot_voting_rule":              "majority-of-3",
    "kshot_no_majority":              "null",   # treat as abstention
    "kshot_k":                        3,
    "failure_handling":               "drop-and-report",
}
# The query-style intervention template (replaces the correction-style template
# used in the pilot runs). Lives at prompts/condition_e_feedback_query.txt.
FEEDBACK_TEMPLATE_PATH = REPO_ROOT / "prompts/condition_e_feedback_query.txt"

# Pilot-style template kept on disk for reproducibility of legacy E/B_prime_E
# runs (the prior 8-condition study used this one). Not used by any primary
# study condition — only by the documented pilot-diagnostic runs.
FEEDBACK_TEMPLATE_PILOT_PATH = REPO_ROOT / "prompts/condition_e_feedback.txt"

# ──────────────────────────────────────────────────────────────────────────────
# Pilot+full max-workers per condition. Picked to match the same numbers each
# individual condition notebook uses, so results are comparable across runs.
# k-shot uses fewer workers because each vignette dispatches k=3 parallel
# Sonnet calls per tool call (~3.6 calls/vignette × k=3 ≈ 11 concurrent
# Sonnet calls per vignette worker).
# ──────────────────────────────────────────────────────────────────────────────
WORKERS = {
    "A":                             128,
    "B":                              64,
    "B_prime":                        64,
    "C":                              64,
    "D":                              32,   # Sonnet verifier per vignette
    "B_prime_E":                      32,   # upfront full-schema Sonnet pressure
    "B_prime_E_reactive":             32,   # reactive: ~3.6 Sonnet calls/vignette, sequential
    "B_prime_E_reactive_citations":   32,   # same as reactive; citation validation is local
    "B_prime_E_reactive_kshot":       16,   # k=3 × ~3.6 calls/vignette → halve workers
}

PILOT_N = 10


def _system_for(cond: str) -> str | None:
    """Return the locked system prompt for the condition, or None for no system prompt.

    None → adapter.run(system=None) → no system message in the chatml prompt.
    All four primary study conditions (B_prime_E and variants) share the
    locked B' system prompt — the only inter-condition variation lives in
    the extractor pipeline, NOT in the system prompt.
    """
    PRIMARY_SYSTEM = REPO_ROOT / "prompts/condition_b_prime.txt"
    mapping = {
        "A":                             None,
        "B":                             None,
        "B_prime":                       PRIMARY_SYSTEM,
        "C":                             REPO_ROOT / "prompts/condition_c.txt",
        "D":                             None,   # Sonnet verifier has its own prompt
        # All four primary study conditions share the locked B' system prompt.
        # The only inter-condition variation lives in the extractor pipeline.
        "B_prime_E":                     PRIMARY_SYSTEM,
        "B_prime_E_reactive":            PRIMARY_SYSTEM,
        "B_prime_E_reactive_citations":  PRIMARY_SYSTEM,
        "B_prime_E_reactive_kshot":      PRIMARY_SYSTEM,
    }
    p = mapping.get(cond)
    return p.read_text().strip() if p is not None else None

# COMMAND ----------

# DBTITLE 1,Schema-gating helper (filter violations to active calculator's required fields)
# ──────────────────────────────────────────────────────────────────────────────
# The pre-registered design schema-gates the verifier: only flag fields that
# the active calculator declares as required. Without this filter, the
# verifier flags fields the calculator never reads (58.6% of pilot flags
# fell on non-required fields — see paper §methods_pilots). This helper is
# applied to event_info["violations"] in every primary-study adapter,
# AFTER monitor.verify() runs.
# ──────────────────────────────────────────────────────────────────────────────
def _required_fields_for_calculator(schema_dump: dict, calc_name: str) -> set[str] | None:
    """Look up the required-fields set for a calculator from the MCP schema
    dump. Returns None if the calculator isn't in the dump (caller decides
    whether to fail open or closed)."""
    per_calc = schema_dump.get("per_calc_schemas", {}) or {}
    entry = per_calc.get(calc_name)
    if entry is None:
        return None
    schema = entry.get("schema") or {}
    # input_data is the clinical args; the required list lives one level deeper
    # in the per-calculator schema structure.
    if "input_data" in schema and isinstance(schema["input_data"], dict):
        req = schema["input_data"].get("required") or []
    else:
        req = schema.get("required") or []
    return set(req)


def _schema_gate_violations(violations, call_obj, schema_dump):
    """Filter a violations list to only those on fields the active calculator
    declares as required. Pass-through if we can't identify the calculator
    (fail open, with a marker — analysis can detect this)."""
    if not violations:
        return violations
    if not isinstance(call_obj, dict):
        return violations
    if call_obj.get("name") != "medical_calculator_output":
        return violations  # non-clinical tool call — no schema to gate against
    args = call_obj.get("arguments") or {}
    input_data = args.get("input_data") or {}
    # The calculator name lives in the input_data under "calculator_name" or
    # similar — check both common conventions used in the MCP responses.
    calc_name = (
        args.get("calculator_name")
        or input_data.get("calculator_name")
        or args.get("name")
    )
    if calc_name is None:
        return violations
    required = _required_fields_for_calculator(schema_dump, calc_name)
    if required is None:
        return violations  # calculator not in dump — fail open
    return [v for v in violations if v.field in required]

# COMMAND ----------

# DBTITLE 1,Plain adapters (A/B/C/B', D)
def _build_plain_adapter(cond: str):
    """A/B/C/B': plain Qwen3Adapter; use_tools toggled by whether the
    condition exposes tools (A has none, B/C/B' do). The locked system prompt
    that differentiates B/C/B' is threaded in by run_eval via _system_for,
    NOT by the adapter itself."""
    return Qwen3Adapter(
        base_url=server.base_url,
        use_tools=(cond != "A"),
    )


def _build_posthoc_adapter():
    """D: Qwen3 primary + Sonnet post-hoc verifier. The verifier receives
    (case, candidate answer) and flags inconsistencies; on flag the primary
    is asked to revise once. Token counts and model calls include the
    verifier + revision (see harness/verifier/posthoc.py)."""
    from harness.verifier import PostHocVerifierAdapter
    primary = Qwen3Adapter(base_url=server.base_url, use_tools=True)
    return PostHocVerifierAdapter(
        primary=primary,
        verifier_prompt_path=str(REPO_ROOT / "prompts/condition_d.txt"),
        verifier_model="claude-sonnet-4-6",
    )

# COMMAND ----------

# DBTITLE 1,_VerifiedAdapter builder (E, B_prime_E)
def _build_verified_adapter():
    """E and B_prime_E: Qwen3 tool-calling loop with ClinicalInputMonitor intercept.

    B_prime_E (exploratory) shares this exact adapter with E. The only
    difference vs E is the system prompt (handled by _system_for, threaded
    through run_eval → adapter.run(prompt, system=...)). Sharing the adapter
    is the methodological point: we isolate the system-prompt effect on top
    of the E mechanics, with no other code-path delta.

    Architecture (mirrors Qwen3Adapter.run() exactly, adds one step):
      1. Call /v1/completions, stop at </tool_call>
      2. If no tool call → return final answer
      3. If tool call → run ClinicalInputMonitor.fix():
           violations found → inject feedback into prompt, re-generate (no MCP call)
           no violations    → dispatch MCP tool, inject real tool_response, continue
           malformed JSON   → inject correction request, re-generate
      4. Loop until final answer or MAX_TOOL_TURNS exhausted
    """
    from harness.extraction import FactExtractor
    from harness.monitors import ClinicalInputMonitor
    from harness.karma_adapter.mcp_tools import fetch_tool_schemas
    from harness.karma_adapter.qwen3 import Qwen3Response
    from transformers import AutoTokenizer
    from openai import OpenAI
    import asyncio, re
    import nest_asyncio
    nest_asyncio.apply()

    _MODEL_ID   = "Qwen/Qwen3-30B-A3B-Thinking-2507"
    _MAX_TURNS  = 10
    _THINK_RE   = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

    # The schema-aligned runtime extractor prompt is generated by the
    # preflight cell from MCP's per-calculator schemas. No fallback — if
    # missing, preflight didn't run and we halt rather than silently use a
    # stale/incompatible schema.
    if not RUNTIME_EXTRACTOR_PATH.exists():
        raise FileNotFoundError(
            f"Runtime extractor prompt not found at {RUNTIME_EXTRACTOR_PATH}. "
            f"The schema-dump preflight cell (§4.5) must run before E/B_prime_E. "
            f"Run that cell, or delete /dbfs/results/provenance/mcp_calculator_schemas.json "
            f"to force a fresh fetch + regeneration."
        )
    extractor         = FactExtractor(
        prompt_path=str(RUNTIME_EXTRACTOR_PATH),
        model="claude-sonnet-4-6",
        temperature=PREREG_CONFIG["sonnet_extraction_temperature"],
    )
    # Pre-registered: query-style intervention template (not the correction-style
    # template used in the pilot runs). See PREREG_CONFIG.
    feedback_template = FEEDBACK_TEMPLATE_PATH.read_text()
    tools             = fetch_tool_schemas()
    tokenizer         = AutoTokenizer.from_pretrained(_MODEL_ID)
    client            = OpenAI(base_url=server.base_url, api_key="EMPTY")

    # Load schema dump for the schema-gating filter (paper §methods_prereg).
    import json as _vj
    if not SCHEMA_DUMP_PATH.exists():
        raise FileNotFoundError(
            f"MCP schema dump not found at {SCHEMA_DUMP_PATH}. "
            f"The preflight (§4.5) must run before any primary study condition."
        )
    schema_dump_local = _vj.loads(SCHEMA_DUMP_PATH.read_text())

    class _VerifiedAdapter:
        """Qwen3 completions loop with per-tool-call semantic verification."""

        def __init__(self):
            self._last_monitor = None   # populated after each run(); inspect for smoke tests

        def run(self, prompt, system=None):
            import time as _time

            # ── 1. Extract patient facts (Sonnet, deterministic) ──────────────
            t_extract = _time.time()
            facts = extractor.extract(prompt)
            extractor_elapsed_s = _time.time() - t_extract
            extractor_prompt_tokens = facts.prompt_tokens
            extractor_completion_tokens = facts.completion_tokens

            # ── 2. Render prompt with MCP tool schemas ────────────────────────
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            rendered = tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=False,
            )

            # ── 3. Per-vignette monitor (stateful; one per run() call) ─────────
            monitor = ClinicalInputMonitor(
                patient_facts=facts,
                feedback_template=feedback_template,
            )
            self._last_monitor = monitor
            loop = asyncio.get_event_loop()

            n_tool_calls    = 0     # successful MCP dispatches
            n_model_calls   = 0     # LLM calls (>tool_calls when verifier fires)
            n_reprompts_applied = 0  # post-gating verifier re-prompts; capped by PREREG_CONFIG
            qwen3_prompt_tokens     = 0
            qwen3_completion_tokens = 0
            rolling_prompt  = rendered

            # ── 4. Tool-calling loop (timed; tokens summed from vLLM usage) ───
            t_qwen3 = _time.time()
            for _ in range(_MAX_TURNS + 1):
                resp = client.completions.create(
                    model=_MODEL_ID,
                    prompt=rolling_prompt,
                    max_tokens=4096,
                    temperature=0.0,
                    top_p=1.0,
                    stop=["</tool_call>"],
                )
                n_model_calls += 1
                if resp.usage is not None:
                    qwen3_prompt_tokens     += getattr(resp.usage, "prompt_tokens",     0) or 0
                    qwen3_completion_tokens += getattr(resp.usage, "completion_tokens", 0) or 0

                choice    = resp.choices[0]
                generated = choice.text or ""

                # vLLM strips the stop string; re-attach so regex sees full block.
                if (choice.finish_reason == "stop"
                        and "<tool_call>" in generated
                        and "</tool_call>" not in generated):
                    generated += "</tool_call>"

                rolling_prompt += generated

                # No tool call → final answer.
                if "<tool_call>" not in generated:
                    break

                # Tool call detected: verify, then dispatch or inject feedback.
                monitor.metrics.n_steps_seen += 1
                event_info: dict = {}
                loop.run_until_complete(monitor.verify(
                    chunk=generated,
                    token_index=0,
                    event=asyncio.Event(),  # signal not needed; event_info carries results
                    event_info=event_info,
                ))

                # Pre-registered: schema-gate the violations to the active
                # calculator's required fields. Without this, the verifier
                # flags fields the calculator never reads (paper §methods_prereg).
                if event_info.get("violations"):
                    event_info["violations"] = _schema_gate_violations(
                        event_info["violations"],
                        event_info.get("call_obj"),
                        schema_dump_local,
                    )

                # Pre-registered: re-prompt cap (PREREG_CONFIG["verifier_reprompt_cap"]).
                # Use a local counter that only increments when violations
                # SURVIVE schema-gating — flagged-then-gated violations don't
                # count. Cap = max re-prompts allowed; demote on the (cap+1)th.
                if event_info.get("violations"):
                    if n_reprompts_applied >= PREREG_CONFIG["verifier_reprompt_cap"]:
                        event_info["violations"] = []   # demote: cap reached
                    else:
                        n_reprompts_applied += 1

                # fix() returns the new rolling_prompt:
                #   violations → feedback injected, no MCP call
                #   clean call  → MCP dispatched, tool_response injected
                #   malformed   → correction request injected
                rolling_prompt = loop.run_until_complete(
                    monitor.fix(rolling_prompt, event_info)
                )

                # Count only real MCP dispatches (no violations, not malformed).
                if not event_info.get("violations") and not event_info.get("malformed"):
                    n_tool_calls += 1
            qwen3_elapsed_s = _time.time() - t_qwen3

            # ── 5. Extract final answer text ──────────────────────────────────
            tail = rolling_prompt[len(rendered):]
            if "<|im_start|>assistant" in tail:
                tail = tail.rsplit("<|im_start|>assistant", 1)[-1]
            tail = tail.split("<|im_end|>", 1)[0]
            text = _THINK_RE.sub("", tail).strip()

            return Qwen3Response(
                text=text,
                n_tool_calls=n_tool_calls,
                raw_completion=rolling_prompt[len(rendered):],
                stop_reason="stop",
                # Honest totals = extractor (Sonnet) + Qwen3 (on-GPU vLLM).
                prompt_tokens=extractor_prompt_tokens + qwen3_prompt_tokens,
                completion_tokens=extractor_completion_tokens + qwen3_completion_tokens,
                n_model_calls=n_model_calls,
                n_verifier_fires=monitor.metrics.n_verifier_fires,
                n_fixes_applied=monitor.metrics.n_fixes_applied,
                extractor_prompt_tokens=extractor_prompt_tokens,
                extractor_completion_tokens=extractor_completion_tokens,
                extractor_elapsed_s=extractor_elapsed_s,
                qwen3_prompt_tokens=qwen3_prompt_tokens,
                qwen3_completion_tokens=qwen3_completion_tokens,
                qwen3_elapsed_s=qwen3_elapsed_s,
                violations_history=list(monitor.metrics.violations_history),
                extracted_facts=dict(facts.raw) if facts.extractor_ok else {},
            )

    return _VerifiedAdapter()

# COMMAND ----------

# DBTITLE 1,_ReactiveVerifiedAdapter builder (B_prime_E_reactive)
def _build_reactive_adapter():
    """B_prime_E_reactive: reactive per-tool-call focused extraction.

    Follow-up to B_prime_E. That condition showed a large negative effect
    (Δ=-9.0 pp vs B_prime, McNemar p≈1e-15) attributed to upfront extractor
    unreliability on the ~500-field schema (sex alone was 31% of all
    verifier flags). This condition tests whether per-tool-call focused
    extraction — same B' system prompt, same verifier, same model — fixes
    the harm.

    Architecture diff vs _VerifiedAdapter:
      - NO upfront extractor.extract(prompt) at vignette start
      - At each tool call: parse input_data keys → build focused prompt
        covering only those fields → Sonnet extract with that prompt →
        mutate monitor.patient_facts to the focused result → verify
      - All other mechanics (loop, fix, MCP dispatch, response shape) identical

    Same Qwen3Response schema as B_prime_E so the downstream runner and
    analysis stages handle it without changes. extracted_facts is the union
    of all per-tool-call extractions; extractor_* totals sum across calls.
    """
    from harness.extraction import FactExtractor
    from harness.extraction.extractor import PatientFacts
    from harness.extraction.prompt_builder import render_focused_prompt
    from harness.monitors import ClinicalInputMonitor
    from harness.karma_adapter.mcp_tools import fetch_tool_schemas
    from harness.karma_adapter.qwen3 import Qwen3Response
    from transformers import AutoTokenizer
    from openai import OpenAI
    import asyncio, json as _rj, re
    import nest_asyncio
    nest_asyncio.apply()

    _MODEL_ID   = "Qwen/Qwen3-30B-A3B-Thinking-2507"
    _MAX_TURNS  = 10
    _THINK_RE   = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
    _TOOL_RE    = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    # FactExtractor.__init__ requires a readable prompt file; we override
    # the system prompt per call via extract_with_prompt, but still need a
    # valid path here. The runtime extractor prompt is the natural choice.
    if not RUNTIME_EXTRACTOR_PATH.exists():
        raise FileNotFoundError(
            f"Runtime extractor prompt not found at {RUNTIME_EXTRACTOR_PATH}. "
            f"The schema-dump preflight cell (§4.5) must run before B_prime_E_reactive."
        )
    extractor         = FactExtractor(
        prompt_path=str(RUNTIME_EXTRACTOR_PATH),
        model="claude-sonnet-4-6",
        temperature=PREREG_CONFIG["sonnet_extraction_temperature"],
    )

    # Load the schema dump once — focused prompts are derived from it per tool call.
    if not SCHEMA_DUMP_PATH.exists():
        raise FileNotFoundError(
            f"MCP schema dump not found at {SCHEMA_DUMP_PATH}. "
            f"The preflight (§4.5) must run before B_prime_E_reactive."
        )
    schema_dump       = _rj.loads(SCHEMA_DUMP_PATH.read_text())

    # Pre-registered: query-style intervention template.
    feedback_template = FEEDBACK_TEMPLATE_PATH.read_text()
    tools             = fetch_tool_schemas()
    tokenizer         = AutoTokenizer.from_pretrained(_MODEL_ID)
    client            = OpenAI(base_url=server.base_url, api_key="EMPTY")

    class _ReactiveVerifiedAdapter:
        """Qwen3 tool-calling loop with per-tool-call focused fact extraction."""

        def __init__(self):
            self._last_monitor = None   # populated after each run() for smoke tests

        def run(self, prompt, system=None):
            import time as _time

            # ── 1. Render Qwen3 prompt (no upfront extraction) ────────────────
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            rendered = tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=False,
            )

            # ── 2. Monitor with empty facts; we populate per-tool-call ────────
            # PatientFacts is mutable enough that we swap it between tool calls.
            # The monitor reads self.patient_facts at verify time.
            placeholder_facts = PatientFacts(raw={}, extractor_ok=True)
            monitor = ClinicalInputMonitor(
                patient_facts=placeholder_facts,
                feedback_template=feedback_template,
            )
            self._last_monitor = monitor
            loop = asyncio.get_event_loop()

            n_tool_calls    = 0
            n_model_calls   = 0
            n_reprompts_applied = 0   # post-gating re-prompts; capped by PREREG_CONFIG
            qwen3_prompt_tokens     = 0
            qwen3_completion_tokens = 0
            # Reactive: extractor metrics accumulate ACROSS tool calls.
            extractor_prompt_tokens     = 0
            extractor_completion_tokens = 0
            extractor_elapsed_s         = 0.0
            accumulated_facts: dict = {}     # union of per-tool-call extractions
            rolling_prompt  = rendered

            # ── 3. Tool-calling loop ──────────────────────────────────────────
            t_qwen3 = _time.time()
            for _ in range(_MAX_TURNS + 1):
                resp = client.completions.create(
                    model=_MODEL_ID,
                    prompt=rolling_prompt,
                    max_tokens=4096,
                    temperature=0.0,
                    top_p=1.0,
                    stop=["</tool_call>"],
                )
                n_model_calls += 1
                if resp.usage is not None:
                    qwen3_prompt_tokens     += getattr(resp.usage, "prompt_tokens",     0) or 0
                    qwen3_completion_tokens += getattr(resp.usage, "completion_tokens", 0) or 0

                choice    = resp.choices[0]
                generated = choice.text or ""
                if (choice.finish_reason == "stop"
                        and "<tool_call>" in generated
                        and "</tool_call>" not in generated):
                    generated += "</tool_call>"
                rolling_prompt += generated

                if "<tool_call>" not in generated:
                    break

                # ── 3a. Reactive focused extraction (the new bit) ──────────────
                # Parse the tool call to learn which fields the model wants.
                # If it's a medical_calculator_output call with an input_data
                # dict, extract only those fields from the case. For other
                # tool calls (drug search, calculator metadata), skip
                # extraction — they don't carry clinical inputs to verify.
                tool_match = _TOOL_RE.search(generated)
                if tool_match is not None:
                    try:
                        tool_obj = _rj.loads(tool_match.group(1))
                    except _rj.JSONDecodeError:
                        tool_obj = None
                    if (isinstance(tool_obj, dict)
                            and tool_obj.get("name") == "medical_calculator_output"
                            and isinstance(tool_obj.get("arguments"), dict)
                            and isinstance(tool_obj["arguments"].get("input_data"), dict)):
                        wanted_fields = list(tool_obj["arguments"]["input_data"].keys())
                        if wanted_fields:
                            focused_prompt = render_focused_prompt(schema_dump, wanted_fields)
                            t_extract = _time.time()
                            focused_facts = extractor.extract_with_prompt(prompt, focused_prompt)
                            extractor_elapsed_s         += _time.time() - t_extract
                            extractor_prompt_tokens     += focused_facts.prompt_tokens
                            extractor_completion_tokens += focused_facts.completion_tokens
                            if focused_facts.extractor_ok:
                                accumulated_facts.update(focused_facts.raw or {})
                                monitor.patient_facts = focused_facts
                            else:
                                # Extractor failure on this call — empty facts
                                # so verifier conservatively skips (no false
                                # positives from a broken call).
                                monitor.patient_facts = PatientFacts(raw={}, extractor_ok=True)
                        else:
                            monitor.patient_facts = PatientFacts(raw={}, extractor_ok=True)
                    else:
                        # Non-clinical tool call — empty facts, verifier no-ops
                        monitor.patient_facts = PatientFacts(raw={}, extractor_ok=True)

                # ── 3b. Verify + fix (same mechanics as _VerifiedAdapter) ─────
                monitor.metrics.n_steps_seen += 1
                event_info: dict = {}
                loop.run_until_complete(monitor.verify(
                    chunk=generated,
                    token_index=0,
                    event=asyncio.Event(),
                    event_info=event_info,
                ))

                # Pre-registered: schema-gate violations + enforce re-prompt cap.
                # Local n_reprompts_applied counter ensures cap fires only on
                # violations that SURVIVE schema-gating (not on gated-away flags).
                if event_info.get("violations"):
                    event_info["violations"] = _schema_gate_violations(
                        event_info["violations"],
                        event_info.get("call_obj"),
                        schema_dump,
                    )
                if event_info.get("violations"):
                    if n_reprompts_applied >= PREREG_CONFIG["verifier_reprompt_cap"]:
                        event_info["violations"] = []   # demote: cap reached
                    else:
                        n_reprompts_applied += 1

                rolling_prompt = loop.run_until_complete(
                    monitor.fix(rolling_prompt, event_info)
                )
                if not event_info.get("violations") and not event_info.get("malformed"):
                    n_tool_calls += 1
            qwen3_elapsed_s = _time.time() - t_qwen3

            # ── 4. Extract final answer text ──────────────────────────────────
            tail = rolling_prompt[len(rendered):]
            if "<|im_start|>assistant" in tail:
                tail = tail.rsplit("<|im_start|>assistant", 1)[-1]
            tail = tail.split("<|im_end|>", 1)[0]
            text = _THINK_RE.sub("", tail).strip()

            return Qwen3Response(
                text=text,
                n_tool_calls=n_tool_calls,
                raw_completion=rolling_prompt[len(rendered):],
                stop_reason="stop",
                # Honest totals: extractor (sum over tool calls) + Qwen3 (sum over turns)
                prompt_tokens=extractor_prompt_tokens + qwen3_prompt_tokens,
                completion_tokens=extractor_completion_tokens + qwen3_completion_tokens,
                n_model_calls=n_model_calls,
                n_verifier_fires=monitor.metrics.n_verifier_fires,
                n_fixes_applied=monitor.metrics.n_fixes_applied,
                extractor_prompt_tokens=extractor_prompt_tokens,
                extractor_completion_tokens=extractor_completion_tokens,
                extractor_elapsed_s=extractor_elapsed_s,
                qwen3_prompt_tokens=qwen3_prompt_tokens,
                qwen3_completion_tokens=qwen3_completion_tokens,
                qwen3_elapsed_s=qwen3_elapsed_s,
                violations_history=list(monitor.metrics.violations_history),
                # Union of all per-tool-call focused extractions
                extracted_facts=dict(accumulated_facts),
            )

    return _ReactiveVerifiedAdapter()

# COMMAND ----------

# DBTITLE 1,_build_citation_reactive_adapter (B_prime_E_reactive_citations)
def _build_citation_reactive_adapter():
    """B_prime_E_reactive_citations: reactive scoped extraction returning
    (value, source_span) per field; substring-validated against the vignette.

    Differences vs _build_reactive_adapter:
      - The per-tool-call focused prompt is rendered by
        render_focused_prompt_with_citations() (asks Sonnet to return
        {value, source_span} per field).
      - After extraction, harness.extraction.citations.coerce_citations_to_bare
        validates each source_span as a verbatim substring of the vignette;
        failures coerce the field to null (abstention).
      - The verifier consumes the bare {field: value} dict that survives
        validation. The per-field validation report is stored in
        monitor.metrics so the analysis notebook can compute citation
        acceptance rates.

    All other loop mechanics are byte-identical to the reactive adapter.
    """
    from harness.extraction import FactExtractor, coerce_citations_to_bare
    from harness.extraction.extractor import PatientFacts
    from harness.extraction.prompt_builder import render_focused_prompt_with_citations
    from harness.monitors import ClinicalInputMonitor
    from harness.karma_adapter.mcp_tools import fetch_tool_schemas
    from harness.karma_adapter.qwen3 import Qwen3Response
    from transformers import AutoTokenizer
    from openai import OpenAI
    import asyncio, json as _rj, re
    import nest_asyncio
    nest_asyncio.apply()

    _MODEL_ID  = "Qwen/Qwen3-30B-A3B-Thinking-2507"
    _MAX_TURNS = 10
    _THINK_RE  = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
    _TOOL_RE   = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    if not RUNTIME_EXTRACTOR_PATH.exists():
        raise FileNotFoundError(
            f"Runtime extractor prompt not found at {RUNTIME_EXTRACTOR_PATH}. "
            f"The preflight (§4.5) must run before B_prime_E_reactive_citations."
        )
    extractor = FactExtractor(
        prompt_path=str(RUNTIME_EXTRACTOR_PATH),
        model="claude-sonnet-4-6",
        temperature=PREREG_CONFIG["sonnet_extraction_temperature"],
    )

    if not SCHEMA_DUMP_PATH.exists():
        raise FileNotFoundError(
            f"MCP schema dump not found at {SCHEMA_DUMP_PATH}. "
            f"The preflight (§4.5) must run before B_prime_E_reactive_citations."
        )
    schema_dump = _rj.loads(SCHEMA_DUMP_PATH.read_text())

    feedback_template = FEEDBACK_TEMPLATE_PATH.read_text()
    tools             = fetch_tool_schemas()
    tokenizer         = AutoTokenizer.from_pretrained(_MODEL_ID)
    client            = OpenAI(base_url=server.base_url, api_key="EMPTY")

    class _CitationReactiveAdapter:
        """Reactive adapter with citation-anchored extraction + substring validation.

        Each run() accumulates per-tool-call citation validation reports in a
        local list and attaches them to the returned Qwen3Response via the
        citation_reports field. The runner serializes that field to the parquet
        as a JSON string (see harness/runner.py), so analysis can compute
        per-field acceptance rates across vignettes."""

        def __init__(self):
            self._last_monitor = None
            self._last_citation_reports: list[dict] = []  # mirrored on Qwen3Response.citation_reports

        def run(self, prompt, system=None):
            import time as _time

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            rendered = tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=False,
            )

            placeholder_facts = PatientFacts(raw={}, extractor_ok=True)
            monitor = ClinicalInputMonitor(
                patient_facts=placeholder_facts,
                feedback_template=feedback_template,
            )
            self._last_monitor = monitor
            self._last_citation_reports = []
            loop = asyncio.get_event_loop()

            n_tool_calls = 0
            n_model_calls = 0
            n_reprompts_applied = 0
            qwen3_prompt_tokens = 0
            qwen3_completion_tokens = 0
            extractor_prompt_tokens     = 0
            extractor_completion_tokens = 0
            extractor_elapsed_s         = 0.0
            accumulated_facts: dict = {}
            rolling_prompt = rendered

            t_qwen3 = _time.time()
            for _ in range(_MAX_TURNS + 1):
                resp = client.completions.create(
                    model=_MODEL_ID, prompt=rolling_prompt,
                    max_tokens=4096, temperature=0.0, top_p=1.0,
                    stop=["</tool_call>"],
                )
                n_model_calls += 1
                if resp.usage is not None:
                    qwen3_prompt_tokens     += getattr(resp.usage, "prompt_tokens",     0) or 0
                    qwen3_completion_tokens += getattr(resp.usage, "completion_tokens", 0) or 0

                choice = resp.choices[0]
                generated = choice.text or ""
                if (choice.finish_reason == "stop"
                        and "<tool_call>" in generated
                        and "</tool_call>" not in generated):
                    generated += "</tool_call>"
                rolling_prompt += generated

                if "<tool_call>" not in generated:
                    break

                # Reactive citation-aware focused extraction.
                tool_match = _TOOL_RE.search(generated)
                if tool_match is not None:
                    try:
                        tool_obj = _rj.loads(tool_match.group(1))
                    except _rj.JSONDecodeError:
                        tool_obj = None
                    if (isinstance(tool_obj, dict)
                            and tool_obj.get("name") == "medical_calculator_output"
                            and isinstance(tool_obj.get("arguments"), dict)
                            and isinstance(tool_obj["arguments"].get("input_data"), dict)):
                        wanted_fields = list(tool_obj["arguments"]["input_data"].keys())
                        if wanted_fields:
                            citation_prompt = render_focused_prompt_with_citations(
                                schema_dump, wanted_fields
                            )
                            t_extract = _time.time()
                            citation_facts = extractor.extract_with_prompt(prompt, citation_prompt)
                            extractor_elapsed_s         += _time.time() - t_extract
                            extractor_prompt_tokens     += citation_facts.prompt_tokens
                            extractor_completion_tokens += citation_facts.completion_tokens

                            if citation_facts.extractor_ok and isinstance(citation_facts.raw, dict):
                                bare, report = coerce_citations_to_bare(citation_facts.raw, prompt)
                                self._last_citation_reports.append({
                                    "wanted_fields": wanted_fields,
                                    "report":        report,
                                })
                                accumulated_facts.update(bare)
                                monitor.patient_facts = PatientFacts(raw=bare, extractor_ok=True)
                            else:
                                monitor.patient_facts = PatientFacts(raw={}, extractor_ok=True)
                        else:
                            monitor.patient_facts = PatientFacts(raw={}, extractor_ok=True)
                    else:
                        monitor.patient_facts = PatientFacts(raw={}, extractor_ok=True)

                # Verify + schema-gate + cap + fix.
                monitor.metrics.n_steps_seen += 1
                event_info: dict = {}
                loop.run_until_complete(monitor.verify(
                    chunk=generated, token_index=0,
                    event=asyncio.Event(), event_info=event_info,
                ))
                if event_info.get("violations"):
                    event_info["violations"] = _schema_gate_violations(
                        event_info["violations"],
                        event_info.get("call_obj"),
                        schema_dump,
                    )
                if event_info.get("violations"):
                    if n_reprompts_applied >= PREREG_CONFIG["verifier_reprompt_cap"]:
                        event_info["violations"] = []
                    else:
                        n_reprompts_applied += 1
                rolling_prompt = loop.run_until_complete(
                    monitor.fix(rolling_prompt, event_info)
                )
                if not event_info.get("violations") and not event_info.get("malformed"):
                    n_tool_calls += 1
            qwen3_elapsed_s = _time.time() - t_qwen3

            tail = rolling_prompt[len(rendered):]
            if "<|im_start|>assistant" in tail:
                tail = tail.rsplit("<|im_start|>assistant", 1)[-1]
            tail = tail.split("<|im_end|>", 1)[0]
            text = _THINK_RE.sub("", tail).strip()

            return Qwen3Response(
                text=text,
                n_tool_calls=n_tool_calls,
                raw_completion=rolling_prompt[len(rendered):],
                stop_reason="stop",
                prompt_tokens=extractor_prompt_tokens + qwen3_prompt_tokens,
                completion_tokens=extractor_completion_tokens + qwen3_completion_tokens,
                n_model_calls=n_model_calls,
                n_verifier_fires=monitor.metrics.n_verifier_fires,
                n_fixes_applied=monitor.metrics.n_fixes_applied,
                extractor_prompt_tokens=extractor_prompt_tokens,
                extractor_completion_tokens=extractor_completion_tokens,
                extractor_elapsed_s=extractor_elapsed_s,
                qwen3_prompt_tokens=qwen3_prompt_tokens,
                qwen3_completion_tokens=qwen3_completion_tokens,
                qwen3_elapsed_s=qwen3_elapsed_s,
                violations_history=list(monitor.metrics.violations_history),
                extracted_facts=dict(accumulated_facts),
                # Per-tool-call citation validation reports — serialized by the
                # runner into the parquet's citation_reports column for analysis.
                citation_reports=list(self._last_citation_reports),
            )

    return _CitationReactiveAdapter()


# COMMAND ----------

# DBTITLE 1,_build_kshot_reactive_adapter (B_prime_E_reactive_kshot)
def _build_kshot_reactive_adapter():
    """B_prime_E_reactive_kshot: reactive scoped extraction with k=3
    independent samples per tool call; per-field majority vote.

    Differences vs _build_reactive_adapter:
      - Each per-tool-call focused extraction dispatches k=3 Sonnet calls
        (same prompt, fresh seed per call via temperature 0.7) instead of 1.
      - The k samples are reduced per field by harness.extraction.voting.majority_vote;
        no majority (3-way disagreement) yields null (abstention).
      - The verifier consumes the voted bare dict. Per-field vote reports are
        stored for analysis (per-field agreement rates, per-field abstention).

    All other loop mechanics are byte-identical to the reactive adapter.
    """
    from harness.extraction import FactExtractor, majority_vote
    from harness.extraction.extractor import PatientFacts
    from harness.extraction.prompt_builder import render_focused_prompt
    from harness.monitors import ClinicalInputMonitor
    from harness.karma_adapter.mcp_tools import fetch_tool_schemas
    from harness.karma_adapter.qwen3 import Qwen3Response
    from transformers import AutoTokenizer
    from openai import OpenAI
    import asyncio, json as _rj, re
    import nest_asyncio
    nest_asyncio.apply()

    _MODEL_ID  = "Qwen/Qwen3-30B-A3B-Thinking-2507"
    _MAX_TURNS = 10
    _THINK_RE  = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
    _TOOL_RE   = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    _K         = PREREG_CONFIG["kshot_k"]            # 3

    if not RUNTIME_EXTRACTOR_PATH.exists():
        raise FileNotFoundError(
            f"Runtime extractor prompt not found at {RUNTIME_EXTRACTOR_PATH}."
        )
    extractor = FactExtractor(
        prompt_path=str(RUNTIME_EXTRACTOR_PATH),
        model="claude-sonnet-4-6",
        temperature=PREREG_CONFIG["sonnet_extraction_temperature"],
    )

    if not SCHEMA_DUMP_PATH.exists():
        raise FileNotFoundError(f"MCP schema dump not found at {SCHEMA_DUMP_PATH}.")
    schema_dump = _rj.loads(SCHEMA_DUMP_PATH.read_text())

    feedback_template = FEEDBACK_TEMPLATE_PATH.read_text()
    tools             = fetch_tool_schemas()
    tokenizer         = AutoTokenizer.from_pretrained(_MODEL_ID)
    client            = OpenAI(base_url=server.base_url, api_key="EMPTY")

    class _KShotReactiveAdapter:
        """Reactive adapter with k=3 majority-vote extraction per tool call.

        Each run() accumulates per-tool-call voting reports in a local list and
        attaches them to the returned Qwen3Response via the voting_reports
        field. The runner serializes that field to the parquet as a JSON
        string (see harness/runner.py), so analysis can compute per-field
        agreement / abstention rates across vignettes."""

        def __init__(self):
            self._last_monitor = None
            self._last_voting_reports: list[dict] = []   # mirrored on Qwen3Response.voting_reports

        def run(self, prompt, system=None):
            import time as _time

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            rendered = tokenizer.apply_chat_template(
                messages, tools=tools, add_generation_prompt=True, tokenize=False,
            )

            placeholder_facts = PatientFacts(raw={}, extractor_ok=True)
            monitor = ClinicalInputMonitor(
                patient_facts=placeholder_facts,
                feedback_template=feedback_template,
            )
            self._last_monitor = monitor
            self._last_voting_reports = []
            loop = asyncio.get_event_loop()

            n_tool_calls = 0
            n_model_calls = 0
            n_reprompts_applied = 0
            qwen3_prompt_tokens = 0
            qwen3_completion_tokens = 0
            extractor_prompt_tokens     = 0
            extractor_completion_tokens = 0
            extractor_elapsed_s         = 0.0
            accumulated_facts: dict = {}
            rolling_prompt = rendered

            t_qwen3 = _time.time()
            for _ in range(_MAX_TURNS + 1):
                resp = client.completions.create(
                    model=_MODEL_ID, prompt=rolling_prompt,
                    max_tokens=4096, temperature=0.0, top_p=1.0,
                    stop=["</tool_call>"],
                )
                n_model_calls += 1
                if resp.usage is not None:
                    qwen3_prompt_tokens     += getattr(resp.usage, "prompt_tokens",     0) or 0
                    qwen3_completion_tokens += getattr(resp.usage, "completion_tokens", 0) or 0

                choice = resp.choices[0]
                generated = choice.text or ""
                if (choice.finish_reason == "stop"
                        and "<tool_call>" in generated
                        and "</tool_call>" not in generated):
                    generated += "</tool_call>"
                rolling_prompt += generated

                if "<tool_call>" not in generated:
                    break

                # Reactive k-shot voting extraction.
                tool_match = _TOOL_RE.search(generated)
                if tool_match is not None:
                    try:
                        tool_obj = _rj.loads(tool_match.group(1))
                    except _rj.JSONDecodeError:
                        tool_obj = None
                    if (isinstance(tool_obj, dict)
                            and tool_obj.get("name") == "medical_calculator_output"
                            and isinstance(tool_obj.get("arguments"), dict)
                            and isinstance(tool_obj["arguments"].get("input_data"), dict)):
                        wanted_fields = list(tool_obj["arguments"]["input_data"].keys())
                        if wanted_fields:
                            focused_prompt = render_focused_prompt(schema_dump, wanted_fields)
                            # k=3 independent samples. Sonnet seed is non-pinned;
                            # temperature 0.7 gives sample divergence. We dispatch
                            # sequentially (the Anthropic client doesn't expose a
                            # native batch API and concurrent.futures here would
                            # double-up against the orchestrator's process pool).
                            samples_raw: list[dict] = []
                            t_extract = _time.time()
                            for _k in range(_K):
                                facts_k = extractor.extract_with_prompt(prompt, focused_prompt)
                                extractor_prompt_tokens     += facts_k.prompt_tokens
                                extractor_completion_tokens += facts_k.completion_tokens
                                if facts_k.extractor_ok and isinstance(facts_k.raw, dict):
                                    samples_raw.append(facts_k.raw)
                                else:
                                    samples_raw.append({})    # failed sample = abstain
                            extractor_elapsed_s += _time.time() - t_extract

                            voted, report = majority_vote(samples_raw, min_agreement=2)
                            self._last_voting_reports.append({
                                "wanted_fields": wanted_fields,
                                "report":        report,
                            })
                            accumulated_facts.update(voted)
                            monitor.patient_facts = PatientFacts(raw=voted, extractor_ok=True)
                        else:
                            monitor.patient_facts = PatientFacts(raw={}, extractor_ok=True)
                    else:
                        monitor.patient_facts = PatientFacts(raw={}, extractor_ok=True)

                # Verify + schema-gate + cap + fix.
                monitor.metrics.n_steps_seen += 1
                event_info: dict = {}
                loop.run_until_complete(monitor.verify(
                    chunk=generated, token_index=0,
                    event=asyncio.Event(), event_info=event_info,
                ))
                if event_info.get("violations"):
                    event_info["violations"] = _schema_gate_violations(
                        event_info["violations"],
                        event_info.get("call_obj"),
                        schema_dump,
                    )
                if event_info.get("violations"):
                    if n_reprompts_applied >= PREREG_CONFIG["verifier_reprompt_cap"]:
                        event_info["violations"] = []
                    else:
                        n_reprompts_applied += 1
                rolling_prompt = loop.run_until_complete(
                    monitor.fix(rolling_prompt, event_info)
                )
                if not event_info.get("violations") and not event_info.get("malformed"):
                    n_tool_calls += 1
            qwen3_elapsed_s = _time.time() - t_qwen3

            tail = rolling_prompt[len(rendered):]
            if "<|im_start|>assistant" in tail:
                tail = tail.rsplit("<|im_start|>assistant", 1)[-1]
            tail = tail.split("<|im_end|>", 1)[0]
            text = _THINK_RE.sub("", tail).strip()

            return Qwen3Response(
                text=text,
                n_tool_calls=n_tool_calls,
                raw_completion=rolling_prompt[len(rendered):],
                stop_reason="stop",
                prompt_tokens=extractor_prompt_tokens + qwen3_prompt_tokens,
                completion_tokens=extractor_completion_tokens + qwen3_completion_tokens,
                n_model_calls=n_model_calls,
                n_verifier_fires=monitor.metrics.n_verifier_fires,
                n_fixes_applied=monitor.metrics.n_fixes_applied,
                extractor_prompt_tokens=extractor_prompt_tokens,
                extractor_completion_tokens=extractor_completion_tokens,
                extractor_elapsed_s=extractor_elapsed_s,
                qwen3_prompt_tokens=qwen3_prompt_tokens,
                qwen3_completion_tokens=qwen3_completion_tokens,
                qwen3_elapsed_s=qwen3_elapsed_s,
                violations_history=list(monitor.metrics.violations_history),
                extracted_facts=dict(accumulated_facts),
                # Per-tool-call k-shot voting reports — serialized by the runner
                # into the parquet's voting_reports column for analysis.
                voting_reports=list(self._last_voting_reports),
            )

    return _KShotReactiveAdapter()


# COMMAND ----------

# DBTITLE 1,_build_adapter dispatcher (9 conditions)
def _build_adapter(cond: str):
    """Route a condition name to its adapter builder. The 9-condition design
    has three architectural classes:
      - plain          (A, B, B_prime, C)        → Qwen3Adapter
      - post-hoc       (D)                       → PostHocVerifierAdapter
      - verified       (B_prime_E)               → _VerifiedAdapter (upfront extraction)
      - reactive       (B_prime_E_reactive)      → _ReactiveVerifiedAdapter
      - reactive+cit   (..._reactive_citations)  → _CitationReactiveAdapter (stub)
      - reactive+k     (..._reactive_kshot)      → _KShotReactiveAdapter (stub)

    All extractor-using conditions (B_prime_E and variants) share pre-registered
    hygiene: query intervention, schema-gating, prompt-only abstention. See
    PREREG_CONFIG and paper §methods_prereg.
    """
    if cond in ("A", "B", "B_prime", "C"):
        return _build_plain_adapter(cond)
    if cond == "D":
        return _build_posthoc_adapter()
    if cond == "B_prime_E":
        return _build_verified_adapter()
    if cond == "B_prime_E_reactive":
        return _build_reactive_adapter()
    if cond == "B_prime_E_reactive_citations":
        return _build_citation_reactive_adapter()
    if cond == "B_prime_E_reactive_kshot":
        return _build_kshot_reactive_adapter()
    raise ValueError(f"Unknown condition: {cond}")

# COMMAND ----------

# DBTITLE 1,Orchestration helpers (out_dir, idempotency, backups)
def _out_dir(cond: str, kind: str) -> Path:
    return RESULTS_ROOT / f"qwen3_condition_{cond}_{kind}"


def _already_done(cond: str) -> bool:
    summary = _out_dir(cond, "full") / "summary.json"
    return summary.exists()


def _backup_existing_results(cond: str) -> Path | None:
    """If a condition's full-run dir already has data, move it under a
    timestamped _backup_<ts>/ subdir so the new run doesn't overwrite it.
    Returns the backup path used (or None if nothing existed).
    """
    full = _out_dir(cond, "full")
    if not (full / "rows.parquet").exists():
        return None
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_root = full.parent / f"{full.name}_backup_{ts}"
    backup_root.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in full.iterdir():
        target = backup_root / p.name
        p.rename(target)
        moved.append(target.name)
    print(f"[{cond}] backed up {len(moved)} files from {full} → {backup_root}")
    return backup_root

# COMMAND ----------

# DBTITLE 1,run_condition — per-condition pilot + full
def run_condition(cond: str) -> dict:
    """Run pilot + full for one condition, idempotent + error-tolerant."""
    record = {"condition": cond, "started_at": time.time()}

    # Idempotency unless explicitly overridden
    if _already_done(cond) and not FORCE_RERUN:
        summary = json.loads((_out_dir(cond, "full") / "summary.json").read_text())
        record.update({"status": "skipped", "summary": summary})
        print(f"[{cond}] already complete — skipped (acc={summary.get('accuracy', '?')})")
        return record

    # Preserve any existing per-row data before a fresh run would clobber it
    if BACKUP_LEGACY:
        backup_path = _backup_existing_results(cond)
        if backup_path is not None:
            record["legacy_backup"] = str(backup_path)

    try:
        print(f"[{cond}] PILOT (n={PILOT_N}, workers={WORKERS[cond]})")
        adapter = _build_adapter(cond)
        system = _system_for(cond)
        pilot = run_eval(
            adapter, n=PILOT_N, max_workers=WORKERS[cond],
            system=system,
            out_dir=str(_out_dir(cond, "pilot")),
        )
        record["pilot"] = {
            "accuracy": pilot.accuracy,
            "n": pilot.n,
            "parse_failures": pilot.n_parse_failures,
            "mean_tool_calls": float(pilot.rows["n_tool_calls"].mean()),
        }
        print(f"[{cond}] pilot acc={pilot.accuracy:.1%}, tool_calls/vignette={record['pilot']['mean_tool_calls']:.2f}")
        if pilot.accuracy == 0.0:
            print(f"[{cond}] WARNING: pilot accuracy is 0% — check adapter before full run. Proceeding anyway.")
        if cond != "A" and record["pilot"]["mean_tool_calls"] == 0.0:
            print(f"[{cond}] WARNING: pilot used 0 tool calls — tool dispatch may be broken.")

        print(f"[{cond}] FULL (n=1066, workers={WORKERS[cond]})")
        full = run_eval(
            adapter, n=None, max_workers=WORKERS[cond],
            system=system,
            out_dir=str(_out_dir(cond, "full")),
        )
        ci = wilson_ci(full.n_correct, full.n)
        record["full"] = {
            "accuracy": full.accuracy,
            "ci_low": ci.lo,
            "ci_high": ci.hi,
            "n": full.n,
            "parse_failures": full.n_parse_failures,
            "mean_tool_calls": float(full.rows["n_tool_calls"].mean()),
            "mean_total_tokens": full.mean_total_tokens,
            "median_latency_s": full.median_latency_seconds,
            "total_run_seconds": full.total_run_seconds,
            "estimated_usd_h100": full.estimated_cost_usd(3.0),
        }
        record["status"] = "ok"
        print(f"[{cond}] FULL DONE: {ci}  ({full.total_run_seconds/60:.1f} min)")
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()
        print(f"[{cond}] ERROR: {record['error']}")

    record["ended_at"] = time.time()
    return record


# COMMAND ----------

# MAGIC %md ## 6. (Optional) Smoke-test verified adapters
# MAGIC
# MAGIC Diagnostic, not required for the run loop below. Verifies the primary
# MAGIC study adapters on 3 vignettes (MCP dispatch, verifier fires,
# MAGIC tool_response injection, schema-gating, query-style intervention).
# MAGIC Skip on a known-good config.
# MAGIC
# MAGIC Also doubles as the **pilot-batch sanity checks** required before the
# MAGIC full N=1066 run (paper §methods_prereg, items 1-5). Run a wider pilot
# MAGIC (n=30) before the overnight run if you've changed anything.

# COMMAND ----------

# DBTITLE 1,Smoke-test primary study adapters
# Run this cell BEFORE the run-all loop to verify primary-condition mechanics.
# Tests all 4 primary study adapters on 3 vignettes each, plus the k-shot
# sample-divergence assertion (pilot-batch sanity check #1).
#
# Per-vignette checks:
#   n_tool_calls > 0   → MCP tools are actually being dispatched
#   n_model_calls      → > n_tool_calls means verifier fired at least once
#   n_verifier_fires   → how many tool calls had violations (post schema-gate)
#   raw_completion     → should contain <tool_response> blocks (not phantom calls)
#
# k-shot specific (the critical caching check):
#   For the k-shot adapter, after one vignette we inspect the per-tool-call
#   vote reports and assert at least one field showed divergent samples
#   across the k=3 calls. If ALL fields agreed across all samples on EVERY
#   tool call, that's a strong signal that Anthropic prompt caching is
#   collapsing the samples — which would silently turn k-shot into k=1
#   at 3× the cost. Fail loudly here, not silently after a 9k-second run.

from datasets import load_dataset as _lds

_ds = _lds("ekacare/medical_calculator_eval", split="test")

_SMOKE_CONDS = [
    "B_prime_E",
    "B_prime_E_reactive",
    "B_prime_E_reactive_citations",
    "B_prime_E_reactive_kshot",
]

for _cond in _SMOKE_CONDS:
    print("\n" + "=" * 60)
    print(f"=== smoke test: {_cond} ===")
    print("=" * 60)
    try:
        _adapter = _build_adapter(_cond)
    except NotImplementedError as _e:
        print(f"  SKIP: {_cond} adapter not yet implemented:\n    {_e}")
        continue
    _system = _system_for(_cond)

    for i in range(3):
        _vignette = _ds[i]["question_text"]
        _resp = _adapter.run(_vignette, system=_system)
        _m = _adapter._last_monitor.metrics
        print(f"\n--- vignette {i} ---")
        print(f"  n_model_calls   : {_resp.n_model_calls}")
        print(f"  n_tool_calls    : {_resp.n_tool_calls}   (MCP dispatches)")
        print(f"  n_steps_seen    : {_m.n_steps_seen}    (tool_call blocks)")
        print(f"  n_verifier_fires: {_m.n_verifier_fires} (post schema-gate)")
        print(f"  n_fixes_applied : {_m.n_fixes_applied}  (re-prompts; capped at {PREREG_CONFIG['verifier_reprompt_cap']})")
        print(f"  answer text     : {_resp.text[:120]}")
        has_response = "<tool_response>" in _resp.raw_completion
        print(f"  tool_response in raw_completion: {has_response}  (✓ real dispatch if True)")
        if _m.violations_history:
            print(f"  violations_history[0]: {_m.violations_history[0]}")

    # ── Adapter-specific sanity checks ────────────────────────────────────
    if _cond == "B_prime_E_reactive_kshot":
        # PILOT SANITY CHECK #1: k-shot samples must diverge.
        # If samples never disagree, prompt caching is active and k-shot
        # silently collapses to k=1. Fail loudly.
        reports = getattr(_adapter, "_last_voting_reports", [])
        n_tool_calls_with_votes = len(reports)
        n_divergent_fields = 0
        n_total_fields     = 0
        for r in reports:
            for field_name, field_report in r.get("report", {}).items():
                n_total_fields += 1
                samples = field_report.get("samples", [])
                if len(set(map(repr, samples))) > 1:
                    n_divergent_fields += 1
        print(f"\n  [k-shot divergence check]")
        print(f"  tool-calls observed: {n_tool_calls_with_votes}")
        print(f"  total field-votes:   {n_total_fields}")
        print(f"  divergent votes:     {n_divergent_fields} "
              f"({100*n_divergent_fields/max(n_total_fields,1):.1f}%)")
        if n_total_fields > 0 and n_divergent_fields == 0:
            raise AssertionError(
                "k-shot samples never diverged across 3 vignettes. "
                "This strongly suggests Anthropic prompt caching is collapsing "
                "the samples — k-shot would silently behave like k=1 at 3× cost. "
                "Verify temperature is actually being applied server-side, and "
                "that cache_control is not set on the Sonnet system prompt."
            )

    if _cond == "B_prime_E_reactive_citations":
        # Sanity: citation validation should accept some spans and reject
        # some (real vignettes have a mix). If ALL spans are accepted on
        # 3 vignettes, the validator is too permissive; if NONE are, it's
        # too strict (or Sonnet is fabricating spans).
        reports = getattr(_adapter, "_last_citation_reports", [])
        n_total   = 0
        n_valid   = 0
        for r in reports:
            for field_name, field_report in r.get("report", {}).items():
                n_total += 1
                if field_report.get("valid"):
                    n_valid += 1
        print(f"\n  [citation validation rates]")
        print(f"  total fields seen: {n_total}")
        print(f"  valid spans:       {n_valid} "
              f"({100*n_valid/max(n_total,1):.1f}%)")
        # Soft warning rather than assertion — low validity could be Sonnet
        # quality issue, not a code bug.
        if n_total > 0 and n_valid == 0:
            print("  WARNING: 0% citation validity across smoke vignettes. "
                  "Inspect a sample citation_report to debug.")
        if n_total > 0 and n_valid == n_total:
            print("  NOTE: 100% citation validity — possible but worth eyeballing one report.")

print("\nSmoke tests done. Verify: n_tool_calls > 0 and tool_response True on any vignette.")
print("If the k-shot divergence assertion passed, sampling is genuinely stochastic.")

# COMMAND ----------

# MAGIC %md ## 7. Run all conditions

# COMMAND ----------

# DBTITLE 1,Run all conditions
run_records: list[dict] = []
for cond in CONDITIONS_TO_RUN:
    print("=" * 60)
    print(f"=== {cond} ===")
    print("=" * 60)
    record = run_condition(cond)
    run_records.append(record)
    # Persist after each condition so a mid-run failure still leaves a partial summary on disk.
    (RESULTS_ROOT / "_orchestrator_progress.json").write_text(json.dumps(run_records, indent=2, default=str))

print("\n\n=== ALL CONDITIONS ATTEMPTED ===")
for r in run_records:
    line = f"{r['condition']}: {r['status']}"
    if r["status"] == "ok":
        line += f"  acc={r['full']['accuracy']:.1%}  ({r['full']['total_run_seconds']/60:.1f} min)"
    elif r["status"] == "skipped":
        line += f"  acc={r['summary'].get('accuracy', '?'):.1%} (cached)"
    else:
        line += f"  {r.get('error', '')}"
    print(line)

# COMMAND ----------

# MAGIC %md ## 8. Save run records

# COMMAND ----------

# DBTITLE 1,Save aggregated run records
out_path = RESULTS_ROOT / "_AGGREGATED_RESULTS.json"
out_path.write_text(json.dumps({"run_records": run_records, "completed_at": time.time()}, indent=2, default=str))
print(f"Run records written to: {out_path}")
print("\nNext: open 03_analysis and run all cells for the full analysis + export bundle.")

# COMMAND ----------

# MAGIC %md ## 9. Cleanup

# COMMAND ----------

# DBTITLE 1,Stop vLLM server
server.stop()
print("server stopped:", not server.is_alive())
print(f"\nDone. Results at: {RESULTS_ROOT}")
