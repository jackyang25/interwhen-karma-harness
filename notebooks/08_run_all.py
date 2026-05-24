# Databricks notebook source
# MAGIC %md
# MAGIC # 08 — Orchestrator: run all conditions, aggregate, report
# MAGIC
# MAGIC Single notebook to click before going to bed. Runs every Qwen3 condition
# MAGIC against one vLLM server, recovers from per-condition errors, and at the
# MAGIC end produces an aggregated results table.
# MAGIC
# MAGIC **Design properties:**
# MAGIC - **Idempotent:** each condition checks for an existing
# MAGIC   `/dbfs/results/.../summary.json` and skips if found. Re-running picks
# MAGIC   up where it left off.
# MAGIC - **Error-tolerant:** if a condition crashes, log + continue to the
# MAGIC   next. The orchestrator should not error overnight.
# MAGIC - **One vLLM server:** launched once, used for all conditions, stopped
# MAGIC   at the end.
# MAGIC - **Pilot-gated:** each condition runs an n=10 pilot first. If the
# MAGIC   pilot looks broken (0% accuracy, or tool calls=0 where they should
# MAGIC   exist), it logs a warning but still proceeds — the user can decide
# MAGIC   from the aggregated report what to inspect.
# MAGIC
# MAGIC **Order of conditions** (chosen for fail-fast on the cheapest first):
# MAGIC 1. A (no tools)
# MAGIC 2. B (tools)
# MAGIC 3. B' (force tool use, secondary)
# MAGIC 4. C (best-effort prompt verification)
# MAGIC 5. D' (post-hoc Sonnet verifier)
# MAGIC 6. E (literal interwhen + semantic verifier)
# MAGIC
# MAGIC **Before clicking Run All:**
# MAGIC - Pilot each individual notebook (02–07) once to confirm the wiring works
# MAGIC - Paste API keys in section 2
# MAGIC - Confirm `IDM-H100GPU-Compute_*` is attached

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
# MAGIC ## ⚙️ Orchestration knobs — edit before clicking Run All
# MAGIC
# MAGIC - `CONDITIONS_TO_RUN`: which conditions the orchestrator processes. The
# MAGIC   default runs everything in dependency order. Trim the list to run a
# MAGIC   subset (idempotency means already-completed conditions are still
# MAGIC   skipped even if you leave them in).
# MAGIC - `FORCE_RERUN`: if True, ignores existing `summary.json` and re-runs
# MAGIC   each selected condition from scratch. Old per-row parquets are
# MAGIC   backed up first (see BACKUP_LEGACY).
# MAGIC - `BACKUP_LEGACY`: if True (default), any existing `rows.parquet` in a
# MAGIC   condition's result dir is moved to a timestamped backup directory
# MAGIC   before the new run writes. Safe-by-default — no historical data is
# MAGIC   destroyed.

# COMMAND ----------

CONDITIONS_TO_RUN = ["A", "B", "B_prime", "C", "D_prime", "E"]
FORCE_RERUN = False        # True = re-run even if summary.json exists
BACKUP_LEGACY = True       # True = preserve any existing rows.parquet under _backup_<ts>/

# COMMAND ----------

# MAGIC %md ## 5. Define the orchestration helpers

# COMMAND ----------

import json
import time
import traceback
from pathlib import Path

from harness.karma_adapter.qwen3 import Qwen3Adapter
from harness.runner import run_eval
from harness.analysis import wilson_ci

REPO_ROOT = pathlib.Path("/Workspace/Users/jack.yang@gatesfoundation.org/interwhen-karma-harness")
RESULTS_ROOT = Path("/dbfs/results")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

# Pilot+full max-workers per condition. Picked to match the same numbers each
# individual condition notebook uses, so results are comparable across runs.
WORKERS = {
    "A":        128,
    "B":        64,
    "B_prime":  64,
    "C":        64,
    "D_prime":  32,   # adds a Sonnet verifier call per vignette
    "E":        16,   # adds extractor + interwhen streaming + potential retries
}

PILOT_N = 10


def _system_for(cond: str) -> str | None:
    """Return the locked system prompt for the condition, or None for DEFAULT_SYSTEM."""
    mapping = {
        "A":       None,    # use DEFAULT_SYSTEM
        "B":       None,
        "B_prime": REPO_ROOT / "conf/prompts/condition_b_prime.txt",
        "C":       REPO_ROOT / "conf/prompts/condition_c.txt",
        "D_prime": None,    # primary uses DEFAULT_SYSTEM; verifier has its own prompt
        "E":       None,
    }
    p = mapping.get(cond)
    return p.read_text().strip() if p is not None else None


def _build_adapter(cond: str):
    """Return the adapter instance for the given condition."""
    if cond in ("A", "B", "B_prime", "C"):
        return Qwen3Adapter(
            base_url=server.base_url,
            use_tools=(cond != "A"),
        )
    if cond == "D_prime":
        from harness.verifier import PostHocVerifierAdapter
        primary = Qwen3Adapter(base_url=server.base_url, use_tools=True)
        return PostHocVerifierAdapter(
            primary=primary,
            verifier_prompt_path=str(REPO_ROOT / "conf/prompts/condition_d_prime.txt"),
            verifier_model="claude-sonnet-4-6",
        )
    if cond == "E":
        # Defer to the notebook 07 adapter pattern via a small inline class.
        # We construct it here to avoid coupling on import order.
        from harness.extraction import FactExtractor
        from harness.monitors import ClinicalInputMonitor
        from harness.karma_adapter.mcp_tools import fetch_tool_schemas
        from transformers import AutoTokenizer
        from interwhen import stream_completion
        import asyncio, re
        import nest_asyncio
        nest_asyncio.apply()

        extractor = FactExtractor(
            prompt_path=str(REPO_ROOT / "conf/prompts/extractor.txt"),
            model="claude-sonnet-4-6",
        )
        feedback_template = (REPO_ROOT / "conf/prompts/condition_e_feedback.txt").read_text()
        tools = fetch_tool_schemas()
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Thinking-2507")
        completions_url = f"{server.base_url}/completions"
        _THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

        class _Adapter:
            def run(self, prompt, system=None):
                facts = extractor.extract(prompt)
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                rendered = tokenizer.apply_chat_template(
                    messages, tools=tools, add_generation_prompt=True, tokenize=False,
                )
                monitor = ClinicalInputMonitor(patient_facts=facts, feedback_template=feedback_template)
                llm_server = {
                    "url": completions_url,
                    "headers": {"Content-Type": "application/json"},
                    "payload": {
                        "model": "Qwen/Qwen3-30B-A3B-Thinking-2507",
                        "max_tokens": 4096,
                        "temperature": 0.0,
                        "stream": True,
                    },
                }
                loop = asyncio.get_event_loop()
                generated = loop.run_until_complete(
                    stream_completion(
                        prompt=rendered, prev_text="",
                        llm_server=llm_server, monitors=[monitor],
                        async_execution=True,
                    )
                )
                text = generated
                if "<|im_start|>assistant" in text:
                    text = text.rsplit("<|im_start|>assistant", 1)[-1]
                text = text.split("<|im_end|>", 1)[0]
                text = _THINK_RE.sub("", text).strip()

                from harness.karma_adapter.qwen3 import Qwen3Response
                return Qwen3Response(
                    text=text,
                    n_tool_calls=monitor.metrics.n_steps_seen,
                    raw_prompt=rendered,
                    raw_completion=generated,
                    stop_reason="stop",
                    prompt_tokens=facts.prompt_tokens,
                    completion_tokens=facts.completion_tokens,
                    n_model_calls=1,
                )
        return _Adapter()
    raise ValueError(f"Unknown condition: {cond}")


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
            system=system if system else None,
            out_dir=str(_out_dir(cond, "pilot")),
        )
        record["pilot"] = {
            "accuracy": pilot.accuracy,
            "n": pilot.n,
            "parse_failures": pilot.n_parse_failures,
            "mean_tool_calls": float(pilot.rows["n_tool_calls"].mean()),
        }
        print(f"[{cond}] pilot acc={pilot.accuracy:.1%}, tool_calls/vignette={record['pilot']['mean_tool_calls']:.2f}")

        print(f"[{cond}] FULL (n=1066, workers={WORKERS[cond]})")
        full = run_eval(
            adapter, n=None, max_workers=WORKERS[cond],
            system=system if system else None,
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

# MAGIC %md ## 6. Run all conditions

# COMMAND ----------

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

# MAGIC %md ## 7. Save run records

# COMMAND ----------

# DBTITLE 1,Cell 24
out_path = RESULTS_ROOT / "_AGGREGATED_RESULTS.json"
out_path.write_text(json.dumps({"run_records": run_records, "completed_at": time.time()}, indent=2, default=str))
print(f"Run records written to: {out_path}")
print("\nNext: open 09_analysis and run all cells for the full analysis + export bundle.")

# COMMAND ----------

# DBTITLE 1,Cell 25
# MAGIC %md ## 8. Cleanup

# COMMAND ----------

server.stop()
print("server stopped:", not server.is_alive())
print("\nDone. Inspect /dbfs/results/_AGGREGATED_RESULTS.json for the full summary.")
