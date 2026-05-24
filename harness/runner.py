"""Eval runner for medical_calculator_eval.

Loads the dataset, iterates vignettes through an adapter, scores against ground
truth using the dataset's per-row tolerance, and writes results to parquet.

Dataset schema (from ekacare/medical_calculator_eval README):
  - question_text: clinical vignette
  - confinement_instruction: prompt suffix instructing JSON output
  - expected_output: JSON string containing primary_field + extras
  - primary_field: the key in expected_output to compare
  - tolerance: absolute deviation threshold

Scoring: parse model output as JSON, extract primary_field, score 1 if
|prediction - expected| <= tolerance else 0. Parse failures score 0.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from datasets import load_dataset

DATASET_NAME = "ekacare/medical_calculator_eval"

DEFAULT_SYSTEM = (
    "You are an expert clinician answering medical calculator questions for "
    "Indian patients. Read the case carefully, use any available tools if "
    "useful, and reply strictly with the JSON object the user requests."
)


class Adapter(Protocol):
    def run(self, prompt: str, system: str | None = None) -> Any: ...


@dataclass
class EvalResults:
    rows: pd.DataFrame
    accuracy: float
    n: int
    n_correct: int
    n_parse_failures: int
    # Cost/time aggregates (per-vignette means + run totals)
    mean_prompt_tokens: float = 0.0
    mean_completion_tokens: float = 0.0
    mean_total_tokens: float = 0.0
    median_latency_seconds: float = 0.0
    total_run_seconds: float = 0.0   # wall-clock for the whole run (not sum of per-vignette)
    meta: dict[str, Any] = field(default_factory=dict)

    def estimated_cost_usd(self, gpu_hourly_rate_usd: float = 3.0) -> float:
        """Estimated USD cost for the whole run on a self-hosted single H100.

        Uses total wall-clock × hourly rate. Comparable across conditions
        when the same hardware and worker count are used. Document the rate
        used in any paper figure.
        """
        return self.total_run_seconds * gpu_hourly_rate_usd / 3600.0

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.rows.to_parquet(out / "rows.parquet", index=False)
        summary = {
            "accuracy": self.accuracy,
            "n": self.n,
            "n_correct": self.n_correct,
            "n_parse_failures": self.n_parse_failures,
            "mean_prompt_tokens": self.mean_prompt_tokens,
            "mean_completion_tokens": self.mean_completion_tokens,
            "mean_total_tokens": self.mean_total_tokens,
            "median_latency_seconds": self.median_latency_seconds,
            "total_run_seconds": self.total_run_seconds,
            **self.meta,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return out


def run_eval(
    adapter: Adapter,
    n: int | None = None,
    split: str = "test",
    system: str | None = DEFAULT_SYSTEM,
    out_dir: str | Path | None = None,
    progress_every: int = 25,
    max_workers: int = 1,
) -> EvalResults:
    """Run an adapter on medical_calculator_eval and return scored results.

    n: if set, limit to the first n rows (for pilot runs). None = full set.
    max_workers: thread pool size. 1 = sequential. Set higher to parallelize
        across vignettes for the full 1,066-row run. The adapter must be
        thread-safe at this concurrency (Anthropic client + a fresh MCP
        connection per call are both fine).
    """
    ds = load_dataset(DATASET_NAME, split=split)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    total = len(ds)

    run_t0 = time.time()
    if max_workers <= 1:
        results_by_idx = {}
        for i, row in enumerate(ds):
            results_by_idx[i] = _score_one(adapter, row, system)
            if (i + 1) % progress_every == 0:
                acc = sum(r["correct"] for r in results_by_idx.values()) / len(results_by_idx)
                print(f"  [{i + 1}/{total}]  acc={acc:.3f}")
    else:
        results_by_idx = _run_parallel(
            adapter, ds, system, max_workers=max_workers, progress_every=progress_every
        )
    total_run_seconds = time.time() - run_t0

    rows = [results_by_idx[i] for i in range(total)]

    df = pd.DataFrame(rows)
    results = EvalResults(
        rows=df,
        accuracy=df["correct"].mean(),
        n=len(df),
        n_correct=int(df["correct"].sum()),
        n_parse_failures=int(df["parse_failed"].sum()),
        mean_prompt_tokens=float(df["prompt_tokens"].mean()),
        mean_completion_tokens=float(df["completion_tokens"].mean()),
        mean_total_tokens=float(df["total_tokens"].mean()),
        median_latency_seconds=float(df["elapsed_seconds"].median()),
        total_run_seconds=total_run_seconds,
        meta={"dataset": DATASET_NAME, "split": split, "max_workers": max_workers},
    )
    if out_dir is not None:
        results.save(out_dir)
    return results


def _score_one(adapter: Adapter, row: dict[str, Any], system: str | None) -> dict[str, Any]:
    """Run + score a single vignette. Pure function; safe to call from threads."""
    prompt = f"{row['question_text']}\n\n{row['confinement_instruction']}"
    t0 = time.time()
    try:
        resp = adapter.run(prompt, system=system)
        output_text = resp.text
        n_tool_calls = getattr(resp, "n_tool_calls", 0)
        stop_reason = getattr(resp, "stop_reason", "n/a")
        prompt_tokens = getattr(resp, "prompt_tokens", 0)
        completion_tokens = getattr(resp, "completion_tokens", 0)
        n_model_calls = getattr(resp, "n_model_calls", 0)
        n_verifier_fires = getattr(resp, "n_verifier_fires", 0)
        n_fixes_applied = getattr(resp, "n_fixes_applied", 0)
        violations_history = getattr(resp, "violations_history", []) or []
        # Cost / latency breakouts (populated by the E adapter; 0 elsewhere).
        # These let cost_latency_table separate Sonnet-API cost from on-GPU
        # inference, so deployment readers can see what's billed where.
        extractor_prompt_tokens = getattr(resp, "extractor_prompt_tokens", 0)
        extractor_completion_tokens = getattr(resp, "extractor_completion_tokens", 0)
        extractor_elapsed_s = getattr(resp, "extractor_elapsed_s", 0.0)
        qwen3_prompt_tokens = getattr(resp, "qwen3_prompt_tokens", 0)
        qwen3_completion_tokens = getattr(resp, "qwen3_completion_tokens", 0)
        qwen3_elapsed_s = getattr(resp, "qwen3_elapsed_s", 0.0)
        # D-specific breakouts (populated by PostHocVerifierAdapter; 0 elsewhere).
        primary_prompt_tokens = getattr(resp, "primary_prompt_tokens", 0)
        primary_completion_tokens = getattr(resp, "primary_completion_tokens", 0)
        primary_elapsed_s = getattr(resp, "primary_elapsed_s", 0.0)
        verifier_prompt_tokens = getattr(resp, "verifier_prompt_tokens", 0)
        verifier_completion_tokens = getattr(resp, "verifier_completion_tokens", 0)
        verifier_elapsed_s = getattr(resp, "verifier_elapsed_s", 0.0)
        adapter_error = None
    except Exception as e:
        output_text = ""
        n_tool_calls = 0
        stop_reason = "adapter_error"
        prompt_tokens = 0
        completion_tokens = 0
        n_model_calls = 0
        n_verifier_fires = 0
        n_fixes_applied = 0
        violations_history = []
        extractor_prompt_tokens = 0
        extractor_completion_tokens = 0
        extractor_elapsed_s = 0.0
        qwen3_prompt_tokens = 0
        qwen3_completion_tokens = 0
        qwen3_elapsed_s = 0.0
        primary_prompt_tokens = 0
        primary_completion_tokens = 0
        primary_elapsed_s = 0.0
        verifier_prompt_tokens = 0
        verifier_completion_tokens = 0
        verifier_elapsed_s = 0.0
        adapter_error = f"{type(e).__name__}: {e}"
    elapsed_seconds = time.time() - t0

    try:
        expected = json.loads(row["expected_output"])
        expected_val = float(expected[row["primary_field"]])
        expected_ok = True
    except Exception:
        expected_val = float("nan")
        expected_ok = False

    predicted_val, parse_ok = _parse_prediction(output_text, row["primary_field"])
    tolerance = float(row["tolerance"])
    correct = bool(parse_ok and expected_ok and abs(predicted_val - expected_val) <= tolerance)

    return {
        "id": row.get("id"),
        "category": row.get("category"),
        "expected_calculator": row.get("expected_calculator"),
        "primary_field": row["primary_field"],
        "tolerance": tolerance,
        "expected": expected_val,
        "predicted": predicted_val if parse_ok else None,
        "correct": correct,
        "parse_failed": not parse_ok,
        "n_tool_calls": n_tool_calls,
        "n_model_calls": n_model_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "elapsed_seconds": elapsed_seconds,
        "stop_reason": stop_reason,
        "adapter_error": adapter_error,
        "n_verifier_fires": n_verifier_fires,
        "n_fixes_applied": n_fixes_applied,
        # JSON-encoded so the column survives both parquet and CSV cleanly.
        # Empty list "[]" for non-interwhen conditions.
        "violations_history": json.dumps(violations_history),
        # Honest cost / latency breakouts. Zero for non-E conditions.
        "extractor_prompt_tokens": extractor_prompt_tokens,
        "extractor_completion_tokens": extractor_completion_tokens,
        "extractor_elapsed_s": extractor_elapsed_s,
        "qwen3_prompt_tokens": qwen3_prompt_tokens,
        "qwen3_completion_tokens": qwen3_completion_tokens,
        "qwen3_elapsed_s": qwen3_elapsed_s,
        "primary_prompt_tokens": primary_prompt_tokens,
        "primary_completion_tokens": primary_completion_tokens,
        "primary_elapsed_s": primary_elapsed_s,
        "verifier_prompt_tokens": verifier_prompt_tokens,
        "verifier_completion_tokens": verifier_completion_tokens,
        "verifier_elapsed_s": verifier_elapsed_s,
        "raw_output": output_text,
    }


def _run_parallel(
    adapter: Adapter,
    ds,
    system: str | None,
    max_workers: int,
    progress_every: int,
) -> dict[int, dict[str, Any]]:
    """Dispatch vignettes across a thread pool, preserving dataset order on return."""
    total = len(ds)
    results: dict[int, dict[str, Any]] = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {ex.submit(_score_one, adapter, ds[i], system): i for i in range(total)}
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            results[i] = fut.result()
            completed += 1
            if completed % progress_every == 0 or completed == total:
                acc = sum(r["correct"] for r in results.values()) / completed
                print(f"  [{completed}/{total}]  acc={acc:.3f}")
    return results


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_prediction(text: str, primary_field: str) -> tuple[float, bool]:
    """Extract the numeric prediction for `primary_field` from a model response.

    Strategy:
      1. Find the last fenced ```json``` block, parse it, look up primary_field.
      2. Else find the last {...} blob in the text, parse it, look up primary_field.
      3. Else fall back to the last number in the text.
    """
    # Strategy 1: fenced JSON
    for m in reversed(list(re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL))):
        v = _try_json_extract(m.group(1), primary_field)
        if v is not None:
            return v, True

    # Strategy 2: any {...} object
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))):
        v = _try_json_extract(m.group(0), primary_field)
        if v is not None:
            return v, True

    # Strategy 3: last number
    nums = _NUM_RE.findall(text)
    if nums:
        try:
            return float(nums[-1]), True
        except ValueError:
            pass
    return float("nan"), False


def _try_json_extract(blob: str, key: str) -> float | None:
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    val = obj.get(key)
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        m = _NUM_RE.search(val)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None
