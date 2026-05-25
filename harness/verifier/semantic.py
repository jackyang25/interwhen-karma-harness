"""Deterministic semantic verifier (Condition E).

Compares the clinical arguments of a `medical_calculator_output` tool call
against `PatientFacts` extracted by Sonnet. Returns a list of `Violation`s
when the model's planned values clearly contradict the case.

Design — schema-aligned, no bridging logic:
    * The extractor schema is generated from MCP's per-calculator input
      schemas (see `harness.extraction.prompt_builder`). PatientFacts.raw
      is a flat dict keyed by exactly the MCP field names.
    * The verifier looks up each tool-call argument by exact string match
      in `facts.raw`. No alias dictionary, no synonym table, no unit
      stripping, no LLM judgment in the bridge.
    * Only `medical_calculator_output` calls are verified — the other MCP
      tools (drug search, protocol search, etc.) take routing arguments,
      not patient values. Detected by the presence of an `input_data` dict.

Conservative flagging principle: flag only when the case has CLEAR
evidence (extractor produced a non-null value) AND the planned value
clearly contradicts it. Anything else — extractor returned null, type
ambiguous — is silently passed.

The split between LLM extraction (Sonnet) and deterministic comparison
(this module) is the methodological defense against AI-verifying-AI
circularity (TESTING.md §5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.extraction.extractor import PatientFacts


@dataclass
class Violation:
    field: str
    planned: Any
    expected: Any
    note: str

    def __str__(self) -> str:
        return (
            f"{self.field}: you passed {self.planned!r}, case states {self.expected!r}. {self.note}"
        )


# Enum normalization — flatten case/punctuation/abbrev synonyms before
# comparison. Kept narrow: covers the variants the case text might use
# even when MCP and the extractor both declare a normalized enum value.
_ENUM_SYNONYMS: dict[str, str] = {
    # sex
    "m": "male", "male": "male", "man": "male",
    "f": "female", "female": "female", "woman": "female",
    # activity level
    "sedentary": "sedentary", "low": "sedentary", "inactive": "sedentary",
    "lightly active": "lightly_active", "lightly_active": "lightly_active", "light": "lightly_active",
    "moderately active": "moderately_active", "moderately_active": "moderately_active", "moderate": "moderately_active",
    "very active": "very_active", "very_active": "very_active", "active": "very_active", "athlete": "very_active",
    "extra active": "extra_active", "extra_active": "extra_active",
    # smoking
    "never": "never", "non-smoker": "never", "nonsmoker": "never",
    "former": "former", "ex-smoker": "former",
    "current": "current", "smoker": "current", "active smoker": "current",
    # alcohol
    "none": "none", "no alcohol": "none",
    "occasional": "occasional", "social": "occasional",
    "regular": "regular", "heavy": "heavy", "alcoholic": "heavy",
}


def verify(arguments: dict[str, Any], facts: PatientFacts) -> list[Violation]:
    """Check each clinical argument against PatientFacts; return violations.

    Only acts on `medical_calculator_output`-shaped calls (i.e., calls
    with a dict-valued `input_data` argument). Other tool calls (drug
    search, protocol search, calculator metadata) have no patient values
    to verify — returns [] immediately.
    """
    if not isinstance(arguments, dict):
        return []
    if not facts.extractor_ok:
        return []

    # Unwrap medical_calculator_output's clinical args from input_data.
    # Calls without input_data are not clinical computations — pass.
    clinical_args = arguments.get("input_data")
    if not isinstance(clinical_args, dict):
        return []

    violations: list[Violation] = []
    for field_name, planned_value in clinical_args.items():
        if planned_value is None:
            continue   # model didn't actually choose a value

        expected_value = facts.raw.get(field_name)
        if expected_value is None:
            continue   # case doesn't state this field — conservative skip

        violation = _compare(field_name, planned_value, expected_value)
        if violation is not None:
            violations.append(violation)

    return violations


def _compare(field: str, planned: Any, expected: Any) -> Violation | None:
    """Return a Violation if planned clearly contradicts expected, else None."""
    # Numeric: 1% relative tolerance + small absolute floor for integers
    if _is_number(planned) and _is_number(expected):
        p, e = float(planned), float(expected)
        if abs(e) < 1e-9:
            if abs(p) < 1e-6:
                return None
        elif abs(p - e) / max(abs(e), 1e-9) <= 0.01:
            return None
        elif abs(p - e) <= 0.5:
            return None
        return Violation(
            field=field,
            planned=planned,
            expected=expected,
            note=f"case states {expected}, planned call passes {planned}",
        )

    # String enum
    if isinstance(planned, str) and isinstance(expected, str):
        if _normalize_enum(planned) == _normalize_enum(expected):
            return None
        return Violation(
            field=field,
            planned=planned,
            expected=expected,
            note=f"case states {expected!r}, planned call passes {planned!r}",
        )

    # Boolean
    if isinstance(planned, bool) and isinstance(expected, bool):
        if planned == expected:
            return None
        return Violation(
            field=field,
            planned=planned,
            expected=expected,
            note=f"case states {expected}, planned call passes {planned}",
        )

    # Type mismatch — flag with note (cautious; might be string-int coercion)
    return Violation(
        field=field,
        planned=planned,
        expected=expected,
        note=f"value type differs from the case ({type(expected).__name__} vs {type(planned).__name__})",
    )


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _normalize_enum(s: str) -> str:
    key = s.strip().lower().replace("-", " ").replace("_", " ")
    return _ENUM_SYNONYMS.get(key, key.replace(" ", "_"))


def format_feedback(template: str, violations: list[Violation]) -> str:
    """Render the violation list into the feedback template's {violations} slot."""
    bullets = "\n".join(f"- {v}" for v in violations)
    return template.replace("{violations}", bullets)
