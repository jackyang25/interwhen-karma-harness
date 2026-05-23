"""Deterministic semantic verifier (Condition E).

Takes a planned tool-call's arguments and the structured patient facts from
`FactExtractor`, and checks whether each argument is consistent with the
case. Returns a list of `Violation`s — each names the field, what the case
states, and what the model planned to pass.

This module performs ONLY deterministic comparison. No LLM calls live here.
The deterministic-comparison-after-LLM-extraction split is the methodological
defense against "AI verifying AI" circularity (TESTING.md §5).

Scope (intentionally conservative):
- Numeric comparison with 1% relative tolerance (rounding/precision)
- Enum normalization (lowercase, underscore-vs-space, common synonyms)
- Field-name matching via a fixed alias table + a fallback heuristic
- Only flag when the case has clear evidence AND the planned value clearly
  contradicts it. Missing evidence → no flag (avoid false positives that
  would derail correct answers).
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


# Map tool-call argument names → path in PatientFacts.
# Conservative: only fields we're confident about; missing aliases just don't fire.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    # demographics
    "age": ("age_years",),
    "age_years": ("age_years",),
    "age_y": ("age_years",),
    "patient_age": ("age_years",),
    "sex": ("sex",),
    "gender": ("sex",),
    # body measurements
    "weight": ("weight_kg",),
    "weight_kg": ("weight_kg",),
    "body_weight": ("weight_kg",),
    "height": ("height_cm",),
    "height_cm": ("height_cm",),
    "body_height": ("height_cm",),
    "bmi": ("bmi",),
    # lifestyle
    "activity_level": ("lifestyle", "activity_level"),
    "physical_activity": ("lifestyle", "activity_level"),
    "smoking": ("lifestyle", "smoking"),
    "smoking_status": ("lifestyle", "smoking"),
    "alcohol": ("lifestyle", "alcohol"),
    "alcohol_use": ("lifestyle", "alcohol"),
    # vitals
    "systolic_bp": ("vitals", "bp_systolic_mmHg"),
    "bp_systolic": ("vitals", "bp_systolic_mmHg"),
    "sbp": ("vitals", "bp_systolic_mmHg"),
    "diastolic_bp": ("vitals", "bp_diastolic_mmHg"),
    "bp_diastolic": ("vitals", "bp_diastolic_mmHg"),
    "dbp": ("vitals", "bp_diastolic_mmHg"),
    "heart_rate": ("vitals", "heart_rate_bpm"),
    "hr": ("vitals", "heart_rate_bpm"),
    "respiratory_rate": ("vitals", "respiratory_rate"),
    "rr": ("vitals", "respiratory_rate"),
    "temperature": ("vitals", "temperature_C"),
    "temp": ("vitals", "temperature_C"),
    "spo2": ("vitals", "spo2_pct"),
    # pregnancy
    "pregnancy_weeks": ("pregnancy_weeks",),
    "gestational_age_weeks": ("pregnancy_weeks",),
}

# Enum normalization — flatten case/punctuation/synonyms before comparison.
_ENUM_SYNONYMS: dict[str, str] = {
    # sex
    "m": "male",
    "male": "male",
    "man": "male",
    "f": "female",
    "female": "female",
    "woman": "female",
    # activity level
    "sedentary": "sedentary",
    "low": "sedentary",
    "inactive": "sedentary",
    "lightly active": "lightly_active",
    "lightly_active": "lightly_active",
    "light": "lightly_active",
    "moderately active": "moderately_active",
    "moderately_active": "moderately_active",
    "moderate": "moderately_active",
    "very active": "very_active",
    "very_active": "very_active",
    "active": "very_active",
    "athlete": "very_active",
    "extra active": "extra_active",
    "extra_active": "extra_active",
    # smoking
    "never": "never",
    "non-smoker": "never",
    "nonsmoker": "never",
    "former": "former",
    "ex-smoker": "former",
    "current": "current",
    "smoker": "current",
    "active smoker": "current",
    # alcohol
    "none": "none",
    "no alcohol": "none",
    "occasional": "occasional",
    "social": "occasional",
    "regular": "regular",
    "heavy": "heavy",
    "alcoholic": "heavy",
}


def verify(arguments: dict[str, Any], facts: PatientFacts) -> list[Violation]:
    """Check each tool-call argument against PatientFacts; return violations.

    Conservative: only flag when both (a) we know how to look up the field
    in facts and (b) the case has a definite value that clearly contradicts
    the planned value. Anything ambiguous → not flagged.
    """
    if not facts.extractor_ok or not isinstance(arguments, dict):
        return []

    violations: list[Violation] = []
    for arg_name, planned_value in arguments.items():
        key = arg_name.lower()
        path = _FIELD_ALIASES.get(key)
        if path is None:
            continue   # unknown field — don't flag

        expected_value = facts.get(*path)
        if expected_value is None:
            continue   # case doesn't state this — don't flag

        violation = _compare(arg_name, planned_value, expected_value)
        if violation is not None:
            violations.append(violation)

    return violations


def _compare(field: str, planned: Any, expected: Any) -> Violation | None:
    """Return a Violation if planned clearly contradicts expected, else None."""
    # Both numeric: compare with 1% relative tolerance + small absolute floor
    if _is_number(planned) and _is_number(expected):
        p, e = float(planned), float(expected)
        if abs(e) < 1e-9:
            if abs(p) < 1e-6:
                return None
        elif abs(p - e) / max(abs(e), 1e-9) <= 0.01:
            return None
        elif abs(p - e) <= 0.5:   # absolute floor for small integers
            return None
        return Violation(
            field=field,
            planned=planned,
            expected=expected,
            note=f"case states {expected}, planned call passes {planned}",
        )

    # Both stringy (enum-like): normalize and compare
    if isinstance(planned, str) and isinstance(expected, str):
        p = _normalize_enum(planned)
        e = _normalize_enum(expected)
        if p == e:
            return None
        return Violation(
            field=field,
            planned=planned,
            expected=expected,
            note=f"case states {expected!r}, planned call passes {planned!r}",
        )

    # Type mismatch (e.g., string vs number) — be cautious: flag as note,
    # the verifier prompt will let the model decide if it's a real issue.
    return Violation(
        field=field,
        planned=planned,
        expected=expected,
        note=f"value type or representation differs from the case ({type(expected).__name__} vs {type(planned).__name__})",
    )


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _normalize_enum(s: str) -> str:
    key = s.strip().lower().replace("-", " ").replace("_", " ")
    return _ENUM_SYNONYMS.get(key, key.replace(" ", "_"))


def format_feedback(template: str, violations: list[Violation]) -> str:
    """Render the list of violations into the feedback template's {violations} slot."""
    bullets = "\n".join(f"- {v}" for v in violations)
    return template.replace("{violations}", bullets)
