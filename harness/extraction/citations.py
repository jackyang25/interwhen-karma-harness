"""Citation handling for the reactive + citations variant.

Two responsibilities:
1. Validate that each Sonnet-returned source_span is a verbatim substring of
   the vignette text (no paraphrase, no normalization).
2. Coerce a citation-shaped extraction dict back to the bare-value dict the
   verifier expects, dropping any field whose citation failed validation
   (treated as abstention — verifier no-ops on missing fields).

The pre-registered design (paper §methods_extractor_citations) treats
validation failure as null/abstention: rather than guess whether the value
is right when the citation is wrong, the field is removed from the verifier's
reference. The validator is intentionally strict (byte-identical substring,
not paraphrase or unit-normalized).
"""
from __future__ import annotations

from typing import Any


def validate_substring(span: str, vignette: str) -> bool:
    """Return True iff `span` is a verbatim substring of `vignette`.

    Strict: no whitespace normalization, no case folding, no unit conversion.
    The pre-registered citation validity rule is byte-identical match. Any
    softening here would introduce LLM-judgment back into a step whose whole
    point is to be deterministic.
    """
    if not isinstance(span, str) or not isinstance(vignette, str):
        return False
    if span == "":
        return False
    return span in vignette


def coerce_citations_to_bare(
    citation_dict: dict[str, Any],
    vignette: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Reduce a {field: {value, source_span}} dict to a bare {field: value}
    dict, dropping fields whose source_span fails substring validation.

    Returns (bare_values, validation_report). The validation_report maps
    every input field name to {"value", "source_span", "valid"} so the
    per-vignette log can record what was accepted vs rejected, and analysis
    can compute citation-acceptance rates per field.

    Tolerance for non-citation shapes: if a field's value isn't a
    {value, source_span} dict, the field is dropped and recorded with
    valid=False, reason="malformed_shape". The adapter then treats it as
    abstention.
    """
    bare: dict[str, Any] = {}
    report: dict[str, dict[str, Any]] = {}

    if not isinstance(citation_dict, dict):
        return bare, report

    for field, payload in citation_dict.items():
        if not isinstance(payload, dict):
            report[field] = {
                "value":       payload,
                "source_span": None,
                "valid":       False,
                "reason":      "malformed_shape",
            }
            continue
        value = payload.get("value")
        span  = payload.get("source_span")
        if value is None:
            report[field] = {
                "value":       None,
                "source_span": span,
                "valid":       False,
                "reason":      "null_value",
            }
            continue
        is_valid = validate_substring(span, vignette) if span is not None else False
        report[field] = {
            "value":       value,
            "source_span": span,
            "valid":       is_valid,
            "reason":      "ok" if is_valid else "span_not_in_vignette",
        }
        if is_valid:
            bare[field] = value

    return bare, report
