"""Deterministic extractor-prompt generator.

Takes the MCP calculator schema dump (produced by the preflight in
notebooks/02_run_all.py) and emits a Sonnet system prompt that asks for
the *exact* MCP input vocabulary as a flat JSON object.

This module contains no LLM calls and no clinical-vocabulary judgment.
Field names, types, and descriptions are pulled directly from EkaCare's
per-calculator JSON Schemas. The output is a deterministic function of
the input dump: same dump → same prompt, byte-for-byte.

Design properties:
- Source of truth: MCP schemas. The extractor's vocabulary is generated
  *from* MCP, never independently authored.
- Observational extraction discipline: the prompt rules ("explicitly
  states", "omit if absent", "do not infer", "use case-stated units")
  apply uniformly to every field in the generated schema.
- Field deduplication: when multiple calculators declare a field with
  the same name, the union takes the first-seen schema metadata.
  Collisions in semantic meaning across calculators would surface as
  ambiguous descriptions; verify by inspecting the dump if it matters.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# The behavioral discipline header. Identical principles to the prior
# extractor prompt — only the schema body below changes.
_PROMPT_HEADER = """\
You are a medical fact extractor. Read the patient vignette and return a strict JSON object capturing every value the case explicitly states for the fields below.

Return ONLY a JSON object (no prose, no markdown fences). Include ONLY the fields the case explicitly states a value for. Omit any field the case does not state — do not emit null for absent fields. The schema below is the vocabulary of allowed field names; you do not need to emit every field.

Rules:
- Include only what the case explicitly states. If a value must be inferred, computed, or guessed, omit the field.
- Use the units the case provides. Do not convert units. If the case says "creatinine 130 µmol/L", store 130, not the converted mg/dL value.
- For ranges (e.g. "BP 140/90"), split into the appropriate fields.
- For descriptors that map to a calculator's enum (e.g. "sedentary lifestyle" → "sedentary"), use the enum value if the mapping is unambiguous; otherwise omit.
- If the case uses Hinglish or non-standard phrasing, interpret in standard clinical context where unambiguous; otherwise omit.
- Return only valid JSON. No commentary.

Schema (allowed field names with type/range/enum constraints):
"""


def _summarize_field_schema(field_schema: dict[str, Any]) -> str:
    """Render a single field's schema fragment as a one-line annotation.

    We keep this terse — the goal is to give Sonnet enough type/range
    info to extract correctly without bloating the prompt.
    """
    parts: list[str] = []

    # Type (handle anyOf for nullable fields)
    type_str = _extract_type(field_schema)
    if type_str:
        parts.append(type_str)

    # Description
    desc = field_schema.get("description")
    if desc:
        parts.append(str(desc).strip())

    # Enum constraint (inline)
    enum_values = _extract_enum(field_schema)
    if enum_values:
        parts.append(f"enum: {enum_values}")

    # Numeric range
    rng = _extract_range(field_schema)
    if rng:
        parts.append(rng)

    return " — ".join(parts) if parts else "(no description)"


def _extract_type(field_schema: dict[str, Any]) -> str:
    """Resolve the field's primitive type, handling JSON Schema variants."""
    if "type" in field_schema:
        return str(field_schema["type"])
    # anyOf variant (commonly used for nullable: [{"type": "number"}, {"type": "null"}])
    if "anyOf" in field_schema:
        non_null = [v.get("type") for v in field_schema["anyOf"]
                    if isinstance(v, dict) and v.get("type") not in (None, "null")]
        if non_null:
            return str(non_null[0])
    # $ref to a $defs entry (commonly used for enums)
    if "$ref" in field_schema:
        return "string"   # most refs in this dataset are string enums
    return ""


def _extract_enum(field_schema: dict[str, Any]) -> list[str] | None:
    """Pull enum values if present, either inline or via anyOf."""
    if "enum" in field_schema:
        return list(field_schema["enum"])
    if "anyOf" in field_schema:
        for v in field_schema["anyOf"]:
            if isinstance(v, dict) and "enum" in v:
                return list(v["enum"])
    return None


def _extract_range(field_schema: dict[str, Any]) -> str:
    """Render numeric range constraints if present."""
    parts = []
    for key, sym in (("minimum", "≥"), ("exclusiveMinimum", ">"),
                     ("maximum", "≤"), ("exclusiveMaximum", "<")):
        if key in field_schema:
            parts.append(f"{sym}{field_schema[key]}")
    return ", ".join(parts) if parts else ""


def _resolve_field(field_schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """If the field is a $ref to $defs, resolve to the actual definition."""
    if "$ref" in field_schema:
        ref = field_schema["$ref"]
        # Typical form: "#/$defs/Sex"
        if ref.startswith("#/$defs/"):
            def_name = ref[len("#/$defs/"):]
            if def_name in defs:
                resolved = dict(defs[def_name])
                # Preserve the original description if present
                if "description" in field_schema and "description" not in resolved:
                    resolved["description"] = field_schema["description"]
                return resolved
    return field_schema


def build_field_union(schema_dump: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Walk per_calc_schemas; return {field_name: resolved_field_schema}.

    First-seen wins on collision. Resolves $ref into $defs inline.
    """
    field_union: dict[str, dict[str, Any]] = {}
    per_calc = schema_dump.get("per_calc_schemas") or {}
    for calc_name, calc_data in per_calc.items():
        schema = (calc_data or {}).get("schema") or {}
        defs = schema.get("$defs") or {}
        properties = schema.get("properties") or {}
        if not isinstance(properties, dict):
            continue
        for field_name, field_schema in properties.items():
            if field_name in field_union:
                continue
            if not isinstance(field_schema, dict):
                continue
            field_union[field_name] = _resolve_field(field_schema, defs)
    return field_union


def render_extractor_prompt(schema_dump: dict[str, Any]) -> str:
    """Generate the full Sonnet system prompt from a schema dump.

    Deterministic: same input dump → identical output prompt.
    """
    field_union = build_field_union(schema_dump)

    # Sort for deterministic ordering — prompt is byte-stable across runs
    # when the dump is unchanged.
    sorted_fields = sorted(field_union.items())

    # Render the schema body as a JSON-like object with comments
    lines: list[str] = ["{"]
    for i, (field_name, field_schema) in enumerate(sorted_fields):
        annotation = _summarize_field_schema(field_schema)
        comma = "," if i < len(sorted_fields) - 1 else ""
        lines.append(f'  "{field_name}": null{comma}  // {annotation}')
    lines.append("}")
    schema_body = "\n".join(lines)

    return _PROMPT_HEADER + schema_body + "\n"


def render_focused_prompt(
    schema_dump: dict[str, Any],
    field_names: list[str] | set[str] | tuple[str, ...],
) -> str:
    """Generate a focused Sonnet system prompt limited to the specified fields.

    Used by the reactive extraction architecture in B_prime_E_reactive: when the
    model emits a tool call with specific input fields, the verifier extracts
    *only* those fields from the case via a small focused prompt. This avoids
    the long-schema attention dilution the full extractor prompt suffers from.

    The behavioral discipline (omit absent fields, no inference, use case-stated
    units) is identical to the full extractor prompt — only the schema body
    differs.

    Deterministic: same dump + same field_names → identical output. Unknown
    field names (not in the dump's field union) are silently skipped.
    """
    field_union = build_field_union(schema_dump)
    wanted = set(field_names)
    focused = {k: v for k, v in field_union.items() if k in wanted}

    sorted_fields = sorted(focused.items())
    lines: list[str] = ["{"]
    for i, (field_name, field_schema) in enumerate(sorted_fields):
        annotation = _summarize_field_schema(field_schema)
        comma = "," if i < len(sorted_fields) - 1 else ""
        lines.append(f'  "{field_name}": null{comma}  // {annotation}')
    lines.append("}")
    schema_body = "\n".join(lines)

    return _PROMPT_HEADER + schema_body + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# Citation-grounded prompt (reactive + citations variant).
# ──────────────────────────────────────────────────────────────────────────────
_CITATION_PROMPT_HEADER = """\
You are a medical fact extractor. Read the patient vignette and return a strict JSON object capturing every value the case explicitly states for the fields below, with the exact source text that supports each value.

Return ONLY a JSON object (no prose, no markdown fences). For each field you can populate, return an object of the form {"value": <extracted_value>, "source_span": "<verbatim quote from the vignette>"}. The source_span must be a VERBATIM substring of the patient vignette — copy the supporting text exactly, with original spacing and punctuation. Do NOT paraphrase, normalize, or summarize the source_span.

Include ONLY the fields the case explicitly states a value for. Omit any field the case does not state — do not emit null for absent fields, and do not invent a source_span for a field whose value you cannot anchor to a specific quote.

Rules:
- Include only what the case explicitly states. If a value must be inferred, computed, or guessed, omit the field.
- Use the units the case provides. Do not convert units. If the case says "creatinine 130 µmol/L", store 130 and quote "creatinine 130 µmol/L" or similar.
- For ranges (e.g. "BP 140/90"), split into the appropriate fields with source_spans pointing to the same phrase.
- For descriptors that map to a calculator's enum (e.g. "sedentary lifestyle" → "sedentary"), use the enum value and quote the descriptor verbatim.
- If the source_span you would write is not a verbatim substring of the vignette, omit the field entirely.
- Return only valid JSON. No commentary.

Schema (allowed field names with type/range/enum constraints):
"""


def render_focused_prompt_with_citations(
    schema_dump: dict[str, Any],
    field_names: list[str] | set[str] | tuple[str, ...],
) -> str:
    """Generate a focused Sonnet system prompt that asks for (value, source_span)
    per field. Used by the B_prime_E_reactive_citations condition.

    The returned schema body is the same as render_focused_prompt() — the only
    differences vs the bare variant are (a) the prompt header instructs Sonnet
    to return {value, source_span} objects per field and (b) the source_span
    must be a verbatim substring of the vignette (validated downstream by
    harness.extraction.citations.validate_substring; mismatches coerce the
    field to null at the adapter layer, mirroring the abstention path).
    """
    field_union = build_field_union(schema_dump)
    wanted = set(field_names)
    focused = {k: v for k, v in field_union.items() if k in wanted}

    sorted_fields = sorted(focused.items())
    lines: list[str] = ["{"]
    for i, (field_name, field_schema) in enumerate(sorted_fields):
        annotation = _summarize_field_schema(field_schema)
        comma = "," if i < len(sorted_fields) - 1 else ""
        # Note: schema body still shows the field-name vocabulary; the prompt
        # header above tells Sonnet to wrap each value in {value, source_span}.
        lines.append(f'  "{field_name}": null{comma}  // {annotation}')
    lines.append("}")
    schema_body = "\n".join(lines)

    return _CITATION_PROMPT_HEADER + schema_body + "\n"


def regenerate_extractor_prompt(
    dump_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Read the schema dump, regenerate the extractor prompt, write to disk.

    Returns a small metadata dict (n_fields, dump_path, output_path) so
    the preflight can log what happened.
    """
    dump_path = Path(dump_path)
    output_path = Path(output_path)

    schema_dump = json.loads(dump_path.read_text())
    prompt_text = render_extractor_prompt(schema_dump)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt_text)

    field_union = build_field_union(schema_dump)
    return {
        "n_fields":      len(field_union),
        "n_calculators": schema_dump.get("n_schemas_fetched", 0),
        "dump_path":     str(dump_path),
        "output_path":   str(output_path),
        "prompt_chars":  len(prompt_text),
    }
