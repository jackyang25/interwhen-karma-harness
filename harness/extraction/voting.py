"""Majority-vote reducer for the k-shot extraction variant.

The pre-registered k-shot design samples the extractor k=3 times at
temperature 0.7 (paper §methods_extractor_kshot) and reduces per field by
majority vote: a value present in >=2 of 3 samples wins; three-way
disagreement yields null (abstention).

This module contains no LLM calls and no I/O. It operates on already-extracted
dicts produced by FactExtractor.extract_with_prompt.
"""
from __future__ import annotations

from collections import Counter
from typing import Any


def _hashable(value: Any) -> Any:
    """Return a hashable representation of `value` for vote-counting.

    Lists/dicts are not hashable in general, but extractor field values are
    primitives in practice (numbers, strings, booleans). For defensive
    handling, fall back to repr() — values with the same string form vote
    together. Floats compare structurally (1.4 == 1.40 returns True in
    Python, which `Counter` uses for equality).
    """
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return repr(value)


def majority_vote(
    samples: list[dict[str, Any]],
    min_agreement: int = 2,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Reduce a list of per-sample extraction dicts to a single dict by
    per-field majority vote.

    A field's value wins iff at least `min_agreement` of the samples agree
    on it (default 2-of-3). If no value reaches the threshold, the field is
    omitted (treated as abstention by the verifier).

    Returns (voted, report). `voted` is the bare {field: value} dict the
    verifier consumes. `report` is per-field {samples, winner, count,
    accepted} so analysis can compute per-field agreement rates and the
    paper can characterize when the extractor is stable vs. unstable.

    A field not present in a sample is treated as ABSENT for voting (the
    extractor abstained on that field for that sample). Absent counts do
    not vote for any value — they reduce the effective sample size for the
    field, which in turn raises the bar for agreement.
    """
    report: dict[str, dict[str, Any]] = {}
    voted: dict[str, Any] = {}

    if not samples:
        return voted, report

    # Union of all field names across samples.
    all_fields: set[str] = set()
    for s in samples:
        if isinstance(s, dict):
            all_fields.update(s.keys())

    for field in all_fields:
        present_values: list[Any] = []
        for s in samples:
            if isinstance(s, dict) and field in s and s[field] is not None:
                present_values.append(_hashable(s[field]))
        if not present_values:
            report[field] = {
                "samples":  [],
                "winner":   None,
                "count":    0,
                "accepted": False,
                "reason":   "all_samples_absent",
            }
            continue

        counts = Counter(present_values)
        winner, count = counts.most_common(1)[0]
        accepted = count >= min_agreement
        report[field] = {
            "samples":  present_values,
            "winner":   winner,
            "count":    int(count),
            "accepted": accepted,
            "reason":   "ok" if accepted else "no_majority",
        }
        if accepted:
            voted[field] = winner

    return voted, report
