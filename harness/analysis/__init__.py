"""Post-run statistical analysis (paper §methods_stats).

Currently exposes the statistical primitives — Wilson CIs, McNemar's test, and
Bonferroni correction. Error-type decomposition and verifier characterization
land here once the relevant conditions have produced data.
"""
from harness.analysis.stats import (
    McNemarResult,
    WilsonInterval,
    bonferroni,
    mcnemar,
    wilson_ci,
)

__all__ = [
    "McNemarResult",
    "WilsonInterval",
    "bonferroni",
    "mcnemar",
    "wilson_ci",
]
