"""Post-run statistical analysis.

Reads result JSON / traces from results/ (or DBFS), computes:
- per-condition accuracy with Wilson confidence intervals,
- error-type decomposition (wrong calculator vs wrong input; enum / unit sub-types),
- paired McNemar tests for the confirmatory comparison family,
- Bonferroni-adjusted p-values,
- verifier characterization (firing rate, precision, recall, correction success,
  false-positive cost).

See TESTING.md Section 7 for the full statistical plan.
"""
