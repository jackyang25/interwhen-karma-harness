"""Statistical primitives for the apparatus-validation and condition comparisons.

These functions don't depend on any condition's results — they're pure stats and
can be unit-tested independently. They implement the paper's statistical
analysis (§methods_stats): Wilson confidence intervals for per-condition
accuracy, McNemar's test for paired same-question comparisons, and Bonferroni
correction across the six confirmatory comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint


@dataclass
class WilsonInterval:
    accuracy: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.accuracy:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n})"


def wilson_ci(n_correct: int, n_total: int, alpha: float = 0.05) -> WilsonInterval:
    """Wilson score CI for a binomial proportion."""
    if n_total == 0:
        return WilsonInterval(accuracy=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    lo, hi = proportion_confint(n_correct, n_total, alpha=alpha, method="wilson")
    return WilsonInterval(accuracy=n_correct / n_total, lo=float(lo), hi=float(hi), n=n_total)


@dataclass
class McNemarResult:
    b: int  # X correct, Y wrong
    c: int  # X wrong, Y correct
    statistic: float
    pvalue: float

    def __str__(self) -> str:
        return f"McNemar b={self.b}, c={self.c}, p={self.pvalue:.4g}"


def mcnemar(x_correct: np.ndarray, y_correct: np.ndarray, exact: bool | None = None) -> McNemarResult:
    """McNemar's test on paired correctness vectors.

    x_correct and y_correct are 0/1 arrays of equal length, indexed by the same
    questions. Uses exact binomial when b+c < 25 (the standard threshold),
    chi-squared with continuity correction otherwise — unless overridden.
    """
    x = np.asarray(x_correct).astype(bool)
    y = np.asarray(y_correct).astype(bool)
    if x.shape != y.shape:
        raise ValueError("x_correct and y_correct must have the same shape")
    b = int(((x) & (~y)).sum())
    c = int(((~x) & (y)).sum())
    n = b + c
    if n == 0:
        return McNemarResult(b=0, c=0, statistic=float("nan"), pvalue=1.0)
    use_exact = exact if exact is not None else (n < 25)
    if use_exact:
        # Two-sided exact binomial: probability of seeing |b - c| or more extreme
        # under H0 that b is Binomial(n, 0.5).
        pvalue = float(stats.binomtest(min(b, c), n, p=0.5).pvalue)
        statistic = float(min(b, c))
    else:
        statistic = float((abs(b - c) - 1) ** 2 / n)
        pvalue = float(stats.chi2.sf(statistic, df=1))
    return McNemarResult(b=b, c=c, statistic=statistic, pvalue=pvalue)


def bonferroni(pvalues: dict[str, float], n_tests: int | None = None) -> dict[str, float]:
    """Bonferroni-corrected p-values, capped at 1.0.

    The §7 family is the four confirmatory comparisons (B vs A, E vs B, E vs C,
    E vs D'). Pass that family in as a dict; n_tests defaults to len(pvalues).
    """
    k = n_tests if n_tests is not None else len(pvalues)
    return {name: min(1.0, p * k) for name, p in pvalues.items()}
