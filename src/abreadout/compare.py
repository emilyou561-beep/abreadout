"""Comparing a metric between control and treatment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats


def fmt_p(p: float) -> str:
    return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"


@dataclass
class Comparison:
    metric: str
    control: float
    treatment: float
    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    method: str
    alpha: float = 0.05

    @property
    def lift(self) -> float:
        """Relative change, treatment vs. control."""
        return self.diff / self.control if self.control else float("nan")

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def __str__(self) -> str:
        level = f"{1 - self.alpha:.0%}"
        if self.method == "two-proportion z-test":
            # rates read better as percentages and differences as points
            return (
                f"{self.metric}: {self.control:.2%} -> {self.treatment:.2%} "
                f"({self.diff * 100:+.2f} pts, {level} CI [{self.ci_low * 100:+.2f}, {self.ci_high * 100:+.2f}], "
                f"lift {self.lift:+.1%}, {fmt_p(self.p_value)})"
            )
        return (
            f"{self.metric}: {self.control:.4g} -> {self.treatment:.4g} "
            f"(diff {self.diff:+.4g}, {level} CI [{self.ci_low:+.4g}, {self.ci_high:+.4g}], "
            f"lift {self.lift:+.1%}, {fmt_p(self.p_value)}, Welch t-test)"
        )


def proportion_test(
    successes_control: int,
    n_control: int,
    successes_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
    metric: str = "conversion",
) -> Comparison:
    """Two-proportion z-test.

    The p-value uses the pooled standard error (the usual null of "no
    difference"); the confidence interval uses the unpooled one, which is the
    right width when the rates actually differ.
    """
    for s, n in [(successes_control, n_control), (successes_treatment, n_treatment)]:
        if n <= 0 or not 0 <= s <= n:
            raise ValueError("need 0 <= successes <= n and n > 0")

    p_c = successes_control / n_control
    p_t = successes_treatment / n_treatment
    diff = p_t - p_c

    pooled = (successes_control + successes_treatment) / (n_control + n_treatment)
    se_null = np.sqrt(pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment))
    z = diff / se_null if se_null > 0 else 0.0
    p_value = 2 * stats.norm.sf(abs(z))

    se = np.sqrt(p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treatment)
    half = stats.norm.ppf(1 - alpha / 2) * se
    return Comparison(metric, p_c, p_t, diff, diff - half, diff + half, float(p_value), "two-proportion z-test", alpha)


def mean_test(control, treatment, alpha: float = 0.05, metric: str = "mean") -> Comparison:
    """Welch's t-test for a difference in means (no equal-variance assumption)."""
    x = np.asarray(control, dtype=float)
    y = np.asarray(treatment, dtype=float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        raise ValueError("need at least two observations per group")

    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx**2 / (len(x) - 1) + vy**2 / (len(y) - 1))
    diff = y.mean() - x.mean()
    p_value = stats.ttest_ind(y, x, equal_var=False).pvalue
    half = stats.t.ppf(1 - alpha / 2, df) * se
    return Comparison(metric, x.mean(), y.mean(), diff, diff - half, diff + half, float(p_value), "Welch t-test", alpha)


@dataclass
class BootstrapResult:
    metric: str
    diff: float
    ci_low: float
    ci_high: float
    prob_treatment_better: float
    n_resamples: int

    def __str__(self) -> str:
        return (
            f"{self.metric}: diff {self.diff:+.4g}, bootstrap CI [{self.ci_low:+.4g}, {self.ci_high:+.4g}], "
            f"treatment higher in {self.prob_treatment_better:.1%} of {self.n_resamples:,} resamples"
        )


def bootstrap_diff(
    control,
    treatment,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 5000,
    alpha: float = 0.05,
    seed: int | None = 0,
    metric: str = "statistic",
) -> BootstrapResult:
    """Percentile bootstrap for any statistic (median, 90th percentile, ...).

    Useful when the metric is skewed (revenue, session length) and a t-test's
    normal approximation is shaky.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(control, dtype=float)
    y = np.asarray(treatment, dtype=float)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        diffs[i] = statistic(rng.choice(y, len(y))) - statistic(rng.choice(x, len(x)))
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        metric=metric,
        diff=float(statistic(y) - statistic(x)),
        ci_low=float(lo),
        ci_high=float(hi),
        prob_treatment_better=float((diffs > 0).mean()),
        n_resamples=n_resamples,
    )


def holm(p_values, alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni: which of several metrics are still significant.

    Reading five metrics at alpha = 0.05 gives roughly a 23% chance that at
    least one looks significant by luck. Holm keeps that family-wide rate at
    alpha and is never less powerful than plain Bonferroni.
    """
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject.tolist()
