"""CUPED: use a pre-experiment covariate to shrink metric variance.

If you know each user's value of a metric from before the test started (last
month's sessions, say), most of the noise in the test metric is predictable
from it. Subtracting the predictable part leaves the treatment effect
unchanged on average but makes the confidence interval narrower, often a lot
narrower. Deng et al. (2013), "Improving the Sensitivity of Online Controlled
Experiments by Utilizing Pre-Experiment Data".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CupedResult:
    adjusted: np.ndarray
    theta: float
    variance_reduction: float

    def __str__(self) -> str:
        return f"CUPED: theta = {self.theta:.4g}, variance reduced by {self.variance_reduction:.1%}"


def cuped(y, x_pre, theta: float | None = None) -> CupedResult:
    """Return y - theta * (x_pre - mean(x_pre)).

    Pass both groups together (pooled), so theta and the covariate mean are the
    same for control and treatment. Estimating them separately per group can
    bias the effect.

    y:     the metric during the test, one value per user
    x_pre: the same users' pre-test covariate (must not be affected by treatment)
    theta: leave as None to use the variance-minimising value cov(y, x) / var(x)
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x_pre, dtype=float)
    if y.shape != x.shape:
        raise ValueError("y and x_pre must have the same length")
    if np.isnan(y).any() or np.isnan(x).any():
        raise ValueError("fill or drop missing values before CUPED")

    var_x = x.var(ddof=1)
    if theta is None:
        theta = 0.0 if var_x == 0 else float(np.cov(y, x, ddof=1)[0, 1] / var_x)
    adjusted = y - theta * (x - x.mean())
    reduction = 1 - adjusted.var(ddof=1) / y.var(ddof=1) if y.var(ddof=1) > 0 else 0.0
    return CupedResult(adjusted=adjusted, theta=float(theta), variance_reduction=float(reduction))
