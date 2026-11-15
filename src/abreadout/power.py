"""Sample size and power for conversion-rate tests (normal approximation)."""

from __future__ import annotations

import math

from scipy import stats


def _check(baseline: float, mde: float) -> float:
    if not 0 < baseline < 1:
        raise ValueError("baseline must be a rate between 0 and 1")
    target = baseline + mde
    if not 0 < target < 1 or mde == 0:
        raise ValueError("baseline + mde must be a rate between 0 and 1, and mde can't be 0")
    return target


def sample_size(baseline: float, mde: float, alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0) -> tuple[int, int]:
    """Users needed per group to detect an absolute change of `mde` in a rate.

    baseline: control rate, e.g. 0.19 for 19% retention
    mde:      smallest absolute change worth detecting, e.g. -0.01 for a 1-point drop
    ratio:    treatment size / control size (1 = even split)

    Returns (n_control, n_treatment), rounded up.
    """
    target = _check(baseline, mde)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    var = baseline * (1 - baseline) + target * (1 - target) / ratio
    n_control = (z_a + z_b) ** 2 * var / mde**2
    return math.ceil(n_control), math.ceil(n_control * ratio)


def power(baseline: float, mde: float, n_control: int, n_treatment: int | None = None, alpha: float = 0.05) -> float:
    """Chance of detecting an absolute change of `mde` with the given group sizes."""
    target = _check(baseline, mde)
    n_treatment = n_treatment or n_control
    se = math.sqrt(baseline * (1 - baseline) / n_control + target * (1 - target) / n_treatment)
    z_a = stats.norm.ppf(1 - alpha / 2)
    shift = abs(mde) / se
    return float(stats.norm.cdf(shift - z_a) + stats.norm.cdf(-shift - z_a))


def minimum_detectable_effect(baseline: float, n_control: int, n_treatment: int | None = None, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute change (in either direction) the test can reliably detect.

    Uses the baseline variance for both groups, which is close enough when the
    effect is small relative to the rate.
    """
    if not 0 < baseline < 1:
        raise ValueError("baseline must be a rate between 0 and 1")
    n_treatment = n_treatment or n_control
    se = math.sqrt(baseline * (1 - baseline) * (1 / n_control + 1 / n_treatment))
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se)
