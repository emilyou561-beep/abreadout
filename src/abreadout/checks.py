"""Checks to run before reading any metric: is the split what it was supposed to be?"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SRMResult:
    counts: tuple[int, ...]
    expected_share: tuple[float, ...]
    observed_share: tuple[float, ...]
    chi2: float
    p_value: float
    alpha: float

    @property
    def mismatch(self) -> bool:
        return self.p_value < self.alpha

    def __str__(self) -> str:
        obs = " / ".join(f"{s:.1%}" for s in self.observed_share)
        exp = " / ".join(f"{s:.1%}" for s in self.expected_share)
        verdict = "SAMPLE RATIO MISMATCH" if self.mismatch else "ok"
        return f"SRM check: {obs} observed vs {exp} expected, chi2 = {self.chi2:.2f}, {'p < 0.0001' if self.p_value < 0.0001 else f'p = {self.p_value:.4f}'} -> {verdict}"


def srm_check(counts: Sequence[int], expected: Sequence[float] | None = None, alpha: float = 0.01) -> SRMResult:
    """Chi-square test of whether users split across groups as designed.

    A mismatch usually means a bug in assignment or logging (bots filtered from
    one arm, a redirect that drops users, a crash in the new version), and it
    can bias every metric in the test. Check it first.

    counts:   users per group, e.g. (44700, 45489)
    expected: intended shares, defaults to an even split
    alpha:    threshold for flagging. 0.01 by default; many teams use a stricter
              0.001 because they run this on every test and want few false alarms.
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 1 or len(counts) < 2:
        raise ValueError("counts needs at least two groups")
    if (counts < 0).any() or counts.sum() == 0:
        raise ValueError("counts must be non-negative and not all zero")

    if expected is None:
        expected = np.full(len(counts), 1 / len(counts))
    expected = np.asarray(expected, dtype=float)
    if len(expected) != len(counts):
        raise ValueError("expected must have one share per group")
    expected = expected / expected.sum()

    chi2, p = stats.chisquare(counts, f_exp=expected * counts.sum())
    return SRMResult(
        counts=tuple(int(c) for c in counts),
        expected_share=tuple(float(e) for e in expected),
        observed_share=tuple(float(c) for c in counts / counts.sum()),
        chi2=float(chi2),
        p_value=float(p),
        alpha=alpha,
    )
