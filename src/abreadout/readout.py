"""One call that runs the usual checks and tests on a user-level DataFrame."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from abreadout.checks import SRMResult, srm_check
from abreadout.compare import Comparison, holm, mean_test, proportion_test


@dataclass
class Readout:
    control: str
    treatment: str
    srm: SRMResult
    results: list[Comparison] = field(default_factory=list)
    still_significant: list[bool] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for r, keep in zip(self.results, self.still_significant):
            rows.append(
                {
                    "metric": r.metric,
                    "control": r.control,
                    "treatment": r.treatment,
                    "diff": r.diff,
                    "ci_low": r.ci_low,
                    "ci_high": r.ci_high,
                    "lift": r.lift,
                    "p_value": r.p_value,
                    "significant_after_holm": keep,
                    "test": r.method,
                }
            )
        return pd.DataFrame(rows)

    def __str__(self) -> str:
        lines = [f"A/B readout: {self.treatment} vs {self.control}", "", str(self.srm)]
        if self.srm.mismatch:
            lines.append("  ! Fix or explain the split before trusting the numbers below.")
        lines.append("")
        for r, keep in zip(self.results, self.still_significant):
            mark = "*" if keep else " "
            lines.append(f"{mark} {r}")
        if len(self.results) > 1:
            lines += ["", "* = significant after Holm correction across all metrics above"]
        return "\n".join(lines)


def readout(
    df: pd.DataFrame,
    group: str,
    metrics: Sequence[str],
    control: str,
    treatment: str,
    expected_split: Sequence[float] | None = None,
    alpha: float = 0.05,
    srm_alpha: float = 0.01,
) -> Readout:
    """Run an SRM check, then compare each metric between two groups.

    df:       one row per user (or whatever the randomisation unit is)
    group:    column holding the group label
    metrics:  columns to compare. 0/1 columns get a two-proportion z-test,
              anything else gets Welch's t-test.
    expected_split: intended (control, treatment) shares, default 50/50
    """
    missing = [c for c in [group, *metrics] if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")

    c = df[df[group] == control]
    t = df[df[group] == treatment]
    if c.empty or t.empty:
        raise ValueError(f"no rows for {control!r} or {treatment!r} in column {group!r}")

    srm = srm_check([len(c), len(t)], expected_split, alpha=srm_alpha)

    results = []
    for m in metrics:
        xc, xt = c[m].dropna(), t[m].dropna()
        values = pd.concat([xc, xt]).unique()
        if set(values) <= {0, 1}:
            results.append(proportion_test(int(xc.sum()), len(xc), int(xt.sum()), len(xt), alpha=alpha, metric=m))
        else:
            results.append(mean_test(xc, xt, alpha=alpha, metric=m))

    keep = holm([r.p_value for r in results], alpha=alpha)
    return Readout(control=control, treatment=treatment, srm=srm, results=results, still_significant=keep)
