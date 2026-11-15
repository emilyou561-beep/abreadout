# abreadout

The checks and tests I run every time I read an A/B test, in one small Python package.

Most A/B write-ups go straight to "is the difference significant?". The mistakes usually happen before that: the split wasn't actually 50/50, the test never had the power to see the effect, or five metrics were checked and one came up significant by luck. `abreadout` runs those checks in the order they matter and prints a readout you could paste into a doc.

[![tests](https://github.com/emilyou561-beep/abreadout/actions/workflows/tests.yml/badge.svg)](https://github.com/emilyou561-beep/abreadout/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-00274C?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-00274C?style=flat-square)

```bash
pip install git+https://github.com/emilyou561-beep/abreadout
```

## Example: the Cookie Cats test

90,189 players of a mobile game, first gate at level 30 (control) or 40 (treatment). Full analysis in [cookie-cats-ab-test](https://github.com/emilyou561-beep/cookie-cats-ab-test); here's the same readout in one call ([`examples/cookie_cats.py`](examples/cookie_cats.py)):

```python
import pandas as pd
import abreadout as ab

df = pd.read_csv("cookie_cats.csv")   # one row per player
print(ab.readout(df, group="version", metrics=["retention_1", "retention_7", "sum_gamerounds"],
                 control="gate_30", treatment="gate_40"))
```

```text
A/B readout: gate_40 vs gate_30

SRM check: 49.6% / 50.4% observed vs 50.0% / 50.0% expected, chi2 = 6.90, p = 0.0086 -> SAMPLE RATIO MISMATCH
  ! Fix or explain the split before trusting the numbers below.

  retention_1: 44.82% -> 44.23% (-0.59 pts, 95% CI [-1.24, +0.06], lift -1.3%, p = 0.0744)
* retention_7: 19.02% -> 18.20% (-0.82 pts, 95% CI [-1.33, -0.31], lift -4.3%, p = 0.0016)
  sum_gamerounds: 52.46 -> 51.3 (diff -1.157, 95% CI [-3.72, +1.405], lift -2.2%, p = 0.3759, Welch t-test)

* = significant after Holm correction across all metrics above
```

What it did:

1. **Checked the split first.** It's 49.6 / 50.4 when it should be 50 / 50. That's small, but with 90k players it's very unlikely to be chance (p = 0.009). In a real test I'd raise this with engineering before trusting anything else.
2. **Picked the test per metric.** Retention is 0/1, so it used a two-proportion z-test. Rounds played is a count, so it used Welch's t-test.
3. **Corrected for reading three metrics at once** (Holm). Day-7 retention is still significant after the correction. Day-1 retention isn't, even before it.

Then check the test could actually have seen an effect this size:

```python
>>> ab.power(baseline=0.19, mde=-0.008, n_control=44_700, n_treatment=45_489)
0.870...
>>> ab.minimum_detectable_effect(0.19, 44_700, 45_489)   # in rate units, so 0.73 points
0.0073...
```

## What's in it

| Function | What it's for |
|---|---|
| `srm_check(counts, expected=None)` | Sample ratio mismatch: did users split the way the test was designed? Chi-square test. |
| `proportion_test(s_c, n_c, s_t, n_t)` | Conversion or retention rates. Pooled z-test for the p-value, unpooled CI. |
| `mean_test(control, treatment)` | Continuous metrics. Welch's t-test, so no equal-variance assumption. |
| `bootstrap_diff(control, treatment, statistic=np.median)` | Any statistic (medians, percentiles) when the data is skewed. Also returns the share of resamples where treatment came out ahead. |
| `holm(p_values)` | Holm-Bonferroni correction when you read several metrics. |
| `sample_size(baseline, mde)`, `power(...)`, `minimum_detectable_effect(...)` | Planning a test, or checking afterwards whether a "no effect" result was ever able to see one. |
| `cuped(y, x_pre)` | Variance reduction using each user's pre-test data. |
| `readout(df, group, metrics, control, treatment)` | All of the above in one go on a user-level DataFrame. |

Every result prints as a readable line and keeps its numbers as attributes (`r.diff`, `r.ci_low`, `r.p_value`, `r.lift`, ...). `Readout.to_frame()` gives a table.

## CUPED

If you know each user's value of the metric from before the test, most of the noise in the test metric is predictable from it. CUPED subtracts the predictable part (Deng, Xu, Kohavi and Walker, 2013, *Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data*). The effect estimate stays the same on average, and the confidence interval gets narrower.

Simulated test with a known effect of +0.3 sessions per user ([`examples/cuped_demo.py`](examples/cuped_demo.py)):

```text
sessions:         9.021 -> 9.366 (diff +0.3454, 95% CI [+0.1478, +0.5431], p = 0.0006)
sessions (CUPED): 9.017 -> 9.37  (diff +0.3523, 95% CI [+0.2688, +0.4358], p < 0.0001)
CUPED: theta = 0.9018, variance reduced by 82.1%
CI is 58% narrower; that's like running the test with 5.6x the users
```

Two things to get right, both noted in the docstring: compute CUPED on both groups pooled, and only use a covariate measured *before* the test started. Anything the treatment can move will bias the result.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

27 tests. The z-test, Welch test and SRM check are compared against `statsmodels` and `scipy` results, power is checked by round-tripping `sample_size` → `power`, and CUPED is tested on simulated data where the true effect is known. CI runs them on Python 3.10 to 3.13.

## Notes

- `srm_check` flags at p < 0.01 by default. Teams that run it on every test often use 0.001 to keep false alarms down. Pass `alpha=` to change it.
- Power and sample size use the normal approximation, which is fine at the sample sizes A/B tests usually have. For a few hundred users per group, simulate instead.
- `readout` treats a column as binary if it only contains 0/1 or True/False. Everything else gets a t-test. For ratio metrics (revenue per session, where the unit of analysis isn't the unit of randomisation) you need the delta method, which isn't in here yet.
