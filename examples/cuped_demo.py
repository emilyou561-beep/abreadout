"""How much does CUPED help? A simulated test where we know the true effect.

20,000 users, each with last month's sessions (the pre-period covariate).
The new feature adds 0.3 sessions per user. Compare the confidence interval
with and without CUPED.
"""

import numpy as np

import abreadout as ab

rng = np.random.default_rng(42)
n = 20_000
last_month = rng.gamma(shape=2.0, scale=5.0, size=n)            # skewed, like real usage
treated = rng.random(n) < 0.5
this_month = 0.9 * last_month + rng.normal(0, 3, n) + 0.3 * treated

raw = ab.mean_test(this_month[~treated], this_month[treated], metric="sessions")
adj = ab.cuped(this_month, last_month)
cup = ab.mean_test(adj.adjusted[~treated], adj.adjusted[treated], metric="sessions (CUPED)")

print(raw)
print(cup)
print(adj)
w_raw, w_cup = raw.ci_high - raw.ci_low, cup.ci_high - cup.ci_low
print(f"CI is {1 - w_cup / w_raw:.0%} narrower; that's like running the test with {(w_raw / w_cup) ** 2:.1f}x the users")
