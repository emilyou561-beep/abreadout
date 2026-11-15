import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest

import abreadout as ab

# --- SRM -------------------------------------------------------------------

def test_srm_even_split_passes():
    r = ab.srm_check([10_000, 10_050])
    assert not r.mismatch
    assert r.p_value > 0.5


def test_srm_flags_skewed_split():
    r = ab.srm_check([10_000, 10_600])
    assert r.mismatch
    assert r.p_value < 1e-4


def test_srm_uneven_design():
    # a 90/10 test that came out 90/10 is fine
    assert not ab.srm_check([90_120, 9_880], expected=[0.9, 0.1]).mismatch


def test_srm_matches_scipy():
    r = ab.srm_check([44_700, 45_489])
    assert r.p_value == pytest.approx(stats.chisquare([44_700, 45_489]).pvalue)


@pytest.mark.parametrize("bad", [[10], [-1, 5], [0, 0]])
def test_srm_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        ab.srm_check(bad)


# --- proportions -----------------------------------------------------------

def test_proportion_p_value_matches_statsmodels():
    r = ab.proportion_test(8_502, 44_700, 8_279, 45_489)
    _, p = proportions_ztest([8_279, 8_502], [45_489, 44_700])
    assert r.p_value == pytest.approx(p)


def test_proportion_ci_contains_diff_and_has_right_width():
    r = ab.proportion_test(500, 5_000, 560, 5_000)
    assert r.ci_low < r.diff < r.ci_high
    p_c, p_t = 0.1, 0.112
    se = np.sqrt(p_c * (1 - p_c) / 5000 + p_t * (1 - p_t) / 5000)
    assert (r.ci_high - r.ci_low) / 2 == pytest.approx(1.959964 * se, rel=1e-5)


def test_proportion_no_difference():
    r = ab.proportion_test(100, 1_000, 100, 1_000)
    assert r.diff == 0
    assert r.p_value == pytest.approx(1.0)


def test_proportion_rejects_impossible_counts():
    with pytest.raises(ValueError):
        ab.proportion_test(11, 10, 5, 10)


# --- means and bootstrap ---------------------------------------------------

def test_mean_test_matches_scipy_welch():
    rng = np.random.default_rng(1)
    x, y = rng.normal(10, 2, 300), rng.normal(10.5, 3, 250)
    r = ab.mean_test(x, y)
    assert r.p_value == pytest.approx(stats.ttest_ind(y, x, equal_var=False).pvalue)
    ci = stats.ttest_ind(y, x, equal_var=False).confidence_interval()
    assert (r.ci_low, r.ci_high) == pytest.approx((ci.low, ci.high))


def test_bootstrap_is_reproducible_and_sensible():
    rng = np.random.default_rng(2)
    x, y = rng.exponential(1.0, 2_000), rng.exponential(1.3, 2_000)
    a = ab.bootstrap_diff(x, y, statistic=np.median, n_resamples=1_000, seed=7)
    b = ab.bootstrap_diff(x, y, statistic=np.median, n_resamples=1_000, seed=7)
    assert a == b
    assert a.ci_low < a.diff < a.ci_high
    assert a.prob_treatment_better > 0.95


# --- Holm ------------------------------------------------------------------

def test_holm_known_case():
    # sorted p: .001 <= .05/4, .01 <= .05/3, .03 > .05/2 -> stop
    assert ab.holm([0.01, 0.04, 0.03, 0.001]) == [True, False, False, True]


def test_holm_all_large():
    assert ab.holm([0.2, 0.5]) == [False, False]


# --- power -----------------------------------------------------------------

def test_power_matches_statsmodels():
    # statsmodels works in Cohen's h; compare on a case where the two agree closely
    p1, p2, n = 0.19, 0.182, 45_000
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    expected = NormalIndPower().power(effect_size=h, nobs1=n, alpha=0.05)
    assert ab.power(p1, p2 - p1, n) == pytest.approx(expected, abs=0.01)


def test_sample_size_gives_target_power():
    n_c, n_t = ab.sample_size(0.10, 0.01, power=0.8)
    assert ab.power(0.10, 0.01, n_c, n_t) == pytest.approx(0.8, abs=0.005)


def test_sample_size_uneven_ratio():
    n_c, n_t = ab.sample_size(0.10, 0.01, ratio=0.5)
    assert n_t == pytest.approx(n_c * 0.5, abs=1)


def test_mde_round_trip():
    mde = ab.minimum_detectable_effect(0.2, 20_000)
    assert ab.power(0.2, mde, 20_000) == pytest.approx(0.8, abs=0.02)


@pytest.mark.parametrize("baseline,mde", [(0, 0.1), (0.5, 0), (0.95, 0.1)])
def test_power_rejects_bad_rates(baseline, mde):
    with pytest.raises(ValueError):
        ab.sample_size(baseline, mde)


# --- CUPED -----------------------------------------------------------------

def test_cuped_reduces_variance_and_keeps_effect():
    rng = np.random.default_rng(3)
    n = 20_000
    pre = rng.normal(50, 10, n)
    treated = rng.integers(0, 2, n).astype(bool)
    y = 0.8 * pre + rng.normal(0, 4, n) + 2.0 * treated

    r = ab.cuped(y, pre)
    assert r.variance_reduction > 0.7
    raw = y[treated].mean() - y[~treated].mean()
    adj = r.adjusted[treated].mean() - r.adjusted[~treated].mean()
    assert adj == pytest.approx(2.0, abs=0.15)
    assert abs(adj - 2.0) <= abs(raw - 2.0) + 0.15
    assert ab.mean_test(r.adjusted[~treated], r.adjusted[treated]).ci_high - ab.mean_test(
        r.adjusted[~treated], r.adjusted[treated]
    ).ci_low < ab.mean_test(y[~treated], y[treated]).ci_high - ab.mean_test(y[~treated], y[treated]).ci_low


def test_cuped_constant_covariate_does_nothing():
    y = np.arange(10, dtype=float)
    r = ab.cuped(y, np.ones(10))
    assert r.theta == 0
    np.testing.assert_allclose(r.adjusted, y)


# --- readout ---------------------------------------------------------------

def make_df(n=4_000, seed=4):
    rng = np.random.default_rng(seed)
    group = rng.choice(["a", "b"], n)
    converted = rng.random(n) < np.where(group == "b", 0.14, 0.10)
    revenue = rng.exponential(20, n)
    return pd.DataFrame({"group": group, "converted": converted, "revenue": revenue})


def test_readout_picks_the_right_test():
    r = ab.readout(make_df(), "group", ["converted", "revenue"], control="a", treatment="b")
    methods = {x.metric: x.method for x in r.results}
    assert methods == {"converted": "two-proportion z-test", "revenue": "Welch t-test"}
    assert r.still_significant[0] is True
    assert len(r.to_frame()) == 2
    assert "Holm" in str(r)


def test_readout_warns_on_srm():
    df = make_df()
    df = pd.concat([df, df[df.group == "b"].head(400)])
    r = ab.readout(df, "group", ["converted"], control="a", treatment="b")
    assert r.srm.mismatch
    assert "Fix or explain the split" in str(r)


def test_readout_bad_inputs():
    df = make_df()
    with pytest.raises(KeyError):
        ab.readout(df, "group", ["nope"], control="a", treatment="b")
    with pytest.raises(ValueError):
        ab.readout(df, "group", ["converted"], control="a", treatment="zzz")
