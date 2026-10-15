"""
Tests for ``bma_daysub`` — BMA plus daylight-only background subtraction.

The method exists because on the Columbia/WA-OR shelf (79,849 along-track bins,
2026-08-06) the detected noise floor tracks solar elevation (Spearman rho =
+0.85) and leaving daylight uncorrected biases daytime Kd low: the season x
latitude matched day/night Kd ratio is 0.751 without correction and 1.025 with
daylight-only subtraction. Subtracting at night instead makes things worse,
because ``fit_beers_law_bma`` drops zero-count bins and therefore cannot report
a floor below 1 photon/bin — 34.4% of night floors sit at that artificial limit.

The contract these tests lock down:
  1. Night bins are bit-identical to plain ``bma`` — this is an opt-in method
     that must not perturb the night half of any dataset.
  2. Daylight bins with a detected floor come back with a STEEPER Kd.
  3. Missing / non-finite solar elevation degrades to plain ``bma`` rather than
     raising or silently subtracting.
  4. The threshold is honoured on the correct side of the comparison.
  5. The other fit methods never see the solar kwargs (dispatcher pops them).
  6. Subtraction failures fall back to the valid un-subtracted BMA Kd and say so.

Run:
    pytest tests/test_bma_daysub.py -v
"""
import numpy as np
import pandas as pd
import pytest

from kd_utils.Kd_analysis import (
    KD_FIT_METHODS,
    _fit_kd_with_method,
    fit_beers_law_bma,
    fit_beers_law_bma_daysub,
)

FIT = dict(min_breakpoint_depth=2.0, min_decay_range_m=5.0, min_decay_photons=30,
           min_decay_bins=8, min_total_photons=30, min_total_range_m=5.0)


def make_hist(kd=0.30, e0=400.0, floor=1.4, bp_depth=8.0, zmax=14.0, res=0.25):
    """Exponential decay down to ``bp_depth``, flat ``floor`` below it."""
    z = np.arange(0.0, zmax + res, res)
    counts = np.where(z < bp_depth,
                      e0 * np.exp(-kd * z) + floor,
                      floor)
    counts = np.maximum(np.round(counts), 1.0)
    return pd.DataFrame({'zdepth': z, 'photon_counts': counts})


@pytest.fixture
def hist():
    return make_hist()


def test_method_is_registered():
    assert 'bma_daysub' in KD_FIT_METHODS


def test_night_is_identical_to_plain_bma(hist):
    """Contract 1: the night half of a dataset must not move."""
    base = fit_beers_law_bma(hist, **FIT)
    for elev in (-90.0, -47.8, -6.1):
        got = fit_beers_law_bma_daysub(hist, solar_elevation=elev, **FIT)
        assert got[0] == base[0], f'Kd moved at solar elevation {elev}'
        assert got[4] == base[4]


def test_daylight_steepens_kd(hist):
    """Contract 2: removing a constant floor removes the flattening at depth."""
    base = fit_beers_law_bma(hist, **FIT)
    day = fit_beers_law_bma_daysub(hist, solar_elevation=45.0, **FIT)
    assert np.isfinite(base[3]) and base[3] > 0, 'fixture must have a floor'
    assert day[0] > base[0], 'daylight subtraction should increase Kd'
    # breakpoint and floor are passed through untouched
    assert day[2] == base[2]
    assert day[3] == base[3]


@pytest.mark.parametrize('elev', [None, np.nan])
def test_unknown_sun_angle_degrades_to_bma(hist, elev):
    """Contract 3: never subtract on the strength of a missing sun angle."""
    base = fit_beers_law_bma(hist, **FIT)
    got = fit_beers_law_bma_daysub(hist, solar_elevation=elev, **FIT)
    assert got[0] == base[0]


def test_threshold_boundary_is_exclusive(hist):
    """Contract 4: subtract strictly above the threshold, not at it."""
    base = fit_beers_law_bma(hist, **FIT)
    at = fit_beers_law_bma_daysub(hist, solar_elevation=-6.0,
                                  solar_subtract_threshold=-6.0, **FIT)
    above = fit_beers_law_bma_daysub(hist, solar_elevation=-5.9,
                                     solar_subtract_threshold=-6.0, **FIT)
    assert at[0] == base[0]
    assert above[0] > base[0]


def test_custom_threshold_is_honoured(hist):
    base = fit_beers_law_bma(hist, **FIT)
    # 20 deg sun, but the cut is at 40 -> still "night" for this purpose
    got = fit_beers_law_bma_daysub(hist, solar_elevation=20.0,
                                   solar_subtract_threshold=40.0, **FIT)
    assert got[0] == base[0]


def test_no_noise_floor_bin_is_untouched():
    """Pure decay -> BMA collapses to a single line; nothing to subtract."""
    z = np.arange(0.0, 12.25, 0.25)
    hist = pd.DataFrame({'zdepth': z,
                         'photon_counts': np.maximum(np.round(500 * np.exp(-0.35 * z)), 1.0)})
    base = fit_beers_law_bma(hist, **FIT)
    day = fit_beers_law_bma_daysub(hist, solar_elevation=60.0, **FIT)
    assert day[0] == base[0]
    assert day[4] == base[4]


def test_dispatcher_hides_solar_kwargs_from_other_methods(hist):
    """Contract 5: adding the kwargs must not break the existing methods."""
    for method in ('log_linear', 'bg_subtract', 'breakpoint', 'nonlinear', 'bma'):
        kw = dict(FIT) if method in ('breakpoint', 'bma') else {}
        kd, e0, nf, reason = _fit_kd_with_method(
            hist, method=method, solar_elevation=45.0,
            solar_subtract_threshold=-6.0, **kw)
        assert np.isfinite(kd), f'{method} returned non-finite Kd'


def test_dispatcher_routes_daysub(hist):
    kd_day, _, _, _ = _fit_kd_with_method(hist, method='bma_daysub',
                                          solar_elevation=45.0, **FIT)
    kd_night, _, _, _ = _fit_kd_with_method(hist, method='bma_daysub',
                                            solar_elevation=-45.0, **FIT)
    assert kd_day > kd_night


def test_subtraction_failure_falls_back_and_reports():
    """Contract 6: a floor that eats the decay zone must not produce NaN Kd."""
    # Floor sits just under the shallowest count, so almost nothing survives.
    z = np.arange(0.0, 14.25, 0.25)
    counts = np.where(z < 8.0, np.maximum(np.round(12 * np.exp(-0.5 * z)), 3.0), 3.0)
    hist = pd.DataFrame({'zdepth': z, 'photon_counts': counts})
    base = fit_beers_law_bma(hist, **FIT)
    if not np.isfinite(base[3]):
        pytest.skip('fixture did not produce a detected floor')
    got = fit_beers_law_bma_daysub(hist, solar_elevation=60.0, **FIT)
    assert np.isfinite(got[0]), 'must fall back to a valid Kd, never NaN'
    if got[4] != base[4]:
        assert got[4] in ('daysub_too_few_bins', 'daysub_nonneg_slope')
        assert got[0] == base[0]
