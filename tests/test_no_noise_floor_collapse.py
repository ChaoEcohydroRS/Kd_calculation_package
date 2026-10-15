"""
Tests for the post-GEBCO / no-noise-floor handling added to
``fit_beers_law_bma`` (todo item 9).

When the GEBCO/ATL24 seafloor filter strips the seabed return from a
near-shore bin, the resulting histogram is pure exponential decay with no
flat tail. Pre-fix, the BMA fitter either rejected such bins outright
(``insufficient_range`` for column < 5 m) or returned spurious garbage in
the ``breakpoint`` and ``noise_floor`` fields (the BMA-weighted average
absorbed candidates that placed bp at the deepest 1-2 bins, mistaking
real signal counts for noise).

Post-fix:
  * Range gate is decoupled — ``min_total_range_m_floor=2.5`` is the hard
    rejection floor; ``min_total_range_m=5.0`` controls whether piecewise
    candidates are even considered (below it, only the no-breakpoint
    single-line baseline runs).
  * A spurious-tail filter rejects piecewise candidates whose noise mean
    lies on the decay's extrapolated trajectory (within 2 σ of the decay
    residual std). For pure-decay data, this filter eliminates almost all
    chance-survivors; what's left is then collapsed by the BMA-weight
    detector.
  * If the no-breakpoint candidate carries > 95 % of the BMA weight, the
    fitter returns ``(kd, e0, NaN, NaN, 'no_noise_floor')`` — explicitly
    signalling that the data is pure decay rather than mis-reporting
    breakpoint/noise-floor values that are not statistically supported.

Run:
    pytest tests/test_no_noise_floor_collapse.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kd_utils.Kd_analysis import fit_beers_law_bma  # noqa: E402
from kd_utils.quality_flags import _flag_fit_no_breakpoint  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
TRUE_KD = 0.4
TRUE_E0 = 1000.0
TRUE_NOISE = 5.0
TRUE_BP = 6.0
VERTICAL_RES = 0.25  # matches production


def make_pure_decay(col_depth_m: float, seed: int = 0,
                    kd: float = TRUE_KD, e0: float = TRUE_E0):
    """Pure exponential decay over [0, col_depth_m] — no noise floor.

    Mirrors the post-GEBCO/post-ATL24 near-shore regime where the seabed
    return has been stripped. Zero-count bins are dropped (this matches
    how ``fit_beers_law_bma`` itself preprocesses the histogram), so the
    histogram never carries an artificial floor at log(1).
    """
    rng = np.random.default_rng(seed)
    z_full = np.arange(VERTICAL_RES / 2, col_depth_m + VERTICAL_RES / 2, VERTICAL_RES)
    counts = rng.poisson(e0 * np.exp(-kd * z_full)).astype(float)
    mask = counts > 0
    return pd.DataFrame({'zdepth': z_full[mask], 'photon_counts': counts[mask]})


def make_flat_floor(seed: int = 0):
    """Decay then flat noise floor — must NOT collapse."""
    rng = np.random.default_rng(seed)
    z = np.arange(VERTICAL_RES / 2, 12.125, VERTICAL_RES)
    expected = np.where(z < TRUE_BP,
                        TRUE_E0 * np.exp(-TRUE_KD * z) + TRUE_NOISE,
                        TRUE_NOISE)
    counts = np.maximum(rng.poisson(expected).astype(float), 1.0)
    return pd.DataFrame({'zdepth': z, 'photon_counts': counts})


def make_partial_floor(seed: int = 0, n_clipped: int = 2):
    """Flat-floor data with the deepest n bins removed (e.g., GEBCO clipping
    a few bins of an otherwise-real noise tail). Should NOT collapse — the
    surviving floor segment is still recognisable."""
    df = make_flat_floor(seed=seed)
    return df.iloc[:-n_clipped].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pure-decay recovery
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('col_depth_m', [3.0, 5.0, 6.0, 8.0])
def test_pure_decay_collapses_to_no_noise_floor(col_depth_m):
    """Pure-decay histograms across a range of column depths should collapse
    to 'no_noise_floor', with Kd preserved within ~15 % of truth across
    Poisson seeds.
    """
    n_seeds = 10
    collapsed = 0
    kd_recovered = []
    for seed in range(n_seeds):
        df = make_pure_decay(col_depth_m, seed=seed)
        if len(df) < 6:
            continue  # too_few_bins; not the case under test
        kd, e0, bp, nf, reason = fit_beers_law_bma(df)
        if reason == 'no_noise_floor':
            collapsed += 1
            assert np.isnan(bp), f'seed {seed}: bp should be NaN but is {bp}'
            assert np.isnan(nf), f'seed {seed}: nf should be NaN but is {nf}'
            assert kd > 0, f'seed {seed}: kd should be positive but is {kd}'
            kd_recovered.append(kd)
    # Vast majority should collapse
    assert collapsed >= n_seeds - 1, (
        f'col={col_depth_m} m: only {collapsed}/{n_seeds} collapsed; '
        f'expect ≥ {n_seeds - 1}'
    )
    # Where Kd was recovered, it should be close to truth
    if kd_recovered:
        median_kd = float(np.median(kd_recovered))
        assert abs(median_kd - TRUE_KD) / TRUE_KD < 0.15, (
            f'col={col_depth_m} m: median recovered kd {median_kd:.3f} '
            f'differs from truth {TRUE_KD} by > 15 %'
        )


def test_too_shallow_rejected_as_insufficient_range():
    """Column shorter than the 2.5 m hard floor must reject as
    'insufficient_range'."""
    df = make_pure_decay(col_depth_m=2.0, seed=0)
    if len(df) < 6:
        pytest.skip("too_few_bins, not the case under test")
    _, _, _, _, reason = fit_beers_law_bma(df)
    assert reason == 'insufficient_range', (
        f"expected 'insufficient_range', got {reason!r}"
    )


# ---------------------------------------------------------------------------
# Floor-present must NOT regress
# ---------------------------------------------------------------------------
def test_flat_floor_must_not_collapse():
    """Real flat-floor data must continue to recover bp and nf — this is
    the must-not-break-existing-behaviour guard."""
    n_seeds = 10
    for seed in range(n_seeds):
        df = make_flat_floor(seed=seed)
        kd, e0, bp, nf, reason = fit_beers_law_bma(df)
        assert reason is None, (
            f'seed {seed}: flat-floor data should not collapse; reason={reason!r}'
        )
        assert not np.isnan(bp), f'seed {seed}: bp should be finite'
        assert not np.isnan(nf), f'seed {seed}: nf should be finite'
        # bp should be within 1.5 m of truth (BMA averages neighbouring
        # candidates so it can drift from the exact truth value)
        assert abs(bp - TRUE_BP) < 1.5, (
            f'seed {seed}: bp {bp:.2f} more than 1.5 m off truth {TRUE_BP}'
        )
        assert abs(kd - TRUE_KD) / TRUE_KD < 0.15, (
            f'seed {seed}: kd {kd:.3f} more than 15 % off truth {TRUE_KD}'
        )


def test_partial_floor_must_not_collapse():
    """Flat-floor data with the deepest 2 bins clipped should still find a
    bp — the noise segment becomes shorter but is still distinguishable
    from the decay extrapolation."""
    for seed in range(5):
        df = make_partial_floor(seed=seed, n_clipped=2)
        kd, e0, bp, nf, reason = fit_beers_law_bma(df)
        assert reason != 'no_noise_floor', (
            f'seed {seed}: partial-floor data should not collapse '
            f'(reason={reason!r}, kd={kd}, bp={bp})'
        )


# ---------------------------------------------------------------------------
# Quality-flag wiring
# ---------------------------------------------------------------------------
def test_no_noise_floor_routes_to_bit16():
    """`compute_quality_flags` should treat 'no_noise_floor' the same as
    'no_breakpoint_fallback' / 'no_breakpoint_found' — bit 16."""
    kd_df = pd.DataFrame({
        'lat_bins': [0.0, 1.0, 2.0, 3.0],
        'fit_failure_reason': [None, 'no_breakpoint_fallback',
                               'no_noise_floor', 'no_breakpoint_found'],
    })
    flags = _flag_fit_no_breakpoint(kd_df)
    assert flags == {0.0: False, 1.0: True, 2.0: True, 3.0: True}, flags
