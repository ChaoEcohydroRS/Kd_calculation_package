"""Tests for the post-refraction re-binning of the Beer's-law depth axis.

Guards the v5 fix: height_bins must be re-cut on refraction-corrected
photon_height so the fitter (which reads height_bins, not photon_height)
works in true metres -- while lat_bins and the surface Gaussian stay on the
uncorrected coordinates they are defined in.
"""
import numpy as np
import pandas as pd
import pytest

from kd_utils.bathy_processing import (
    rebin_height_after_refraction,
    rescale_depth_limits_after_refraction,
)

N_WATER = 1.34  # refractive index used to synthesize the corrected depths


def _synthetic_photons(n_per_bin=400, n_lat_bins=3, seed=0):
    """Two-column truth set: apparent depths and their n-scaled corrected twin."""
    rng = np.random.default_rng(seed)
    rows = []
    for lb in range(n_lat_bins):
        surface = 0.5 * lb  # a different surface height per bin
        # apparent depths from 1 m (the clip floor) to 20 m
        dep_apparent = rng.uniform(1.0, 20.0, n_per_bin)
        dep_true = dep_apparent / N_WATER
        rows.append(pd.DataFrame({
            'lat_bins': lb,
            'sea_surface_height': surface,
            'photon_height_pre_refraction': surface - dep_apparent,
            'photon_height': surface - dep_true,
        }))
    df = pd.concat(rows, ignore_index=True)
    # height_bins as step 4 would have left it: cut on the APPARENT heights
    h = df['photon_height_pre_refraction']
    n_bins = max(1, round(abs(h.max() - h.min()) / 0.25))
    df['height_bins'] = pd.cut(
        h, bins=n_bins,
        labels=np.round(np.linspace(h.min(), h.max(), num=n_bins), decimals=1)
    ).astype(float)
    return df


def test_rebin_puts_depth_axis_in_true_metres():
    df = _synthetic_photons()
    before = df['height_bins'].max() - df['height_bins'].min()
    n_before = df['height_bins'].nunique()
    out = rebin_height_after_refraction(df, vertical_res=0.25)
    after = out['height_bins'].max() - out['height_bins'].min()
    n_after = out['height_bins'].nunique()
    # The axis must shrink by the refractive index, not stay put. Tolerance is
    # 0.5 m on the span because labels are rounded to 0.1 m at both ends
    # (inherited from horizontal_vertical_bin_dataset); the bin count is the
    # tighter check.
    assert after == pytest.approx(before / N_WATER, abs=0.5)
    assert n_after == pytest.approx(n_before / N_WATER, rel=0.03)


def test_rebin_tracks_corrected_not_apparent_height():
    df = _synthetic_photons()
    out = rebin_height_after_refraction(df, vertical_res=0.25)
    err_post = (out['height_bins'] - out['photon_height']).abs().median()
    err_pre = (out['height_bins'] - out['photon_height_pre_refraction']).abs().median()
    assert err_post < 0.25          # within one bin of the corrected height
    assert err_post < err_pre / 3   # and decisively closer than to the old axis


def test_rebin_preserves_bin_width_and_lat_bins():
    df = _synthetic_photons()
    out = rebin_height_after_refraction(df, vertical_res=0.25)
    # Mean spacing, not median: labels are rounded to 0.1 m, so a 0.25 m grid
    # alternates 0.2/0.3 between neighbours while averaging to 0.25.
    widths = np.diff(np.sort(out['height_bins'].dropna().unique()))
    assert widths.mean() == pytest.approx(0.25, abs=0.02)
    assert widths.min() >= 0.15 and widths.max() <= 0.35
    # lat_bins must be untouched -- re-keying it silently breaks every
    # per-bin diagnostic merge downstream.
    pd.testing.assert_series_equal(out['lat_bins'], df['lat_bins'])
    assert len(out) == len(df)


def test_rebin_is_a_noop_on_degenerate_input():
    empty = pd.DataFrame(columns=['photon_height', 'height_bins'])
    assert rebin_height_after_refraction(empty).empty
    single = pd.DataFrame({'photon_height': [-3.0], 'height_bins': [-3.0],
                           'photon_height_pre_refraction': [-4.0]})
    assert rebin_height_after_refraction(single)['height_bins'].tolist() == [-3.0]
    missing = pd.DataFrame({'lat_bins': [0, 1]})
    pd.testing.assert_frame_equal(rebin_height_after_refraction(missing), missing)


def test_rebin_skipped_when_refraction_did_not_run():
    """apply_optional_refraction_correction has early returns that hand back the
    frame untouched; re-binning must not fire on those apparent depths."""
    df = _synthetic_photons().drop(columns=['photon_height_pre_refraction'])
    out = rebin_height_after_refraction(df, vertical_res=0.25)
    pd.testing.assert_frame_equal(out, df)


def test_depth_limits_rescale_to_true_metres():
    df = _synthetic_photons()
    limits = {0: 20.0, 1: 15.0, 2: 5.0}
    out = rescale_depth_limits_after_refraction(df, limits)
    for lb, apparent in limits.items():
        assert out[lb] == pytest.approx(apparent / N_WATER, rel=0.02)


def test_depth_limits_unchanged_without_the_pre_refraction_column():
    df = _synthetic_photons().drop(columns=['photon_height_pre_refraction'])
    limits = {0: 20.0}
    assert rescale_depth_limits_after_refraction(df, limits) == limits
    assert rescale_depth_limits_after_refraction(df, {}) == {}
