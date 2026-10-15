"""Tests for sea-surface flattening.

The headline test is the one that would have caught the inverted sign shipped in
the original implementation: flattening must make photon depths CONSISTENT with a
common datum. Getting the sign backwards still "does something" to every photon
and still produces plausible Kd values, so only a test that checks the direction
of the correction catches it.
"""
import numpy as np
import pandas as pd
import pytest

from kd_utils.bathy_processing import apply_sea_surface_flattening

UNDULATION_M = 0.5          # amplitude of the synthetic sea-surface wave
N_BINS, PER_BIN = 20, 200


def _synthetic(seed=0):
    """Photons at a KNOWN depth below an undulating surface.

    Returns (photon frame, per-bin surface heights, true depth below the local
    surface). photon_height is an elevation, so depth = surface - photon_height.
    """
    rng = np.random.default_rng(seed)
    lat_bins, ph, truth, ssh = [], [], [], []
    for b in range(N_BINS):
        z_local = UNDULATION_M * np.sin(2 * np.pi * b / 10)
        d = rng.uniform(2, 10, PER_BIN)
        lat_bins += [b] * PER_BIN
        ph += list(z_local - d)
        truth += list(d)
        ssh.append(z_local)
    df = pd.DataFrame({"lat_bins": lat_bins, "photon_height": ph})
    return df, ssh, np.asarray(truth)


def _depth_error_std(photon_height, ssh, truth):
    """Spread of (depth measured against the window datum) - (true local depth)."""
    z_window = float(np.mean(ssh))
    return float(((z_window - np.asarray(photon_height)) - truth).std())


def test_flattening_removes_the_undulation_not_doubles_it():
    """The regression test for the sign.

    Before flattening the depth error is the undulation itself. After a correct
    flattening it must collapse toward zero; an inverted sign doubles it instead,
    which is exactly what the original implementation did.
    """
    df, ssh, truth = _synthetic()
    before = _depth_error_std(df.photon_height.values, ssh, truth)
    out = apply_sea_surface_flattening(df, ssh, list(range(N_BINS)),
                                       horizontal_res=500, flattening_window_m=10_000)
    after = _depth_error_std(out.photon_height.values, ssh, truth)

    assert before == pytest.approx(np.std(ssh), rel=0.05)   # sanity: error == undulation
    assert after < 1e-9, f"depth error should vanish, got {after:.4f} m"
    assert after < before / 10


def test_offset_sign_matches_the_definition():
    """offset = window mean - local surface, and it is ADDED to photon_height."""
    df, ssh, _ = _synthetic()
    out = apply_sea_surface_flattening(df, ssh, list(range(N_BINS)),
                                       horizontal_res=500, flattening_window_m=10_000)
    z_window = float(np.mean(ssh))
    for b in (0, 3, 7):
        rows = out[out.lat_bins == b]
        expected = z_window - ssh[b]
        assert rows["sea_surface_flattening_offset"].iloc[0] == pytest.approx(expected)
        shift = (rows["photon_height"] - rows["photon_height_original"]).iloc[0]
        assert shift == pytest.approx(expected)


def test_window_narrower_than_bin_is_a_noop():
    """A window at or below the bin size gives every bin its own mean -> no shift."""
    df, ssh, _ = _synthetic()
    out = apply_sea_surface_flattening(df, ssh, list(range(N_BINS)),
                                       horizontal_res=500, flattening_window_m=500)
    assert np.allclose(out.photon_height, df.photon_height)


def test_height_bins_are_recut_so_the_fit_sees_the_shift():
    """The Beer's-law fitter reads height_bins, not photon_height."""
    df, ssh, _ = _synthetic()
    df["height_bins"] = np.round(df.photon_height * 4) / 4     # 0.25 m grid
    out = apply_sea_surface_flattening(df, ssh, list(range(N_BINS)),
                                       horizontal_res=500, flattening_window_m=10_000)
    hb = pd.to_numeric(out["height_bins"], errors="coerce")
    assert (hb - out["photon_height"]).abs().median() < 0.25
    assert (hb - out["photon_height_original"]).abs().median() > \
           (hb - out["photon_height"]).abs().median()


def test_empty_and_degenerate_inputs_pass_through():
    empty = pd.DataFrame(columns=["lat_bins", "photon_height"])
    assert apply_sea_surface_flattening(empty, [], [], 500).empty
    df, ssh, _ = _synthetic()
    # mismatched surface-height list -> return unchanged rather than misalign
    out = apply_sea_surface_flattening(df, ssh[:-1], list(range(N_BINS)), 500)
    assert np.allclose(out.photon_height, df.photon_height)
