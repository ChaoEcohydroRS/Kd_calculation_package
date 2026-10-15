"""Unit tests for the ATL09 backg_c -> per-histogram-bin formula.

Reference: SolarBackground/Solarbackground.pdf, final message (Emily Eidam, 5/6/2026)
    bkg_hist = [bin_z * (backg_c / 30 m)] x [bin_x / 280 m]
"""
import numpy as np
import pytest

from kd_utils.SolarBckgrd.atl09_backg_c import (
    ATM_BIN_X_M,
    ATM_BIN_Z_M,
    bkg_per_hist_bin,
)


def test_identity_at_atm_bin_size():
    """With bin_z = ATM_BIN_Z_M and bin_x = ATM_BIN_X_M the scale must be 1."""
    c = np.array([0.0, 1.0, 7.5, 42.0])
    out = bkg_per_hist_bin(c, ATM_BIN_Z_M, ATM_BIN_X_M)
    np.testing.assert_allclose(out, c, rtol=0, atol=1e-12)


def test_round_30m_vertical_alias():
    """Email uses 30 m as the vertical bin. With atm_bin_z_m=30 the scale at 30x280 is 1."""
    c = 10.0
    out = bkg_per_hist_bin(c, 30.0, 280.0, atm_bin_z_m=30.0)
    assert out == pytest.approx(10.0)


def test_vertical_scaling_linear():
    """Halving bin_z halves the per-bin count."""
    c = 8.0
    full = bkg_per_hist_bin(c, ATM_BIN_Z_M, ATM_BIN_X_M)
    half = bkg_per_hist_bin(c, ATM_BIN_Z_M / 2.0, ATM_BIN_X_M)
    assert half == pytest.approx(full / 2.0)


def test_horizontal_scaling_linear():
    """Doubling bin_x doubles the per-bin count."""
    c = 8.0
    full = bkg_per_hist_bin(c, ATM_BIN_Z_M, ATM_BIN_X_M)
    doubled = bkg_per_hist_bin(c, ATM_BIN_Z_M, ATM_BIN_X_M * 2.0)
    assert doubled == pytest.approx(full * 2.0)


def test_pipeline_typical_bin():
    """Repo default fit bin is ~0.25 m (vertical) x 500 m (along-track)."""
    c = 12.0
    out = bkg_per_hist_bin(c, 0.25, 500.0)
    expected = 12.0 * (0.25 / ATM_BIN_Z_M) * (500.0 / ATM_BIN_X_M)
    assert out == pytest.approx(expected)
    assert 0.05 < out < 0.5, f"sanity: typical out ~ 0.1 photons/bin, got {out}"


def test_vector_input_preserves_shape():
    c = np.linspace(0.0, 30.0, 17)
    out = bkg_per_hist_bin(c, 0.5, 250.0)
    assert out.shape == c.shape
    np.testing.assert_allclose(
        out, c * (0.5 / ATM_BIN_Z_M) * (250.0 / ATM_BIN_X_M), rtol=1e-12
    )


def test_zero_backg_c_is_zero():
    assert bkg_per_hist_bin(0.0, 0.25, 500.0) == 0.0


def test_negative_inputs_pass_through():
    """The formula is linear; if the file ever has fill-value sentinels they should not error."""
    c = np.array([-1.0, 0.0, 5.0])
    out = bkg_per_hist_bin(c, 0.25, 500.0)
    assert out.shape == c.shape
    assert out[0] < 0  # negative passes through linearly
