"""
Snapshot regression test for bootstrap_kd_for_bin + bootstrap_kd_uncertainty.

The OLS refactor swaps sklearn LinearRegression for np.polyfit inside
fit_beers_law_bma / fit_beers_law_hybrid (called from the bootstrap inner
loop). Per profile, sklearn overhead is 94% of LR.fit time on tiny inputs,
not the actual math; the math itself is bit-identical for well-conditioned
1-D fits via polyfit.

Because the bootstrap residual resampling (rng.choice at line 1463) runs
BEFORE the fit and takes a fixed seed, the same seed produces the same
residual indices regardless of how the fit is implemented downstream.
The boot_kds list will differ from the sklearn baseline only at ULP level.

This test fixes the RNG seed and snapshots the full dict output of
bootstrap_kd_for_bin (and bootstrap_kd_uncertainty for the multi-bin
driver). Refactor must preserve every numeric field at rtol=1e-9.

To regenerate snapshots:
    python tests/test_bootstrap_rng_equivalence.py --update-snapshots
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kd_utils.Kd_analysis import (  # noqa: E402
    bootstrap_kd_for_bin,
    bootstrap_kd_uncertainty,
)


# ---------------------------------------------------------------------------
# Photon-height fixtures — synthetic photon clouds with deterministic seeding
# ---------------------------------------------------------------------------

def _photons_from_depth_pdf(kd, e0, noise_floor, max_depth, n_photons,
                             surface_z=0.0, fixture_seed=12345):
    """
    Synthesize photon heights from a target Beer's-law depth distribution.

    Returns array of photon_height values (negative below surface_z).
    Uses inverse-CDF sampling so the same fixture_seed produces the same
    photon set every run.
    """
    rng = np.random.default_rng(fixture_seed)
    # Decay PDF proportional to e0 * exp(-kd * z) on [0, z_break]
    # plus a constant noise pad on [z_break, max_depth]
    z_break = -np.log(noise_floor / e0) / kd
    z_break = min(z_break, max_depth)

    # Sample fraction in decay vs. noise via integrated mass
    decay_mass = (e0 / kd) * (1 - np.exp(-kd * z_break))
    noise_mass = noise_floor * (max_depth - z_break)
    p_decay = decay_mass / (decay_mass + noise_mass)

    n_decay = int(round(n_photons * p_decay))
    n_noise = n_photons - n_decay

    # Inverse-CDF for truncated exponential on [0, z_break]
    u = rng.uniform(size=n_decay)
    z_decay = -np.log(1 - u * (1 - np.exp(-kd * z_break))) / kd

    # Uniform on [z_break, max_depth] for noise
    z_noise = rng.uniform(z_break, max_depth, size=n_noise)

    z_all = np.concatenate([z_decay, z_noise])
    rng.shuffle(z_all)
    return surface_z - z_all  # photon_height = below surface


def fx_strong_decay_with_floor():
    """Sharp decay, clear noise floor — bootstrap should converge."""
    return _photons_from_depth_pdf(
        kd=0.5, e0=800.0, noise_floor=6.0, max_depth=18.0,
        n_photons=12000, fixture_seed=12345,
    )


def fx_full_line_fallback():
    """Slow decay, no clear floor — full-line model dominates."""
    return _photons_from_depth_pdf(
        kd=0.15, e0=400.0, noise_floor=2.0, max_depth=14.0,
        n_photons=8000, fixture_seed=12346,
    )


def fx_low_signal():
    """Few photons — bootstrap success rate may dip below 30%."""
    return _photons_from_depth_pdf(
        kd=0.3, e0=200.0, noise_floor=4.0, max_depth=10.0,
        n_photons=180, fixture_seed=12347,
    )


def fx_empty():
    """Empty photon array — must hit nan_result path."""
    return np.array([], dtype=float)


SINGLE_BIN_FIXTURES = {
    'strong_decay_with_floor': fx_strong_decay_with_floor,
    'full_line_fallback': fx_full_line_fallback,
    'low_signal': fx_low_signal,
    'empty': fx_empty,
}

FIT_METHODS = ('bma', 'hybrid')


# ---------------------------------------------------------------------------
# Multi-bin fixture — exercises bootstrap_kd_uncertainty driver
# ---------------------------------------------------------------------------

def fx_multi_bin_df():
    """Three lat_bins with distinct photon profiles → exercises the driver loop."""
    parts = []
    for lat_bin, fixture_seed, kd, e0, nf, n in [
        (0, 22001, 0.4, 600.0, 5.0, 6000),
        (1, 22002, 0.25, 400.0, 4.0, 5000),
        (2, 22003, 0.6, 800.0, 8.0, 4000),
    ]:
        ph = _photons_from_depth_pdf(
            kd=kd, e0=e0, noise_floor=nf, max_depth=15.0,
            n_photons=n, fixture_seed=fixture_seed,
        )
        parts.append(pd.DataFrame({'lat_bins': lat_bin, 'photon_height': ph}))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Snapshot persistence
# ---------------------------------------------------------------------------

SNAPSHOTS_PATH = Path(__file__).parent / 'bootstrap_snapshots.json'


def _to_jsonable(d):
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
        elif isinstance(v, float) and np.isnan(v):
            out[k] = None
        elif isinstance(v, (np.floating, float)):
            out[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _load_snapshots():
    if not SNAPSHOTS_PATH.exists():
        return {}
    with SNAPSHOTS_PATH.open() as f:
        return json.load(f)


def _save_snapshots(snapshots):
    with SNAPSHOTS_PATH.open('w') as f:
        json.dump(snapshots, f, indent=2, sort_keys=True)


def _key_single(method, fx_name):
    return f"single__{method}__{fx_name}"


def _key_multi(method):
    return f"multi__{method}"


def _assert_dict_close(actual, expected, key, rtol=1e-9, atol=1e-12):
    """Compare result dicts; numeric fields rtol-close, NaN matches NaN."""
    assert set(actual.keys()) == set(expected.keys()), (
        f"{key}: keys differ — actual={sorted(actual)}, expected={sorted(expected)}"
    )
    for k, e in expected.items():
        a = actual[k]
        if e is None:
            assert a is None or (isinstance(a, float) and np.isnan(a)), (
                f"{key}: {k} expected None/NaN, got {a!r}"
            )
        elif isinstance(a, float) and np.isnan(a):
            assert e is None, f"{key}: {k} actual NaN, expected {e!r}"
        elif isinstance(e, (int, float)):
            np.testing.assert_allclose(
                float(a), float(e), rtol=rtol, atol=atol,
                err_msg=f"{key}: {k}",
            )
        else:
            assert a == e, f"{key}: {k} expected {e!r}, got {a!r}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", FIT_METHODS)
@pytest.mark.parametrize("fx_name", sorted(SINGLE_BIN_FIXTURES.keys()))
def test_bootstrap_for_bin_matches_snapshot(method, fx_name):
    """bootstrap_kd_for_bin must reproduce all dict fields at rtol=1e-9."""
    snapshots = _load_snapshots()
    key = _key_single(method, fx_name)
    if key not in snapshots:
        pytest.fail(
            f"No snapshot for {key!r}. Run "
            "`python tests/test_bootstrap_rng_equivalence.py --update-snapshots`"
        )

    photon_heights = SINGLE_BIN_FIXTURES[fx_name]()
    rng = np.random.default_rng(42)
    actual = bootstrap_kd_for_bin(
        photon_heights, n_boot=300, vertical_res=0.25,
        rng=rng, fit_method=method,
    )
    _assert_dict_close(_to_jsonable(actual), snapshots[key], key)


@pytest.mark.parametrize("method", FIT_METHODS)
def test_bootstrap_uncertainty_matches_snapshot(method):
    """bootstrap_kd_uncertainty (multi-bin driver) must reproduce DataFrame at rtol=1e-9."""
    snapshots = _load_snapshots()
    key = _key_multi(method)
    if key not in snapshots:
        pytest.fail(
            f"No snapshot for {key!r}. Run "
            "`python tests/test_bootstrap_rng_equivalence.py --update-snapshots`"
        )

    df = fx_multi_bin_df()
    out = bootstrap_kd_uncertainty(
        df, n_boot=200, vertical_res=0.25, random_seed=42,
        fit_method=method,
    )

    expected_rows = snapshots[key]
    assert len(out) == len(expected_rows), (
        f"{key}: row count mismatch — actual={len(out)}, expected={len(expected_rows)}"
    )
    for i, expected in enumerate(expected_rows):
        actual_row = _to_jsonable(out.iloc[i].to_dict())
        _assert_dict_close(actual_row, expected, f"{key}[row {i}]")


# ---------------------------------------------------------------------------
# Snapshot generation
# ---------------------------------------------------------------------------

def update_snapshots():
    snapshots = {}
    for method in FIT_METHODS:
        for fx_name, fx_func in sorted(SINGLE_BIN_FIXTURES.items()):
            key = _key_single(method, fx_name)
            rng = np.random.default_rng(42)
            result = bootstrap_kd_for_bin(
                fx_func(), n_boot=300, vertical_res=0.25,
                rng=rng, fit_method=method,
            )
            snapshots[key] = _to_jsonable(result)
            print(f"  {key}: {snapshots[key]}")

        # Multi-bin driver
        key = _key_multi(method)
        df = fx_multi_bin_df()
        out = bootstrap_kd_uncertainty(
            df, n_boot=200, vertical_res=0.25, random_seed=42,
            fit_method=method,
        )
        snapshots[key] = [_to_jsonable(out.iloc[i].to_dict())
                          for i in range(len(out))]
        print(f"  {key}: {len(out)} rows")

    _save_snapshots(snapshots)
    print(f"\nWrote {len(snapshots)} snapshot entries to {SNAPSHOTS_PATH}")


if __name__ == '__main__':
    if '--update-snapshots' in sys.argv:
        update_snapshots()
    else:
        sys.stderr.write(
            "Usage: python tests/test_bootstrap_rng_equivalence.py --update-snapshots\n"
            "Or run via pytest: pytest tests/test_bootstrap_rng_equivalence.py\n"
        )
        sys.exit(1)
