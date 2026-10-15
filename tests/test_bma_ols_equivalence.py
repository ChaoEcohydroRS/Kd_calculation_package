"""
Snapshot regression test for fit_beers_law_bma + fit_beers_law_hybrid.

The Kd retrieval inner loop currently uses sklearn LinearRegression.
A planned refactor replaces it with np.polyfit for a ~10-20x bootstrap
speedup. Profile (run_2026-05-04_existing_atl03/profiling/profile_task254.out)
shows 94% of LR.fit time is sklearn validation/dispatch overhead, not
actual math; replacing with np.polyfit on numpy arrays is mathematically
equivalent to ULP precision.

This test captures (kd, e0, bp, nf, reason) for hand-built histogram
fixtures against the sklearn-based implementation. After refactor, the
same fixtures must reproduce the snapshots at rtol=1e-10, atol=1e-12.

To (re)generate the snapshot file from scratch:
    python tests/test_bma_ols_equivalence.py --update-snapshots
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kd_utils.Kd_analysis import fit_beers_law_bma, fit_beers_law_hybrid  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — hand-built histograms exercising distinct code paths
# ---------------------------------------------------------------------------

def fx_clean_exp_decay():
    """Clean exponential decay across many bins, no noise floor."""
    z = np.arange(0.25, 14.0, 0.25)  # 55 bins, range 13.5 m
    counts = (np.exp(-0.15 * z) * 400).astype(int)
    return pd.DataFrame({'zdepth': z, 'photon_counts': counts})


def fx_decay_plus_noise():
    """Exp decay → flat noise floor with sharp transition. Breakpoint should win."""
    z = np.arange(0.25, 18.0, 0.25)  # 71 bins, range 17.75 m
    # exp(-0.5*z)*800 hits 6 at z = log(800/6)/0.5 ≈ 9.78 m → bp lives inside window
    decay = np.exp(-0.5 * z) * 800
    counts = np.maximum(decay, 6.0).astype(int)
    return pd.DataFrame({'zdepth': z, 'photon_counts': counts})


def fx_n_equals_6():
    """Minimum n. Span exactly 5 m so passes total-range gate; no bp candidates."""
    z = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])  # 6 bins, range 5.0 m
    counts = np.array([200, 130, 85, 55, 36, 24])
    return pd.DataFrame({'zdepth': z, 'photon_counts': counts})


def fx_one_bp_candidate():
    """Sharp decay→noise transition; narrow window limits surviving bp idxs."""
    z = np.arange(0.25, 7.0, 0.25)  # 27 bins, range 6.75 m
    # exp(-0.8*z)*600 hits 4 at z ≈ 6.27 m → only late bp idxs win
    decay = np.exp(-0.8 * z) * 600
    counts = np.maximum(decay, 4.0).astype(int)
    return pd.DataFrame({'zdepth': z, 'photon_counts': counts})


def fx_many_bp_candidates():
    """Slow decay + soft noise floor → many bp candidates all carry BMA weight."""
    z = np.arange(0.25, 20.0, 0.25)  # 79 bins, range 19.75 m
    # Slower decay extends into window; multiple bp_idx values produce comparable BIC
    decay = np.exp(-0.3 * z) * 700
    counts = np.maximum(decay, 8.0).astype(int)
    return pd.DataFrame({'zdepth': z, 'photon_counts': counts})


def fx_insufficient_photons():
    """Total photons below min_total_photons=30 → reason='insufficient_photons'."""
    z = np.arange(0.5, 6.0, 0.5)  # 11 bins, range 5.0 m
    counts = np.array([2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    return pd.DataFrame({'zdepth': z, 'photon_counts': counts})


def fx_too_few_bins():
    """n < MIN_DEPTH_BINS_FOR_FIT (=6) → reason='too_few_bins'."""
    z = np.array([0.5, 1.5, 2.5, 3.5])
    counts = np.array([100, 80, 65, 50])
    return pd.DataFrame({'zdepth': z, 'photon_counts': counts})


def fx_increasing_counts():
    """Counts increase with depth → all slopes positive → 'no_valid_candidates'."""
    z = np.arange(0.25, 8.0, 0.25)  # 31 bins, range 7.75 m
    counts = (np.arange(len(z)) * 12 + 60).astype(int)
    return pd.DataFrame({'zdepth': z, 'photon_counts': counts})


def fx_with_nan_counts():
    """Mixed valid + NaN counts. NaN bins get filtered by photon_counts>0."""
    z = np.arange(0.25, 12.0, 0.25)  # 47 bins, range 11.75 m
    counts = (np.exp(-0.15 * z) * 350).astype(float)
    counts[3] = np.nan
    counts[10] = np.nan
    counts[20] = np.nan
    return pd.DataFrame({'zdepth': z, 'photon_counts': counts})


FIXTURES = {
    'clean_exp_decay': fx_clean_exp_decay,
    'decay_plus_noise': fx_decay_plus_noise,
    'n_equals_6': fx_n_equals_6,
    'one_bp_candidate': fx_one_bp_candidate,
    'many_bp_candidates': fx_many_bp_candidates,
    'insufficient_photons': fx_insufficient_photons,
    'too_few_bins': fx_too_few_bins,
    'increasing_counts': fx_increasing_counts,
    'with_nan_counts': fx_with_nan_counts,
}

FIT_FUNCTIONS = {
    'bma': fit_beers_law_bma,
    'hybrid': fit_beers_law_hybrid,
}


# ---------------------------------------------------------------------------
# Snapshot persistence
# ---------------------------------------------------------------------------

SNAPSHOTS_PATH = Path(__file__).parent / 'fit_snapshots.json'


def _result_to_json(result):
    """fit_*_bma/hybrid return (kd, e0, bp, nf, reason). NaN → null."""
    def _f(x):
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        return float(x)

    kd, e0, bp, nf, reason = result
    return {'kd': _f(kd), 'e0': _f(e0), 'bp': _f(bp), 'nf': _f(nf),
            'reason': reason}


def _result_from_json(obj):
    def _f(x):
        return np.nan if x is None else float(x)
    return (_f(obj['kd']), _f(obj['e0']), _f(obj['bp']), _f(obj['nf']),
            obj['reason'])


def _load_snapshots():
    if not SNAPSHOTS_PATH.exists():
        return {}
    with SNAPSHOTS_PATH.open() as f:
        return json.load(f)


def _save_snapshots(snapshots):
    with SNAPSHOTS_PATH.open('w') as f:
        json.dump(snapshots, f, indent=2, sort_keys=True)


def _key(fit_name, fx_name):
    return f"{fit_name}__{fx_name}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fit_name", sorted(FIT_FUNCTIONS.keys()))
@pytest.mark.parametrize("fx_name", sorted(FIXTURES.keys()))
def test_fit_matches_snapshot(fit_name, fx_name):
    """Refactor must produce same (kd, e0, bp, nf, reason) at rtol=1e-10."""
    snapshots = _load_snapshots()
    key = _key(fit_name, fx_name)
    if key not in snapshots:
        pytest.fail(
            f"No snapshot for {key!r}. Run "
            "`python tests/test_bma_ols_equivalence.py --update-snapshots` "
            "to generate."
        )

    expected = _result_from_json(snapshots[key])
    actual = FIT_FUNCTIONS[fit_name](FIXTURES[fx_name]())

    # Numeric quad: kd, e0, bp, nf
    for i, label in enumerate(('kd', 'e0', 'bp', 'nf')):
        e, a = expected[i], actual[i]
        e_nan = isinstance(e, float) and np.isnan(e)
        a_nan = isinstance(a, float) and np.isnan(a)
        if e_nan or a_nan:
            assert e_nan and a_nan, (
                f"{key}: {label} NaN mismatch — expected {e!r}, got {a!r}"
            )
        else:
            np.testing.assert_allclose(
                a, e, rtol=1e-10, atol=1e-12,
                err_msg=f"{key}: {label}",
            )

    # Reason string
    assert actual[4] == expected[4], (
        f"{key}: reason mismatch — expected {expected[4]!r}, got {actual[4]!r}"
    )


# ---------------------------------------------------------------------------
# Snapshot generation
# ---------------------------------------------------------------------------

def update_snapshots():
    snapshots = {}
    for fit_name, fit_func in sorted(FIT_FUNCTIONS.items()):
        for fx_name, fx_func in sorted(FIXTURES.items()):
            key = _key(fit_name, fx_name)
            result = fit_func(fx_func())
            snapshots[key] = _result_to_json(result)
            print(f"  {key}: kd={result[0]!r}, e0={result[1]!r}, "
                  f"bp={result[2]!r}, nf={result[3]!r}, "
                  f"reason={result[4]!r}")
    _save_snapshots(snapshots)
    print(f"\nWrote {len(snapshots)} snapshots to {SNAPSHOTS_PATH}")


if __name__ == '__main__':
    if '--update-snapshots' in sys.argv:
        update_snapshots()
    else:
        sys.stderr.write(
            "Usage: python tests/test_bma_ols_equivalence.py --update-snapshots\n"
            "Or run via pytest: pytest tests/test_bma_ols_equivalence.py\n"
        )
        sys.exit(1)
