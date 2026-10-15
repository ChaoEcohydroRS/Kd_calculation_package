"""Tests for coherent_model_selection (region-level decay-vs-flat selection)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kd_utils.Kd_analysis import coherent_model_selection  # noqa: E402


def _height_bins(depth_counts):
    """Expand {depth(+m): count} into a list of photon_height bins (negative)."""
    out = []
    for d, c in depth_counts.items():
        out += [-round(d, 2)] * int(c)
    return out


def _decay_plus_flat(kd=0.35, e0=60, floor=3, dmax=15.0, step=0.5):
    """Two-segment: steep decay to a flat noise tail -> two-segment preferred."""
    dc = {}
    for d in np.arange(step, dmax + step, step):
        sig = e0 * np.exp(-kd * d)
        dc[d] = max(int(round(sig)), 0) + floor
    return dc


def _pure_decay(kd=0.12, e0=45, dmax=15.0, step=0.5):
    """Gentle pure decay, no flat tail -> single-line preferred / no breakpoint."""
    return {d: max(int(round(e0 * np.exp(-kd * d))), 1) for d in np.arange(step, dmax + step, step)}


def _build(n_bp=10, n_coll=10):
    sub_rows, kd_rows = [], []
    for i in range(n_bp + n_coll):
        dc = _decay_plus_flat() if i < n_bp else _pure_decay()
        for hb in _height_bins(dc):
            sub_rows.append({'beam_id': 'gt1l', 'lat_bins': i, 'height_bins': hb})
        kd_rows.append({'beam_id': 'gt1l', 'lat_bins': i, 'kd': 0.2,
                        'kd_std': np.nan, 'kd_ci_low': np.nan, 'kd_ci_high': np.nan,
                        'surface_fit_quality': 0.99,
                        'fit_failure_reason': 'no_noise_floor'})  # mark unstable so coherent acts
    return pd.DataFrame(kd_rows), pd.DataFrame(sub_rows)


def test_columns_added_and_runs():
    kd, sub = _build()
    out = coherent_model_selection(kd, sub, window=5)
    for c in ['kd_single', 'kd_2seg', 'has_breakpoint', 'coherent_2seg',
              'kd_coherent', 'kd_coherent_lo', 'kd_coherent_hi', 'coherent_regularized']:
        assert c in out.columns, c
    assert len(out) == len(kd)


def test_region_commits_to_majority_model():
    kd, sub = _build(n_bp=10, n_coll=10)
    out = coherent_model_selection(kd, sub, window=5).sort_values('lat_bins').reset_index(drop=True)
    bp_region = out[out.lat_bins < 10]      # decay+flat -> should pick two-segment
    coll_region = out[out.lat_bins >= 10]   # pure decay -> should pick single-line
    # majority of the breakpoint region commits to two-segment
    assert bp_region['coherent_2seg'].mean() > 0.5
    # collapse region does not
    assert coll_region['coherent_2seg'].mean() < 0.5
    # and the coherent Kd is higher in the two-segment region than the collapse region
    assert bp_region['kd_coherent'].median() > coll_region['kd_coherent'].median()


def test_coherent_curve_is_smooth_in_region():
    kd, sub = _build()
    out = coherent_model_selection(kd, sub, window=5).sort_values('lat_bins')
    reg = out[out['coherent_regularized']]
    if len(reg) > 3:
        assert np.median(np.abs(np.diff(reg['kd_coherent'].to_numpy()))) < 0.05


def _build_stable_2seg(n=16):
    """A run of individually-STABLE two-segment bins (finite kd_std, no failure
    reason) whose two-segment Kd zig-zags bin-to-bin — the WaxDelta turbidity
    ramp case. The region commits to two-segment but no bin is flagged unstable,
    so only force_smooth_mask makes coherent smooth it."""
    sub_rows, kd_rows = [], []
    for i in range(n):
        kd_true = 0.30 if i % 2 == 0 else 0.42  # jitter the decay rate
        for hb in _height_bins(_decay_plus_flat(kd=kd_true)):
            sub_rows.append({'beam_id': 'gt1l', 'lat_bins': i, 'height_bins': hb})
        kd_rows.append({'beam_id': 'gt1l', 'lat_bins': i, 'kd': kd_true,
                        'kd_std': 0.02, 'kd_ci_low': kd_true - 0.05,
                        'kd_ci_high': kd_true + 0.05, 'surface_fit_quality': 0.99,
                        'fit_failure_reason': None})
    return pd.DataFrame(kd_rows), pd.DataFrame(sub_rows)


def test_committed_2seg_stable_region_is_smoothed_and_banded():
    kd, sub = _build_stable_2seg(16)
    out = coherent_model_selection(kd, sub, window=7,
                                   central_stat='mean').sort_values('lat_bins').reset_index(drop=True)
    # region commits to two-segment
    assert out['coherent_2seg'].mean() > 0.5
    # even though every bin is individually stable, the committed region smooths
    assert out['coherent_regularized'].sum() >= 8
    # the bin-to-bin zig-zag in the selected two-segment Kd is reduced
    raw_rough = np.median(np.abs(np.diff(out['kd_2seg'].to_numpy())))
    sm_rough = np.median(np.abs(np.diff(out['kd_coherent'].to_numpy())))
    assert sm_rough < raw_rough
    # and the band is non-degenerate across the region (no zero-width collapse)
    assert (out['kd_coherent_hi'] - out['kd_coherent_lo']).median() > 0
