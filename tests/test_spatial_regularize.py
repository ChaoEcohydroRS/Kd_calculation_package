"""Tests for spatial_regularize_kd (region-adaptive along-track Kd smoothing)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kd_utils.Kd_analysis import spatial_regularize_kd  # noqa: E402


def _make_df(kd, reason=None, kd_std=None, beam='gt1l'):
    n = len(kd)
    d = {'beam_id': [beam] * n, 'lat_bins': list(range(n)), 'kd': kd}
    d['fit_failure_reason'] = reason if reason is not None else [None] * n
    d['kd_std'] = kd_std if kd_std is not None else [0.02] * n
    return pd.DataFrame(d)


def test_stable_region_is_untouched():
    # All bins stable (no reason, finite small kd_std) -> kd_smooth == kd.
    kd = [0.1, 0.5, 0.12, 0.48, 0.11]  # spiky but "stable" by flags
    out = spatial_regularize_kd(_make_df(kd), window=5)
    assert not out['kd_regularized'].any()
    assert np.allclose(out['kd_smooth'], kd)
    assert np.allclose(out['kd_smooth_lo'], kd)


def _sawtooth_df(pattern_kd, pattern_reason, pattern_std, ci_lo, ci_hi, reps):
    n = len(pattern_kd) * reps
    df = _make_df(pattern_kd * reps, reason=pattern_reason * reps, kd_std=pattern_std * reps)
    df['kd_ci_low'] = ci_lo * reps
    df['kd_ci_high'] = ci_hi * reps
    return df


def test_unstable_sawtooth_is_smoothed_mean_on_balanced():
    # Balanced collapse(0.08)/breakpoint(0.25). A median can't centre a 50/50
    # sawtooth (by design) — that's the one case central_stat='mean' is for.
    df = _sawtooth_df([0.08, 0.25], ['no_noise_floor', None], [np.nan, 0.03],
                      [np.nan, 0.10], [np.nan, 0.45], reps=8)
    out = spatial_regularize_kd(df, window=5, central_stat='mean')
    assert out['kd_regularized'].sum() >= 12
    raw_rough = np.median(np.abs(np.diff(df['kd'].to_numpy())))
    sm_rough = np.median(np.abs(np.diff(out['kd_smooth'].to_numpy())))
    assert sm_rough < raw_rough / 3
    assert 0.08 <= out['kd_smooth'].median() <= 0.25
    reg = out[out['kd_regularized']]
    assert (reg['kd_smooth_hi'] - reg['kd_smooth_lo']).median() > 0


def test_median_default_not_pulled_up_by_high_cluster():
    # Realistic collapse-MAJORITY zone: 2 collapse(0.10) : 1 breakpoint(0.25).
    # Default median should track the collapse majority (~0.10), NOT be dragged
    # up toward the breakpoint cluster the way a mean is.
    df = _sawtooth_df([0.10, 0.10, 0.25], ['no_noise_floor', 'no_noise_floor', None],
                      [np.nan, np.nan, 0.03], [np.nan, np.nan, 0.10],
                      [np.nan, np.nan, 0.45], reps=8)
    med = spatial_regularize_kd(df, window=9, central_stat='median')
    mean = spatial_regularize_kd(df, window=9, central_stat='mean')
    reg_med = med[med['kd_regularized']]['kd_smooth']
    reg_mean = mean[mean['kd_regularized']]['kd_smooth']
    assert reg_med.median() < reg_mean.median()      # median lower than mean
    assert reg_med.median() <= 0.15                  # tracks the ~0.10 majority
    # both still smooth the sawtooth
    assert np.median(np.abs(np.diff(reg_med.to_numpy()))) < 0.05


def test_nan_kd_bins_are_not_regularized():
    kd = [0.1, np.nan, 0.2, np.nan, 0.15]
    out = spatial_regularize_kd(_make_df(kd, kd_std=[np.nan] * 5), window=5)
    nan_rows = out['kd'].isna()
    assert not out.loc[nan_rows, 'kd_regularized'].any()


def test_columns_added_and_order_preserved():
    kd = list(np.linspace(0.5, 0.1, 10))
    df = _make_df(kd)
    out = spatial_regularize_kd(df, window=5)
    for c in ['kd_smooth', 'kd_smooth_lo', 'kd_smooth_hi', 'kd_regularized']:
        assert c in out.columns
    assert list(out['lat_bins']) == list(df['lat_bins'])  # original row order


def test_adaptive_window_scales_with_run_length():
    # Long unstable run (sparse-flip zone) -> heavy smoothing; isolated stable
    # bins untouched. Build a 40-bin alternating collapse/breakpoint run.
    import numpy as np
    kd = ([0.08, 0.25] * 20)
    reason = ['no_noise_floor', None] * 20
    kd_std = [np.nan, 0.03] * 20
    df = _make_df(kd, reason=reason, kd_std=kd_std)
    df['kd_ci_low'] = [np.nan, 0.10] * 20
    df['kd_ci_high'] = [np.nan, 0.45] * 20
    out = spatial_regularize_kd(df, adaptive=True, min_window=7, max_window=25,
                                central_stat='mean')
    reg = out[out['kd_regularized']]
    assert len(reg) >= 30  # most of the long run regularized
    # heavy smoothing: long run -> window hits max -> sawtooth flattened hard
    sm_rough = np.median(np.abs(np.diff(reg['kd_smooth'].to_numpy())))
    assert sm_rough < 0.01
    # central value between the two modes, band non-degenerate
    assert 0.10 <= reg['kd_smooth'].median() <= 0.23
    assert (reg['kd_smooth_hi'] - reg['kd_smooth_lo']).median() > 0


def test_adaptive_leaves_stable_bins_untouched():
    import numpy as np
    kd = list(np.linspace(0.5, 0.1, 12))  # smooth, all stable
    out = spatial_regularize_kd(_make_df(kd), adaptive=True)
    assert not out['kd_regularized'].any()
    assert np.allclose(out['kd_smooth'], kd)


def test_force_smooth_mask_regularizes_stable_bins():
    # Individually-stable bins (no reason, finite kd_std) that still zig-zag.
    # Normally untouched; force_smooth_mask makes the regularizer smooth them
    # (the committed-region case). Balanced sawtooth -> central_stat='mean'.
    kd = [0.30, 0.42] * 8
    df = _make_df(kd)
    df['kd_ci_low'] = [0.25, 0.37] * 8
    df['kd_ci_high'] = [0.35, 0.47] * 8
    base = spatial_regularize_kd(df, window=7)
    assert not base['kd_regularized'].any()  # nothing regularized by default
    forced = spatial_regularize_kd(df, window=7, central_stat='mean',
                                   force_smooth_mask=pd.Series([True] * 16, index=df.index))
    assert forced['kd_regularized'].all()
    raw_rough = np.median(np.abs(np.diff(kd)))
    sm_rough = np.median(np.abs(np.diff(forced['kd_smooth'].to_numpy())))
    assert sm_rough < raw_rough / 2
    assert (forced['kd_smooth_hi'] - forced['kd_smooth_lo']).median() > 0


def test_ci_fallback_band_on_unregularized_bins():
    # Stable bins with a per-bin bootstrap CI but NOT regularized: the band must
    # fall back to that CI, not collapse to the line (the 'no CI in the middle'
    # bug). Without kd_ci_* columns, the band stays == line (back-compat).
    kd = [0.30, 0.42] * 8
    df = _make_df(kd)
    df['kd_ci_low'] = [0.25, 0.37] * 8
    df['kd_ci_high'] = [0.35, 0.47] * 8
    out = spatial_regularize_kd(df, window=7)
    assert not out['kd_regularized'].any()
    assert np.allclose(out['kd_smooth_lo'].to_numpy(), df['kd_ci_low'].to_numpy())
    assert np.allclose(out['kd_smooth_hi'].to_numpy(), df['kd_ci_high'].to_numpy())
    assert (out['kd_smooth_hi'] - out['kd_smooth_lo']).min() > 0
    # No CI columns -> degenerate band preserved (the line itself).
    out2 = spatial_regularize_kd(_make_df(kd), window=7)
    assert np.allclose(out2['kd_smooth_lo'], kd)
