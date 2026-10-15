"""Regression test (2026-10-15): quality flags must be computed per beam.

Before the fix every helper keyed its result by lat_bin alone, so on a
multi-beam table row-based flags kept the last beam processed, photon-based
flags pooled all beams' photons, and the assembly broadcast one value onto
every beam's row (observed: all 18 bits identical across beams in 100% of
shared bins at two sites, although surface_sigma differed in every bin).
"""
import numpy as np
import pandas as pd

from kd_utils.quality_flags import (QualityFlag, compute_quality_flags,
                                    has_flag, per_beam_dict)


def _kd_df():
    # Same lat_bins on two beams; only beam A is wavy / range-rejected at bin 1.
    return pd.DataFrame({
        'beam_id': ['gt1l', 'gt1l', 'gt3l', 'gt3l'],
        'lat_bins': [1, 2, 1, 2],
        'kd': [np.nan, 0.2, 0.3, 0.25],
        'surface_sigma': [0.60, 0.20, 0.20, 0.20],
        'fit_failure_reason': ['insufficient_range', None, None, None],
        'n_boot_success': [0, 200, 200, 200],
    })


def _photons():
    # bin 1: beam A sparse (10 photons), beam B dense (400); bin 2: both dense.
    rows = []
    for beam, lb, n in [('gt1l', 1, 10), ('gt1l', 2, 400), ('gt3l', 1, 400), ('gt3l', 2, 400)]:
        rows.append(pd.DataFrame({
            'beam_id': beam, 'lat_bins': lb,
            'photon_height': -np.linspace(1, 10, n),
            'sea_surface_height': 0.0, 'seafloor_elevation': -30.0,
            'quality_ph': 0, 'is_land_label': 1, 'background_rate': 1.0, 'solar_elevation': 30.0,
        }))
    return pd.concat(rows, ignore_index=True)


def _ctx():
    return {'atl03_version': 6, 'gebco_enabled': True, 'n_boot': 200,
            'adaptive_depth_limits': {('gt1l', 1): 10.0, ('gt1l', 2): 10.0,
                                      ('gt3l', 1): 10.0, ('gt3l', 2): 10.0},
            'snr_per_bin': {}, 'surface_photon_counts': {}}


def test_row_based_flags_are_per_beam():
    kd = _kd_df()
    flags = compute_quality_flags(kd, _photons(), _ctx())
    a1, a2, b1, b2 = flags.values
    assert has_flag(a1, QualityFlag.EXTREME_WAVE)
    assert not has_flag(b1, QualityFlag.EXTREME_WAVE)      # same bin, other beam
    assert has_flag(a1, QualityFlag.FIT_INSUFFICIENT_RANGE)
    assert not has_flag(b1, QualityFlag.FIT_INSUFFICIENT_RANGE)
    assert not has_flag(a2, QualityFlag.EXTREME_WAVE)
    assert not has_flag(b2, QualityFlag.EXTREME_WAVE)


def test_photon_based_flags_do_not_pool_beams():
    flags = compute_quality_flags(_kd_df(), _photons(), _ctx())
    a1, a2, b1, b2 = flags.values
    # 10 photons over a 10 m limit = 0.25 per 0.25 m bin -> sparse on beam A only
    assert has_flag(a1, QualityFlag.FEW_PHOTONS)
    assert not has_flag(b1, QualityFlag.FEW_PHOTONS)
    assert not has_flag(a2, QualityFlag.FEW_PHOTONS)
    assert not has_flag(b2, QualityFlag.FEW_PHOTONS)


def test_single_beam_and_legacy_keys_unchanged():
    kd = _kd_df(); kd = kd[kd.beam_id == 'gt1l'].drop(columns='beam_id')
    ph = _photons(); ph = ph[ph.beam_id == 'gt1l'].drop(columns='beam_id')
    ctx = _ctx(); ctx['adaptive_depth_limits'] = {1: 10.0, 2: 10.0}   # plain keys
    flags = compute_quality_flags(kd, ph, ctx)
    assert has_flag(flags.iloc[0], QualityFlag.FEW_PHOTONS)
    assert has_flag(flags.iloc[0], QualityFlag.EXTREME_WAVE)
    assert not has_flag(flags.iloc[1], QualityFlag.FEW_PHOTONS)


def test_per_beam_dict():
    d = {('gt1l', 1): 5.0, ('gt3l', 1): 7.0, 2: 9.0}
    assert per_beam_dict(d, 'gt1l') == {1: 5.0, 2: 9.0}
    assert per_beam_dict(d, 'gt3l') == {1: 7.0, 2: 9.0}
    assert per_beam_dict(d, None) == {1: 7.0, 2: 9.0}
