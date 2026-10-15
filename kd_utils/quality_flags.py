"""
Per-bin quality flag bitmask system for ICESat-2 Kd pipeline.

Each flag occupies one bit position in an integer bitmask. A bin with
quality_flags = 0 has no known issues. Multiple flags combine via OR:
e.g., SEAFLOOR_SHALLOW | EXTREME_WAVE = 4 | 32 = 36.
"""
import logging
import numpy as np
import pandas as pd
from enum import IntEnum

logger = logging.getLogger(__name__)


class QualityFlag(IntEnum):
    AFTERPULSE_RISK = 0          # >5% of bin photons have quality_ph bit 0
    AFTERPULSE_DEPTH_ZONE = 1    # >10% afterpulse photons in 1.5-4.5m depth zone
    SEAFLOOR_SHALLOW = 2         # GEBCO depth within 2m of depth limit
    BATHYMETRY_UNKNOWN = 3       # no GEBCO/ATL24 coverage
    HIGH_SOLAR_BACKGROUND = 4    # background_rate > 2x day/night beam median
    EXTREME_WAVE = 5             # surface_sigma > threshold (default 0.43m)
    LOW_SNR = 6                  # S/N < 5.0 (borderline)
    SHORT_WATER_COLUMN = 7       # adaptive depth limit < 5m
    FEW_PHOTONS = 8              # < 2 photons per 0.25m depth bin in decay zone
    SHORELINE_DATA_MISSING = 9   # is_land_label == -1 (unknown land status)
    KD_FIT_DIVERGENCE = 10       # Kd is NaN but bin had photons
    SURFACE_PEAK_DOMINATED = 11  # surface/(surface+subsurface) > 0.8
    TEP_RISK = 12                # >5% of bin photons have quality_ph bit 2
    FIT_INSUFFICIENT_PHOTONS = 13  # rejected: total photons < min_total_photons
    FIT_INSUFFICIENT_RANGE = 14    # rejected: total depth range < min_total_range_m
    FIT_INSUFFICIENT_BINS = 15     # rejected: non-empty bin count < min_decay_bins
    FIT_NO_BREAKPOINT = 16         # rejected: BIC found no valid breakpoint
    BOOTSTRAP_UNSTABLE = 17        # bp_std > threshold OR low bootstrap success rate


def has_flag(flags_value, flag):
    """Check if a specific QualityFlag bit is set in a bitmask value."""
    return bool(flags_value & (1 << flag))


def decode_flags(flags_value):
    """Return list of flag names set in a bitmask value."""
    return [f.name for f in QualityFlag if flags_value & (1 << f.value)]


def _flag_afterpulse_risk(subsurface_df, atl03_version, threshold=0.05):
    """Bit 0: >5% of bin photons have quality_ph bit 0 (afterpulse)."""
    result = {}
    if atl03_version < 7 or 'quality_ph' not in subsurface_df.columns:
        for lb in subsurface_df['lat_bins'].unique():
            result[lb] = False
        return result
    for lb, group in subsurface_df.groupby('lat_bins'):
        n_ap = (group['quality_ph'].astype(int) & 1).sum()
        result[lb] = (n_ap / len(group)) > threshold if len(group) > 0 else False
    return result


def _flag_afterpulse_depth_zone(subsurface_df, atl03_version, threshold=0.10):
    """Bit 1: >10% afterpulse photons in 1.5-4.5m depth zone."""
    result = {}
    if atl03_version < 7 or 'quality_ph' not in subsurface_df.columns:
        for lb in subsurface_df['lat_bins'].unique():
            result[lb] = False
        return result
    for lb, group in subsurface_df.groupby('lat_bins'):
        depth = -group['photon_height']
        zone = group[(depth >= 1.5) & (depth <= 4.5)]
        if len(zone) == 0:
            result[lb] = False
            continue
        n_ap = (zone['quality_ph'].astype(int) & 1).sum()
        result[lb] = (n_ap / len(zone)) > threshold
    return result


def _flag_seafloor_shallow(subsurface_df, adaptive_depth_limits, buffer_m=2.0):
    """Bit 2: GEBCO depth within buffer_m of adaptive depth limit."""
    result = {}
    has_col = 'seafloor_elevation' in subsurface_df.columns
    for lb in adaptive_depth_limits:
        if not has_col:
            result[lb] = False
            continue
        bin_data = subsurface_df[subsurface_df['lat_bins'] == lb]
        if bin_data.empty or bin_data['seafloor_elevation'].isna().all():
            result[lb] = False
            continue
        gebco_depth = -bin_data['seafloor_elevation'].median()
        depth_limit = adaptive_depth_limits[lb]
        result[lb] = (gebco_depth - depth_limit) < buffer_m
    return result


def _flag_bathymetry_unknown(subsurface_df, gebco_enabled):
    """Bit 3: no GEBCO/ATL24 coverage."""
    result = {}
    bins = subsurface_df['lat_bins'].unique()
    if not gebco_enabled or 'seafloor_elevation' not in subsurface_df.columns:
        for lb in bins:
            result[lb] = True
        return result
    for lb, group in subsurface_df.groupby('lat_bins'):
        result[lb] = group['seafloor_elevation'].isna().all()
    return result


def _flag_high_solar_background(subsurface_df, threshold_multiplier=2.0):
    """Bit 4: background_rate > 2x day/night beam median."""
    result = {}
    if 'background_rate' not in subsurface_df.columns:
        for lb in subsurface_df['lat_bins'].unique():
            result[lb] = False
        return result
    bin_stats = subsurface_df.groupby('lat_bins').agg(
        bg_rate=('background_rate', 'mean'),
        solar_el=('solar_elevation', 'mean'),
    )
    day_bins = bin_stats[bin_stats['solar_el'] >= 0]
    night_bins = bin_stats[bin_stats['solar_el'] < 0]
    day_median = day_bins['bg_rate'].median() if len(day_bins) > 0 else np.nan
    night_median = night_bins['bg_rate'].median() if len(night_bins) > 0 else np.nan
    for lb, row in bin_stats.iterrows():
        median = day_median if row['solar_el'] >= 0 else night_median
        if np.isnan(median) or np.isnan(row['bg_rate']):
            result[lb] = False
        else:
            result[lb] = row['bg_rate'] > threshold_multiplier * median
    return result


def _flag_extreme_wave(kd_df, threshold=0.43):
    """Bit 5: surface_sigma > threshold.

    Default 0.43 m is the surface sigma corresponding to U10 = 10 m/s -- the
    wind above which the ATL13 ATBD (Release 007, Jasinski et al. 2025, sec.
    4.2.4) reports whitecap/foam scattering "approaches the volume scattering of
    low turbidity natural waters", i.e. contaminates the signal Kd is fitted to.
    The mapping is empirical: the pooled granule-median sigma at U10 = 10 m/s is
    0.433 m across 813 granules (0.424 East coast / 0.460 West coast).

    It is deliberately NOT set from the Kd-bias sweep: that criterion's level is
    a free parameter, and sweeping it over 10-25% of baseline Kd moves the
    implied sigma from 0.22 to 0.43.  The bias and retrieval-loss curves are
    reported as consequences of this threshold, not as its derivation.

    The pre-2026-08 default of 0.3 m predated all of this and fired at the median
    sea state of the East-coast footprint (median sigma 0.297), flagging 49% of
    its bins and 84% of the West coast's.  At 0.43 m: 19% and 50%.
    See 03_analysis/figure_sigma_flag_supplement.py.
    """
    result = {}
    for _, row in kd_df.iterrows():
        lb = row['lat_bins']
        sigma = row.get('surface_sigma', np.nan)
        result[lb] = sigma > threshold if not np.isnan(sigma) else False
    return result


def _flag_low_snr(snr_per_bin, warning_threshold=5.0):
    """Bit 6: S/N < warning_threshold."""
    return {lb: snr < warning_threshold for lb, snr in snr_per_bin.items()}


def _flag_short_water_column(adaptive_depth_limits, min_depth=5.0):
    """Bit 7: adaptive depth limit < min_depth."""
    return {lb: dl < min_depth for lb, dl in adaptive_depth_limits.items()}


def _flag_few_photons(subsurface_df, adaptive_depth_limits, vertical_res=0.25,
                      min_density=2.0):
    """Bit 8: < min_density photons per vertical bin in decay zone."""
    result = {}
    for lb, group in subsurface_df.groupby('lat_bins'):
        depth_limit = adaptive_depth_limits.get(lb, 20.0)
        n_depth_bins = depth_limit / vertical_res
        if n_depth_bins <= 0:
            result[lb] = True
            continue
        density = len(group) / n_depth_bins
        result[lb] = density < min_density
    return result


def _flag_shoreline_data_missing(subsurface_df):
    """Bit 9: any photon in bin has is_land_label == -1."""
    result = {}
    if 'is_land_label' not in subsurface_df.columns:
        for lb in subsurface_df['lat_bins'].unique():
            result[lb] = True
        return result
    for lb, group in subsurface_df.groupby('lat_bins'):
        result[lb] = (group['is_land_label'] == -1).any()
    return result


def _flag_kd_fit_divergence(kd_df, subsurface_df):
    """Bit 10: Kd is NaN but bin had subsurface photons."""
    bins_with_photons = set(subsurface_df['lat_bins'].unique())
    result = {}
    for _, row in kd_df.iterrows():
        lb = row['lat_bins']
        result[lb] = pd.isna(row['kd']) and lb in bins_with_photons
    return result


def _flag_surface_peak_dominated(subsurface_df, surface_photon_counts, threshold=0.8):
    """Bit 11: surface/(surface+subsurface) > threshold."""
    result = {}
    sub_counts = subsurface_df.groupby('lat_bins').size()
    for lb in sub_counts.index:
        surf = surface_photon_counts.get(lb, 0)
        total = surf + sub_counts[lb]
        result[lb] = (surf / total) > threshold if total > 0 else False
    return result


def _flag_tep_risk(subsurface_df, atl03_version, threshold=0.05):
    """Bit 12: >5% of bin photons have quality_ph bit 2 (TEP)."""
    result = {}
    if atl03_version < 7 or 'quality_ph' not in subsurface_df.columns:
        for lb in subsurface_df['lat_bins'].unique():
            result[lb] = False
        return result
    for lb, group in subsurface_df.groupby('lat_bins'):
        n_tep = (group['quality_ph'].astype(int) & 4).astype(bool).sum()
        result[lb] = (n_tep / len(group)) > threshold if len(group) > 0 else False
    return result


def _flag_fit_insufficient_photons(kd_df):
    """Bit 13: fit_failure_reason == 'insufficient_photons'."""
    result = {}
    for _, row in kd_df.iterrows():
        lb = row['lat_bins']
        reason = row.get('fit_failure_reason', None)
        result[lb] = (reason == 'insufficient_photons')
    return result


def _flag_fit_insufficient_range(kd_df):
    """Bit 14: fit_failure_reason == 'insufficient_range'."""
    result = {}
    for _, row in kd_df.iterrows():
        lb = row['lat_bins']
        reason = row.get('fit_failure_reason', None)
        result[lb] = (reason == 'insufficient_range')
    return result


def _flag_fit_insufficient_bins(kd_df):
    """Bit 15: fit_failure_reason == 'too_few_bins'."""
    result = {}
    for _, row in kd_df.iterrows():
        lb = row['lat_bins']
        reason = row.get('fit_failure_reason', None)
        result[lb] = (reason == 'too_few_bins')
    return result


def _flag_fit_no_breakpoint(kd_df):
    """Bit 16: fit_failure_reason indicates no valid breakpoint found.

    Includes 'no_noise_floor' from fit_beers_law_bma — a soft-fail collapse
    where Kd was recovered from the no-breakpoint single-line fit because
    the BMA weight on that candidate exceeded the no-noise-floor threshold
    (typical for post-GEBCO/post-ATL24 near-shore bins where the seabed
    return has been stripped, leaving pure exponential decay).
    """
    result = {}
    for _, row in kd_df.iterrows():
        lb = row['lat_bins']
        reason = row.get('fit_failure_reason', None)
        result[lb] = reason in ('no_breakpoint_found',
                                'no_breakpoint_fallback',
                                'no_noise_floor')
    return result


def _flag_bootstrap_unstable(kd_df, boot_df=None, bp_std_threshold=2.0,
                              n_boot_expected=300, success_fraction=0.5):
    """
    Bit 17: Bootstrap is unstable for this bin.

    Fires when either:
      - bp_std > bp_std_threshold (breakpoint location jumping around
        across bootstrap samples indicates the two-segment structure is
        poorly defined), OR
      - n_boot_success / n_boot_expected < success_fraction (hybrid fit
        failed on many bootstrap samples, indicating the bin is near the
        edge of fittability).
    """
    result = {}
    if boot_df is None or len(boot_df) == 0:
        for _, row in kd_df.iterrows():
            result[row['lat_bins']] = False
        return result

    boot_cols = ['lat_bins'] + [c for c in ('bp_std', 'n_boot_success')
                                if c in boot_df.columns]
    merged = kd_df[['lat_bins']].merge(
        boot_df[boot_cols],
        on='lat_bins', how='left'
    )
    has_n_success = 'n_boot_success' in merged.columns
    for _, row in merged.iterrows():
        lb = row['lat_bins']
        bp_std = row.get('bp_std', np.nan)

        bp_unstable = (not pd.isna(bp_std)) and (bp_std > bp_std_threshold)
        if has_n_success:
            n_success = row['n_boot_success']
            low_success = (not pd.isna(n_success)) and (n_success < success_fraction * n_boot_expected)
        else:
            low_success = False

        result[lb] = bool(bp_unstable or low_success)
    return result


def compute_quality_flags(kd_df, subsurface_df, context):
    """
    Compute per-bin quality flags as a bitmask integer.

    Parameters
    ----------
    kd_df : pd.DataFrame
        Kd output with lat_bins, kd, noise_floor, surface_sigma, etc.
    subsurface_df : pd.DataFrame
        Subsurface photon DataFrame with quality_ph, background_rate,
        is_land_label, photon_height, solar_elevation, etc.
    context : dict
        Pipeline context with keys: atl03_version, gebco_enabled,
        active_filters, adaptive_depth_limits, snr_per_bin,
        surface_photon_counts, bootstrap_df (optional), n_boot (optional;
        actual --bootstrap_iterations, used to scale the low-success check),
        extreme_wave_sigma_threshold (optional; actual
        --extreme_wave_sigma_threshold, the surface sigma above which
        EXTREME_WAVE is set).

    Returns
    -------
    pd.Series
        Integer bitmask indexed like kd_df, named 'quality_flags'.
    """
    # `or 0` guards an unknown/None version (e.g. an ingestion path with no ATL03
    # filename to parse): unknown is treated as pre-v007, so the version-gated
    # afterpulse/TEP flags stay off rather than crashing on a None comparison.
    atl03_version = context.get('atl03_version', 0) or 0
    gebco_enabled = context.get('gebco_enabled', False)
    depth_limits = context.get('adaptive_depth_limits', {})
    snr_per_bin = context.get('snr_per_bin', {})
    surface_counts = context.get('surface_photon_counts', {})
    bootstrap_df = context.get('bootstrap_df', None)
    # Actual bootstrap iteration count (CLI --bootstrap_iterations). The 300
    # fallback matches _flag_bootstrap_unstable's historical default; without
    # threading this, lowering the CLI default (500 -> 200) silently raised
    # the low-success threshold from the intended 50% to 300*0.5/200 = 75%.
    n_boot = context.get('n_boot', 300)
    # Actual CLI --extreme_wave_sigma_threshold; the fallback matches
    # _flag_extreme_wave's own default so an un-threaded caller is unchanged.
    wave_sigma_thr = context.get('extreme_wave_sigma_threshold', 0.43)

    # Flags are computed ONE BEAM AT A TIME (2026-10-15). lat_bins restart per
    # beam, so running the helpers on a multi-beam table keyed every flag by
    # lat_bin alone: row-based flags kept the last beam processed, photon-based
    # flags pooled the beams' photons, and the assembly broadcast that one value
    # onto every beam's row. The helpers themselves are correct for a single
    # beam, so each beam is passed through them in isolation.
    out = pd.Series(0, index=kd_df.index, name='quality_flags', dtype='int64')
    has_beam = 'beam_id' in kd_df.columns
    beams = list(pd.unique(kd_df['beam_id'])) if has_beam else [None]
    for beam in beams:
        kd_b = kd_df if beam is None else kd_df[kd_df['beam_id'] == beam]
        if beam is not None and 'beam_id' in subsurface_df.columns:
            sub_b = subsurface_df[subsurface_df['beam_id'] == beam]
        else:
            sub_b = subsurface_df
        boot_b = bootstrap_df
        if beam is not None and bootstrap_df is not None and 'beam_id' in bootstrap_df.columns:
            boot_b = bootstrap_df[bootstrap_df['beam_id'] == beam]
        masks = _compute_flags_one_beam(
            kd_b, sub_b, boot_b,
            depth_limits=per_beam_dict(depth_limits, beam),
            snr_per_bin=per_beam_dict(snr_per_bin, beam),
            surface_counts=per_beam_dict(surface_counts, beam),
            atl03_version=atl03_version, gebco_enabled=gebco_enabled,
            n_boot=n_boot, wave_sigma_thr=wave_sigma_thr,
        )
        out.loc[kd_b.index] = masks

    n_flagged = int((out != 0).sum())
    logger.info("Quality flags: %d/%d bins flagged, %d clean (%d beam%s)",
                n_flagged, len(out), len(out) - n_flagged,
                len(beams), "" if len(beams) == 1 else "s")
    return out


def per_beam_dict(d, beam):
    """One beam's view of a per-bin dict.

    Production dicts (adaptive_depth_limits, snr_per_bin, surface_photon_counts
    from bathy_processing.process_subsurface_photon_filtering) are keyed by
    (beam_id, lat_bin). Legacy / single-beam callers may pass plain lat_bin keys;
    those pass through unchanged. ``beam=None`` returns plain keys for every
    entry (tuple keys are stripped to their lat_bin).
    """
    out = {}
    for k, v in d.items():
        if isinstance(k, tuple) and len(k) == 2:
            if beam is None or k[0] == beam:
                out[k[1]] = v
        else:
            out[k] = v
    return out


def _compute_flags_one_beam(kd_df, subsurface_df, bootstrap_df, depth_limits,
                            snr_per_bin, surface_counts, atl03_version,
                            gebco_enabled, n_boot, wave_sigma_thr):
    """Bitmask per row of a SINGLE-beam kd_df (lat_bins unique per row)."""
    flag_results = {
        QualityFlag.AFTERPULSE_RISK: _flag_afterpulse_risk(subsurface_df, atl03_version),
        QualityFlag.AFTERPULSE_DEPTH_ZONE: _flag_afterpulse_depth_zone(subsurface_df, atl03_version),
        QualityFlag.SEAFLOOR_SHALLOW: _flag_seafloor_shallow(subsurface_df, depth_limits),
        QualityFlag.BATHYMETRY_UNKNOWN: _flag_bathymetry_unknown(subsurface_df, gebco_enabled),
        QualityFlag.HIGH_SOLAR_BACKGROUND: _flag_high_solar_background(subsurface_df),
        QualityFlag.EXTREME_WAVE: _flag_extreme_wave(kd_df, threshold=wave_sigma_thr),
        QualityFlag.LOW_SNR: _flag_low_snr(snr_per_bin),
        QualityFlag.SHORT_WATER_COLUMN: _flag_short_water_column(depth_limits),
        QualityFlag.FEW_PHOTONS: _flag_few_photons(subsurface_df, depth_limits),
        QualityFlag.SHORELINE_DATA_MISSING: _flag_shoreline_data_missing(subsurface_df),
        QualityFlag.KD_FIT_DIVERGENCE: _flag_kd_fit_divergence(kd_df, subsurface_df),
        QualityFlag.SURFACE_PEAK_DOMINATED: _flag_surface_peak_dominated(subsurface_df, surface_counts),
        QualityFlag.TEP_RISK: _flag_tep_risk(subsurface_df, atl03_version),
        QualityFlag.FIT_INSUFFICIENT_PHOTONS: _flag_fit_insufficient_photons(kd_df),
        QualityFlag.FIT_INSUFFICIENT_RANGE: _flag_fit_insufficient_range(kd_df),
        QualityFlag.FIT_INSUFFICIENT_BINS: _flag_fit_insufficient_bins(kd_df),
        QualityFlag.FIT_NO_BREAKPOINT: _flag_fit_no_breakpoint(kd_df),
        QualityFlag.BOOTSTRAP_UNSTABLE: _flag_bootstrap_unstable(
            kd_df, bootstrap_df, n_boot_expected=n_boot),
    }
    bitmasks = []
    for lb in kd_df['lat_bins'].values:
        mask = 0
        for flag, results_dict in flag_results.items():
            if results_dict.get(lb, False):
                mask |= (1 << flag)
        bitmasks.append(mask)
    return bitmasks
