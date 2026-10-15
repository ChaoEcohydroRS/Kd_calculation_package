# utils/bathy_processing.py
import os
import logging
import pandas as pd
import rasterio
import numpy as np
from kd_utils.sea_photons_analysis import (
    get_sea_surface_height_adaptive,
    get_sea_surface_height_static,
    horizontal_vertical_bin_dataset,
)
from rtree import index
from shapely.geometry import box, Point
from scipy.spatial import cKDTree
from scipy.stats import norm

logger = logging.getLogger(__name__)


def apply_optional_histogram_quality_filter(
    binned_dataset_sea_surface,
    subsurface_photon_dataset,
    sea_surface_height,
    min_subsurface_snr=3.0,
    signal_above_noise=1.5,
    track_reject_fraction=0.80,
    coarse_bin=1.0,
    min_depth=3.0,
    max_depth=50.0,
    tail_margin_m=10.0,
    **_kwargs,
):
    """
    Adaptive histogram quality filter with per-bin depth limit.

    Flowchart step 6 — Three-level quality decision:

    Level 1 (per-bin S/N gate):
        Compare subsurface photon count at 1-2 m depth to the noise floor
        estimated from 15-20 m.  If the subsurface signal-to-noise ratio
        (S/N) is below *min_subsurface_snr*, the bin is rejected as
        ``insufficient_signal`` — the water has too few particles for
        reliable photon-profile Kd estimation (e.g., oligotrophic ocean).

    Level 2 (adaptive depth limit):
        For bins that pass Level 1, scan from the surface downward using
        1 m coarse bins and find the deepest bin whose photon count exceeds
        *signal_above_noise* × noise_floor.  That depth becomes the fit
        limit for Kd (Step 17).  Subsurface photons below that depth are
        trimmed from the dataset.

    Level 3 (track-level gate):
        If more than *track_reject_fraction* of along-track bins are
        rejected as ``insufficient_signal``, the entire beam/track is
        flagged as ``insufficient_particle_load``.  This catches whole
        tracks over clear open ocean (e.g., Hawaii) where per-bin S/N
        is marginal and individual bins may pass by chance.

    Parameters
    ----------
    binned_dataset_sea_surface : pd.DataFrame
        Full binned photon dataset with ``lat_bins`` and ``photon_height``.
    subsurface_photon_dataset : pd.DataFrame
        Subsurface photons (output of Step 7 surface detection).
    sea_surface_height : list
        Per-bin sea-surface elevation from Step 7.
    min_subsurface_snr : float
        Minimum subsurface S/N ratio (1-2 m vs 15-20 m) to keep a bin.
        Default 3.0.
    signal_above_noise : float
        Multiplier for adaptive depth limit — a depth bin must exceed
        noise_floor × this value to be considered signal.  Default 2.0.
    track_reject_fraction : float
        If more than this fraction of bins are rejected, flag the entire
        track.  Default 0.80 (80 %) — relaxed from the original 0.40 so that
        mixed-quality tracks (e.g., a bbox crossing both a turbid river plume
        and adjacent open ocean) keep the good plume bins. Increase toward
        1.0 to fully disable track-level rejection; tighten back toward
        0.40 for stricter open-ocean-only filtering.
    coarse_bin : float
        Width of coarse depth bins (m) for robust photon counting.
    min_depth : float
        Minimum allowed depth limit (m).  Default 3.0.
    max_depth : float
        Maximum depth range to consider (m).  Default 20.0.
    """
    if subsurface_photon_dataset.empty:
        return subsurface_photon_dataset, {}, {}

    grouped = binned_dataset_sea_surface.groupby(['lat_bins'], observed=False)
    lat_bin_keys = list(grouped.groups.keys())

    # sea_surface_height is a positional list aligned to this same sorted
    # lat_bins grouping (built by get_sea_surface_height_adaptive on the same
    # dataframe). Bind it to a lat_bins-keyed dict ONCE so per-bin access is
    # by key, not loop index — a future reorder/skip in the loop can then no
    # longer silently shift every subsequent bin's surface height.
    if len(sea_surface_height) != len(lat_bin_keys):
        logger.warning(
            "sea_surface_height length %d != %d lat_bins groups; "
            "surface-depth alignment may be off.",
            len(sea_surface_height), len(lat_bin_keys),
        )
    surface_by_bin = dict(zip(lat_bin_keys, sea_surface_height))

    depth_limits = {}
    snr_per_bin = {}
    reject_reasons = {}

    for lat_bin in lat_bin_keys:
        surface = surface_by_bin.get(lat_bin, np.nan)
        if surface is None or np.isnan(surface):
            continue
        bin_data = binned_dataset_sea_surface[binned_dataset_sea_surface['lat_bins'] == lat_bin]
        if bin_data.empty:
            continue

        depth = surface - bin_data['photon_height'].values
        depth = depth[depth > 0]
        if len(depth) < 10:
            reject_reasons[lat_bin] = 'too_few_photons'
            continue

        # Coarse histogram (1 m bins)
        coarse_edges = np.arange(0, max_depth + coarse_bin, coarse_bin)
        counts, _ = np.histogram(depth, bins=coarse_edges)
        centers = np.arange(0.5 * coarse_bin, max_depth, coarse_bin)[:len(counts)]

        if len(counts) < 5:
            reject_reasons[lat_bin] = 'too_few_bins'
            continue

        # Reference: 1-2 m depth band (subsurface signal, avoids surface peak)
        ref_count = float(counts[1]) if len(counts) > 1 else 0.0

        # Noise floor: median of deepest 5 m
        noise_bins = counts[-5:] if len(counts) >= 20 else counts[max(len(counts) - 5, 0):]
        noise_floor = max(float(np.median(noise_bins)), 0.5)

        subsurface_snr = ref_count / noise_floor
        snr_per_bin[lat_bin] = subsurface_snr

        # --- Level 1: S/N gate ---
        if subsurface_snr < min_subsurface_snr:
            reject_reasons[lat_bin] = 'insufficient_signal'
            continue

        # --- Level 2: Adaptive depth limit (with BMA-aware tail margin) ---
        # Find the deepest 1 m bin where signal exceeds 2× noise, then extend
        # by `tail_margin_m` (default 10 m) so the BMA fitter has enough flat-
        # noise-floor bins past the signal decay to evaluate the two-segment
        # model. Without the margin, the trim leaves <5 noise bins and the
        # BMA's BIC vote collapses to the single-line (no_noise_floor) model.
        threshold = noise_floor * signal_above_noise
        signal_depth = min_depth
        for j in range(len(counts)):
            if counts[j] > threshold:
                signal_depth = centers[j] + coarse_bin / 2.0
        depth_limits[lat_bin] = max(min_depth, min(max_depth, signal_depth + tail_margin_m))

    # --- Level 3: Track-level gate ---
    n_assessed = len(depth_limits) + len(reject_reasons)
    n_insuf = sum(1 for r in reject_reasons.values() if r == 'insufficient_signal')
    n_pass = len(depth_limits)
    reject_rate = n_insuf / n_assessed if n_assessed > 0 else 0.0

    track_rejected = reject_rate > track_reject_fraction

    logger.info(
        "Adaptive Step 6 (S/N≥%.1f, depth×%.1f noise): "
        "%d/%d bins passed, %d insufficient_signal (%.0f%%)%s",
        min_subsurface_snr, signal_above_noise,
        n_pass, n_assessed, n_insuf, reject_rate * 100,
        " → TRACK REJECTED (insufficient particle load)" if track_rejected else "",
    )
    if depth_limits:
        avg_dl = np.mean(list(depth_limits.values()))
        logger.info("  Adaptive depth limits: avg %.1f m (range %.1f–%.1f m)",
                     avg_dl, min(depth_limits.values()), max(depth_limits.values()))

    if track_rejected:
        logger.warning(
            "Track rejected: %.0f%% of bins have insufficient subsurface "
            "signal (S/N < %.1f). Water may be too clear for photon-profile "
            "Kd estimation.",
            reject_rate * 100, min_subsurface_snr,
        )
        return subsurface_photon_dataset.iloc[0:0].copy(), {}, {}

    # Trim subsurface photons to per-bin adaptive depth limit
    if not depth_limits:
        return subsurface_photon_dataset.iloc[0:0].copy(), {}, {}

    keep_mask = pd.Series(False, index=subsurface_photon_dataset.index)
    for lat_bin, group in subsurface_photon_dataset.groupby('lat_bins', observed=False):
        if lat_bin not in depth_limits:
            continue
        limit = depth_limits[lat_bin]
        if 'sea_surface_height' in group.columns and not group['sea_surface_height'].isna().all():
            surface = group['sea_surface_height'].iloc[0]
        else:
            surface = 0.0
        min_height = surface - limit
        bin_keep = (subsurface_photon_dataset['lat_bins'] == lat_bin) & \
                   (subsurface_photon_dataset['photon_height'] >= min_height)
        keep_mask = keep_mask | bin_keep

    return subsurface_photon_dataset[keep_mask].copy(), depth_limits, snr_per_bin


SURFACE_DIAGNOSTIC_KEYS = (
    'surface_sigma',
    'surface_peak_count',
    'surface_n_within_3sigma',
    'surface_bin_sum_3sigma',
    'surface_fit_quality',
)


def _empty_surface_diagnostics():
    return {k: np.nan for k in SURFACE_DIAGNOSTIC_KEYS}


def _gaussian_fit_r2(edges_lo, edges_hi, bin_counts, mu, sigma):
    """R² between observed histogram counts and the fitted Gaussian over the
    same bins, scaled so that expected counts integrate to the observed total.
    Bin extents (edges_lo, edges_hi) come from each bin's actual photon-height
    range, so they're robust to pd.cut label ambiguity.
    """
    if sigma <= 0 or len(bin_counts) < 3:
        return np.nan
    expected_prob = norm.cdf(edges_hi, loc=mu, scale=sigma) - \
        norm.cdf(edges_lo, loc=mu, scale=sigma)
    total_obs = float(bin_counts.sum())
    expected_total = float(expected_prob.sum())
    if total_obs <= 0 or expected_total <= 0:
        return np.nan
    expected_counts = expected_prob * (total_obs / expected_total)
    obs = bin_counts.astype(float)
    ss_tot = float(((obs - obs.mean()) ** 2).sum())
    ss_res = float(((obs - expected_counts) ** 2).sum())
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def compute_surface_diagnostics(binned_dataset_sea_surface, sea_surface_height,
                                vertical_res=0.25):
    """Compute per-bin surface-peak diagnostics from photons near the surface peak.

    Two-pass Gaussian fit: initial ±1 m window, expanded to ±min(2.5σ, 3 m)
    if the initial sigma > 0.4 m. Returns a dict mapping
    ``lat_bin -> {surface_sigma, surface_peak_count, surface_n_within_3sigma,
    surface_bin_sum_3sigma, surface_fit_quality}``. Bins with fewer than 10
    surface photons get all-NaN diagnostics.

    - ``surface_sigma``: σ of the fitted Gaussian (m).
    - ``surface_peak_count``: max photon count in any single height_bin within
      the lat_bin.
    - ``surface_n_within_3sigma``: photon count with ``μ-3σ ≤ h ≤ μ+3σ``.
    - ``surface_bin_sum_3sigma``: summed count over height_bins whose
      ``[label-vertical_res/2, label+vertical_res/2]`` interval intersects
      ``[μ-3σ, μ+3σ]``.
    - ``surface_fit_quality``: R² between the observed histogram (over the
      σ-refit window) and the fitted Gaussian PDF, scaled to the observed
      total. Typical values are in [0,1]; not clipped — negatives indicate
      a Gaussian fit worse than predicting the mean count.
    """
    grouped = binned_dataset_sea_surface.groupby(['lat_bins'], observed=False)
    lat_bin_keys = list(grouped.groups.keys())
    if len(lat_bin_keys) != len(sea_surface_height):
        return {}

    diagnostics_map = {}
    for lat_bin, surface in zip(lat_bin_keys, sea_surface_height):
        if pd.isna(surface):
            continue
        bin_data = binned_dataset_sea_surface[
            binned_dataset_sea_surface['lat_bins'] == lat_bin
        ]
        if bin_data.empty:
            continue
        diag = _empty_surface_diagnostics()
        # Per-height_bin counts and actual photon-height extents. We use
        # extents (not pd.cut labels) for window/intersection math because
        # horizontal_vertical_bin_dataset attaches arbitrary float labels
        # to height_bins that are not aligned with bin centers.
        if 'height_bins' in bin_data.columns:
            grp = bin_data.groupby('height_bins', observed=True)['photon_height']
            hcounts = grp.size()
            if not hcounts.empty:
                diag['surface_peak_count'] = int(hcounts.max())
                bin_lo = grp.min().reindex(hcounts.index).values
                bin_hi = grp.max().reindex(hcounts.index).values
            else:
                bin_lo = np.array([])
                bin_hi = np.array([])
        else:
            hcounts = pd.Series(dtype=int)
            bin_lo = np.array([])
            bin_hi = np.array([])

        peak_data = bin_data[
            (bin_data['photon_height'] > surface - 1.0) &
            (bin_data['photon_height'] < surface + 1.0)
        ]
        if len(peak_data) > 10:
            mu, sigma = norm.fit(peak_data['photon_height'])
            half_win = 1.0
            if sigma > 0.4:
                half_win2 = min(2.5 * sigma, 3.0)
                peak_data2 = bin_data[
                    (bin_data['photon_height'] > mu - half_win2) &
                    (bin_data['photon_height'] < mu + half_win2)
                ]
                if len(peak_data2) > 10:
                    mu, sigma = norm.fit(peak_data2['photon_height'])
                    half_win = half_win2
            diag['surface_sigma'] = float(sigma)

            heights = bin_data['photon_height'].values
            within = (heights >= mu - 3.0 * sigma) & (heights <= mu + 3.0 * sigma)
            diag['surface_n_within_3sigma'] = int(within.sum())

            if not hcounts.empty:
                lo, hi = mu - 3.0 * sigma, mu + 3.0 * sigma
                # Strict superset of n_within_3sigma: a bin counts if any of
                # its photons lies in [μ-3σ, μ+3σ]. Equivalent to bin_hi > lo
                # AND bin_lo < hi.
                intersect_mask = (bin_hi > lo) & (bin_lo < hi)
                diag['surface_bin_sum_3sigma'] = int(
                    hcounts.values[intersect_mask].sum()
                )
                # R² window: bins whose extents intersect the σ-refit window.
                window_lo, window_hi = mu - half_win, mu + half_win
                window_mask = (bin_hi > window_lo) & (bin_lo < window_hi)
                if window_mask.sum() >= 3:
                    diag['surface_fit_quality'] = _gaussian_fit_r2(
                        bin_lo[window_mask],
                        bin_hi[window_mask],
                        hcounts.values[window_mask],
                        mu, sigma,
                    )
        diagnostics_map[lat_bin] = diag

    return diagnostics_map


def compute_surface_sigma(binned_dataset_sea_surface, sea_surface_height,
                          vertical_res=0.25):
    """Backward-compat wrapper: returns dict[lat_bin] -> sigma.

    Calls compute_surface_diagnostics and projects to just the σ field, which
    is what apply_optional_surface_sigma_filter and the per-photon
    surface_sigma broadcast still consume.
    """
    diag_map = compute_surface_diagnostics(
        binned_dataset_sea_surface, sea_surface_height, vertical_res=vertical_res
    )
    return {k: v['surface_sigma'] for k, v in diag_map.items()}


def apply_optional_surface_sigma_filter(
    binned_dataset_sea_surface,
    subsurface_photon_dataset,
    sea_surface_height,
    sigma_max=0.5
):
    """
    Optional Gaussian surface sigma filter: discard along-track bins if the
    fitted surface Gaussian sigma exceeds sigma_max.

    Flowchart step: 8 — If standard deviation of Gaussian peak is >X m,
    quality-flag this along-track bin as low confidence and discard it
    (addresses surface waves) (Possible discard).
    """
    if subsurface_photon_dataset.empty:
        return subsurface_photon_dataset

    sigma_map = compute_surface_sigma(binned_dataset_sea_surface, sea_surface_height)
    if not sigma_map:
        return subsurface_photon_dataset

    sigma_df = pd.DataFrame(
        list(sigma_map.items()), columns=['lat_bins', 'surface_sigma']
    )
    valid_bins = set(sigma_df.loc[sigma_df['surface_sigma'] <= sigma_max, 'lat_bins'].tolist())
    return subsurface_photon_dataset[subsurface_photon_dataset['lat_bins'].isin(valid_bins)].copy()


def apply_optional_refraction_correction(
    binned_dataset_sea_surface,
    subsurface_photon_dataset,
    sea_surface_height,
    water_temp_c=20.0,
    wavelength_nm=532.0
):
    """
    Optional refraction correction for subsurface photons.
    Updates photon positions/heights using Snell-based geometry.

    Flowchart step: 9 — Compute water depths (distance below surface) and
    correct depths for refraction.
    """
    required_cols = {'lat_bins', 'photon_height', 'ref_elevation', 'ref_azimuth', 'lon', 'lat'}
    if subsurface_photon_dataset.empty or (not required_cols.issubset(set(subsurface_photon_dataset.columns))):
        return subsurface_photon_dataset

    grouped = binned_dataset_sea_surface.groupby(['lat_bins'], observed=False)
    lat_bin_keys = list(grouped.groups.keys())
    if len(lat_bin_keys) != len(sea_surface_height):
        return subsurface_photon_dataset

    surface_df = pd.DataFrame({'lat_bins': lat_bin_keys, 'sea_surface_height': sea_surface_height})
    corrected = subsurface_photon_dataset.merge(surface_df, on='lat_bins', how='left').copy()
    if corrected['sea_surface_height'].isna().all():
        return subsurface_photon_dataset

    corrected['photon_height_pre_refraction'] = corrected['photon_height']
    corrected['lon_pre_refraction'] = corrected['lon']
    corrected['lat_pre_refraction'] = corrected['lat']

    # Refraction index parameterization from legacy module.
    a = -0.000001501562500
    b = 0.000000107084865
    c = -0.000042759374989
    d = -0.000160475520686
    e = 1.398067112092424
    n1 = 1.00029
    n2 = (a * water_temp_c ** 2) + (b * wavelength_nm ** 2) + (c * water_temp_c) + (d * wavelength_nm) + e

    ref_elev = pd.to_numeric(corrected['ref_elevation'], errors='coerce').to_numpy(dtype=float)
    ref_az = pd.to_numeric(corrected['ref_azimuth'], errors='coerce').to_numpy(dtype=float)
    z = pd.to_numeric(corrected['photon_height'], errors='coerce').to_numpy(dtype=float)
    ws = pd.to_numeric(corrected['sea_surface_height'], errors='coerce').to_numpy(dtype=float)
    x = pd.to_numeric(corrected['lon'], errors='coerce').to_numpy(dtype=float)
    y = pd.to_numeric(corrected['lat'], errors='coerce').to_numpy(dtype=float)

    # Convert to radians if values look like degrees.
    if np.nanmax(np.abs(ref_elev)) > (2 * np.pi):
        ref_elev = np.deg2rad(ref_elev)
    if np.nanmax(np.abs(ref_az)) > (2 * np.pi):
        ref_az = np.deg2rad(ref_az)

    valid_mask = np.isfinite(ref_elev) & np.isfinite(ref_az) & np.isfinite(z) & np.isfinite(ws) & (z <= ws)
    if not np.any(valid_mask):
        return subsurface_photon_dataset

    theta1 = (np.pi / 2.0) - ref_elev[valid_mask]
    theta2_arg = (n1 * np.sin(theta1)) / n2
    theta2_arg = np.clip(theta2_arg, -1.0, 1.0)
    theta2 = np.arcsin(theta2_arg)

    D = ws[valid_mask] - z[valid_mask]
    cos_theta1 = np.cos(theta1)
    cos_theta1 = np.where(np.abs(cos_theta1) < 1e-6, np.nan, cos_theta1)
    S = D / cos_theta1
    R = (S * n1) / n2
    Gamma = (np.pi / 2.0) - theta1
    phi = theta1 - theta2
    P_sq = np.maximum(R ** 2 + S ** 2 - 2 * R * S * np.cos(phi), 0.0)
    P = np.sqrt(P_sq)
    alpha_arg = np.divide(R * np.sin(phi), P, out=np.zeros_like(P), where=(P != 0))
    alpha_arg = np.clip(alpha_arg, -1.0, 1.0)
    alpha = np.arcsin(alpha_arg)
    Beta = Gamma - alpha

    DY = P * np.cos(Beta)
    DZ = P * np.sin(Beta)
    DE = DY * np.sin(ref_az[valid_mask])
    DN = DY * np.cos(ref_az[valid_mask])

    x_corr = x[valid_mask] + DE
    y_corr = y[valid_mask] + DN
    z_corr = z[valid_mask] + DZ

    corrected.loc[valid_mask, 'lon'] = x_corr
    corrected.loc[valid_mask, 'lat'] = y_corr
    corrected.loc[valid_mask, 'photon_height'] = z_corr
    return corrected


def rebin_height_after_refraction(subsurface_photon_dataset, vertical_res=0.25):
    """
    Re-cut ``height_bins`` on refraction-corrected photon heights (step 10).

    ``height_bins`` is assigned once at Step 4 (process_sea_photon_binning),
    five steps before the refraction correction, and the Beer's-law fitter
    reads that column directly as its depth axis (Kd_analysis.py
    CalculateKdFromFilteredSubsurfacePhoton / bootstrap_kd_all_bins). Without
    this re-cut the corrected ``photon_height`` reaches the output CSV but
    never reaches the fit, so Kd is reported per *apparent* metre and is low
    by roughly the refractive index (measured: median 1.45x on BohaiSea, IQR
    1.25-1.63; the spread over the geometric 1.34 is BMA breakpoint selection,
    which is not scale-invariant).

    Only ``height_bins`` is rebuilt. ``lat_bins`` is deliberately left alone:
    refraction shifts lat/lon, so re-cutting it would silently re-key every
    per-bin diagnostic (surface_sigma, adaptive_depth_limits, snr_per_bin,
    surface_photon_counts) onto slightly different geographic bins. This is
    the narrow alternative to rebuild_and_refit_surface_after_refraction,
    which rebuilds from the full pre-refraction dataset and therefore also
    discards the Step 6 histogram quality filter and re-fits the surface
    Gaussian on refracted surface photons (see that function's docstring).

    The surface Gaussian (mu, sigma) and the surface clip are NOT redone --
    surface returns never propagated through water, so they are already
    correctly located, and sigma feeds the sea-state estimator H_s = 4*sigma.
    One consequence to be aware of: the clip is applied in apparent
    coordinates, so the ~1.0 m apparent floor corresponds to ~0.75 m of true
    water column, and the fit's zdepth=0 sits there.

    Parameters
    ----------
    subsurface_photon_dataset : pd.DataFrame
        Subsurface photons after apply_optional_refraction_correction, i.e.
        with ``photon_height`` already in true (corrected) metres.
    vertical_res : float
        Depth bin width (m) -- now true metres rather than apparent.

    Returns
    -------
    pd.DataFrame
        Same frame with ``height_bins`` re-cut on the corrected heights.
        Returned unchanged if the frame is empty or lacks the columns.
    """
    if subsurface_photon_dataset.empty or \
            'photon_height' not in subsurface_photon_dataset.columns:
        return subsurface_photon_dataset

    # apply_optional_refraction_correction has several early returns (missing
    # ref_elevation/ref_azimuth, lat_bin/surface length mismatch, no photon
    # passing z <= surface) that hand back the input frame untouched. In those
    # cases photon_height is still apparent, and re-cutting would change the
    # binning relative to step 4 for no reason. photon_height_pre_refraction is
    # written only on the path that actually applies the correction, so its
    # presence is the reliable signal.
    if 'photon_height_pre_refraction' not in subsurface_photon_dataset.columns:
        logger.warning(
            "Skipping post-refraction re-binning: refraction correction did not "
            "run on this beam (no photon_height_pre_refraction column). "
            "Kd for this beam is per apparent metre."
        )
        return subsurface_photon_dataset

    rebinned = subsurface_photon_dataset.copy()
    heights = pd.to_numeric(rebinned['photon_height'], errors='coerce')
    finite = heights[np.isfinite(heights)]
    if len(finite) < 2:
        return subsurface_photon_dataset

    h_min, h_max = float(finite.min()), float(finite.max())
    height_range = abs(h_max - h_min)
    if height_range <= 0:
        return subsurface_photon_dataset

    # Same construction as horizontal_vertical_bin_dataset (step 4) so the
    # label semantics the fitter relies on are unchanged: float labels spanning
    # [min, max], zdepth later taken as (label.max() - label).  Deliberately
    # identical, including the decimals=1 label rounding -- which quantizes a
    # 0.25 m grid to alternating 0.2/0.3 m label steps.  That quantization is
    # pre-existing and applies equally before and after this change, so it
    # keeps the v4-vs-v5 difference attributable to refraction alone.
    n_bins = max(1, round(height_range / vertical_res))
    rebinned['height_bins'] = pd.cut(
        heights, bins=n_bins,
        labels=np.round(np.linspace(h_min, h_max, num=n_bins), decimals=1)
    ).astype(float)

    logger.info(
        "Re-binned height_bins on refraction-corrected depths (step 10): "
        "range %.2f to %.2f m, %d bins at %.2f m",
        h_min, h_max, n_bins, vertical_res
    )
    return rebinned


def rescale_depth_limits_after_refraction(subsurface_photon_dataset,
                                          adaptive_depth_limits):
    """
    Convert per-bin adaptive depth limits from apparent to true metres.

    The Step 6 histogram quality filter runs before the refraction correction,
    so its ``adaptive_depth_limits`` are in apparent metres. Downstream they
    are consumed as physical depths -- as the FEW_PHOTONS density denominator
    (limit / 0.25 m), by SHORT_WATER_COLUMN (limit < 5 m) and by
    SEAFLOOR_SHALLOW (GEBCO depth within 2 m of the limit). Left unscaled
    against a re-binned true-metre axis they run ~34 % too deep, so
    FEW_PHOTONS over-fires and SHORT_WATER_COLUMN under-fires.

    The scale factor is measured per bin from the photons themselves
    (median corrected-depth / apparent-depth over photons deeper than 0.5 m
    apparent), not assumed, so it absorbs the per-photon beam geometry. Bins
    with too few usable photons fall back to the dataset-wide median ratio.
    """
    if not adaptive_depth_limits:
        return adaptive_depth_limits
    needed = {'lat_bins', 'photon_height', 'photon_height_pre_refraction',
              'sea_surface_height'}
    if subsurface_photon_dataset.empty or \
            not needed.issubset(set(subsurface_photon_dataset.columns)):
        return adaptive_depth_limits

    df = subsurface_photon_dataset
    surf = pd.to_numeric(df['sea_surface_height'], errors='coerce')
    dep_pre = surf - pd.to_numeric(df['photon_height_pre_refraction'], errors='coerce')
    dep_post = surf - pd.to_numeric(df['photon_height'], errors='coerce')
    usable = pd.DataFrame({
        'lat_bins': df['lat_bins'],
        'ratio': dep_post / dep_pre,
    })[(dep_pre > 0.5) & np.isfinite(dep_pre) & np.isfinite(dep_post)]
    if usable.empty:
        return adaptive_depth_limits

    global_ratio = float(usable['ratio'].median())
    per_bin = usable.groupby('lat_bins', observed=True)['ratio'].agg(['median', 'size'])
    per_bin = per_bin[per_bin['size'] >= 10]['median'].to_dict()

    rescaled = {}
    for lat_bin, limit in adaptive_depth_limits.items():
        ratio = per_bin.get(lat_bin, global_ratio)
        rescaled[lat_bin] = limit * ratio if np.isfinite(ratio) else limit
    logger.info(
        "Rescaled %d adaptive depth limits to true metres (median factor %.3f)",
        len(rescaled), global_ratio
    )
    return rescaled


def _fine_surface_heights(binned_dataset_sea_surface, fine_res_m, min_photons=20):
    """Detect the sea surface on a FINE along-track grid.

    Runs the same Gaussian surface-peak detection used in Step 7, but on bins
    much finer than the production along-track resolution, so the local sea
    surface is resolved well enough to flatten against. Returns the bin edges
    (needed to place photons on the same grid) and {fine_bin: sea-surface height}.

    Note `lat` here is a PROJECTED coordinate in metres, not degrees, so
    fine_res_m indexes it directly.
    """
    lat = pd.to_numeric(binned_dataset_sea_surface['lat'], errors='coerce')
    lo, hi = float(lat.min()), float(lat.max())
    n = max(1, int(round((hi - lo) / max(fine_res_m, 1e-6))))
    edges = np.linspace(lo, hi, n + 1)

    fine = binned_dataset_sea_surface.copy()
    # Own the binning rather than delegating to pd.cut, so the identical grid can
    # be reapplied to the subsurface photons below.
    fine['lat_bins'] = np.clip(np.digitize(lat.to_numpy(), edges[1:-1]), 0, n - 1)

    ssh, _lbl, _solo, _sub, _cnt = get_sea_surface_height_adaptive(fine)
    keys = sorted(pd.unique(fine['lat_bins']))
    if len(ssh) != len(keys):
        logger.warning("Fine-grid surface detection returned %d heights for %d bins; "
                       "skipping flattening.", len(ssh), len(keys))
        return edges, {}
    return edges, {int(k): float(v) for k, v in zip(keys, ssh)
                   if v is not None and np.isfinite(v)}


def apply_sea_surface_flattening(
    subsurface_photon_dataset,
    sea_surface_height,
    lat_bin_keys,
    horizontal_res,
    flattening_window_m=5000,
    binned_dataset_sea_surface=None,
    fine_res_m=50.0,
):
    """
    Flatten local sea-surface variation to the mean level of a broader window.

    Two-level, as described in the manuscript: the surface-peak detection is
    applied to photon clouds binned much more finely than 100 m (fine_res_m,
    default 50 m) so the sea-surface height is resolved with good along-track
    resolution; the mean is then taken over a broader segment
    (flattening_window_m, default 5 km); and each photon's depth is shifted up or
    down so that its local sea-surface height matches that window mean.

    Resolving the surface finely matters because the shift is only as good as the
    local surface estimate: at the 500 m production resolution a 5 km window holds
    just ten samples, and any tide- or wave-driven tilt inside a production bin is
    invisible. At 50 m there are ~70-110 surface photons per bin at these sites,
    comfortably above the 20-photon minimum the detector needs.

    If binned_dataset_sea_surface is not supplied the function falls back to the
    legacy behaviour of flattening the production-resolution heights, so older
    callers keep working.
    """
    if subsurface_photon_dataset.empty:
        return subsurface_photon_dataset

    use_fine = (binned_dataset_sea_surface is not None
                and not binned_dataset_sea_surface.empty
                and 'lat' in binned_dataset_sea_surface.columns
                and fine_res_m and fine_res_m < horizontal_res)

    flattened = subsurface_photon_dataset.copy()

    if use_fine:
        edges, ssh_map = _fine_surface_heights(binned_dataset_sea_surface, fine_res_m)
        if not ssh_map:
            return subsurface_photon_dataset
        n_fine = len(edges) - 1
        per_window = max(1, int(round(flattening_window_m / max(fine_res_m, 1e-6))))

        # window mean of the finely-resolved surface, then the per-fine-bin offset
        fb = pd.Series(ssh_map, name='ssh').rename_axis('fine_bin').reset_index()
        fb['window'] = fb['fine_bin'] // per_window
        fb = fb.merge(fb.groupby('window')['ssh'].mean().rename('ssh_window'),
                      on='window', how='left')
        fb['offset'] = fb['ssh_window'] - fb['ssh']
        offset_map = dict(zip(fb['fine_bin'], fb['offset']))

        # Place the subsurface photons on the SAME fine grid. Refraction moves
        # lat/lon, so use the pre-refraction coordinate where it exists, which is
        # the one the surface grid was built from.
        src = ('lat_pre_refraction' if 'lat_pre_refraction' in flattened.columns
               else 'lat')
        slat = pd.to_numeric(flattened[src], errors='coerce').to_numpy()
        fidx = np.clip(np.digitize(slat, edges[1:-1]), 0, n_fine - 1)
        off = pd.Series(fidx).map(offset_map).fillna(0.0).to_numpy()

        flattened['sea_surface_flattening_offset'] = off
        logger.info("Sea-surface flattening: %.0f m fine grid (%d bins, %d with a "
                    "surface fit), %.0f m window (%d fine bins each); "
                    "offset median %.3f m, p95 |offset| %.3f m",
                    fine_res_m, n_fine, len(ssh_map), flattening_window_m,
                    per_window, float(np.median(off)),
                    float(np.percentile(np.abs(off), 95)))
    else:
        # ---- legacy path: flatten the production-resolution heights ----
        if len(sea_surface_height) != len(lat_bin_keys):
            return subsurface_photon_dataset
        surface_df = pd.DataFrame({
            'lat_bins': lat_bin_keys,
            'sea_surface_height_local': sea_surface_height
        }).dropna(subset=['sea_surface_height_local']).copy()
        if surface_df.empty:
            return subsurface_photon_dataset
        surface_df['lat_bins_num'] = pd.to_numeric(surface_df['lat_bins'], errors='coerce')
        surface_df = surface_df.dropna(subset=['lat_bins_num'])
        if surface_df.empty:
            return subsurface_photon_dataset
        big_bin_size = max(1, int(round(flattening_window_m / max(horizontal_res, 1))))
        if big_bin_size <= 1:
            logger.warning(
                "Sea surface flattening window (%d m) <= horizontal_res (%d m) -> "
                "big_bin_size=1, flattening has no effect.",
                flattening_window_m, horizontal_res)
        surface_df['big_bin'] = (surface_df['lat_bins_num'].astype(int) // big_bin_size).astype(int)
        mean_surface = (surface_df.groupby('big_bin', observed=False)['sea_surface_height_local']
                        .mean().rename('sea_surface_height_bigbin_mean').reset_index())
        surface_df = surface_df.merge(mean_surface, on='big_bin', how='left')
        surface_df['sea_surface_flattening_offset'] = (
            surface_df['sea_surface_height_bigbin_mean'] - surface_df['sea_surface_height_local'])
        flattened = flattened.merge(
            surface_df[['lat_bins', 'sea_surface_height_local',
                        'sea_surface_height_bigbin_mean', 'sea_surface_flattening_offset']],
            on='lat_bins', how='left')

    # SIGN (fixed 2026-08-26; the original subtracted and so DOUBLED the
    # undulation instead of removing it). A photon sits at depth d below its LOCAL
    # surface: photon_height = z_local - d. To place it at the same depth below the
    # window datum we need z_window - d, i.e.
    #     photon_new = photon_height + (z_window - z_local) = photon_height + offset
    # Verified against a synthetic surface in tests/test_surface_flattening.py:
    # the depth error std must collapse toward zero, not double.
    flattened['photon_height_original'] = flattened['photon_height']
    flattened['photon_height'] = (
        flattened['photon_height'] + flattened['sea_surface_flattening_offset'].fillna(0.0))

    # Re-bin height_bins from the flattened photon_height so the Kd fit (which
    # reads height_bins, not photon_height) actually sees the correction.
    if 'height_bins' in flattened.columns:
        orig_bins = pd.to_numeric(flattened['height_bins'], errors='coerce').dropna().unique()
        if len(orig_bins) >= 2:
            sorted_bins = np.sort(orig_bins)
            diffs = np.diff(sorted_bins)
            v_res = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.25
        else:
            v_res = 0.25
        h_min = flattened['photon_height'].min()
        h_max = flattened['photon_height'].max()
        n_hbins = max(1, int(round((h_max - h_min) / v_res)))
        bin_edges = np.linspace(h_min, h_max, n_hbins + 1)
        bin_labels = np.round((bin_edges[:-1] + bin_edges[1:]) / 2, decimals=1)
        flattened['height_bins'] = pd.cut(
            flattened['photon_height'], bins=bin_edges, labels=bin_labels,
            include_lowest=True)

    return flattened.drop(columns=['lat_bins_num', 'big_bin'], errors='ignore')


# 1. Function to create an R-tree spatial index for raster bounds
def create_spatial_index(gebco_paths):
    """
    Create an R-tree spatial index for raster bounds to quickly find relevant rasters.
    
    Parameters:
        gebco_paths (list): List of paths to GEBCO raster files.
    
    Returns:
        raster_data_dict (dict): Dictionary containing loaded rasters and their respective data.
        spatial_index (rtree.index.Index): R-tree spatial index for raster bounds.
    """
    idx = index.Index()
    raster_data_dict = {}
    for i, path in enumerate(gebco_paths):
        with rasterio.Env(GTIFF_SRS_SOURCE='EPSG'):
            gebco_raster = rasterio.open(path)
            # Do NOT read the full tile: get_seafloor_elevation samples via
            # gebco_raster.sample() (windowed), so the full-array read was dead
            # weight — ~1 GB per global GEBCO tile x 8 tiles = ~7.5 GB per call,
            # loaded once PER main.py invocation. That was the OOM (>8 GB) and the
            # per-row slowdown on the SlideRule ERDDAP array. Keep only the open
            # dataset (for sampling) + bounds (for the index).
            raster_data_dict[path] = (gebco_raster, None)
            bounds = gebco_raster.bounds
            idx.insert(i, (bounds.left, bounds.bottom, bounds.right, bounds.top), obj=path)
    return raster_data_dict, idx

# 2. Function to determine which rasters are needed for a given set of coordinates using spatial index in a batch manner
def get_relevant_rasters_using_index(lons, lats, raster_data_dict, spatial_index):
    """
    Determine which raster datasets are relevant for the given coordinates using spatial index.
    This function uses a more efficient approach by performing a bounding box query for batches of points.
    
    Parameters:
        lons (numpy.ndarray): Array of longitudes.
        lats (numpy.ndarray): Array of latitudes.
        raster_data_dict (dict): Dictionary containing raster datasets and their data.
        spatial_index (rtree.index.Index): R-tree spatial index for raster bounds.
    
    Returns:
        relevant_rasters (list): List of relevant raster datasets and their respective data.
    """
    # Empty input (a beam with no photons at the seafloor-query stage) → no rasters.
    # Without this guard lons.min() raises "zero-size array to reduction operation
    # minimum", crashing the whole run instead of letting that beam drop out. Common
    # for global point-matchups where some beams have no usable subsurface photons.
    if len(lons) == 0:
        return []
    # Create a bounding box that covers all points
    min_lon, max_lon = lons.min(), lons.max()
    min_lat, max_lat = lats.min(), lats.max()
    bounding_box = box(min_lon, min_lat, max_lon, max_lat)

    # Get all rasters that intersect with the bounding box
    matches = list(spatial_index.intersection((bounding_box.bounds), objects=True))
    relevant_paths = {match.object for match in matches}
    relevant_rasters = [(raster_data_dict[path][0], raster_data_dict[path][1]) for path in relevant_paths]
    return relevant_rasters


# 3. Function to get seafloor elevation from the relevant rasters in a vectorized manner
def get_seafloor_elevation(lons, lats, relevant_rasters):
    """
    Get seafloor elevations for a batch of points based on longitude and latitude from relevant rasters.
    
    Parameters:
        lons (numpy.ndarray): Array of longitudes.
        lats (numpy.ndarray): Array of latitudes.
        relevant_rasters (list): List of relevant raster datasets and their respective data.
    
    Returns:
        seafloor_elevations (numpy.ndarray): Array of seafloor elevations.
    """
    points = np.vstack((lons, lats)).T
    seafloor_elevations = np.full(len(lons), np.nan)
    
    for gebco_raster, raster_data in relevant_rasters:
        # Use rasterio.sample to get values for multiple points in a batch
        values = list(gebco_raster.sample(points))
        for idx, value in enumerate(values):
            if np.isnan(seafloor_elevations[idx]) and value is not None:
                seafloor_elevations[idx] = value[0]
    
    return seafloor_elevations


def query_gebco_seafloor_elevation(sea_photon_dataset, gebco_paths):
    """
    Query GEBCO raster tiles and attach a 'seafloor_elevation' column to the
    photon dataset. Does not remove any rows.

    Flowchart step: 12 — Retrieve GEBCO data (or other regional raster
    bathymetry data) for the region when ATL24 is unavailable or returns no
    matching points.

    Parameters
    ----------
    sea_photon_dataset : pd.DataFrame
        Must contain 'longitude' and 'latitude' columns.
    gebco_paths : list
        Paths to GEBCO GeoTIFF raster files.

    Returns
    -------
    pd.DataFrame
        Input dataset with 'seafloor_elevation' column added (negative = below
        sea level).
    """
    dataset = sea_photon_dataset.copy()
    # An empty subsurface set can arrive with no columns at all (KeyError on
    # 'longitude') or with empty arrays — either way there is nothing to query.
    # Common for global point-matchups where a beam has no usable subsurface.
    if dataset.empty or 'longitude' not in dataset.columns or 'latitude' not in dataset.columns:
        dataset['seafloor_elevation'] = np.full(len(dataset), np.nan)
        return dataset
    raster_data_dict, spatial_index = create_spatial_index(gebco_paths)
    lons = dataset['longitude'].values
    lats = dataset['latitude'].values
    relevant_rasters = get_relevant_rasters_using_index(lons, lats, raster_data_dict, spatial_index)
    dataset['seafloor_elevation'] = get_seafloor_elevation(lons, lats, relevant_rasters)
    return dataset


def remove_photons_below_seafloor(sea_photon_dataset, buffer_m=0.0):
    """
    Remove photons whose height is at or below the seabed elevation plus a
    safety buffer.

    Flowchart step: 13 — Remove data below seabed (seabed defined by ATL24 or
    GEBCO).  The buffer accounts for the diffuse seafloor return that extends
    above the actual bottom — GEBCO marks the hard surface, but scattered
    photons from the seafloor can appear up to ~1-2 m above it.

    Parameters
    ----------
    sea_photon_dataset : pd.DataFrame
        Must contain 'photon_height' and 'seafloor_elevation' columns.
    buffer_m : float
        Safety margin above the seafloor (metres).  Photons within this
        distance above the GEBCO/ATL24 seafloor are also removed.
        Default 0.0 (no buffer, backward-compatible).

    Returns
    -------
    pd.DataFrame
        Only rows where photon_height > seafloor_elevation + buffer_m.
    """
    cutoff = sea_photon_dataset['seafloor_elevation'] + buffer_m
    return sea_photon_dataset[
        sea_photon_dataset['photon_height'] > cutoff
    ].copy()


def discard_shallow_seabed_bins(sea_photon_dataset, min_depth_m):
    """
    Discard along-track bins where GEBCO confirms the seabed is too shallow
    for reliable Kd fitting.

    Logic:
      - GEBCO depth >= min_depth_m  → keep (enough water column)
      - GEBCO depth <  min_depth_m  → discard (too shallow)
      - No GEBCO coverage (NaN)     → keep (unknown depth, let Step 6
                                       S/N gate handle quality)

    Flowchart step: 14 — Discard photon data in horizontal bins where GEBCO
    confirms seabed < 5 m deep; Kd is not calculated for these bins.

    Parameters
    ----------
    sea_photon_dataset : pd.DataFrame
        Must contain a 'depth' column (= -seafloor_elevation, positive downward).
        NaN depth indicates no GEBCO coverage.
    min_depth_m : float
        Minimum seabed depth (e.g., 5.0 m). Only bins with a known GEBCO
        depth shallower than this are discarded.

    Returns
    -------
    pd.DataFrame
        All rows except those where GEBCO confirms depth < min_depth_m.
    """
    has_gebco = sea_photon_dataset['depth'].notna()
    too_shallow = has_gebco & (sea_photon_dataset['depth'] < min_depth_m)
    return sea_photon_dataset[~too_shallow].copy()


# 4. The main process function that ties everything together
def process_seafloor_data(sea_photon_dataset, gebco_paths, Ignore_Subsurface_Height_Thres,
                          seafloor_buffer_m=0.0):
    """
    Query GEBCO bathymetry then apply seabed and shallow-water filters.

    Delegates to three single-step helpers:
      query_gebco_seafloor_elevation  → step 12
      remove_photons_below_seafloor   → step 13
      discard_shallow_seabed_bins     → step 14

    Parameters:
        gebco_paths (list): List of path to GEBCO raster files.
        sea_photon_dataset (pandas.DataFrame): Dataset containing longitude,
            latitude, and photon height.
        Ignore_Subsurface_Height_Thres (float): Minimum depth fallback when
            GEBCO is unavailable.
        seafloor_buffer_m (float): Buffer above GEBCO seafloor (metres) to
            exclude the diffuse seafloor return.

    Returns:
        filtered_sea_photon_dataset (pandas.DataFrame): Filtered dataset
            containing points above the seafloor in water deeper than threshold.
    """
    # Empty subsurface set (a beam with no usable subsurface photons) can arrive
    # with no columns at all; short-circuit so the seafloor chain never dereferences
    # a missing column ('longitude'/'photon_height'/...). The caller / main.py then
    # detect .empty and exit "No subsurface photons" (rc=3) gracefully. Common for
    # global point-matchups (SlideRule ERDDAP driver).
    if sea_photon_dataset.empty:
        logger.info("GEBCO seafloor filter: 0 photons (empty subsurface set) — skipping.")
        out = sea_photon_dataset.copy()
        out["depth"] = np.full(len(out), np.nan)
        return out
    dataset = query_gebco_seafloor_elevation(sea_photon_dataset, gebco_paths)  # step 12
    dataset['depth'] = -dataset['seafloor_elevation']
    n_before = len(dataset)
    dataset = remove_photons_below_seafloor(dataset, buffer_m=seafloor_buffer_m)  # step 13
    n_after_seafloor = len(dataset)
    dataset = discard_shallow_seabed_bins(dataset, abs(Ignore_Subsurface_Height_Thres))  # step 14
    n_after_shallow = len(dataset)
    logger.info("GEBCO seafloor filter: %d photons → %d (seafloor+buffer) → %d (shallow fallback)",
                n_before, n_after_seafloor, n_after_shallow)
    return dataset


def load_atl24_points(atl24_file_path):
    """
    Load ATL24 point dataset and normalize to columns: longitude, latitude,
    seafloor_elevation_atl24.

    Flowchart step: 12 — Query ATL24 data for bathymetry match; if no match,
    retrieve GEBCO data (or other regional raster bathymetry data) for region.
    """
    if (not atl24_file_path) or (not os.path.exists(atl24_file_path)):
        return pd.DataFrame(columns=['longitude', 'latitude', 'seafloor_elevation_atl24'])

    lower_path = atl24_file_path.lower()
    if lower_path.endswith(('.h5', '.hdf5')):
        try:
            import h5py
            beams = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
            frames = []
            with h5py.File(atl24_file_path, 'r') as f:
                for beam in beams:
                    if beam not in f:
                        continue
                    grp = f[beam]
                    if not all(k in grp for k in ('lon_ph', 'lat_ph', 'ortho_h', 'class_ph')):
                        continue
                    class_ph = grp['class_ph'][:]
                    bathy_mask = class_ph == 40  # ATL24 bathymetry class
                    # Apply low_confidence_flag filter if available
                    if 'low_confidence_flag' in grp:
                        bathy_mask = bathy_mask & (grp['low_confidence_flag'][:] == 0)
                    if bathy_mask.sum() == 0:
                        continue
                    frames.append(pd.DataFrame({
                        'longitude': grp['lon_ph'][:][bathy_mask].astype(float),
                        'latitude':  grp['lat_ph'][:][bathy_mask].astype(float),
                        'seafloor_elevation_atl24': grp['ortho_h'][:][bathy_mask].astype(float),
                    }))
            if not frames:
                return pd.DataFrame(columns=['longitude', 'latitude', 'seafloor_elevation_atl24'])
            return pd.concat(frames, ignore_index=True).dropna()
        except Exception:
            return pd.DataFrame(columns=['longitude', 'latitude', 'seafloor_elevation_atl24'])
    elif lower_path.endswith(('.csv', '.txt')):
        atl24_df = pd.read_csv(atl24_file_path)
    elif lower_path.endswith('.parquet'):
        atl24_df = pd.read_parquet(atl24_file_path)
    elif lower_path.endswith(('.gpkg', '.shp', '.geojson')):
        try:
            import geopandas as gpd
            atl24_gdf = gpd.read_file(atl24_file_path)
            atl24_df = pd.DataFrame(atl24_gdf)
            if ('longitude' not in atl24_df.columns or 'latitude' not in atl24_df.columns) and ('geometry' in atl24_gdf.columns):
                atl24_df['longitude'] = atl24_gdf.geometry.x
                atl24_df['latitude'] = atl24_gdf.geometry.y
        except Exception:
            return pd.DataFrame(columns=['longitude', 'latitude', 'seafloor_elevation_atl24'])
    else:
        return pd.DataFrame(columns=['longitude', 'latitude', 'seafloor_elevation_atl24'])

    lon_candidates = ['longitude', 'lon', 'x', 'LONGITUDE', 'LON']
    lat_candidates = ['latitude', 'lat', 'y', 'LATITUDE', 'LAT']
    elev_candidates = [
        'seafloor_elevation', 'bottom_elevation', 'bathymetry', 'elevation', 'z',
        'depth', 'water_depth'
    ]

    lon_col = next((c for c in lon_candidates if c in atl24_df.columns), None)
    lat_col = next((c for c in lat_candidates if c in atl24_df.columns), None)
    elev_col = next((c for c in elev_candidates if c in atl24_df.columns), None)
    if lon_col is None or lat_col is None or elev_col is None:
        return pd.DataFrame(columns=['longitude', 'latitude', 'seafloor_elevation_atl24'])

    out_df = pd.DataFrame({
        'longitude': pd.to_numeric(atl24_df[lon_col], errors='coerce'),
        'latitude': pd.to_numeric(atl24_df[lat_col], errors='coerce'),
        'atl24_raw_bathy': pd.to_numeric(atl24_df[elev_col], errors='coerce')
    }).dropna(subset=['longitude', 'latitude', 'atl24_raw_bathy']).copy()

    if 'depth' in elev_col.lower() and ('elev' not in elev_col.lower()):
        out_df['seafloor_elevation_atl24'] = -out_df['atl24_raw_bathy'].abs()
    else:
        out_df['seafloor_elevation_atl24'] = out_df['atl24_raw_bathy']

    return out_df[['longitude', 'latitude', 'seafloor_elevation_atl24']]


def process_seafloor_data_atl24(
    sea_photon_dataset,
    atl24_file_path,
    Ignore_Subsurface_Height_Thres,
    max_match_distance_deg=0.01
):
    """
    Match ATL24 bathymetry points to photons, then filter below seafloor and
    shallow bins. Returns filtered dataset and number of matched photons.

    Flowchart steps:
      12 — Query ATL24 data for bathymetry match; if no match, retrieve GEBCO data.
      13 — Remove data below seabed (seabed defined by ATL24 or GEBCO).
      14 — Discard all photon data in horizontal bins where the seabed is <5 m
           deep; Kd is not calculated for these bins — label discarded bins as
           low confidence.
    """
    atl24_points = load_atl24_points(atl24_file_path)
    if atl24_points.empty:
        return sea_photon_dataset, 0

    photon_coords = sea_photon_dataset[['longitude', 'latitude']].to_numpy(dtype=float)
    atl24_coords = atl24_points[['longitude', 'latitude']].to_numpy(dtype=float)
    if len(photon_coords) == 0 or len(atl24_coords) == 0:
        return sea_photon_dataset, 0

    tree = cKDTree(atl24_coords)
    distances, indices = tree.query(photon_coords, k=1, distance_upper_bound=max_match_distance_deg)
    valid_match = np.isfinite(distances) & (indices < len(atl24_points))
    matched_count = int(valid_match.sum())
    if matched_count == 0:
        return sea_photon_dataset, 0

    matched_dataset = sea_photon_dataset.copy()
    matched_dataset['seafloor_elevation'] = np.nan
    matched_dataset.loc[valid_match, 'seafloor_elevation'] = atl24_points['seafloor_elevation_atl24'].to_numpy()[indices[valid_match]]
    matched_dataset = matched_dataset.dropna(subset=['seafloor_elevation']).copy()
    matched_dataset['depth'] = -matched_dataset['seafloor_elevation']

    above_floor = remove_photons_below_seafloor(matched_dataset)                          # step 13
    filtered_dataset = discard_shallow_seabed_bins(above_floor, abs(Ignore_Subsurface_Height_Thres))  # step 14
    return filtered_dataset, matched_count


def rebuild_and_refit_surface_after_refraction(
    binned_dataset_sea_surface,
    sea_surface_height,
    water_temp_c,
    wavelength_nm,
    horizontal_res,
    vertical_res,
):
    """
    Apply refraction correction to the full binned dataset, rebuild histograms
    on the corrected depths, and re-fit the Gaussian surface peak.

    Flowchart steps:
      10 — Re-build histograms based on corrected depths
           (apply_optional_refraction_correction → horizontal_vertical_bin_dataset).
      11 — Re-fit Gaussian curve to surface to identify surface peak; compute
           standard deviation and remove histogram data within three standard
           deviations (get_sea_surface_height_adaptive on the rebinned data).

    Parameters
    ----------
    binned_dataset_sea_surface : pd.DataFrame
        Full binned photon dataset before refraction correction.
    sea_surface_height : list
        Per-bin surface heights from step 7.
    water_temp_c : float
        Water temperature for the refraction index calculation.
    wavelength_nm : float
        Laser wavelength (nm) for the refraction index calculation.
    horizontal_res : float
        Along-track bin size (m) for histogram rebuild.
    vertical_res : float
        Vertical bin size (m) for histogram rebuild.

    Returns
    -------
    tuple
        (sea_surface_height, sea_surface_height_abnormal_label,
         solo_sea_surface_label, subsurface_photon_dataset) — same 4-tuple as
        get_sea_surface_height_adaptive.
    """
    corrected_full_dataset = apply_optional_refraction_correction(
        binned_dataset_sea_surface,
        binned_dataset_sea_surface.copy(),
        sea_surface_height,
        water_temp_c=water_temp_c,
        wavelength_nm=wavelength_nm,
    )
    # Drop old Categorical bin columns so pd.cut() can create fresh categories
    # from the refraction-corrected height range.
    cols_to_drop = [c for c in ('height_bins', 'lat_bins') if c in corrected_full_dataset.columns]
    corrected_full_dataset = corrected_full_dataset.drop(columns=cols_to_drop)
    rebinned_corrected = horizontal_vertical_bin_dataset(   # step 10
        corrected_full_dataset, horizontal_res, vertical_res
    )
    return get_sea_surface_height_adaptive(rebinned_corrected)  # step 11


# 5. The function to get the subsurface photon dataset
def get_subsurface_photon(
    binned_dataset_sea_surface,
    GEBCO_paths,
    subsurface_thresh,
    Ignore_Subsurface_Height_Thres,
    use_atl24_filter=False,
    atl24_file_path='',
    atl24_max_match_distance_deg=0.01,
    use_gebco_filter=False,
    seafloor_buffer_m=0.0,
    apply_histogram_quality_filter=False,
    signal_above_noise=1.5,
    apply_surface_sigma_filter=False,
    surface_sigma_max=0.5,
    apply_refraction_correction=False,
    refraction_water_temp_c=20.0,
    refraction_wavelength_nm=532.0,
    apply_post_refraction_refit=False,
    apply_refraction_rebin=True,
    apply_flattening=False,
    flattening_window_m=500,
    flattening_fine_res_m=50.0,
    surface_floor_m=1.0,
    horizontal_res=500,
    vertical_res=0.25
):
    """
    Orchestrate per-beam subsurface photon extraction and all optional quality filters.

    Flowchart steps executed in order:
      7  — Fit Gaussian curve to identify surface elevation; compute std dev of
           Gaussian peak (via get_sea_surface_height_adaptive).  The surface
           peak is removed here, on UNCORRECTED heights: surface returns never
           propagated through water, so they are already correctly located,
           and the clip threshold max(2σ, 1.0 m) is defined by the surface
           elevation distribution, which lives in uncorrected coordinates.
      6  — Review histograms for quality; quality-flag low-confidence bins and
           exclude (optional, apply_histogram_quality_filter).
      8  — If Gaussian std dev > X m, quality-flag and discard bin (optional,
           apply_surface_sigma_filter).
      9  — Compute water depths and correct for refraction (optional,
           apply_refraction_correction).  Applied only to the surviving
           subsurface photons, and only where z <= sea surface.
      10 — Re-build histograms based on corrected depths.  Default path is
           apply_refraction_rebin → rebin_height_after_refraction, which
           re-cuts height_bins only.  The legacy apply_post_refraction_refit →
           rebuild_and_refit_surface_after_refraction also rebuilds lat_bins
           and re-detects the surface from the full pre-refraction dataset;
           it is retained for reproducing older runs and is NOT recommended
           (it discards this function's Step 6 result and re-keys lat_bins).
      11 — (legacy only) Re-fit Gaussian curve to identify surface peak.  Not
           performed on the default path: refracting surface photons
           compresses the peak and biases σ, which feeds H_s = 4σ and the
           EXTREME_WAVE flag.
      12 — Query ATL24 / retrieve GEBCO bathymetry data (optional,
           use_atl24_filter / use_gebco_filter).
      13 — Remove data below seabed (optional).
      14 — Discard bins where seabed < 5 m deep; label as low confidence
           (applied unconditionally via Ignore_Subsurface_Height_Thres).
    """
    # adaptive threshold
    sea_surface_height, sea_surface_label, solo_sea_surface_label, subsurface_photon_dataset, \
        surface_photon_counts = get_sea_surface_height_adaptive(binned_dataset_sea_surface,
                                                                surface_floor_m=surface_floor_m)

    # Always compute per-bin surface diagnostics (sigma + 4 surface-peak metrics)
    # for quality flagging, wave-adaptive fit, and the bin-level diagnostic merge
    # into the AddedKd CSV downstream.
    surface_diagnostics = compute_surface_diagnostics(
        binned_dataset_sea_surface, sea_surface_height, vertical_res=vertical_res
    )
    sigma_map = {k: v['surface_sigma'] for k, v in surface_diagnostics.items()}

    if apply_histogram_quality_filter:
        subsurface_photon_dataset, adaptive_depth_limits, snr_per_bin = \
            apply_optional_histogram_quality_filter(
                binned_dataset_sea_surface,
                subsurface_photon_dataset,
                sea_surface_height,
                signal_above_noise=signal_above_noise,
            )
    else:
        adaptive_depth_limits = {}
        snr_per_bin = {}

    if apply_surface_sigma_filter:
        subsurface_photon_dataset = apply_optional_surface_sigma_filter(
            binned_dataset_sea_surface,
            subsurface_photon_dataset,
            sea_surface_height,
            sigma_max=surface_sigma_max
        )

    if apply_refraction_correction:
        subsurface_photon_dataset = apply_optional_refraction_correction(
            binned_dataset_sea_surface,
            subsurface_photon_dataset,
            sea_surface_height,
            water_temp_c=refraction_water_temp_c,
            wavelength_nm=refraction_wavelength_nm
        )
        if apply_refraction_rebin:
            # Step 10 (narrow form): put the Beer's-law depth axis into true
            # metres. Without this the corrected photon_height never reaches
            # the fitter, which reads height_bins. lat_bins is untouched, so
            # every per-bin key stays valid; the depth limits from Step 6 are
            # converted alongside it.
            subsurface_photon_dataset = rebin_height_after_refraction(
                subsurface_photon_dataset, vertical_res=vertical_res
            )
            adaptive_depth_limits = rescale_depth_limits_after_refraction(
                subsurface_photon_dataset, adaptive_depth_limits
            )
        if apply_post_refraction_refit:
            sea_surface_height, sea_surface_label, solo_sea_surface_label, subsurface_photon_dataset, \
                _refit_surface_counts = \
                rebuild_and_refit_surface_after_refraction(   # steps 10 & 11
                    binned_dataset_sea_surface,
                    sea_surface_height,
                    water_temp_c=refraction_water_temp_c,
                    wavelength_nm=refraction_wavelength_nm,
                    horizontal_res=horizontal_res,
                    vertical_res=vertical_res,
                )

    # Safety-net depth cutoff: remove extreme deep noise (e.g., -70 m cosmic rays)
    # without clipping real signal OR the noise-floor sampling region needed by
    # the BMA fitter. Set to -50 m (2026-05-14): the previous -20 m starved the
    # adaptive trim (Step 6) of tail bins; the BMA's BIC model selection needs
    # 5+ noise bins past the signal decay to support a two-segment fit. -50 m
    # still removes cosmic-ray photons (-70 m and below) and is well past where
    # even clear ocean (Kd≈0.05) carries useful signal (<0.0002% at 50 m).
    # The fine-grained depth filtering is handled by GEBCO/ATL24 (Step 13) and
    # the adaptive decay zone threshold (Step 16).
    SAFETY_NET_DEPTH = -50.0
    if (not subsurface_photon_dataset.empty) and ('photon_height' in subsurface_photon_dataset.columns):
        subsurface_photon_dataset = subsurface_photon_dataset[
            subsurface_photon_dataset['photon_height'] >= SAFETY_NET_DEPTH
        ].copy()

    if use_atl24_filter:
        atl24_filtered, atl24_match_count = process_seafloor_data_atl24(
            subsurface_photon_dataset,
            atl24_file_path,
            abs(Ignore_Subsurface_Height_Thres),
            max_match_distance_deg=atl24_max_match_distance_deg
        )
        if atl24_match_count > 0:
            filtered_seafloor_subsurface_photon_dataset = atl24_filtered
        elif use_gebco_filter:
            filtered_seafloor_subsurface_photon_dataset = process_seafloor_data(
                subsurface_photon_dataset, GEBCO_paths, abs(Ignore_Subsurface_Height_Thres),
                seafloor_buffer_m=seafloor_buffer_m
            )
        else:
            filtered_seafloor_subsurface_photon_dataset = subsurface_photon_dataset
    elif use_gebco_filter:
        filtered_seafloor_subsurface_photon_dataset = process_seafloor_data(
            subsurface_photon_dataset, GEBCO_paths, abs(Ignore_Subsurface_Height_Thres),
            seafloor_buffer_m=seafloor_buffer_m
        )
    else:
        filtered_seafloor_subsurface_photon_dataset = subsurface_photon_dataset

    if apply_flattening:
        lat_bin_keys = list(
            binned_dataset_sea_surface.groupby(['lat_bins'], observed=False).groups.keys()
        )
        filtered_seafloor_subsurface_photon_dataset = apply_sea_surface_flattening(
            filtered_seafloor_subsurface_photon_dataset,
            sea_surface_height,
            lat_bin_keys,
            horizontal_res=horizontal_res,
            flattening_window_m=flattening_window_m,
            binned_dataset_sea_surface=binned_dataset_sea_surface,
            fine_res_m=flattening_fine_res_m,
        )
    # Attach per-bin surface_sigma as a quality flag column
    filtered_seafloor_subsurface_photon_dataset = filtered_seafloor_subsurface_photon_dataset.copy()
    if sigma_map and 'lat_bins' in filtered_seafloor_subsurface_photon_dataset.columns:
        filtered_seafloor_subsurface_photon_dataset['surface_sigma'] = \
            filtered_seafloor_subsurface_photon_dataset['lat_bins'].map(sigma_map)
    else:
        filtered_seafloor_subsurface_photon_dataset['surface_sigma'] = np.nan

    return sea_surface_height, sea_surface_label, filtered_seafloor_subsurface_photon_dataset, \
        adaptive_depth_limits, snr_per_bin, surface_photon_counts, surface_diagnostics


# Main processing function to apply the subsurface photon filtering beam-by-beam
def process_subsurface_photon_filtering(
    binned_dataset_sea_surface,
    GEBCO_paths,
    subsurface_thresh,
    Ignore_Subsurface_Height_Thres,
    use_atl24_filter=False,
    atl24_file_path='',
    atl24_max_match_distance_deg=0.01,
    use_gebco_filter=False,
    seafloor_buffer_m=0.0,
    apply_histogram_quality_filter=False,
    signal_above_noise=1.5,
    apply_surface_sigma_filter=False,
    surface_sigma_max=0.5,
    apply_refraction_correction=False,
    refraction_water_temp_c=20.0,
    refraction_wavelength_nm=532.0,
    apply_post_refraction_refit=False,
    apply_refraction_rebin=True,
    apply_flattening=False,
    flattening_window_m=500,
    flattening_fine_res_m=50.0,
    surface_floor_m=1.0,
    horizontal_res=500,
    vertical_res=0.25
):
    """
    Beam-by-beam wrapper that calls get_subsurface_photon for every beam and
    concatenates the results.

    Flowchart steps 6–14 are all executed inside this function (delegated to
    get_subsurface_photon per beam). See get_subsurface_photon for the
    step-by-step breakdown.
    """
    # Initialize lists to store results for each beam
    sea_surface_heights = []
    sea_surface_labels = []
    filtered_beam_datasets = []
    all_depth_limits = {}
    all_snr_per_bin = {}
    all_surface_photon_counts = {}
    surface_diagnostics_per_beam = []

    # Group the binned dataset by 'beam_id' and process each group separately
    for beam_id, beam_data in binned_dataset_sea_surface.groupby('beam_id'):
        print(f'Processing subsurface filtering for beam: {beam_id}')

        # Apply get_subsurface_photon to the current beam's dataset
        sea_surface_height, sea_surface_label, filtered_seafloor_subsurface_photon_dataset, \
            beam_depth_limits, beam_snr, beam_surface_counts, beam_surface_diagnostics = \
            get_subsurface_photon(
                beam_data,
                GEBCO_paths,
                subsurface_thresh,
                Ignore_Subsurface_Height_Thres,
                use_atl24_filter=use_atl24_filter,
                atl24_file_path=atl24_file_path,
                atl24_max_match_distance_deg=atl24_max_match_distance_deg,
                use_gebco_filter=use_gebco_filter,
                seafloor_buffer_m=seafloor_buffer_m,
                apply_histogram_quality_filter=apply_histogram_quality_filter,
                signal_above_noise=signal_above_noise,
                apply_surface_sigma_filter=apply_surface_sigma_filter,
                surface_sigma_max=surface_sigma_max,
                apply_refraction_correction=apply_refraction_correction,
                refraction_water_temp_c=refraction_water_temp_c,
                refraction_wavelength_nm=refraction_wavelength_nm,
                apply_post_refraction_refit=apply_post_refraction_refit,
                apply_refraction_rebin=apply_refraction_rebin,
                apply_flattening=apply_flattening,
                flattening_window_m=flattening_window_m,
                flattening_fine_res_m=flattening_fine_res_m,
                surface_floor_m=surface_floor_m,
                horizontal_res=horizontal_res,
                vertical_res=vertical_res
            )

        # Append each result to the lists
        sea_surface_heights.append(sea_surface_height)
        sea_surface_labels.append(sea_surface_label)
        filtered_beam_datasets.append(filtered_seafloor_subsurface_photon_dataset)
        # Key the merged per-bin dicts by (beam_id, lat_bin). lat_bins restart
        # per beam, so plain lat_bin keys collided across beams and the last
        # beam processed silently overwrote the others (2026-10-15 fix; the
        # quality-flag bitmask built from these dicts was identical across
        # beams because of it). Per-beam consumers use
        # quality_flags.per_beam_dict() to recover one beam's view.
        all_depth_limits.update({(beam_id, lb): v for lb, v in beam_depth_limits.items()})
        all_snr_per_bin.update({(beam_id, lb): v for lb, v in beam_snr.items()})
        all_surface_photon_counts.update({(beam_id, lb): v for lb, v in beam_surface_counts.items()})
        surface_diagnostics_per_beam.append(beam_surface_diagnostics)

    # Combine all filtered beam datasets into a single DataFrame
    if filtered_beam_datasets:
        combined_filtered_dataset = pd.concat(filtered_beam_datasets, ignore_index=True)
    else:
        # binned_dataset_sea_surface had no beams (empty input). Preserve schema.
        combined_filtered_dataset = pd.DataFrame(columns=list(binned_dataset_sea_surface.columns) + ['surface_sigma'])

    return sea_surface_heights, sea_surface_labels, combined_filtered_dataset, \
        all_depth_limits, all_snr_per_bin, all_surface_photon_counts, \
        surface_diagnostics_per_beam
