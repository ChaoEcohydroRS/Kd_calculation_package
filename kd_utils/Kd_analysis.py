# utils/Kd_analysis.py
# updated to perform a linear fit in log-space, just like MATLAB's polyfitn(zdepth, y, 1) for a first-order polynomial.

import numpy as np
import pandas as pd
import logging
import warnings
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

# Suppress np.polyfit RankWarning once at module scope; BMA/hybrid hot path
# fires ~1.6M polyfit calls per granule, so per-call warning overhead would
# dominate. Pathological inputs (zero z-variance) are guarded upstream by
# MIN_DEPTH_BINS_FOR_FIT and min_total_range_m gates.
warnings.filterwarnings("ignore", message=".*Polyfit may be poorly conditioned.*")

MIN_DEPTH_BINS_FOR_FIT = 6  # hybrid fitter requires at least this many non-zero bins


def compute_physics_noise_floor(df, horizontal_res, iss_velocity,
                                vertical_res=0.25):
    """Compute expected noise floor per horizontal bin from ATL03 bckgrd_rate.

    Background photons are distributed uniformly across the detector's
    altimetric range window (bckgrd_int_height from ATL03).  The expected
    noise per depth bin is:

        expected_nf = median(bckgrd_rate) * dwell_time * (vertical_res / median(bckgrd_int_height))

    This is a diagnostic/validation value — not used as a fitter constraint.

    Parameters
    ----------
    df : pd.DataFrame
        Subsurface photon dataset with 'lat_bins', 'background_rate',
        and 'bckgrd_int_height'.
    horizontal_res : float
        Horizontal bin size in meters (e.g. 500).
    iss_velocity : float
        ISS ground velocity in m/s (e.g. 7000).
    vertical_res : float
        Depth bin height in meters (default 0.25).

    Returns
    -------
    pd.Series
        Indexed by lat_bins, values are expected noise floor (photon counts
        per depth bin). NaN for bins with missing data.
    """
    if 'background_rate' not in df.columns or 'bckgrd_int_height' not in df.columns:
        return pd.Series(dtype=float)

    dwell_time = horizontal_res / iss_velocity

    grouped = df.groupby('lat_bins', observed=False)
    median_bg = grouped['background_rate'].median()
    median_int_height = grouped['bckgrd_int_height'].median()

    expected_nf = median_bg * dwell_time * (vertical_res / median_int_height)

    return expected_nf


# def log_model(z, kd, e0):
#     return np.log(e0) - kd * z

## This is wrong because the input is already a bined data, 
## it is not necessary to do a histogram again
# def CalculateKdFromFilteredSubsurfacePhoton(df, vertical_res=0.8):
#     if df.empty or 'lat_bins' not in df.columns:
#         return pd.DataFrame({'lat_bins': [np.nan], 'kd': [np.nan], 'e0': [np.nan], 'latitude': [np.nan], 'longitude': [np.nan]})
    
#     # Get the latitude bin value
#     lat_bin_value = df['lat_bins'].iloc[0] if not df['lat_bins'].empty else np.nan
#     latitude = df['latitude'].mean() if 'latitude' in df.columns else df['lat'].mean() if 'lat' in df.columns else np.nan
#     longitude = df['longitude'].mean() if 'longitude' in df.columns else df['lon'].mean() if 'lon' in df.columns else np.nan
    
#     # Calculate photon height range
#     photon_height_min = df['photon_height'].min()
#     photon_height_max = df['photon_height'].max()
    
#     # Check for sufficient data range
#     if np.isnan(photon_height_min) or np.isnan(photon_height_max) or photon_height_min == photon_height_max:
#         return pd.DataFrame({'lat_bins': [lat_bin_value], 'kd': [np.nan], 'e0': [np.nan]})
    
#     height_bins_range = abs(photon_height_max - photon_height_min)    
#     height_bins_number = round(height_bins_range / vertical_res)
    
#     # Ensure there are enough bins
#     if height_bins_number < 5:
#         return pd.DataFrame({'lat_bins': [lat_bin_value], 'kd': [np.nan], 'e0': [np.nan]})
    
#     # Create histogram of photon heights
#     bin_edges = np.linspace(photon_height_min, photon_height_max, num=height_bins_number)
#     counts, _ = np.histogram(df['photon_height'], bins=bin_edges)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
#     # Store histogram data
#     hist_df = pd.DataFrame({'zdepth': bin_centers, 'photon_counts': counts})
    
#     # x value for model
#     # Reverse zdepth to align with the MATLAB approach
#     hist_df['zdepth'] = hist_df['zdepth'].max() - hist_df['zdepth']
    
#     # Log-transform photon counts, replacing zeros with NaN
#     # y value for model
#     hist_df['log_photon_counts'] = np.log(hist_df['photon_counts'].replace(0, np.nan))
#     hist_df.loc[np.isinf(hist_df['log_photon_counts']), 'log_photon_counts'] = np.nan
    
#     # Filter out rows with NaNs in either column for regression
#     valid_data = hist_df.dropna(subset=['zdepth', 'log_photon_counts'])
    
#     # Check for enough valid data points
#     # Skip the regression if there are fewer than 5 datapoints
#     if valid_data['log_photon_counts'].notna().sum() > 3:
#         # Drop NaNs for regression
#         zdepth_valid = valid_data['zdepth'].values.reshape(-1, 1)
#         log_counts_valid = valid_data['log_photon_counts'].values

#         # Perform linear regression
#         model = LinearRegression()
#         model.fit(zdepth_valid, log_counts_valid)
        
#         # Extract kd as the negative of the slope and e0 from the intercept
#         kd = -model.coef_[0]
        
#         print('zdepth_valid:',zdepth_valid)
#         print('photon_counts:',hist_df['photon_counts'])
#         print('kd:',kd)
        
#         e0 = np.exp(model.intercept_)
        
#         # Set kd to NaN if negative
#         if kd < 0:
#             kd = np.nan
#     else:
#         kd, e0 = np.nan, np.nan
    
#     return pd.DataFrame({
#         'lat_bins': [lat_bin_value],
#         'kd': [kd],
#         'e0': [e0],
#         'latitude': [latitude],
#         'longitude': [longitude]
#     })


# # one solution is to adjust the vertical_res to 0.25
# def CalculateKdFromFilteredSubsurfacePhoton(df, vertical_res=0.25):
#     if df.empty or 'lat_bins' not in df.columns:
#         return pd.DataFrame({'lat_bins': [np.nan], 'kd': [np.nan], 'e0': [np.nan], 'latitude': [np.nan], 'longitude': [np.nan]})
    
#     # Get the latitude bin value
#     lat_bin_value = df['lat_bins'].iloc[0] if not df['lat_bins'].empty else np.nan
#     latitude = df['latitude'].mean() if 'latitude' in df.columns else df['lat'].mean() if 'lat' in df.columns else np.nan
#     longitude = df['longitude'].mean() if 'longitude' in df.columns else df['lon'].mean() if 'lon' in df.columns else np.nan
    
#     # Calculate photon height range
#     photon_height_min = df['photon_height'].min()
#     photon_height_max = df['photon_height'].max()
    
#     # Check for sufficient data range
#     if np.isnan(photon_height_min) or np.isnan(photon_height_max) or photon_height_min == photon_height_max:
#         return pd.DataFrame({'lat_bins': [lat_bin_value], 'kd': [np.nan], 'e0': [np.nan]})
    
#     height_bins_range = abs(photon_height_max - photon_height_min)    
#     height_bins_number = round(height_bins_range / vertical_res)
    
#     # Ensure there are enough bins
#     if height_bins_number < 5:
#         return pd.DataFrame({'lat_bins': [lat_bin_value], 'kd': [np.nan], 'e0': [np.nan]})
    
#     # Create histogram of photon heights
#     bin_edges = np.linspace(photon_height_min, photon_height_max, num=height_bins_number)
#     counts, _ = np.histogram(df['photon_height'], bins=bin_edges)
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
#     # Store histogram data
#     hist_df = pd.DataFrame({'zdepth': bin_centers, 'photon_counts': counts})
    
#     # x value for model
#     # Reverse zdepth to align with the MATLAB approach
#     hist_df['zdepth'] = hist_df['zdepth'].max() - hist_df['zdepth']
    
#     # Log-transform photon counts, replacing zeros with NaN
#     # y value for model
#     hist_df['log_photon_counts'] = np.log(hist_df['photon_counts'].replace(0, np.nan))
#     hist_df.loc[np.isinf(hist_df['log_photon_counts']), 'log_photon_counts'] = np.nan
    
#     # Filter out rows with NaNs in either column for regression
#     valid_data = hist_df.dropna(subset=['zdepth', 'log_photon_counts'])
    
#     # Check for enough valid data points
#     # Skip the regression if there are fewer than 5 datapoints
#     if valid_data['log_photon_counts'].notna().sum() > 3:
#         # Drop NaNs for regression
#         zdepth_valid = valid_data['zdepth'].values.reshape(-1, 1)
#         log_counts_valid = valid_data['log_photon_counts'].values

#         # Perform linear regression
#         model = LinearRegression()
#         model.fit(zdepth_valid, log_counts_valid)
        
#         # Extract kd as the negative of the slope and e0 from the intercept
#         kd = -model.coef_[0]
        
#         print('zdepth_valid:',zdepth_valid)
#         print('photon_counts:',hist_df['photon_counts'])
#         print('kd:',kd)
        
#         e0 = np.exp(model.intercept_)
        
#         # Set kd to NaN if negative
#         if kd < 0:
#             kd = np.nan
#     else:
#         kd, e0 = np.nan, np.nan
    
#     return pd.DataFrame({
#         'lat_bins': [lat_bin_value],
#         'kd': [kd],
#         'e0': [e0],
#         'latitude': [latitude],
#         'longitude': [longitude]
#     })


# Photons per bin below which a depth bin is treated as noise rather than signal
# when measuring the vertical extent available to the fit. A small absolute
# floor, not a background model: it only has to exclude isolated stray photons.
FIT_SPAN_MIN_COUNT = 3


def signal_depth_span(hist_df, min_count=FIT_SPAN_MIN_COUNT):
    """Vertical extent (m) over which the histogram actually carries signal.

    This is the range the decay rate is measured across. Measured on Bohai gt3l,
    every bin returning Kd > 1.0 m^-1 spanned a median of 1.0 m with 5 bins,
    against 4.25 m and 9 bins for bins returning Kd < 0.3 -- i.e. the extreme
    values were slopes extrapolated from a depth range barely wider than the bin
    size, not measurements. --min_fit_span_m rejects those; see
    _fit_kd_with_method.

    Returns np.nan when fewer than two bins clear ``min_count``.
    """
    if hist_df is None or 'zdepth' not in hist_df or 'photon_counts' not in hist_df:
        return np.nan
    z = np.asarray(hist_df['zdepth'], float)
    c = np.asarray(hist_df['photon_counts'], float)
    sig = np.isfinite(z) & np.isfinite(c) & (c >= min_count)
    if sig.sum() < 2:
        return np.nan
    return float(z[sig].max() - z[sig].min())


def find_exponential_decay_zone(hist_df, decay_threshold=0.0):
    """
    Identify the depth bins that lie within the exponential decay zone.

    Flowchart step: 16 — Determine depth where signal decays to 1% of incoming
    signal or where photon decay is no longer exponential. Bins with zero photon
    counts (log undefined) are excluded; the remaining contiguous non-zero bins
    define the exponential decay zone.

    Parameters
    ----------
    hist_df : pd.DataFrame
        Columns: 'zdepth' (depth from surface, ascending), 'photon_counts'.
    decay_threshold : float, optional
        Fraction of peak photon count below which bins are excluded.
        0.0 (default) preserves original behaviour (only zero-count bins removed).
        E.g. 0.01 removes bins with < 1% of the peak signal.

    Returns
    -------
    pd.DataFrame
        Filtered rows with a 'log_photon_counts' column added; only bins inside
        the exponential decay zone are kept (photon_counts > 0 and log finite).
    """
    df = hist_df.copy()

    # Apply decay threshold: exclude bins below threshold fraction of peak signal
    if decay_threshold > 0.0:
        peak_count = df['photon_counts'].max()
        if peak_count > 0:
            df.loc[df['photon_counts'] < decay_threshold * peak_count, 'photon_counts'] = 0

    df['log_photon_counts'] = np.log(df['photon_counts'].replace(0, np.nan))
    df.loc[np.isinf(df['log_photon_counts']), 'log_photon_counts'] = np.nan
    return df.dropna(subset=['zdepth', 'log_photon_counts'])


def fit_beers_law(valid_data):
    """
    Fit Beer's Law exponential decay curve to the identified decay zone.

    Flowchart step: 17 — Fit exponential decay curve (Beer's Law) to data
    within the zone of exponential decay. Implemented as linear regression on
    log(photon_counts) vs depth; kd = -slope, e0 = exp(intercept).

    Parameters
    ----------
    valid_data : pd.DataFrame
        Output of find_exponential_decay_zone; columns 'zdepth' and
        'log_photon_counts'.

    Returns
    -------
    tuple[float, float]
        (kd, e0). kd is set to np.nan if negative or if fewer than 4 points.
    """
    if valid_data['log_photon_counts'].notna().sum() <= 3:
        return np.nan, np.nan

    zdepth_valid = valid_data['zdepth'].values.reshape(-1, 1)
    log_counts_valid = valid_data['log_photon_counts'].values

    model = LinearRegression()
    model.fit(zdepth_valid, log_counts_valid)

    kd = -model.coef_[0]
    e0 = np.exp(model.intercept_)

    if kd < 0:
        kd = np.nan
    return kd, e0


# ---------------------------------------------------------------------------
#  Strategy A: Background subtraction + log-linear fit
# ---------------------------------------------------------------------------
def fit_beers_law_bg_subtract(hist_df, bg_fraction=0.2):
    """
    Estimate the noise floor from the deepest bins, subtract it, then fit
    Beer's Law in log-space on the noise-subtracted counts.

    Parameters
    ----------
    hist_df : pd.DataFrame
        Columns: 'zdepth' (ascending from surface), 'photon_counts'.
    bg_fraction : float
        Fraction of the deepest bins used to estimate the background level.

    Returns
    -------
    tuple[float, float, float]
        (kd, e0, noise_floor).
    """
    df = hist_df.copy().sort_values('zdepth')
    n_bins = len(df)
    if n_bins < 5:
        return np.nan, np.nan, np.nan

    # Estimate noise floor from the deepest bg_fraction of bins
    n_bg = max(int(n_bins * bg_fraction), 2)
    noise_floor = float(df.tail(n_bg)['photon_counts'].median())

    # Subtract noise and keep only positive residuals
    df['signal'] = df['photon_counts'] - noise_floor
    df = df[df['signal'] > 0].copy()
    if len(df) < 4:
        return np.nan, np.nan, noise_floor

    df['log_signal'] = np.log(df['signal'])

    zdepth = df['zdepth'].values.reshape(-1, 1)
    log_signal = df['log_signal'].values

    model = LinearRegression()
    model.fit(zdepth, log_signal)

    kd = -model.coef_[0]
    e0 = np.exp(model.intercept_)

    if kd < 0:
        kd = np.nan
    return kd, e0, noise_floor


# ---------------------------------------------------------------------------
#  Strategy B: Breakpoint / segmented regression
# ---------------------------------------------------------------------------
def fit_beers_law_breakpoint(hist_df, min_breakpoint_depth=2.0,
                             min_decay_range_m=5.0,
                             min_decay_photons=30,
                             min_decay_bins=8,
                             min_total_photons=30,
                             min_total_range_m=5.0):
    """
    Find the optimal breakpoint between exponential decay and noise floor,
    then fit Beer's Law only to the decay segment.

    The breakpoint is chosen by testing every candidate position and selecting
    the one that minimises total residual sum of squares of a two-segment
    model: linear slope above the breakpoint, flat constant below.

    Parameters
    ----------
    hist_df : pd.DataFrame
        Columns: 'zdepth' (ascending from surface), 'photon_counts'.
    min_breakpoint_depth : float
        Minimum depth (m) below surface for a valid breakpoint. Default 2.0 m.
    min_decay_range_m : float
        Minimum span (m) covered by the decay segment. Prevents short noisy
        decay fits in low-signal bins. Default 5.0 m.
    min_decay_photons : int
        Minimum number of photons that must lie within the decay segment.
        Default 30.
    min_decay_bins : int
        Minimum number of non-empty histogram bins in the decay segment.
        Default 8.
    min_total_photons : int
        Minimum total photons in the bin for the fit to be attempted at all.
        Default 30.
    min_total_range_m : float
        Minimum total histogram span (m) for the fit to be attempted at all.
        Default 5.0 m.

    Returns
    -------
    tuple[float, float, float, float, str or None]
        (kd, e0, breakpoint_depth, noise_floor, failure_reason).
        failure_reason is None on success, otherwise a short string
        explaining why Kd is NaN (e.g. 'too_few_bins', 'insufficient_photons').
    """
    df = hist_df.copy().sort_values('zdepth')
    df = df[df['photon_counts'] > 0].copy()
    if len(df) < 5:
        return np.nan, np.nan, np.nan, np.nan, 'too_few_bins'

    # NEW: reject the entire bin if insufficient data for any meaningful fit.
    total_photons = int(df['photon_counts'].sum())
    total_range = df['zdepth'].max() - df['zdepth'].min()
    if total_photons < min_total_photons:
        return np.nan, np.nan, np.nan, np.nan, 'insufficient_photons'
    if total_range < min_total_range_m:
        return np.nan, np.nan, np.nan, np.nan, 'insufficient_range'

    df['log_counts'] = np.log(df['photon_counts'])
    depths = df['zdepth'].values
    log_counts = df['log_counts'].values
    counts = df['photon_counts'].values
    n = len(depths)

    best_rss = np.inf
    best_bp_idx = None

    # Try each candidate breakpoint (need >= min_decay_bins in decay segment,
    # >= 1 in noise segment)
    for bp_idx in range(max(4, min_decay_bins), n - 1):
        if depths[bp_idx] < min_breakpoint_depth:
            continue

        # NEW: decay zone must span at least min_decay_range_m meters
        decay_range = depths[bp_idx - 1] - depths[0]
        if decay_range < min_decay_range_m:
            continue

        # NEW: decay zone must contain at least min_decay_photons photons
        decay_photons = int(counts[:bp_idx].sum())
        if decay_photons < min_decay_photons:
            continue

        # Decay segment: linear fit on bins 0..bp_idx-1
        z_decay = depths[:bp_idx].reshape(-1, 1)
        lc_decay = log_counts[:bp_idx]
        model = LinearRegression()
        model.fit(z_decay, lc_decay)
        rss_decay = float(np.sum((model.predict(z_decay) - lc_decay) ** 2))

        # Noise segment: flat at the mean of bins bp_idx..end
        lc_noise = log_counts[bp_idx:]
        noise_mean = lc_noise.mean()
        rss_noise = float(np.sum((lc_noise - noise_mean) ** 2))

        total_rss = rss_decay + rss_noise
        if total_rss < best_rss:
            best_rss = total_rss
            best_bp_idx = bp_idx

    if best_bp_idx is None:
        return np.nan, np.nan, np.nan, np.nan, 'no_breakpoint_found'

    # Final fit on the decay segment
    z_decay = depths[:best_bp_idx].reshape(-1, 1)
    lc_decay = log_counts[:best_bp_idx]
    model = LinearRegression()
    model.fit(z_decay, lc_decay)

    kd = -model.coef_[0]
    e0 = np.exp(model.intercept_)
    breakpoint_depth = float(depths[best_bp_idx])
    noise_floor = float(np.exp(log_counts[best_bp_idx:].mean()))

    if kd < 0:
        return np.nan, np.nan, np.nan, np.nan, 'negative_slope'
    return kd, e0, breakpoint_depth, noise_floor, None


# ---------------------------------------------------------------------------
#  Strategy C: Nonlinear fit  C(z) = A * exp(-Kd * z) + N
# ---------------------------------------------------------------------------
def _beer_plus_noise(z, A, kd, N):
    """Model: signal = A * exp(-kd * z) + N."""
    return A * np.exp(-kd * z) + N


def fit_beers_law_nonlinear(hist_df):
    """
    Fit the physically correct model C(z) = A·exp(-Kd·z) + N directly
    to raw photon counts using nonlinear least-squares (no log transform).

    Two-stage approach for robust convergence:
      Stage 1: Run bg_subtract to get reliable (Kd, A, N) estimates.
               The log-linear fit on noise-subtracted counts gives a good
               Kd, and the noise floor from deep bins gives N.
      Stage 2: Use Stage 1 results as initial guesses for the nonlinear fit,
               with N constrained to [0, 3 * N_stage1].  This refines the
               physically correct model starting from a good solution.

    Parameters
    ----------
    hist_df : pd.DataFrame
        Columns: 'zdepth' (ascending from surface), 'photon_counts'.

    Returns
    -------
    tuple[float, float, float]
        (kd, e0, noise_floor).
    """
    df = hist_df.copy().sort_values('zdepth')
    depths = df['zdepth'].values
    counts = df['photon_counts'].values.astype(float)

    if len(depths) < 5:
        return np.nan, np.nan, np.nan

    # Stage 1: bg_subtract for Kd and N estimates
    kd_init, _, N_init = fit_beers_law_bg_subtract(hist_df)

    if np.isnan(N_init) or N_init < 0:
        N_init = float(np.median(counts[len(counts) * 3 // 4:]))
    if np.isnan(kd_init) or kd_init <= 0:
        kd_init = 0.1

    # A0 from actual data: surface count minus noise, extrapolated to z=0
    # A = (counts[0] - N) / exp(-Kd * z[0])
    A_init = max(float(counts[0] - N_init), 1.0) / max(np.exp(-kd_init * depths[0]), 1e-6)

    N_upper = max(N_init * 3.0, 1.0)

    # Stage 2: nonlinear refinement
    try:
        popt, _ = curve_fit(
            _beer_plus_noise, depths, counts,
            p0=[A_init, kd_init, N_init],
            bounds=([0, 0.001, 0], [np.inf, np.inf, N_upper]),
            maxfev=5000
        )
        A_fit, kd_fit, N_fit = popt
        if kd_fit <= 0:
            kd_fit = np.nan
        return float(kd_fit), float(A_fit), float(N_fit)
    except (RuntimeError, ValueError):
        # If nonlinear fails, return bg_subtract result
        return kd_init, A_init, N_init


# ---------------------------------------------------------------------------
#  Hybrid: Stabilised breakpoint zone detection + log-linear fit on decay only
# ---------------------------------------------------------------------------
def fit_beers_law_hybrid(hist_df, min_breakpoint_depth=2.0,
                         expected_noise_floor=None,
                         noise_floor_tolerance=3.0,
                         min_decay_range_m=5.0,
                         min_decay_photons=30,
                         min_decay_bins=8,
                         min_total_photons=30,
                         min_total_range_m=5.0):
    """
    Two-stage approach matching the flowchart intent:
      Step 16 — Find where exponential decay transitions to noise floor
                (stabilised breakpoint with BIC selection).
      Step 17 — Fit Beer's Law log-linear ONLY on the decay segment.

    Stabilisation constraints:
      * Minimum breakpoint depth prevents shallow breakpoints that yield
        unstable regression from too few decay bins.
      * Decay segment slope must be negative (Kd > 0).
      * BIC model selection penalises over-fitting, preventing the breakpoint
        from drifting too deep (where the decay segment gets long and noisy).
      * When expected_noise_floor is provided (from beam-level median),
        candidate breakpoints whose noise segment deviates too far from the
        expected value are penalised.
      * Falls back to full log-linear if no valid breakpoint improves BIC
        over a single-line model.
      * NEW: whole-bin rejection when insufficient total photons or total
        range exist to support any meaningful fit.
      * NEW: decay-zone constraints (minimum range, minimum photons, minimum
        non-empty bins) prevent spurious high-Kd fits in low-signal bins.

    Parameters
    ----------
    hist_df : pd.DataFrame
        Columns: 'zdepth' (ascending from surface), 'photon_counts'.
    min_breakpoint_depth : float
        Minimum depth (m) below surface for a valid breakpoint.  Prevents
        breakpoints landing in the first few bins where regression is
        unstable.  Default 2.0 m.
    expected_noise_floor : float or None
        If provided, the beam-level median noise floor (photon counts).
        Candidate breakpoints whose noise segment mean deviates by more
        than ``noise_floor_tolerance`` times from the expected value
        receive a BIC penalty.
    noise_floor_tolerance : float
        Factor controlling how far the candidate noise segment mean may
        deviate from ``expected_noise_floor`` before a penalty is applied.
        Default 3.0 (allow 3x variation).
    min_decay_range_m : float
        Minimum span (m) that the decay segment must cover. Prevents
        short decay fits dominated by noise. Default 5.0 m.
    min_decay_photons : int
        Minimum number of photons that must lie within the decay segment.
        Default 30.
    min_decay_bins : int
        Minimum number of non-empty histogram bins in the decay segment.
        Default 8.
    min_total_photons : int
        Minimum total photons in the bin for the fit to be attempted at all.
        Bins with fewer photons are rejected outright. Default 30.
    min_total_range_m : float
        Minimum total histogram span (m) for the fit to be attempted at all.
        Bins with a narrower span are rejected outright. Default 5.0 m.

    Returns
    -------
    tuple[float, float, float, float, str or None]
        (kd, e0, breakpoint_depth, noise_floor, failure_reason).
        breakpoint_depth and noise_floor are np.nan if fallback to full fit.
        failure_reason is None on success, otherwise a short string
        explaining why Kd is NaN or the fit fell back
        (e.g. 'too_few_bins', 'insufficient_photons',
        'no_breakpoint_fallback', 'no_breakpoint_found').
    """
    df = hist_df.copy().sort_values('zdepth')
    df = df[df['photon_counts'] > 0].copy()
    n = len(df)
    if n < 6:
        return np.nan, np.nan, np.nan, np.nan, 'too_few_bins'

    # NEW: Reject the entire bin if insufficient data for any meaningful fit.
    # Returns (nan, nan, nan, nan, reason_string) — reason is used by quality
    # flag system to explain why Kd is NaN.
    total_photons = int(df['photon_counts'].sum())
    total_range = df['zdepth'].max() - df['zdepth'].min()
    if total_photons < min_total_photons:
        return np.nan, np.nan, np.nan, np.nan, 'insufficient_photons'
    if total_range < min_total_range_m:
        return np.nan, np.nan, np.nan, np.nan, 'insufficient_range'

    df['log_counts'] = np.log(df['photon_counts'])
    depths = df['zdepth'].values
    log_counts = df['log_counts'].values
    counts = df['photon_counts'].values

    # ------------------------------------------------------------------
    # Reference: BIC for a single-line model (no breakpoint)
    # ------------------------------------------------------------------
    slope_full, intercept_full = np.polyfit(depths, log_counts, 1)
    pred_full = slope_full * depths + intercept_full
    rss_full = float(np.sum((pred_full - log_counts) ** 2))
    k_full = 2  # slope + intercept
    bic_full = n * np.log(rss_full / n + 1e-10) + k_full * np.log(n)

    # ------------------------------------------------------------------
    # Search: best breakpoint using BIC
    # ------------------------------------------------------------------
    best_bic = bic_full  # must beat the single-line model
    best_bp_idx = None
    best_slope = None
    best_intercept = None

    for bp_idx in range(max(4, min_decay_bins), n - 1):
        # -- Minimum breakpoint depth constraint --
        if depths[bp_idx] < min_breakpoint_depth:
            continue

        # NEW: decay zone must span at least min_decay_range_m meters
        decay_range = depths[bp_idx - 1] - depths[0]
        if decay_range < min_decay_range_m:
            continue

        # NEW: decay zone must contain at least min_decay_photons photons
        decay_photons = int(counts[:bp_idx].sum())
        if decay_photons < min_decay_photons:
            continue

        # -- Decay segment (surface to breakpoint) --
        z_decay = depths[:bp_idx]
        lc_decay = log_counts[:bp_idx]
        slope, intercept = np.polyfit(z_decay, lc_decay, 1)

        # Physical constraint: slope must be negative
        if slope >= 0:
            continue

        pred_decay = slope * z_decay + intercept
        rss_decay = float(np.sum((pred_decay - lc_decay) ** 2))

        # -- Noise segment (breakpoint to bottom) --
        lc_noise = log_counts[bp_idx:]
        noise_mean = float(lc_noise.mean())
        rss_noise = float(np.sum((lc_noise - noise_mean) ** 2))

        # -- BIC for piecewise model. 4 params: slope, intercept, noise
        # level, AND the breakpoint location itself — bp is estimated by
        # searching every candidate position, so it consumes a degree of
        # freedom like any fitted parameter. Counting k=3 under-penalised
        # the piecewise model by log(n), letting marginal two-segment fits
        # beat the single line too easily on low-SNR bins.
        rss_total = rss_decay + rss_noise
        k_bp = 4
        bic_bp = n * np.log(rss_total / n + 1e-10) + k_bp * np.log(n)

        # -- Noise floor consistency penalty --
        # When we have a beam-level expected noise floor, penalise
        # candidates whose noise segment deviates substantially.
        if expected_noise_floor is not None and expected_noise_floor > 0:
            candidate_nf = float(np.exp(noise_mean))
            ratio = candidate_nf / expected_noise_floor
            if ratio > noise_floor_tolerance or ratio < 1.0 / noise_floor_tolerance:
                # Add a penalty proportional to log-deviation
                bic_bp += n * abs(np.log(ratio))

        if bic_bp < best_bic:
            best_bic = bic_bp
            best_bp_idx = bp_idx
            best_slope = slope
            best_intercept = intercept

    # ------------------------------------------------------------------
    # Result
    # ------------------------------------------------------------------
    if best_bp_idx is None:
        # No breakpoint beats the single-line model — fall back
        kd = -slope_full
        e0 = np.exp(intercept_full)
        if kd < 0:
            # Full fit also produces a non-physical slope; fitter cannot
            # recover a Kd estimate from this bin.
            return np.nan, np.nan, np.nan, np.nan, 'no_breakpoint_found'
        return kd, e0, np.nan, np.nan, 'no_breakpoint_fallback'

    kd = -best_slope
    e0 = np.exp(best_intercept)
    breakpoint_depth = float(depths[best_bp_idx])
    noise_floor = float(np.exp(log_counts[best_bp_idx:].mean()))

    if kd < 0:
        return np.nan, np.nan, np.nan, np.nan, 'no_breakpoint_found'
    return kd, e0, breakpoint_depth, noise_floor, None


def fit_beers_law_bma(hist_df, min_breakpoint_depth=2.0,
                      expected_noise_floor=None,
                      noise_floor_tolerance=3.0,
                      min_decay_range_m=5.0,
                      min_decay_photons=30,
                      min_decay_bins=8,
                      min_total_photons=30,
                      min_total_range_m=5.0,
                      min_total_range_m_floor=2.5,
                      no_noise_floor_threshold=0.95,
                      min_noise_floor_photons_per_bin=0.0,
                      bkg_hist=None,
                      horizontal_res=None,
                      poisson_weighted=False,
                      return_candidates=False):
    """
    Bayesian Model-Averaged Kd: weighted average across all candidate breakpoints.

    Instead of picking the single best BIC breakpoint (which is unstable),
    compute Kd for every candidate and weight by BIC probability:
        w_i = exp(-0.5 * (BIC_i - BIC_min))
        kd = sum(w_i * kd_i) / sum(w_i)

    When BIC is confident (one clear winner), gives same result as hybrid.
    When BIC is ambiguous, gives a smoothed, more stable Kd.

    Range gates are decoupled into two thresholds so that shallow histograms
    with no flat noise floor (the typical post-GEBCO/post-ATL24 near-shore
    case) can still recover Kd:
      * total_range < min_total_range_m_floor → hard reject as
        'insufficient_range'.
      * min_total_range_m_floor ≤ total_range < min_total_range_m → fit only
        the no-breakpoint single-line model; piecewise candidates are skipped.
      * total_range ≥ min_total_range_m → run the full BMA over piecewise
        candidates plus the no-breakpoint baseline.

    Pure-decay (no noise floor) detection: after BMA weights are normalised,
    if the no-breakpoint candidate's weight exceeds no_noise_floor_threshold,
    there is no statistical evidence for a flat-tail segment. The fitter
    collapses to the single-line solution and returns
    (kd, e0, NaN, NaN, 'no_noise_floor') — analogous to the existing
    'no_breakpoint_fallback' path in fit_beers_law_hybrid.

    Parameters
    ----------
    hist_df : pd.DataFrame
        Columns: 'zdepth' (ascending from surface), 'photon_counts'.
    min_breakpoint_depth : float
        Minimum depth (m) below surface for a valid breakpoint. Default 2.0 m.
    expected_noise_floor : float or None
        If provided, the beam-level median noise floor (photon counts).
        Candidate breakpoints whose noise segment mean deviates by more
        than ``noise_floor_tolerance`` times from the expected value
        receive a BIC penalty.
    noise_floor_tolerance : float
        Factor controlling how far the candidate noise segment mean may
        deviate from ``expected_noise_floor`` before a penalty is applied.
        Default 3.0.
    min_decay_range_m : float
        Minimum span (m) that the decay segment must cover. Default 5.0 m.
    min_decay_photons : int
        Minimum number of photons in the decay segment. Default 30.
    min_decay_bins : int
        Minimum number of non-empty histogram bins in the decay segment.
        Default 8.
    min_total_photons : int
        Minimum total photons for the fit to be attempted. Default 30.
    min_total_range_m : float
        Total histogram span (m) below which piecewise candidates are
        skipped (the fitter reduces to a single-line fit). Default 5.0.
    min_total_range_m_floor : float
        Hard rejection threshold; below this the bin is dropped as
        'insufficient_range'. Default 2.5 m.
    no_noise_floor_threshold : float
        BMA weight on the no-breakpoint candidate above which the fitter
        reports 'no_noise_floor' and returns NaN for breakpoint and noise
        floor. Default 0.95.
    poisson_weighted : bool
        Opt-in (default False, preserving legacy behaviour). When True,
        all log-space fits use Poisson weights: var(log c) ~ 1/c, so
        residuals are weighted by sqrt(c) (np.polyfit's ``w`` is 1/sigma)
        and the BIC RSS terms become sum(c * resid^2) for BOTH the
        single-line and piecewise models, keeping their BIC comparable.
        Unweighted fits give a 1-photon tail bin the same vote as a
        100-photon near-surface bin, letting the sparse tail dominate the
        slope — one driver of the low-SNR decay-vs-flat instability. The
        2-sigma spurious-tail separation check intentionally stays in
        unweighted units so its gate semantics are unchanged.

    Returns
    -------
    tuple[float, float, float, float, str or None]
        (kd, e0, breakpoint_avg, noise_floor_avg, failure_reason).
        Same signature as fit_beers_law_hybrid for drop-in replacement.
        failure_reason values: None on success; 'too_few_bins',
        'insufficient_photons', 'insufficient_range', 'no_valid_candidates',
        'negative_kd', or 'no_noise_floor' on a soft-fail collapse where
        kd/e0 are still recovered from the single-line fit.
    """
    df = hist_df.copy().sort_values('zdepth')
    df = df[df['photon_counts'] > 0].copy()
    n = len(df)
    if n < MIN_DEPTH_BINS_FOR_FIT:
        return np.nan, np.nan, np.nan, np.nan, 'too_few_bins'

    # Effective Poisson-floor threshold. Gate is OFF when
    # min_noise_floor_photons_per_bin <= 0 (default), even if bkg_hist or
    # horizontal_res are provided — those only scale an already-enabled gate.
    # TwoDayNightTime diagnostic scripts opt in by passing
    # min_noise_floor_photons_per_bin > 0.
    if min_noise_floor_photons_per_bin <= 0:
        eff_min_nf = 0.0
    elif bkg_hist is not None and bkg_hist > 0:
        eff_min_nf = max(float(min_noise_floor_photons_per_bin),
                         float(bkg_hist) + 2.0 * float(np.sqrt(bkg_hist)))
    elif horizontal_res is not None and horizontal_res > 0:
        eff_min_nf = float(min_noise_floor_photons_per_bin) * (float(horizontal_res) / 500.0)
    else:
        eff_min_nf = float(min_noise_floor_photons_per_bin)

    # Whole-bin rejection vs. piecewise eligibility — see docstring.
    total_photons = int(df['photon_counts'].sum())
    total_range = df['zdepth'].max() - df['zdepth'].min()
    if total_photons < min_total_photons:
        return np.nan, np.nan, np.nan, np.nan, 'insufficient_photons'
    if total_range < min_total_range_m_floor:
        return np.nan, np.nan, np.nan, np.nan, 'insufficient_range'
    allow_piecewise = total_range >= min_total_range_m

    df['log_counts'] = np.log(df['photon_counts'])
    depths = df['zdepth'].values
    log_counts = df['log_counts'].values
    counts = df['photon_counts'].values

    # Optional Poisson weights for the log-space fits: var(log c) ~ 1/c,
    # and np.polyfit's ``w`` multiplies residuals (w = 1/sigma = sqrt(c)).
    # None -> legacy unweighted behaviour.
    w_fit = np.sqrt(counts) if poisson_weighted else None

    def _rss(pred, obs, wts):
        resid = pred - obs
        if wts is not None:
            return float(np.sum((wts * resid) ** 2))
        return float(np.sum(resid ** 2))

    # ------------------------------------------------------------------
    # Full single-line model (no breakpoint) as baseline candidate
    # ------------------------------------------------------------------
    slope_full, intercept_full = np.polyfit(depths, log_counts, 1, w=w_fit)
    pred_full = slope_full * depths + intercept_full
    rss_full = _rss(pred_full, log_counts, w_fit)
    k_full = 2  # slope + intercept
    bic_full = n * np.log(rss_full / n + 1e-10) + k_full * np.log(n)
    kd_full = -slope_full
    e0_full = np.exp(intercept_full)

    candidates = []

    # Include the full-line model as a candidate (no breakpoint)
    if kd_full > 0:
        candidates.append({
            'bic': bic_full, 'kd': kd_full, 'e0': e0_full,
            'bp': np.nan, 'nf': np.nan,
        })

    # ------------------------------------------------------------------
    # Breakpoint candidates: same search as hybrid (gated by allow_piecewise)
    # ------------------------------------------------------------------
    if allow_piecewise:
        # Noise segment must be at least 4 bins (1 m at 0.25 m vertical_res)
        # to support a statistically meaningful "flat tail" claim. Without
        # this, bp_idx near n-1 lets the BIC overfit 1-2 tail bins as a
        # spurious flat segment.
        min_noise_bins = 4
        for bp_idx in range(max(4, min_decay_bins), n - min_noise_bins + 1):
            if depths[bp_idx] < min_breakpoint_depth:
                continue

            # Decay zone must span at least min_decay_range_m meters
            decay_range = depths[bp_idx - 1] - depths[0]
            if decay_range < min_decay_range_m:
                continue

            # Decay zone must contain at least min_decay_photons photons
            decay_photons = int(counts[:bp_idx].sum())
            if decay_photons < min_decay_photons:
                continue

            # Fit linear decay on surface-to-breakpoint segment
            z_decay = depths[:bp_idx]
            lc_decay = log_counts[:bp_idx]
            w_decay = w_fit[:bp_idx] if w_fit is not None else None
            slope, intercept = np.polyfit(z_decay, lc_decay, 1, w=w_decay)

            # Physical constraint: slope must be negative (Kd > 0)
            if slope >= 0:
                continue

            pred_decay = slope * z_decay + intercept
            rss_decay = _rss(pred_decay, lc_decay, w_decay)
            # Unweighted decay RSS for the 2-sigma spurious-tail check
            # below — keeps the gate in log-count units regardless of
            # the BIC weighting.
            rss_decay_unw = float(np.sum((pred_decay - lc_decay) ** 2))

            # Noise segment (breakpoint to bottom)
            lc_noise = log_counts[bp_idx:]
            if w_fit is not None:
                noise_mean = float(np.average(lc_noise,
                                              weights=counts[bp_idx:]))
            else:
                noise_mean = float(lc_noise.mean())
            w_noise = w_fit[bp_idx:] if w_fit is not None else None
            rss_noise = _rss(np.full(len(lc_noise), noise_mean),
                             lc_noise, w_noise)

            # Reject "spurious flat tail" candidates: a real noise floor sits
            # off the decay's extrapolated trajectory (the floor flattens
            # either above or below where the decay line would predict —
            # depending on whether the floor is reached after the decay
            # passes through it or still ahead of it). For pure decay,
            # noise_mean lies on the extrapolated decay line, so the
            # absolute deviation is consistent with Poisson scatter. Require
            # 2 stds of *absolute* separation to distinguish a real flat
            # floor from a tail that is just decay continuing. This catches
            # the post-GEBCO/post-ATL24 near-shore regime (seabed stripped,
            # only decay remains -> floor lies on the decay line -> rejected)
            # without disrupting either flavour of true flat-floor data.
            z_noise = depths[bp_idx:]
            extrapolated_log_mean = float((slope * z_noise + intercept).mean())
            decay_dof = max(1, bp_idx - 2)
            decay_resid_std = float(np.sqrt(rss_decay_unw / decay_dof))
            if abs(extrapolated_log_mean - noise_mean) < 2.0 * decay_resid_std:
                continue  # tail is consistent with decay continuing, not a floor

            # Poisson-floor distinguishability gate (disabled by default).
            # When eff_min_nf > 0, reject piecewise candidates whose noise
            # plateau mean (np.exp(noise_mean)) falls below the threshold —
            # used by TwoDayNightTime diagnostic scripts to suppress BMA
            # branch-flip jumps where fake plateaus at nf~1.2 photons/bin
            # pass the 2σ separation check above. Production pipeline keeps
            # eff_min_nf=0 to preserve legacy BMA behavior on coastal data
            # where real noise plateaus naturally sit at 1-2 photons/bin.
            if eff_min_nf > 0 and float(np.exp(noise_mean)) < eff_min_nf:
                continue

            # BIC for piecewise model. 4 params: slope, intercept, noise
            # level, AND the breakpoint location (estimated by candidate
            # search — see the k_bp comment in fit_beers_law_hybrid).
            rss_total = rss_decay + rss_noise
            k_bp = 4
            bic_bp = n * np.log(rss_total / n + 1e-10) + k_bp * np.log(n)

            # Noise floor consistency penalty (same as hybrid)
            if expected_noise_floor is not None and expected_noise_floor > 0:
                candidate_nf = float(np.exp(noise_mean))
                ratio = candidate_nf / expected_noise_floor
                if ratio > noise_floor_tolerance or ratio < 1.0 / noise_floor_tolerance:
                    bic_bp += n * abs(np.log(ratio))

            candidates.append({
                'bic': bic_bp, 'kd': -slope, 'e0': np.exp(intercept),
                'bp': float(depths[bp_idx]),
                'nf': float(np.exp(noise_mean)),
            })

    # Coherent model selection (and other diagnostics) need the raw per-bin
    # candidate set, not the BMA average. Return it here so they reuse this
    # exact enumeration with no duplication. Early-reject bins (insufficient
    # photons/range/bins) return their nan 5-tuple above, before candidates
    # exist — callers treat a non-list result as "no candidates".
    if return_candidates:
        return candidates

    if not candidates:
        return np.nan, np.nan, np.nan, np.nan, 'no_valid_candidates'

    # ------------------------------------------------------------------
    # BIC weights: w_i = exp(-0.5 * (BIC_i - BIC_min)), normalised
    # ------------------------------------------------------------------
    cdf = pd.DataFrame(candidates)
    bic_min = cdf['bic'].min()
    cdf['weight'] = np.exp(-0.5 * (cdf['bic'] - bic_min))
    cdf['weight'] /= cdf['weight'].sum()

    # No-noise-floor collapse: if the single-line candidate dominates the
    # BMA weight, there is no statistical evidence for a flat-tail segment.
    # Return the single-line fit and signal the condition via 'no_noise_floor'.
    full_line_mask = cdf['bp'].isna()
    w_full = float(cdf.loc[full_line_mask, 'weight'].sum()) if full_line_mask.any() else 0.0
    if w_full > no_noise_floor_threshold:
        full_row = cdf[full_line_mask].iloc[0]
        return (float(full_row['kd']), float(full_row['e0']),
                np.nan, np.nan, 'no_noise_floor')

    kd_avg = float((cdf['weight'] * cdf['kd']).sum())
    e0_avg = float((cdf['weight'] * cdf['e0']).sum())

    # Weighted average breakpoint (excluding NaN for full-line model)
    bp_valid = cdf.dropna(subset=['bp'])
    if len(bp_valid) > 0 and bp_valid['weight'].sum() > 0.01:
        bp_avg = float((bp_valid['weight'] * bp_valid['bp']).sum() / bp_valid['weight'].sum())
        nf_avg = float((bp_valid['weight'] * bp_valid['nf']).sum() / bp_valid['weight'].sum())
    else:
        bp_avg = np.nan
        nf_avg = np.nan

    if kd_avg <= 0:
        return np.nan, np.nan, np.nan, np.nan, 'negative_kd'

    return kd_avg, e0_avg, bp_avg, nf_avg, None


# ---------------------------------------------------------------------------
#  BMA + daylight-only background subtraction
# ---------------------------------------------------------------------------
MIN_DECAY_BINS_AFTER_SUBTRACT = 4


def fit_beers_law_bma_daysub(hist_df, solar_elevation=None,
                             solar_subtract_threshold=-6.0,
                             expected_noise_floor=None,
                             poisson_weighted=False, **kwargs):
    """
    BMA, then re-fit the decay zone on background-subtracted counts — but only
    in daylight.

    Rationale (Columbia/WA-OR, 79,849 along-track bins, 2026-08-06; see
    ``A_OOI_Glider_Kd_WA_OR/docs/FINDINGS.md``):

      * The detected floor IS the solar background — Spearman rho = +0.85 against
        solar elevation, median 1.06 photons/bin below -20 deg rising to 1.57
        above 40 deg.
      * ``fit_beers_law_bma`` drops zero-count bins, so ``noise_floor`` is a
        geometric mean over OCCUPIED bins and cannot report below 1.0 photon/bin.
        34.4% of night floors sit under 1.05 (vs 0.7% of day floors) — at night
        the "asymptote" is largely a counting artefact, and subtracting it
        removes real signal and inflates Kd by ~33% spuriously.
      * Leaving daylight uncorrected biases daytime Kd low: the season x latitude
        matched day/night Kd ratio is 0.751. Subtracting in daylight only brings
        it to 1.025 and cuts the spread from 0.286 to 0.075.

    A continuous ``max(0, nf - dark)`` correction was tested and rejected — it
    under-corrects daylight (ratio 0.847), because the ~1.05 dark component
    flattens the daytime profile too.

    The result is insensitive to the threshold: any cut from -18 to +10 deg gives
    a day/night ratio of 0.97-0.99. The cost is a step at the terminator, which
    is why this is opt-in and not the default.

    Parameters
    ----------
    solar_elevation : float or None
        Bin-median solar elevation in degrees. ``None`` or non-finite disables
        the subtraction entirely (result is identical to ``bma``).
    solar_subtract_threshold : float
        Subtract only where ``solar_elevation`` exceeds this. Default -6.0
        (civil twilight).

    Returns
    -------
    tuple
        ``(kd, e0, breakpoint_depth, noise_floor, failure_reason)`` — the same
        shape as :func:`fit_beers_law_bma`.  ``failure_reason`` gains two values
        specific to this method: ``'daysub_too_few_bins'`` and
        ``'daysub_nonneg_slope'``.  In both cases ``kd`` falls back to the
        un-subtracted BMA estimate, which is still a valid fit.
    """
    kd, e0, bp, nf, reason = fit_beers_law_bma(
        hist_df, expected_noise_floor=expected_noise_floor,
        poisson_weighted=poisson_weighted, **kwargs
    )

    # Night, unknown sun angle, or no asymptote to subtract -> plain BMA.
    if solar_elevation is None or not np.isfinite(solar_elevation):
        return kd, e0, bp, nf, reason
    if solar_elevation <= solar_subtract_threshold:
        return kd, e0, bp, nf, reason
    if not (np.isfinite(bp) and np.isfinite(nf) and nf > 0):
        return kd, e0, bp, nf, reason

    # Daylight with a detected floor: re-fit Beer's law over the SAME decay zone
    # on (counts - noise_floor), keeping only positive residuals.
    df = hist_df[hist_df['photon_counts'] > 0].copy().sort_values('zdepth')
    decay = df[df['zdepth'] < bp]
    z = decay['zdepth'].values
    signal = decay['photon_counts'].values.astype(float) - float(nf)
    keep = signal > 0
    if keep.sum() < MIN_DECAY_BINS_AFTER_SUBTRACT or np.ptp(z[keep]) <= 0:
        return kd, e0, bp, nf, 'daysub_too_few_bins'

    slope, intercept = np.polyfit(z[keep], np.log(signal[keep]), 1)
    if slope >= 0:
        return kd, e0, bp, nf, 'daysub_nonneg_slope'

    return -float(slope), float(np.exp(intercept)), bp, nf, reason


# ---------------------------------------------------------------------------
#  Dispatcher: select fitting strategy by name
# ---------------------------------------------------------------------------
KD_FIT_METHODS = ('log_linear', 'bg_subtract', 'breakpoint', 'nonlinear',
                  'hybrid', 'bma', 'bma_daysub')


def _fit_kd_with_method(hist_df, method='log_linear', decay_threshold=0.0,
                        expected_noise_floor=None, **kwargs):
    """
    Run the requested fitting strategy on a single histogram.

    Extra keyword arguments are forwarded to the underlying fitter (e.g.
    ``min_decay_range_m``, ``min_decay_photons``, ``min_total_photons``,
    ``min_total_range_m`` for the hybrid and breakpoint methods).

    Returns
    -------
    tuple[float, float, float, str or None]
        (kd, e0, noise_floor, failure_reason).  noise_floor is np.nan for
        log_linear.  failure_reason is None for methods that do not report
        one (log_linear, bg_subtract, nonlinear).
    """
    # poisson_weighted is only supported by the BMA fitters, and the two solar
    # arguments only by bma_daysub; strip them from the generic kwargs so the
    # other methods don't receive an unexpected arg.
    poisson_weighted = kwargs.pop('poisson_weighted', False)
    solar_elevation = kwargs.pop('solar_elevation', None)
    solar_subtract_threshold = kwargs.pop('solar_subtract_threshold', -6.0)

    # Minimum vertical extent the decay rate must be measured over. Checked
    # before dispatch because no fitter can recover a decay length shorter than
    # the window it was measured in -- there is nothing to fit, not a fit to
    # repair. Method-agnostic on purpose: it is a property of the data.
    # BMA-only: shallowest allowed breakpoint depth. Popped here so non-BMA
    # fitters never see it.
    min_breakpoint_depth = kwargs.pop('min_breakpoint_depth', None)
    _bp_kw = {'min_breakpoint_depth': float(min_breakpoint_depth)} if min_breakpoint_depth is not None else {}

    min_fit_span_m = kwargs.pop('min_fit_span_m', 0.0)
    if min_fit_span_m and min_fit_span_m > 0:
        span = signal_depth_span(hist_df)
        if not np.isfinite(span) or span < min_fit_span_m:
            return np.nan, np.nan, np.nan, 'fit_span_too_short'

    if method == 'log_linear':
        valid = find_exponential_decay_zone(hist_df, decay_threshold=decay_threshold)
        kd, e0 = fit_beers_law(valid)
        return kd, e0, np.nan, None

    elif method == 'bg_subtract':
        kd, e0, nf = fit_beers_law_bg_subtract(hist_df)
        return kd, e0, nf, None

    elif method == 'breakpoint':
        kd, e0, _, nf, reason = fit_beers_law_breakpoint(hist_df, **kwargs)
        return kd, e0, nf, reason

    elif method == 'nonlinear':
        kd, e0, nf = fit_beers_law_nonlinear(hist_df)
        return kd, e0, nf, None

    elif method == 'hybrid':
        kd, e0, _, nf, reason = fit_beers_law_hybrid(
            hist_df, expected_noise_floor=expected_noise_floor, **kwargs
        )
        return kd, e0, nf, reason

    elif method == 'bma':
        kd, e0, _, nf, reason = fit_beers_law_bma(
            hist_df, expected_noise_floor=expected_noise_floor,
            poisson_weighted=poisson_weighted, **_bp_kw, **kwargs
        )
        return kd, e0, nf, reason

    elif method == 'bma_daysub':
        kd, e0, _, nf, reason = fit_beers_law_bma_daysub(
            hist_df, solar_elevation=solar_elevation,
            solar_subtract_threshold=solar_subtract_threshold,
            expected_noise_floor=expected_noise_floor,
            poisson_weighted=poisson_weighted, **kwargs
        )
        return kd, e0, nf, reason

    else:
        raise ValueError(f"Unknown kd_fit_method: {method!r}. "
                         f"Choose from {KD_FIT_METHODS}")


# another solution is to calculate kd without hist
def CalculateKdFromFilteredSubsurfacePhoton(df, vertical_res=0.8, decay_zone_threshold=0.0,
                                            kd_fit_method='log_linear', expected_noise_floor=None,
                                            surface_sigma=None,
                                            wave_exclusion_multiplier=0.0,
                                            wave_sigma_calm_threshold=0.1,
                                            **fit_kwargs):
    """
    Calculate Kd for a single along-track bin by fitting Beer's Law in log-space.

    Orchestrates steps 16 and 17 by calling find_exponential_decay_zone and
    fit_beers_law in sequence.

    Extra keyword arguments in ``fit_kwargs`` are forwarded to the underlying
    fitter (hybrid/breakpoint constraint parameters such as
    ``min_decay_range_m``, ``min_decay_photons``, ``min_decay_bins``,
    ``min_total_photons``, ``min_total_range_m``).

    Flowchart steps:
      16 — find_exponential_decay_zone: determine depth range of exponential decay.
      17 — fit_beers_law: fit Beer's Law curve; kd = -slope, e0 = exp(intercept).
    """
    # Early exit if DataFrame is empty or missing required column
    if df.empty or 'lat_bins' not in df.columns:
        return pd.DataFrame({'lat_bins': [np.nan], 'kd': [np.nan], 'e0': [np.nan],
                             'noise_floor': [np.nan], 'surface_sigma': [np.nan],
                             'latitude': [np.nan], 'longitude': [np.nan],
                             'fit_failure_reason': [None]})

    # Retrieve latitude and longitude
    lat_bin_value = df['lat_bins'].iloc[0] if not df['lat_bins'].empty else np.nan
    latitude = df['latitude'].mean() if 'latitude' in df.columns else df['lat'].mean() if 'lat' in df.columns else np.nan
    longitude = df['longitude'].mean() if 'longitude' in df.columns else df['lon'].mean() if 'lon' in df.columns else np.nan

    # Use value_counts to get photon counts in each height bin
    height_counts = df['height_bins'].value_counts().sort_index()
    bin_centers = height_counts.index.astype(float)

    # Create a DataFrame for the height bins and counts
    hist_df = pd.DataFrame({'zdepth': bin_centers, 'photon_counts': height_counts.values})

    # Reverse zdepth for model alignment
    hist_df['zdepth'] = hist_df['zdepth'].max() - hist_df['zdepth']

    # Wave-adaptive fit: skip shallow bins contaminated by wave smearing/bubbles.
    # zdepth=0 in the histogram corresponds to the adaptive surface detection
    # threshold, which since 2026-05-14 is max(2*sigma, 1.0 m) below the
    # Gaussian surface peak (get_sea_surface_height_adaptive:315,369) -- NOT
    # the 3*sigma this comment claimed until 2026-08-12.  Note the threshold is
    # applied in apparent coordinates, so with the step-10 re-binning enabled
    # zdepth=0 sits at ~0.75 * that value in true metres (~0.75 m at the floor).
    #
    # KNOWN INCONSISTENCY (unchanged pending a sensitivity re-run): adaptive_offset
    # below still assumes the old 3*sigma boundary, so for sigma > 1/3 m the total
    # exclusion is 2*sigma + (k-3)*sigma = 3*sigma rather than the intended
    # k*sigma = 4*sigma.  Only reachable via --enable_wave_adaptive_fit, which is
    # off in every production run to date.
    if surface_sigma is not None and wave_exclusion_multiplier > 0:
        if not np.isnan(surface_sigma) and surface_sigma > wave_sigma_calm_threshold:
            adaptive_offset = max(3.0 * surface_sigma, 1.0)
            wave_skip = max(0.0, wave_exclusion_multiplier * surface_sigma - adaptive_offset)
            trimmed = hist_df[hist_df['zdepth'] >= wave_skip]
            if len(trimmed) >= 4:
                logging.info("Wave-adaptive trim: sigma=%.3f, adaptive_offset=%.2f m, "
                             "skip=%.2f m, %d->%d bins",
                             surface_sigma, adaptive_offset, wave_skip,
                             len(hist_df), len(trimmed))
                hist_df = trimmed
            # else: keep original hist_df (too few bins after trim)

    # bma_daysub needs this bin's sun angle. Take it from the photon group
    # itself so no extra plumbing is needed through calculate_kd; the dispatcher
    # discards it for every other method. An explicit value in fit_kwargs (e.g.
    # from a diagnostic caller) wins over the photon-derived one.
    if 'solar_elevation' not in fit_kwargs and 'solar_elevation' in df.columns:
        solar_med = pd.to_numeric(df['solar_elevation'], errors='coerce').median()
        fit_kwargs = {**fit_kwargs, 'solar_elevation': float(solar_med)
                      if np.isfinite(solar_med) else None}

    kd, e0, noise_floor, fit_failure_reason = _fit_kd_with_method(
        hist_df, method=kd_fit_method, decay_threshold=decay_zone_threshold,
        expected_noise_floor=expected_noise_floor, **fit_kwargs
    )

    return pd.DataFrame({
        'lat_bins': [lat_bin_value],
        'kd': [kd],
        'e0': [e0],
        'noise_floor': [noise_floor],
        'surface_sigma': [surface_sigma if surface_sigma is not None else np.nan],
        'latitude': [latitude],
        'longitude': [longitude],
        'fit_failure_reason': [fit_failure_reason],
    })


# Original kd calculation function remains unchanged
def calculate_kd(filtered_seafloor_subsurface_photon_dataset, decay_zone_threshold=0.0, kd_fit_method='log_linear',
                 wave_exclusion_multiplier=0.0, wave_sigma_calm_threshold=0.1,
                 backg_c_df=None, vertical_res=None,
                 **fit_kwargs):
    """
    Loop over along-track bins and call CalculateKdFromFilteredSubsurfacePhoton
    for each bin.

    For the 'hybrid' method, a two-pass approach is used:
      Pass 1 — Fit each bin independently to estimate per-bin noise floors.
      Pass 2 — Compute the beam-level median noise floor, then re-fit each
               bin with the expected noise floor as a stabilisation constraint.

    After fitting, IQR-based outlier filtering removes extreme Kd values
    (for hybrid method only).

    Extra keyword arguments in ``fit_kwargs`` (e.g. ``min_decay_range_m``,
    ``min_decay_photons``, ``min_total_photons``) are forwarded to the
    underlying fitter to tune the data-sufficiency constraints.

    Flowchart steps:
      16 — Determine depth of exponential decay zone (per bin, inside
           CalculateKdFromFilteredSubsurfacePhoton).
      17 — Fit Beer's Law exponential decay curve (per bin).
      18 — Save K_dph attenuation coefficient value and statistical terms
           (kd, e0 columns in the returned DataFrame).
    """
    logging.info("Calculating Kd from filtered subsurface photon dataset (method=%s)", kd_fit_method)

    empty_df = pd.DataFrame({'lat_bins': [], 'kd': [], 'e0': [], 'noise_floor': [],
                             'surface_sigma': [], 'latitude': [], 'longitude': [],
                             'fit_failure_reason': []})

    groups = list(filtered_seafloor_subsurface_photon_dataset.groupby('lat_bins', observed=False))
    if not groups:
        return empty_df

    # Build per-bin bkg_hist computer when ATL09 data is available. Per-bin
    # value: median backg_c over the group's latitude range, scaled to a
    # histogram bin of (horizontal_res, vertical_res). Falls back to None
    # when prerequisites are absent.
    horizontal_res = fit_kwargs.get('horizontal_res')
    can_compute_bkg = (
        backg_c_df is not None and not backg_c_df.empty
        and vertical_res is not None and horizontal_res is not None
    )

    def _bkg_hist_for_group(group_df):
        if not can_compute_bkg:
            return None
        if 'latitude' in group_df.columns:
            lat_series = group_df['latitude']
        elif 'lat' in group_df.columns:
            lat_series = group_df['lat']
        else:
            return None
        lat_lo, lat_hi = float(lat_series.min()), float(lat_series.max())
        from kd_utils.SolarBckgrd.pair_atl09 import median_backg_c_in_lat_range
        from kd_utils.SolarBckgrd.atl09_backg_c import bkg_per_hist_bin
        bg = median_backg_c_in_lat_range(backg_c_df, lat_lo, lat_hi)
        if not np.isfinite(bg):
            return None
        return float(bkg_per_hist_bin(bg, vertical_res, horizontal_res))

    if kd_fit_method == 'hybrid':
        # ------------------------------------------------------------------
        # Pass 1: estimate per-bin noise floors (no expected_noise_floor)
        # ------------------------------------------------------------------
        logging.info("Hybrid pass 1: estimating per-bin noise floors")
        pass1_results = []
        for _, group in groups:
            sigma = float(group['surface_sigma'].iloc[0]) if ('surface_sigma' in group.columns and len(group) > 0) else None
            bkg_hist = _bkg_hist_for_group(group)
            extra = {'bkg_hist': bkg_hist} if bkg_hist is not None else {}
            pass1_results.append(CalculateKdFromFilteredSubsurfacePhoton(
                group, decay_zone_threshold=decay_zone_threshold,
                kd_fit_method=kd_fit_method,
                surface_sigma=sigma,
                wave_exclusion_multiplier=wave_exclusion_multiplier,
                wave_sigma_calm_threshold=wave_sigma_calm_threshold,
                **fit_kwargs, **extra,
            ))
        if not pass1_results:
            return empty_df
        pass1_df = pd.concat(pass1_results, ignore_index=True)

        # ------------------------------------------------------------------
        # Compute sliding-window noise floor for pass 2
        # ------------------------------------------------------------------
        HALF_WINDOW = 10
        MIN_LOCAL = 3

        nf_array = pass1_df['noise_floor'].values.copy()
        n_bins = len(nf_array)

        valid_nf_all = pass1_df['noise_floor'].dropna()
        valid_nf_all = valid_nf_all[valid_nf_all > 0]
        beam_median_nf = float(valid_nf_all.median()) if len(valid_nf_all) >= MIN_LOCAL else None

        if beam_median_nf is not None:
            expected_nf_per_bin = np.full(n_bins, np.nan)
            for i in range(n_bins):
                lo = max(0, i - HALF_WINDOW)
                hi = min(n_bins, i + HALF_WINDOW + 1)
                window_nf = nf_array[lo:hi]
                valid = window_nf[~np.isnan(window_nf) & (window_nf > 0)]
                if len(valid) >= MIN_LOCAL:
                    expected_nf_per_bin[i] = float(np.median(valid))
                else:
                    expected_nf_per_bin[i] = beam_median_nf

            logging.info("Hybrid pass 2: sliding window noise floor "
                         "(half_window=%d bins, beam_median=%.3f, "
                         "local range=%.3f-%.3f)",
                         HALF_WINDOW, beam_median_nf,
                         float(np.nanmin(expected_nf_per_bin)),
                         float(np.nanmax(expected_nf_per_bin)))
        else:
            expected_nf_per_bin = None
            logging.info("Hybrid: insufficient noise floor estimates (%d), "
                         "skipping pass 2", len(valid_nf_all))

        # ------------------------------------------------------------------
        # Pass 2: re-fit with per-bin expected noise floor constraint
        # ------------------------------------------------------------------
        if expected_nf_per_bin is not None:
            results = []
            for idx, (_, group) in enumerate(groups):
                sigma = float(group['surface_sigma'].iloc[0]) if ('surface_sigma' in group.columns and len(group) > 0) else None
                bkg_hist = _bkg_hist_for_group(group)
                extra = {'bkg_hist': bkg_hist} if bkg_hist is not None else {}
                results.append(CalculateKdFromFilteredSubsurfacePhoton(
                    group, decay_zone_threshold=decay_zone_threshold,
                    kd_fit_method=kd_fit_method,
                    expected_noise_floor=float(expected_nf_per_bin[idx]),
                    surface_sigma=sigma,
                    wave_exclusion_multiplier=wave_exclusion_multiplier,
                    wave_sigma_calm_threshold=wave_sigma_calm_threshold,
                    **fit_kwargs, **extra,
                ))
            SubsurfacePhotonDFAddedKd = pd.concat(results, ignore_index=True)
        else:
            SubsurfacePhotonDFAddedKd = pass1_df

        # ------------------------------------------------------------------
        # IQR-based outlier filtering (hybrid only)
        # Uses 3×IQR to accommodate real spatial variation (e.g. turbidity
        # gradients near river mouths) while removing fitting artefacts.
        # ------------------------------------------------------------------
        kd_vals = SubsurfacePhotonDFAddedKd['kd'].dropna()
        if len(kd_vals) >= 5:
            q1 = float(kd_vals.quantile(0.25))
            q3 = float(kd_vals.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 3.0 * iqr
            upper = q3 + 3.0 * iqr
            # Also apply physical cap: Kd > 5.0 m⁻¹ is unrealistic
            upper = min(upper, 5.0)
            outlier_mask = (SubsurfacePhotonDFAddedKd['kd'] < lower) | (SubsurfacePhotonDFAddedKd['kd'] > upper)
            n_outliers = int(outlier_mask.sum())
            if n_outliers > 0:
                logging.info("Hybrid IQR filter: removed %d outliers (bounds: [%.4f, %.4f])",
                             n_outliers, lower, upper)
                SubsurfacePhotonDFAddedKd.loc[outlier_mask, 'kd'] = np.nan

    else:
        # ------------------------------------------------------------------
        # Non-hybrid methods: single pass, no IQR filtering
        # ------------------------------------------------------------------
        results = []
        for _, group in groups:
            sigma = float(group['surface_sigma'].iloc[0]) if ('surface_sigma' in group.columns and len(group) > 0) else None
            bkg_hist = _bkg_hist_for_group(group)
            extra = {'bkg_hist': bkg_hist} if bkg_hist is not None else {}
            results.append(CalculateKdFromFilteredSubsurfacePhoton(
                group, decay_zone_threshold=decay_zone_threshold, kd_fit_method=kd_fit_method,
                surface_sigma=sigma,
                wave_exclusion_multiplier=wave_exclusion_multiplier,
                wave_sigma_calm_threshold=wave_sigma_calm_threshold,
                **fit_kwargs, **extra,
            ))
        if not results:
            return empty_df
        SubsurfacePhotonDFAddedKd = pd.concat(results, ignore_index=True)

    return SubsurfacePhotonDFAddedKd


# Updated function to apply kd calculation beam-by-beam
def process_kd_calculation(Final_filtered_subsurface_photon_dataset, decay_zone_threshold=0.0, kd_fit_method='log_linear',
                           wave_exclusion_multiplier=0.0, wave_sigma_calm_threshold=0.1,
                           backg_c_lookup_per_beam=None, vertical_res=None,
                           **fit_kwargs):
    """
    Beam-by-beam wrapper that calls calculate_kd for every beam and concatenates
    the results into a single Kd output DataFrame.

    Extra keyword arguments in ``fit_kwargs`` (e.g. ``min_decay_range_m``,
    ``min_decay_photons``, ``min_total_photons``) are forwarded to the
    underlying fitter via ``calculate_kd``.

    Parameters
    ----------
    backg_c_lookup_per_beam : dict[str, pd.DataFrame] or None
        Optional dict mapping beam_id (e.g. 'gt1r') to a DataFrame with columns
        ('latitude', 'backg_c') from ATL09. Forwarded to calculate_kd; only
        affects BMA when min_noise_floor_photons_per_bin > 0 in fit_kwargs.

    Flowchart steps:
      16 — Determine depth of exponential decay zone (delegated to
           CalculateKdFromFilteredSubsurfacePhoton).
      17 — Fit Beer's Law exponential decay curve (delegated).
      18 — Save K_dph attenuation coefficient value (slope of the Beer's Law
           curve) and statistical terms (kd, e0) — the returned DataFrame is
           written to CSV by run_pipeline.
    """
    # Initialize list to store results for each beam
    kd_beam_datasets = []

    # Group by 'beam_id' to process each beam independently
    for beam_id, beam_data in Final_filtered_subsurface_photon_dataset.groupby('beam_id'):
        logging.info(f"Calculating Kd for beam: {beam_id}")

        beam_backg_df = None
        if backg_c_lookup_per_beam:
            beam_backg_df = backg_c_lookup_per_beam.get(beam_id)

        # Apply the calculate_kd function to the current beam's dataset
        SubsurfacePhotonDFAddedKd = calculate_kd(
            beam_data, decay_zone_threshold=decay_zone_threshold, kd_fit_method=kd_fit_method,
            wave_exclusion_multiplier=wave_exclusion_multiplier,
            wave_sigma_calm_threshold=wave_sigma_calm_threshold,
            backg_c_df=beam_backg_df, vertical_res=vertical_res,
            **fit_kwargs,
        )

        # Add a column to track the beam_id in the results
        SubsurfacePhotonDFAddedKd['beam_id'] = beam_id

        # Append the result to the list
        kd_beam_datasets.append(SubsurfacePhotonDFAddedKd)

    if not kd_beam_datasets:
        logging.warning(
            "process_kd_calculation: no beam data remained after filtering. "
            "All bins may have been discarded by an upstream filter (e.g. histogram quality). "
            "Returning empty DataFrame."
        )
        return pd.DataFrame(columns=['lat_bins', 'kd', 'e0', 'noise_floor', 'surface_sigma',
                                     'latitude', 'longitude', 'fit_failure_reason', 'beam_id'])

    # Combine results from all beams into a single DataFrame
    combined_kd_dataset = pd.concat(kd_beam_datasets, ignore_index=True)
    return combined_kd_dataset


def bootstrap_kd_for_bin_loglinear(photon_heights, n_boot=500, vertical_res=0.25, rng=None):
    """
    Legacy log-linear residual bootstrap (kept for reference).

    Build the depth histogram, fit log-linear Beer's Law to get baseline Kd,
    compute residuals, and resample residuals N times to get a distribution
    of bootstrap Kd values. Reports std + 95% CI.

    This does NOT capture BIC breakpoint uncertainty. Use ``bootstrap_kd_for_bin``
    for the hybrid-method bootstrap that re-runs the full fit (BIC breakpoint +
    log-linear decay fit) on every bootstrap iteration.

    Parameters
    ----------
    photon_heights : np.ndarray
        Photon heights (negative below surface) for one along-track bin.
    n_boot : int
        Number of bootstrap iterations.
    vertical_res : float
        Depth bin width (m).
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    dict with keys: kd_baseline, kd_std, kd_ci_low, kd_ci_high, n_boot_success
        All values are NaN if the fit cannot be performed (insufficient data).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    nan_result = {'kd_baseline': np.nan, 'kd_std': np.nan,
                  'kd_ci_low': np.nan, 'kd_ci_high': np.nan,
                  'n_boot_success': 0}

    if len(photon_heights) < 20:
        return nan_result

    h_min, h_max = float(np.min(photon_heights)), float(np.max(photon_heights))
    if h_max <= h_min:
        return nan_result

    bin_edges = np.arange(h_min, h_max + vertical_res, vertical_res)
    counts, edges = np.histogram(photon_heights, bins=bin_edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    mask = counts > 0
    if mask.sum() < 4:
        return nan_result

    counts_used = counts[mask]
    bin_centers_used = bin_centers[mask]
    # zdepth convention: 0 at surface-most bin, increasing downward
    zdepth = bin_centers_used.max() - bin_centers_used
    log_counts = np.log(counts_used.astype(float))

    # Baseline log-linear fit
    model = LinearRegression()
    model.fit(zdepth.reshape(-1, 1), log_counts)
    kd_base = -float(model.coef_[0])
    fitted = model.predict(zdepth.reshape(-1, 1)).flatten()
    residuals = log_counts - fitted

    if kd_base < 0 or np.isnan(kd_base):
        return {**nan_result, 'kd_baseline': kd_base}

    # Residual bootstrap
    boot_kds = []
    n_points = len(residuals)
    for _ in range(n_boot):
        boot_resid = rng.choice(residuals, size=n_points, replace=True)
        boot_log = fitted + boot_resid
        m = LinearRegression().fit(zdepth.reshape(-1, 1), boot_log)
        kd_b = -float(m.coef_[0])
        if not np.isnan(kd_b) and kd_b > 0:
            boot_kds.append(kd_b)

    if len(boot_kds) < n_boot * 0.5:
        return {**nan_result, 'kd_baseline': kd_base, 'n_boot_success': len(boot_kds)}

    boot_kds = np.asarray(boot_kds)
    return {
        'kd_baseline': kd_base,
        'kd_std': float(np.std(boot_kds)),
        'kd_ci_low': float(np.percentile(boot_kds, 2.5)),
        'kd_ci_high': float(np.percentile(boot_kds, 97.5)),
        'n_boot_success': len(boot_kds),
    }


def bootstrap_kd_for_bin(photon_heights, n_boot=300, vertical_res=0.25, rng=None,
                         fit_method='bma',
                         min_decay_range_m=5.0, min_decay_photons=30,
                         min_decay_bins=8, min_total_photons=30,
                         min_total_range_m=5.0,
                         min_noise_floor_photons_per_bin=0.0,
                         hist_df=None,
                         poisson_weighted=False,
                         min_breakpoint_depth=2.0):
    """
    Residual bootstrap for single-bin Kd uncertainty.

    Each bootstrap iteration re-runs the fit procedure (BIC breakpoint
    search + log-linear fit on decay segment), capturing BOTH slope uncertainty
    AND breakpoint-location uncertainty.

    Procedure:
      1. Build depth histogram from photons in the bin.
      2. Run fit on the original histogram to get baseline kd, breakpoint,
         e0, and noise floor.
      3. Compute residuals for each depth bin (log_counts - fitted_value), where
         the fitted value is the log-linear slope for the decay segment and
         log(noise_floor) for the noise segment (if a breakpoint was found).
      4. For each of N bootstrap iterations:
         a. Resample residuals with replacement.
         b. Add resampled residuals to fitted values to produce bootstrap log counts.
         c. Run the fit on the bootstrap histogram to get bootstrap kd
            and bp.
      5. Report kd_std (std of bootstrap kds) + 95% CI, plus bp_std (std of
         bootstrap breakpoints when available).

    Parameters
    ----------
    photon_heights : np.ndarray
        Photon heights (negative below surface) for one along-track bin.
    n_boot : int
        Number of bootstrap iterations. Default 300.
    vertical_res : float
        Depth bin width (m). Default 0.25.
    rng : np.random.Generator or None
        Random number generator for reproducibility.
    fit_method : str
        Fitting method to use for baseline and bootstrap iterations.
        'bma' (default) or 'hybrid'.
    min_decay_range_m, min_decay_photons, min_decay_bins, min_total_photons,
    min_total_range_m : forwarded to the fit function.
    hist_df : pd.DataFrame or None
        Pre-built histogram (with columns 'zdepth', 'photon_counts'). When
        provided, skip the photon_heights → histogram step and use this
        directly. Required by ``bootstrap_kd_uncertainty`` so the bootstrap
        baseline matches the main pipeline's baseline kd (which uses
        height_bins value_counts, not raw photon_heights). Pass an empty
        np.array for ``photon_heights`` when ``hist_df`` is given.

    Returns
    -------
    dict with keys: kd_baseline, kd_std, kd_ci_low, kd_ci_high,
        n_boot_success, bp_std. All values are NaN if the fit cannot be
        performed (insufficient data, or the bin is rejected by the
        fit constraints).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    nan_result = {
        'kd_baseline': np.nan, 'kd_std': np.nan,
        'kd_ci_low': np.nan, 'kd_ci_high': np.nan,
        'n_boot_success': 0, 'bp_std': np.nan,
    }

    if hist_df is not None:
        hist = hist_df.copy()
    else:
        if len(photon_heights) == 0:
            return nan_result

        h_min = float(np.min(photon_heights))
        h_max = float(np.max(photon_heights))
        if h_max <= h_min:
            return nan_result

        bin_edges = np.arange(h_min, h_max + vertical_res, vertical_res)
        counts, edges = np.histogram(photon_heights, bins=bin_edges)
        bin_centers = (edges[:-1] + edges[1:]) / 2.0
        # zdepth convention: 0 at surface-most bin, increasing downward
        zdepth_all = bin_centers.max() - bin_centers
        hist = pd.DataFrame({'zdepth': zdepth_all, 'photon_counts': counts})

    if len(hist) < 6:
        return nan_result

    # Select fit function based on method
    fit_kwargs = dict(
        min_decay_range_m=min_decay_range_m,
        min_decay_photons=min_decay_photons,
        min_decay_bins=min_decay_bins,
        min_total_photons=min_total_photons,
        min_total_range_m=min_total_range_m,
    )
    if fit_method == 'bma':
        fit_func = fit_beers_law_bma
        # The BMA Poisson-floor patch is opt-in by passing a positive
        # min_noise_floor_photons_per_bin into the BMA fit. Forward whatever
        # the main pipeline used here so the bootstrap baseline matches.
        if min_noise_floor_photons_per_bin > 0:
            fit_kwargs['min_noise_floor_photons_per_bin'] = (
                float(min_noise_floor_photons_per_bin)
            )
        if poisson_weighted:
            fit_kwargs['poisson_weighted'] = True
        fit_kwargs['min_breakpoint_depth'] = float(min_breakpoint_depth)
    else:
        fit_func = fit_beers_law_hybrid

    # Baseline fit
    kd_base, e0_base, bp_base, nf_base, reason = fit_func(
        hist, **fit_kwargs,
    )

    # Soft-fail reasons still return a valid kd from the single-line fit
    # (BMA 'no_noise_floor' collapse, hybrid 'no_breakpoint_fallback').
    # These bins deserve an uncertainty estimate like any other retrieval —
    # the bootstrap's per-replicate refit naturally captures their
    # model-flip ambiguity as a wide CI. Skipping them here guaranteed
    # kd_std=NaN for every collapsed bin, which downstream consumers
    # (figure quality gates keyed on kd_std.notna()) read as "untrusted"
    # even though no uncertainty was ever attempted.
    SOFT_FAIL_REASONS = ('no_noise_floor', 'no_breakpoint_fallback')
    if np.isnan(kd_base) or (reason is not None
                             and reason not in SOFT_FAIL_REASONS):
        # Bin rejected by fit constraints — no uncertainty to report.
        return {**nan_result, 'kd_baseline': kd_base}

    # Build residuals from the baseline fit on non-empty bins.
    df = hist[hist['photon_counts'] > 0].copy().sort_values('zdepth').reset_index(drop=True)
    df['log_counts'] = np.log(df['photon_counts'].astype(float))

    if not np.isnan(bp_base):
        # Two-segment model: linear decay + constant noise floor.
        decay_mask = df['zdepth'] < bp_base
        noise_mask = ~decay_mask

        fitted = np.empty(len(df))
        # Decay: log_counts ~ log(e0) - kd*zdepth
        fitted[decay_mask.values] = (
            np.log(e0_base) - kd_base * df.loc[decay_mask, 'zdepth'].values
        )
        # Noise segment: constant log(noise_floor) (fallback to empirical mean).
        if not np.isnan(nf_base) and nf_base > 0:
            fitted[noise_mask.values] = np.log(nf_base)
        else:
            fitted[noise_mask.values] = (
                df.loc[noise_mask, 'log_counts'].mean() if noise_mask.any() else 0.0
            )
    else:
        # Full-line fallback (no breakpoint selected).
        fitted = np.log(e0_base) - kd_base * df['zdepth'].values

    residuals = df['log_counts'].values - fitted
    zdepth_fit = df['zdepth'].values

    # Within-segment residual resampling. The decay segment and the flat
    # noise-floor segment have different log-space variances (the tail is
    # far noisier), so pooling them into one urn gives decay bins
    # tail-sized residuals and vice versa — inflating kd_std for clean
    # decays and understating tail variability. Resample each segment from
    # its own residuals. Single-segment bins (full-line fallback / soft-fail
    # collapse) reduce to one group, so their draws are bit-identical to the
    # old global resample.
    if not np.isnan(bp_base):
        decay_sel = df['zdepth'].values < bp_base
        seg_indices = [np.flatnonzero(decay_sel), np.flatnonzero(~decay_sel)]
    else:
        seg_indices = [np.arange(len(residuals))]

    # Bootstrap: resample residuals, add to fitted, re-run fit.
    boot_kds = []
    boot_bps = []
    for _ in range(n_boot):
        boot_resid = np.empty(len(residuals))
        for idx in seg_indices:
            if len(idx) > 0:
                boot_resid[idx] = rng.choice(residuals[idx], size=len(idx),
                                             replace=True)
        boot_log = fitted + boot_resid
        boot_counts = np.exp(boot_log)
        # Clip to avoid zero or negative counts; histogram fit uses log.
        boot_counts = np.clip(boot_counts, 0.5, None)
        boot_hist = pd.DataFrame({'zdepth': zdepth_fit,
                                  'photon_counts': boot_counts})
        kd_b, _e0_b, bp_b, _nf_b, _reason_b = fit_func(
            boot_hist, **fit_kwargs,
        )
        if not np.isnan(kd_b) and kd_b > 0:
            boot_kds.append(kd_b)
            if not np.isnan(bp_b):
                boot_bps.append(bp_b)

    # Hybrid fit is more stringent than log-linear — use a lower success threshold.
    if len(boot_kds) < n_boot * 0.3:
        return {**nan_result, 'kd_baseline': kd_base,
                'n_boot_success': len(boot_kds)}

    boot_kds = np.asarray(boot_kds)
    return {
        'kd_baseline': kd_base,
        'kd_std': float(np.std(boot_kds)),
        'kd_ci_low': float(np.percentile(boot_kds, 2.5)),
        'kd_ci_high': float(np.percentile(boot_kds, 97.5)),
        'n_boot_success': len(boot_kds),
        'bp_std': float(np.std(boot_bps)) if len(boot_bps) > 1 else np.nan,
    }


def bootstrap_kd_uncertainty(subsurface_photon_df, n_boot=300, vertical_res=0.25,
                              random_seed=42,
                              fit_method='bma',
                              min_decay_range_m=5.0, min_decay_photons=30,
                              min_decay_bins=8, min_total_photons=30,
                              min_total_range_m=5.0,
                              min_noise_floor_photons_per_bin=0.0,
                              poisson_weighted=False,
                              min_breakpoint_depth=2.0):
    """
    Apply bootstrap to every along-track bin in a subsurface photon DataFrame.

    Parameters
    ----------
    subsurface_photon_df : pd.DataFrame
        Must contain 'lat_bins' and 'photon_height' columns.
    n_boot : int
        Bootstrap iterations per bin. Set to 0 to skip bootstrap entirely
        (returns an empty DataFrame).
    vertical_res : float
        Depth bin width (m).
    random_seed : int
        Random seed for reproducibility.
    fit_method : str
        Fitting method for baseline and bootstrap iterations ('bma' or 'hybrid').
    min_decay_range_m, min_decay_photons, min_decay_bins, min_total_photons,
    min_total_range_m : forwarded to the fit function via
        ``bootstrap_kd_for_bin`` so defaults match the pipeline fit.

    Returns
    -------
    pd.DataFrame
        Columns: lat_bins, kd_baseline, kd_std, kd_ci_low, kd_ci_high,
        n_boot_success, bp_std. One row per unique lat_bin in the input.
        Empty DataFrame if n_boot=0.
    """
    empty_cols = ['lat_bins', 'kd_baseline', 'kd_std',
                  'kd_ci_low', 'kd_ci_high', 'n_boot_success', 'bp_std']
    if n_boot <= 0:
        return pd.DataFrame(columns=empty_cols)

    rng = np.random.default_rng(random_seed)
    results = []
    # Group by (beam_id, lat_bins) to match process_kd_calculation, which
    # iterates per beam → per lat_bin. Mixing beams' photons into one
    # bootstrap histogram produced a CI that didn't bracket any single beam's
    # kd. Falls back to lat_bins-only grouping when beam_id is absent
    # (legacy / test fixtures).
    have_height_bins = 'height_bins' in subsurface_photon_df.columns
    have_beam_id = 'beam_id' in subsurface_photon_df.columns
    group_cols = ['beam_id', 'lat_bins'] if have_beam_id else ['lat_bins']
    for keys, bin_data in subsurface_photon_df.groupby(group_cols, sort=True):
        bin_hist_df = None
        if have_height_bins:
            height_counts = bin_data['height_bins'].value_counts().sort_index()
            if len(height_counts) > 0:
                bin_centers = height_counts.index.astype(float)
                bin_hist_df = pd.DataFrame({
                    'zdepth': bin_centers,
                    'photon_counts': height_counts.values,
                })
                # Reverse zdepth for model alignment (matches main fit).
                bin_hist_df['zdepth'] = bin_hist_df['zdepth'].max() - bin_hist_df['zdepth']
        r = bootstrap_kd_for_bin(
            bin_data['photon_height'].values if not have_height_bins else np.array([]),
            n_boot=n_boot,
            vertical_res=vertical_res, rng=rng,
            fit_method=fit_method,
            min_decay_range_m=min_decay_range_m,
            min_decay_photons=min_decay_photons,
            min_decay_bins=min_decay_bins,
            min_total_photons=min_total_photons,
            min_total_range_m=min_total_range_m,
            min_noise_floor_photons_per_bin=min_noise_floor_photons_per_bin,
            hist_df=bin_hist_df,
            poisson_weighted=poisson_weighted,
            min_breakpoint_depth=min_breakpoint_depth,
        )
        if have_beam_id:
            r['beam_id'], r['lat_bins'] = keys
        else:
            r['lat_bins'] = keys if not isinstance(keys, tuple) else keys[0]
        results.append(r)

    if not results:
        # Empty input (e.g. track rejected upstream) — preserve the schema
        # so callers' df[cols] / merge calls don't blow up with KeyError.
        return pd.DataFrame(columns=empty_cols)
    df = pd.DataFrame(results)
    base_cols = ['lat_bins', 'kd_baseline', 'kd_std', 'kd_ci_low', 'kd_ci_high',
                 'n_boot_success', 'bp_std']
    cols = (['beam_id'] + base_cols) if 'beam_id' in df.columns else base_cols
    return df[cols]

def spatial_regularize_kd(kd_df, window=13, unstable_frac_threshold=0.5,
                          adaptive=False, min_window=7, max_window=25,
                          central_stat='median', force_smooth_mask=None,
                          ci_low_col='kd_ci_low', ci_high_col='kd_ci_high',
                          beam_col='beam_id', bin_col='lat_bins', kd_col='kd'):
    """Region-adaptive along-track regularization of per-bin Kd.

    In low-SNR water the per-bin Kd is bistable: adjacent ~0.5 km bins flip
    between the two-segment fit (decay -> noise floor) and the single-line
    collapse ('no_noise_floor'), because the decay-vs-flat model choice is
    *decisive per bin but spatially incoherent*. That is a spatial-coherence
    artifact, NOT per-bin uncertainty — verified offline that no per-bin
    treatment (conditioned bootstrap, BMA-variance CI) smooths it, since each
    bin's model weight is decisive. The defensible fix is to regularize the
    OUTPUT along-track, keeping every bin's own fit:

      * Flag each bin unstable from its fit diagnostics (collapse reason,
        absent/NaN kd_std, or relative CI width > 1).
      * Where a centred window is majority-unstable, replace the spiky per-bin
        Kd with the local rolling MEAN and report the local 16-84 percentile
        as the honest scatter band. Stable regions (dense near-shore signal)
        are left untouched, preserving real structure (e.g. river-mouth peaks).
        Mean (not median) is used so a balanced collapse/breakpoint sawtooth
        averages to its centre instead of tracking the local majority, and so
        the estimate doesn't inherit the collapse bins' low bias; large spikes
        live only in stable near-shore bins, which are never regularized, so
        mean is safe inside regularized regions.

    This is NOT photon merging (which blends heterogeneous profiles and was
    shown to make the instability worse); each bin's fit is preserved, only
    the reported curve is smoothed where the per-bin estimate is unreliable.

    Parameters
    ----------
    kd_df : pd.DataFrame
        Per-bin Kd output (one row per beam x lat_bin). Needs ``kd_col`` and
        ``bin_col``; uses ``fit_failure_reason``, ``kd_std``, ``kd_ci_low``,
        ``kd_ci_high`` when present to flag instability.
    window : int
        Centred rolling window size in bins for the median/region test.
        Default 5 (~2.5 km at 0.5 km binning).
    unstable_frac_threshold : float
        A bin is regularized when the unstable fraction within its window is
        >= this. Default 0.5 (majority unstable).
    force_smooth_mask : pd.Series or array-like of bool, optional
        Bins to regularize regardless of the per-bin instability flag (aligned
        to ``kd_df`` by index when a Series, else positionally). Used by
        coherent_model_selection to smooth a region it has *already* committed
        to one model, even where each bin's own fit is individually stable
        (e.g. a turbidity ramp of well-determined two-segment bins that would
        otherwise zig-zag and carry no band).
    ci_low_col, ci_high_col : str
        Per-bin bootstrap CI columns. Where a bin is NOT regularized but has a
        finite per-bin CI, the reported band falls back to that CI instead of
        collapsing to the line — so the curve always carries honest uncertainty.

    Returns
    -------
    pd.DataFrame
        Copy of ``kd_df`` with added columns: ``kd_smooth``, ``kd_smooth_lo``,
        ``kd_smooth_hi`` (the band: rolling 16-84 pct in regularized bins, the
        per-bin bootstrap CI in un-regularized bins with a finite CI, else the
        line), and boolean ``kd_regularized``.
    """
    out = kd_df.copy()
    out['kd_smooth'] = out[kd_col]
    out['kd_smooth_lo'] = out[kd_col]
    out['kd_smooth_hi'] = out[kd_col]
    out['kd_regularized'] = False
    if len(out) == 0 or bin_col not in out.columns:
        return out

    if central_stat not in ('median', 'mean'):
        raise ValueError(f"central_stat must be 'median' or 'mean', got {central_stat!r}")
    # 'median' (default) is robust to the second mode: in a low-SNR zone the
    # unstable bins split into a collapse population (low) and a breakpoint
    # population (high), and a MEAN is dragged toward the high cluster. Median
    # tracks the majority instead of being pulled up. The 16-84 pct band is
    # reported regardless and brackets both modes. (A balanced 50/50 sawtooth
    # is the one case where median can't pick a centre — rare on real data,
    # which is collapse-majority; use central_stat='mean' there.)
    _central_np = np.median if central_stat == 'median' else np.mean

    band_window = 2 * window - 1
    group_cols = [beam_col] if beam_col in out.columns else []

    if force_smooth_mask is not None and not isinstance(force_smooth_mask, pd.Series):
        force_smooth_mask = pd.Series(np.asarray(force_smooth_mask), index=out.index)

    def _ci_fallback_band(g, use_arr):
        """Where a bin is NOT regularized but carries a finite per-bin bootstrap
        CI, report that CI as the band instead of the degenerate line==band, so
        the smoothed curve always shows honest uncertainty."""
        if ci_low_col not in g.columns or ci_high_col not in g.columns:
            return
        cilo = g[ci_low_col].to_numpy(dtype=float)
        cihi = g[ci_high_col].to_numpy(dtype=float)
        kdv = g[kd_col].to_numpy(dtype=float)
        fb = (~np.asarray(use_arr)) & ~np.isnan(cilo) & ~np.isnan(cihi) & ~np.isnan(kdv)
        if not fb.any():
            return
        lo_b = g['kd_smooth_lo'].to_numpy(dtype=float).copy()
        hi_b = g['kd_smooth_hi'].to_numpy(dtype=float).copy()
        lo_b[fb] = np.maximum(cilo[fb], 0.0)
        hi_b[fb] = cihi[fb]
        g['kd_smooth_lo'] = lo_b
        g['kd_smooth_hi'] = hi_b

    def _bin_unstable(g):
        u = pd.Series(False, index=g.index)
        if 'fit_failure_reason' in g.columns:
            u = u | g['fit_failure_reason'].notna()
        if 'kd_std' in g.columns:
            u = u | g['kd_std'].isna()
        if 'kd_ci_low' in g.columns and 'kd_ci_high' in g.columns:
            ciw = g['kd_ci_high'] - g['kd_ci_low']
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = ciw / g[kd_col].replace(0, np.nan)
            u = u | (rel > 1.0)
        # A bin with no Kd at all is not "regularizable" (nothing to smooth).
        return u & g[kd_col].notna()

    pieces = []
    grouped = out.groupby(group_cols, sort=False) if group_cols else [((), out)]
    for _, g in grouped:
        g = g.sort_values(bin_col).copy()
        kd = g[kd_col]
        unstable = _bin_unstable(g)
        frac = unstable.rolling(window, center=True, min_periods=1).mean()
        region = (frac >= unstable_frac_threshold) & kd.notna()
        if force_smooth_mask is not None:
            forced = force_smooth_mask.reindex(g.index).fillna(False).astype(bool)
            region = region | (forced & kd.notna())

        if adaptive:
            # Per-bin window scaled to the local unstable-run length, so a long
            # sparse-flip zone self-selects heavy smoothing while a short blip
            # gets a light touch — no global window to tune per site. Window is
            # clipped to [min_window, max_window] and forced odd.
            kd_arr = kd.to_numpy(dtype=float)
            reg_arr = region.to_numpy()
            n = len(kd_arr)
            eff = np.zeros(n, dtype=int)
            i = 0
            while i < n:
                if reg_arr[i]:
                    j = i
                    while j < n and reg_arr[j]:
                        j += 1
                    w = int(np.clip(j - i, min_window, max_window))
                    if w % 2 == 0:
                        w += 1
                    eff[i:j] = w
                    i = j
                else:
                    i += 1
            kd_sm = kd_arr.copy()
            lo_arr = kd_arr.copy()
            hi_arr = kd_arr.copy()
            did = np.zeros(n, dtype=bool)
            for k in range(n):
                if eff[k] == 0:
                    continue
                h = eff[k] // 2
                seg = kd_arr[max(0, k - h):min(n, k + h + 1)]
                seg = seg[~np.isnan(seg)]
                if len(seg) >= 2:
                    kd_sm[k] = float(_central_np(seg))
                    lo_arr[k] = float(np.percentile(seg, 16))
                    hi_arr[k] = float(np.percentile(seg, 84))
                    did[k] = True
            g['kd_smooth'] = kd_sm
            g['kd_smooth_lo'] = lo_arr
            g['kd_smooth_hi'] = hi_arr
            g['kd_regularized'] = did
            _ci_fallback_band(g, did)
        else:
            roll = kd.rolling(window, center=True, min_periods=2)
            kd_roll = roll.median() if central_stat == 'median' else roll.mean()
            lo = kd.rolling(band_window, center=True, min_periods=3).quantile(0.16)
            hi = kd.rolling(band_window, center=True, min_periods=3).quantile(0.84)
            use = region & kd_roll.notna()
            kd_sm = np.where(use, kd_roll, kd)
            g['kd_smooth'] = kd_sm
            g['kd_smooth_lo'] = np.where(use & lo.notna(), lo, kd_sm)
            g['kd_smooth_hi'] = np.where(use & hi.notna(), hi, kd_sm)
            g['kd_regularized'] = use.to_numpy()
            _ci_fallback_band(g, use.to_numpy())
        pieces.append(g)

    return pd.concat(pieces).reindex(out.index)


def coherent_model_selection(kd_df, subsurface_df, window=13,
                             unstable_frac_threshold=0.5, central_stat='median',
                             adaptive=False, min_window=7, max_window=25,
                             poisson_weighted=False,
                             min_breakpoint_depth=2.0, min_decay_range_m=5.0,
                             min_decay_photons=30, min_decay_bins=8,
                             min_total_photons=30, min_total_range_m=5.0,
                             min_fit_span_m=0.0,
                             beam_col='beam_id', bin_col='lat_bins', kd_col='kd'):
    """Coherent region-level decay-vs-flat model selection (resolves the
    per-bin BMA bistability at its source rather than averaging the output).

    The per-bin BMA decisively picks the two-segment (decay->noise-floor)
    model or the single-line collapse, and that choice flips bin-to-bin on
    marginal low-SNR data. Here we:

      1. Re-enumerate each bin's BMA candidates (via fit_beers_law_bma
         return_candidates=True — the *same* enumeration, no duplication) to
         get the single-line fit (kd_single, bic_single) and the best
         two-segment fit (kd_2seg, bic_2seg).
      2. Pool the per-bin model *preference* (which BIC wins) over a centred
         window and commit the whole region to its majority model — so a zone
         is internally self-consistent (all two-segment OR all single-line),
         with no bimodal mix to average.
      3. Smooth the region-selected Kd within unstable regions by delegating
         to spatial_regularize_kd (median by default), leaving stable bins
         untouched.

    Requires the original subsurface photons (re-enumerates candidates).
    Adds columns: kd_single, kd_2seg, bic_single, bic_2seg, has_breakpoint,
    coherent_2seg (region chose two-segment), kd_coherent, kd_coherent_lo,
    kd_coherent_hi, coherent_regularized. Raw kd/kd_std are preserved.
    """
    out = kd_df.copy()
    have_beam = (beam_col in subsurface_df.columns) and (beam_col in out.columns)
    gcols = [beam_col, bin_col] if have_beam else [bin_col]
    if 'height_bins' not in subsurface_df.columns:
        raise ValueError("coherent_model_selection needs 'height_bins' in subsurface_df")

    fk = dict(min_breakpoint_depth=min_breakpoint_depth,
              min_decay_range_m=min_decay_range_m, min_decay_photons=min_decay_photons,
              min_decay_bins=min_decay_bins, min_total_photons=min_total_photons,
              min_total_range_m=min_total_range_m, poisson_weighted=poisson_weighted)

    recs = []
    for keys, g in subsurface_df.groupby(gcols, sort=False):
        hc = g['height_bins'].value_counts().sort_index()
        rec = {}
        if isinstance(keys, tuple):
            for c, k in zip(gcols, keys):
                rec[c] = k
        else:
            rec[gcols[0]] = keys
        cands = []
        if len(hc) > 0:
            bc = hc.index.astype(float)
            hist = pd.DataFrame({'zdepth': bc.max() - bc, 'photon_counts': hc.values})
            # Same span gate as _fit_kd_with_method (--min_fit_span_m): a bin whose
            # signal spans less than the window cannot yield a decay rate, so it gets
            # no candidates here either (kd_single/kd_2seg NaN, consistent with raw kd).
            span_ok = True
            if min_fit_span_m and min_fit_span_m > 0:
                span = signal_depth_span(hist)
                span_ok = bool(np.isfinite(span) and span >= min_fit_span_m)
            if span_ok:
                res = fit_beers_law_bma(hist, return_candidates=True, **fk)
                cands = res if isinstance(res, list) else []
        full = [c for c in cands if pd.isna(c['bp'])]
        bp = [c for c in cands if not pd.isna(c['bp'])]
        rec['kd_single'] = float(full[0]['kd']) if full else np.nan
        rec['bic_single'] = float(full[0]['bic']) if full else np.nan
        if bp:
            b = min(bp, key=lambda c: c['bic'])
            rec['kd_2seg'] = float(b['kd']); rec['bic_2seg'] = float(b['bic']); rec['has_breakpoint'] = True
        else:
            rec['kd_2seg'] = np.nan; rec['bic_2seg'] = np.nan; rec['has_breakpoint'] = False
        recs.append(rec)
    cand = pd.DataFrame(recs)
    out = out.merge(cand, on=gcols, how='left')
    if 'has_breakpoint' not in out.columns:
        out['has_breakpoint'] = False
    out['has_breakpoint'] = out['has_breakpoint'].fillna(False)

    # Per-bin model preference, then region majority over the window.
    pref2 = out['has_breakpoint'] & (out['bic_2seg'] < out['bic_single'])
    out['_pref2'] = pref2.fillna(False)
    pieces = []
    grouped = out.groupby([beam_col], sort=False) if have_beam else [((), out)]
    for _, g in grouped:
        g = g.sort_values(bin_col).copy()
        frac2 = g['_pref2'].rolling(window, center=True, min_periods=1).mean()
        g['coherent_2seg'] = frac2 >= 0.5
        sel = np.where(g['coherent_2seg'] & g['has_breakpoint'], g['kd_2seg'], g['kd_single'])
        sel = pd.Series(sel, index=g.index)
        g['kd_sel'] = sel.where(sel.notna(), g[kd_col])  # fall back to original kd
        pieces.append(g)
    out = pd.concat(pieces).reindex(out.index)

    # Within-region smoothing of the model-selected Kd (reuse the regularizer).
    # force_smooth_mask = the committed two-segment bins: the region has already
    # been judged coherent (one model), so smooth its curve + report a band even
    # where each bin's own two-segment fit is individually stable (otherwise the
    # per-bin two-segment Kd zig-zags with a degenerate zero-width band — the
    # WaxDelta turbidity-ramp case). spatial_regularize_kd also now falls back to
    # each bin's bootstrap CI for any remaining un-regularized bin.
    sm = spatial_regularize_kd(out, window=window,
                               unstable_frac_threshold=unstable_frac_threshold,
                               adaptive=adaptive, min_window=min_window, max_window=max_window,
                               central_stat=central_stat,
                               force_smooth_mask=out['coherent_2seg'],
                               beam_col=beam_col, bin_col=bin_col, kd_col='kd_sel')
    out['kd_coherent'] = sm['kd_smooth']
    out['kd_coherent_lo'] = sm['kd_smooth_lo']
    out['kd_coherent_hi'] = sm['kd_smooth_hi']
    out['coherent_regularized'] = sm['kd_regularized']
    return out.drop(columns=['_pref2', 'kd_sel'])
