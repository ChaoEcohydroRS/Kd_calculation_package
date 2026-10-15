
import numpy as np
import geopandas as gpd
from datetime import datetime
import pandas as pd
import netCDF4
from scipy.signal import find_peaks
from scipy.stats import norm

# Function to apply binning beam-by-beam by calling the function of horizontal_vertical_bin_dataset
def process_sea_photon_binning(sea_photon_dataset, horizontal_res, vertical_res):
    """
    Apply horizontal and vertical binning to each beam.

    Flowchart steps:
      3 — Choose along-track (x) and vertical (z) bin sizes
          (e.g., 500 m in x, 0.25 m in z).
      4 — Build vertical histograms of photon counts using selected x, z bin sizes.
    """
    # Initialize list to store results from each beam
    binned_beam_datasets = []

    # Group the dataset by 'beam_id' and process each group separately
    for beam_id, beam_data in sea_photon_dataset.groupby('beam_id'):
        print(f'Processing binning for beam: {beam_id}')

        # Apply binning to the current beam dataset
        binned_beam_data = horizontal_vertical_bin_dataset(beam_data, horizontal_res, vertical_res)

        # Append the binned data for the current beam to the list
        binned_beam_datasets.append(binned_beam_data)

    if not binned_beam_datasets:
        # No beams produced rows (e.g. strong beam track has no sea photons in bbox).
        # Return empty DataFrame with the schema a normal binned output would have.
        return pd.DataFrame(columns=list(sea_photon_dataset.columns) + ['lat_bins', 'height_bins'])

    # Combine all binned beam datasets into a single DataFrame
    binned_dataset_sea_surface = pd.concat(binned_beam_datasets, ignore_index=True)

    return binned_dataset_sea_surface


# Bin data along vertical and horizontal scales
def horizontal_vertical_bin_dataset(dataset, lat_res, vertical_res):
    """
    Bin data along vertical and horizontal scales for later segmentation.

    Flowchart steps:
      3 — Choose along-track (x) and vertical (z) bin sizes
          (e.g., 500 m in x, 0.25 m in z).
      4 — Build vertical histograms of photon counts using selected x, z bin sizes.
      10 — Re-build histograms based on corrected depths (called again after
           refraction correction when apply_post_refraction_refit is enabled).
    """
    
    # Filter values within the range (-50, 10), because photons elevation outside this range will be real noise
    valid_range = (-70, 5)
    valid_mask = (dataset['photon_height'] > valid_range[0]) & (dataset['photon_height'] < valid_range[1])

    # Apply the valid_mask to filter unwanted values
    # and create a copy to avoid SettingWithCopyWarning
    filtered_dataset = dataset[valid_mask].copy()

    if filtered_dataset.empty:
        # Beam has no photons inside the valid_range — return empty schema-preserving frame.
        # round(NaN) downstream would otherwise raise ValueError("cannot convert float NaN to integer").
        return pd.DataFrame(columns=list(dataset.columns) + ['lat_bins', 'height_bins'])

    # Calculate the number of height bins
    height_range = abs(filtered_dataset['photon_height'].max() - filtered_dataset['photon_height'].min())
    height_bin_number = max(1, round(height_range / vertical_res))  # Ensure at least one bin

    # Calculate the number of latitude bins
    lat_range = abs(filtered_dataset['lat'].max() - filtered_dataset['lat'].min())
    lat_bin_number = max(1, round(lat_range / lat_res))  # Ensure at least one bin

    # Create bins for latitude
    lat_bins = pd.cut(filtered_dataset['lat'], bins=lat_bin_number, labels=np.arange(lat_bin_number))

    # Create bins for height
    height_bins = pd.cut(filtered_dataset['photon_height'], bins=height_bin_number,
                         labels=np.round(np.linspace(filtered_dataset['photon_height'].min(),
                                                     filtered_dataset['photon_height'].max(),
                                                     num=height_bin_number), decimals=1))

    # Add bins to dataframe using .loc to avoid SettingWithCopyWarning
    filtered_dataset.loc[:, 'lat_bins'] = lat_bins
    filtered_dataset.loc[:, 'height_bins'] = height_bins
    filtered_dataset = filtered_dataset.reset_index(drop=True)

    return filtered_dataset


def get_sea_surface_height_static(binned_data, threshold):
    """
    Calculate sea surface height and filter subsurface photons using static threshold.

    Flowchart step: 7 — Fit Gaussian curve to identify surface elevation; compute
    standard deviation of Gaussian peak.

    Parameters:
        binned_data (pandas.DataFrame): Binned photon data with lat, height_bins, photon_height

    Returns:
        final_sea_surface_height (list): Detected sea surface heights
        sea_surface_height_abnormal_label (array): Labels for abnormal heights
        sea_surface_dominated_label (array): Labels for surface-dominated bins
        PhotonDFBelowSurface (pandas.DataFrame): Photons below detected surface
    """
    
    #set flag for the df save
    firstTimeIndex = True
    
    # Create sea height list
    sea_surface_height = []
    mean_lat_bins_seq = []
    sea_surface_subsurface_photons_ratio = []


    # Group dataset along horizental (latitude bins)
    grouped_data = binned_data.groupby(['lat_bins'], group_keys=True, observed=False)
    data_groups = dict(list(grouped_data))

    # Loop through groups to detect sea surface
    for k, v in data_groups.items():
        # based on lat_utm
        lat_bin_average = v['lat'].mean()

        # Create new dataframe based on occurrence of photons per height bin
        new_df = pd.DataFrame(v.groupby(['height_bins'], observed=False).count())

        # Check if new_df is not empty before finding the bin with the highest photon count
        if not new_df.empty:        
            # Find the vertical bin with the highest photon count
            largest_h_bin = new_df['lat'].argmax()
    
            # Select the index of the bin with the highest count
            largest_h_index = new_df.index[largest_h_bin]
    
            # Calculate the median value of all photon height values within this bin
            photons_sea_surface = v.loc[v['height_bins'] == largest_h_index, 'photon_height']
            lat_bin_sea_median = photons_sea_surface.median()
    
            # Append to sea height list
            sea_surface_height.append(lat_bin_sea_median)
            mean_lat_bins_seq.append(lat_bin_average)
            del new_df
    
            # Get all photons below sea surface
            # to determine segment type of each subsurface water column
            # Use calculated sea height to determine photons at 0.5m below peak
            photons_sea_surface_up = \
                v.loc[(v['photon_height'] > (lat_bin_sea_median - threshold)) &
                      (v['photon_height'] < (lat_bin_sea_median + 2*threshold))]
    
            # Calculate the photon ratio between surface and whole photons
            if v['photon_height'].shape[0] > 0:
                new_photons_ratio_sea_surface = \
                    photons_sea_surface_up.shape[0] / v['photon_height'].shape[0]
            else:
                new_photons_ratio_sea_surface = np.nan
    
            sea_surface_subsurface_photons_ratio.append((1 - new_photons_ratio_sea_surface))
            
        else:
            # Append NaNs if the group is empty
            sea_surface_height.append(np.nan)
            mean_lat_bins_seq.append(np.nan)
            sea_surface_subsurface_photons_ratio.append(np.nan)

    # Filter out sea height bin values outside 2 SD of mean.
    mean = np.nanmean(sea_surface_height, axis=0)
    sd = np.nanstd(sea_surface_height, axis=0)

    final_sea_surface_height = np.where((sea_surface_height > (mean + 2 * sd)) | 
                                        (sea_surface_height < (mean - 2 * sd)),
                                        np.nan,
                                        sea_surface_height).tolist()
    
    sea_surface_height_abnormal_label = np.where(np.isnan(final_sea_surface_height), 0, 1)

    # Determine label based on ratio of sea surface photons and subsurface photons
    sea_surface_dominated_label = \
        np.where(np.array(sea_surface_subsurface_photons_ratio) >= 0.2, 0, 1)
        
    # Loop through groups again and return photons below 0.5m of sea height
    PhotonDFBelowThresholdPeak = pd.DataFrame()
    for i, (k, v) in enumerate(data_groups.items()):
        
        # Get all values below this bin
        if not np.isnan(final_sea_surface_height[i]):
            NewPhotonDFBelowThresholdPeak = \
                v.loc[v['photon_height'] < (final_sea_surface_height[i] - threshold)]
            
            if firstTimeIndex:
                PhotonDFBelowThresholdPeak = NewPhotonDFBelowThresholdPeak
                firstTimeIndex=False
            else: 
                PhotonDFBelowThresholdPeak=\
                    pd.concat([PhotonDFBelowThresholdPeak, NewPhotonDFBelowThresholdPeak])
    
    
    return final_sea_surface_height, \
        sea_surface_height_abnormal_label, \
        sea_surface_dominated_label,\
        PhotonDFBelowThresholdPeak


def get_sea_surface_height_adaptive(binned_data,
                                    noise_corruption_sd_m=5.0,
                                    surface_floor_m=1.0):
    """
    Calculate sea surface height and filter subsurface photons with enhanced wave removal.

    Flowchart steps:
      7  — Fit Gaussian curve to identify surface elevation; compute standard
           deviation of Gaussian peak.
      11 — Re-fit Gaussian curve to surface to identify surface peak; compute
           standard deviation and remove histogram data within three standard
           deviations. (This function is called a second time after refraction
           correction and histogram rebuild when apply_post_refraction_refit
           is enabled.)

    Outlier filter strategy (2026-05-23):
      The legacy ``mean ± 2σ`` filter on per-bin mu values fails when most
      lat_bins are noise-dominated (the noise itself shifts the mean and
      inflates σ, so the filter passes every noisy mu). To preserve the
      legacy behavior on well-behaved granules while fixing the zigzag in
      noise-dominated granules, the standard deviation of mu values is
      checked first:
        * σ_of_mu < noise_corruption_sd_m: use legacy mean ± 2σ filter.
        * σ_of_mu >= noise_corruption_sd_m: noise has corrupted the
          statistics; fall back to a granule-wide median prior derived
          from the well-populated lat_bins (>= 2× the original 20-photon
          minimum), rejecting mu values more than 3 m from the prior.

    Parameters:
        binned_data (pandas.DataFrame): Binned photon data with lat, height_bins, photon_height
        surface_floor_m (float): minimum depth below the fitted surface peak that is
            excluded as "surface" (default 1.0 m). The boundary is
            min(mu - 2*sigma, mu - surface_floor_m). Turbid water (Kd > 2) has its whole
            decay inside the top metre, so the field-validation notebooks used 0.5 m.
        noise_corruption_sd_m (float): Standard-deviation threshold (m) above
            which mean±2σ is considered noise-corrupted and the granule-prior
            fallback kicks in. Default 5.0 m.

    Returns:
        final_sea_surface_height (list): Detected sea surface heights
        sea_surface_height_abnormal_label (array): Labels for abnormal heights
        sea_surface_dominated_label (array): Labels for surface-dominated bins
        PhotonDFBelowSurface (pandas.DataFrame): Photons below detected surface with waves removed
        surface_photon_counts (dict): Mapping of lat_bin -> number of photons
            in the surface peak region (above the adaptive threshold)
    """

    firstTimeIndex = True
    sea_surface_height = []
    mean_lat_bins_seq = []
    sea_surface_subsurface_photons_ratio = []
    surface_photon_counts = {}

    grouped_data = binned_data.groupby(['lat_bins'], group_keys=True, observed=False)
    data_groups = dict(list(grouped_data))

    for k, v in data_groups.items():
        lat_bin_average = v['lat'].mean()

        if len(v) < 20:  # Increase minimum photon count for robustness
            sea_surface_height.append(np.nan)
            mean_lat_bins_seq.append(np.nan)
            sea_surface_subsurface_photons_ratio.append(np.nan)
            continue

        # Finer histogram for better peak resolution
        hist, bin_edges = np.histogram(v['photon_height'],
                                     bins=100,  # Finer bins
                                     density=True,
                                     range=(v['photon_height'].min(), v['photon_height'].max()))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Enhanced peak detection. Zero-pad both ends first: scipy never reports
        # the first/last sample as a peak, and at night (no solar background above
        # the water) the surface photons ARE the highest photons in the bin, so the
        # surface spike sits in the LAST histogram cell and was silently missed
        # (Great Bay / Klamath 2026-09-14: 19/22 bins returned no surface while
        # the 3 bins with one stray photon above the surface were found).
        hist_padded = np.concatenate(([0.0], hist, [0.0]))
        peaks, properties = find_peaks(hist_padded,
                                     height=np.max(hist)*0.3,  # Stricter peak height
                                     distance=10,  # Wider separation
                                     prominence=np.max(hist)*0.1)  # Require prominent peaks
        peaks = peaks - 1  # back to unpadded indices

        if len(peaks) == 0:
            sea_surface_height.append(np.nan)
            mean_lat_bins_seq.append(np.nan)
            sea_surface_subsurface_photons_ratio.append(np.nan)
            continue

        # Use strongest peak as surface, consider nearby peaks as wave effects
        surface_peak_idx = peaks[np.argmax(hist[peaks])]
        surface_height = bin_centers[surface_peak_idx]

        # Two-pass Gaussian fit: start with ±1 m, widen if sigma is large
        # (the ±1 m window underestimates sigma for SWH > 2 m)
        half_win = 1.0
        peak_data = v[(v['photon_height'] > surface_height - half_win) &
                     (v['photon_height'] < surface_height + half_win)]
        if len(peak_data) > 10:
            mu, sigma = norm.fit(peak_data['photon_height'])
            # Refit with wider window if sigma suggests truncation
            if sigma > 0.4:
                half_win2 = min(2.5 * sigma, 3.0)
                peak_data2 = v[(v['photon_height'] > mu - half_win2) &
                              (v['photon_height'] < mu + half_win2)]
                if len(peak_data2) > 10:
                    mu, sigma = norm.fit(peak_data2['photon_height'])
        else:
            mu, sigma = surface_height, 0.2  # More conservative default

        # NON-MONOTONIC boundary (2026-05-14): cap σ multiplier at 2× while
        # keeping the 1.0 m floor. Previous 3σ became excessive at high σ
        # (σ=0.93 → boundary at mu−2.8m, over-excluding 2m of column), producing
        # a spurious σ-Kd downward slope. The 1.0m floor still protects against
        # broad-peak surface leak at high σ.
        adaptive_threshold = min(mu - 2.0 * sigma, mu - surface_floor_m)  # at least surface_floor_m below

        # Extended surface layer to remove wave effects
        surface_photons = v[(v['photon_height'] > adaptive_threshold) &
                          (v['photon_height'] < mu + 3.0 * sigma)]  # Wider upper bound
        ratio = len(surface_photons) / len(v) if len(v) > 0 else np.nan

        sea_surface_height.append(mu)
        mean_lat_bins_seq.append(lat_bin_average)
        sea_surface_subsurface_photons_ratio.append(1 - ratio)

    # Outlier filtering — conditional on whether mean±2σ statistics look
    # corrupted by noise. Threshold: σ_of_mu < noise_corruption_sd_m → use
    # legacy mean±2σ (preserves prior production behavior); otherwise fall
    # back to a granule-prior filter from well-populated lat_bins (fixes
    # the zigzag in granules where noise dominates the mu distribution).
    ssh_arr = np.asarray(sea_surface_height, dtype=float)
    mean = np.nanmean(ssh_arr)
    sd = np.nanstd(ssh_arr)
    if np.isfinite(sd) and sd < noise_corruption_sd_m:
        final_sea_surface_height = np.where(
            (ssh_arr > mean + 2 * sd) | (ssh_arr < mean - 2 * sd),
            np.nan, ssh_arr).tolist()
    else:
        # Noise-corrupted statistics: derive prior from well-populated lat_bins
        # (>= 2 × the original 20-photon minimum) and reject mu values more
        # than 3 m from the prior.
        bin_counts = np.array([len(v) for _, v in data_groups.items()])
        well_pop = bin_counts >= 40
        ssh_well = ssh_arr[well_pop]
        ssh_well = ssh_well[np.isfinite(ssh_well)]
        prior_mu = float(np.nanmedian(ssh_well)) if len(ssh_well) >= 3 else float(np.nanmedian(ssh_arr))
        final_sea_surface_height = np.where(
            np.abs(ssh_arr - prior_mu) > 3.0,
            np.nan, ssh_arr).tolist()

    sea_surface_height_abnormal_label = np.where(np.isnan(final_sea_surface_height), 0, 1)
    sea_surface_dominated_label = np.where(np.array(sea_surface_subsurface_photons_ratio) >= 0.2, 0, 1)

    # Filter subsurface photons with diagnostics
    PhotonDFBelowSurface = pd.DataFrame()
    for i, (k, v) in enumerate(data_groups.items()):
        if not np.isnan(final_sea_surface_height[i]):
            half_win = 1.0
            peak_data = v[(v['photon_height'] > final_sea_surface_height[i] - half_win) &
                         (v['photon_height'] < final_sea_surface_height[i] + half_win)]
            if len(peak_data) > 10:
                mu, sigma = norm.fit(peak_data['photon_height'])
                if sigma > 0.4:
                    half_win2 = min(2.5 * sigma, 3.0)
                    peak_data2 = v[(v['photon_height'] > mu - half_win2) &
                                  (v['photon_height'] < mu + half_win2)]
                    if len(peak_data2) > 10:
                        mu, sigma = norm.fit(peak_data2['photon_height'])
                adaptive_threshold = min(mu - 2.0 * sigma, mu - surface_floor_m)  # 2σ patched
            else:
                adaptive_threshold = final_sea_surface_height[i] - surface_floor_m

            NewPhotonDFBelowSurface = v[v['photon_height'] < adaptive_threshold]
            surface_photon_counts[k] = len(v) - len(NewPhotonDFBelowSurface)

            if firstTimeIndex:
                PhotonDFBelowSurface = NewPhotonDFBelowSurface
                firstTimeIndex = False
            else:
                PhotonDFBelowSurface = pd.concat([PhotonDFBelowSurface, NewPhotonDFBelowSurface])

    return (final_sea_surface_height,
            sea_surface_height_abnormal_label,
            sea_surface_dominated_label,
            PhotonDFBelowSurface,
            surface_photon_counts)


def get_water_temp(date_year, date_month, date_day, latitude, longitude):
    """
    Pull down surface water temperature along the track from the JPL GHRSST opendap website.

    The GHRSST data are gridded tiles with dimension 17998 x 35999.
    To get the specific grid tile of the SST, you must convert from lat, lon coordinates
    to the gridded tile ratio of the SST data product using the coordinates of the IS2 data.
    """
    # Get date from data filename
    # data_path[-33:-25]
    date = date_year + date_month + date_day
    # date[0:4]
    year = date_year
    # date[4:6]
    month = date_month
    # date[6:8]
    day = date_day
    day_of_year = str(datetime.strptime(date, '%Y%m%d').timetuple().tm_yday)
    # Add zero in front of day of year string
    zero_day_of_year = day_of_year.zfill(3)

    # Calculate ratio of latitude from mid-point of IS2 track
    old_lat = latitude.mean()
    old_lat_min = -90
    old_lat_max = 90
    new_lat_min = 0
    new_lat_max = 17998

    new_lat = round(((old_lat - old_lat_min) / (old_lat_max - old_lat_min)) *
                    (new_lat_max - new_lat_min) + new_lat_min)

    # Calculate ratio of longitude from mid-point of IS2 track
    old_lon = longitude.mean()
    old_lon_min = -180
    old_lon_max = 180
    new_lon_min = 0
    new_lon_max = 35999

    new_lon = round(((old_lon - old_lon_min) / (old_lon_max - old_lon_min)) *
                    (new_lon_max - new_lon_min) + new_lon_min)

    # Access the SST data using the JPL OpenDap interface
    url = 'https://opendap.jpl.nasa.gov/opendap/OceanTemperature/ghrsst/data/GDS2/L4/GLOB/JPL/MUR/v4.1/' \
          + str(year) + '/' + str(zero_day_of_year) + '/' + str(date) \
          + '090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc'

    dataset = netCDF4.Dataset(url)

    # Access the data and convert the temperature from K to C
    water_temp = dataset['analysed_sst'][0, new_lat, new_lon] - 273.15
    return water_temp


def refraction_correction(WTemp, WSmodel, Wavelength,
                          Photon_ref_elev, Ph_ref_azimuth,
                          PhotonZ, PhotonX, PhotonY, Ph_Conf):
    """
    WTemp; there is python library that pulls water temp data
    WSmodel is the value surface height
    Wavelength is fixed
    """

    # Only process photons below water surface model
    PhotonX = PhotonX[PhotonZ <= WSmodel]
    PhotonY = PhotonY[PhotonZ <= WSmodel]
    Photon_ref_elev = Photon_ref_elev[PhotonZ <= WSmodel]
    Ph_ref_azimuth = Ph_ref_azimuth[PhotonZ <= WSmodel]
    Ph_Conf = Ph_Conf[PhotonZ <= WSmodel]
    PhotonZ = PhotonZ[PhotonZ <= WSmodel]

    # water temp for refraction correction
    WaterTemp = WTemp

    # Refraction coefficient #
    a = -0.000001501562500
    b = 0.000000107084865
    c = -0.000042759374989
    d = -0.000160475520686
    e = 1.398067112092424
    wl = Wavelength

    # refractive index of air
    n1 = 1.00029

    # refractive index of water
    n2 = (a * WaterTemp ** 2) + (b * wl ** 2) + (c * WaterTemp) + (d * wl) + e

    # assumption is 0.25416
    # This example is refractionCoef = 0.25449
    # 1.00029 is refraction of air constant
    correction_coef = (1 - (n1 / n2))

    # read photon ref_elev to get theta1
    theta1 = np.pi / 2 - Photon_ref_elev

    # eq 1. Theta2
    theta2 = np.arcsin(((n1 * np.sin(theta1)) / n2))

    # eq 3. S
    # Approximate water Surface = 1.5
    # D  = raw uncorrected depth
    D = WSmodel - PhotonZ

    # For Triangle DTS
    S = D / np.cos(theta1)

    # eq 2. R
    R = (S * n1) / n2
    Gamma = (np.pi / 2) - theta1

    # For triangle RPS
    # phi is an angle needed
    phi = theta1 - theta2

    # P is the difference between raw and corrected YZ location
    P = np.sqrt(R ** 2 + S ** 2 - 2 * R * S * np.cos(phi))

    # alpha is an angle needed
    alpha = np.arcsin((R * np.sin(phi)) / P)

    # Beta angle needed for Delta Y an d Delta Z
    Beta = Gamma - alpha

    # Delta Y
    DY = P * np.cos(Beta)

    # Delta Z
    DZ = P * np.sin(Beta)

    # Delta Easting
    DE = DY * np.sin(Ph_ref_azimuth)

    # Delta Northing
    DN = DY * np.cos(Ph_ref_azimuth)

    outX = PhotonX + DE
    outY = PhotonY + DN
    outZ = PhotonZ + DZ

    '''
        print('For selected Bathy photon:')
        print('lat = ', PhotonY[9000])
        print('long = ', PhotonX[9000])
        print('Raw Depth = ', PhotonZ[9000])
        print('D = ', D[9000])
        
        print('ref_elev = ', Photon_ref_elev[9000])
        
        print('Delta East = ', DE[9000])
        print('Delta North = ', DN[9000])
        print('Delta Z = ', DZ[9000])
        '''
    return (outX, outY, outZ, Ph_Conf, PhotonX, PhotonY, PhotonZ, Ph_ref_azimuth,
            Photon_ref_elev)  # We are most interested in out-x, out-y, out-z

