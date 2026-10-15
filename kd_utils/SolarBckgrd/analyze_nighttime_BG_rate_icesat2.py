import h5py
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from multiprocessing import Pool
from config import get_args
from noise_utils.interpolation import geoid_correction, refraction_correction, interpolate_labels, apply_interpolation
from noise_utils.data_processing import isolate_sea_land_photons

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define model functions
def power_model(x, a, b, c):
    return a * np.power(x, b) + c

def exp_model(x, a, b, c):
    """Exponential model: a * exp(b * x) + c"""
    return a * np.exp(b * x) + c

def twilight_model(x, a, b, c):
    return a * np.exp(-b * x**2) + c  # Gaussian form for twilight

def process_file(args):
    """Process a single ATL03 file, skipping files missing required datasets."""
    file, beam, shoreline_data_path = args
    try:
        with h5py.File(file, 'r') as f:
            # Check for required datasets
            required_keys = [
                f'{beam}/bckgrd_atlas/bckgrd_rate',
                f'{beam}/bckgrd_atlas/delta_time',
                f'{beam}/heights/delta_time',
                f'{beam}/heights/h_ph',
                f'{beam}/geolocation/reference_photon_lat',
                f'{beam}/geolocation/reference_photon_lon',
                f'{beam}/geolocation/solar_azimuth',
                f'{beam}/geolocation/solar_elevation'
            ]
            for key in required_keys:
                if key not in f:
                    raise KeyError(f"Missing dataset: {key}")

            bckgrd_rate = f[f'{beam}/bckgrd_atlas/bckgrd_rate'][:].astype(np.float64)
            bckgrd_time = f[f'{beam}/bckgrd_atlas/delta_time'][:]
            heights = f[f'{beam}/heights/h_ph'][:]
            segment_lat = f[f'{beam}/geolocation/reference_photon_lat'][:]
            segment_lon = f[f'{beam}/geolocation/reference_photon_lon'][:]
            segment_delta_time = f[f'{beam}/geolocation/delta_time'][:]

            Segment_Is_Land = pd.DataFrame({
                'latitude': segment_lat,  # Use unique column names to avoid conflicts
                'longitude': segment_lon,
                'delta_time': segment_delta_time
            })
            
            # Create a GeoDataFrame with geometry
            Segment_Is_Land['geometry'] = gpd.points_from_xy(Segment_Is_Land['longitude'], Segment_Is_Land['latitude'])
            ICESat2_GDF = gpd.GeoDataFrame(Segment_Is_Land, crs="EPSG:4326")

            # Determine land/ocean classification
            Segment_Is_Land_Labels = isolate_sea_land_photons(shoreline_data_path, ICESat2_GDF)
            ICESat2_GDF['is_land'] = Segment_Is_Land_Labels

            # Interpolate is_land labels to bckgrd_time
            interp_is_land = interp1d(
                ICESat2_GDF['delta_time'],
                ICESat2_GDF['is_land'].astype(int),
                bounds_error=False,
                fill_value=(ICESat2_GDF['is_land'].astype(int).iloc[0], ICESat2_GDF['is_land'].astype(int).iloc[-1])
            )
            is_land_interpolated = interp_is_land(bckgrd_time).round().astype(int)
            
            # ##########Debug#################################
            # # Interpolate latitude and longitude to bckgrd_time
            # interp_lat = interp1d(
            #     ICESat2_GDF['delta_time'],
            #     ICESat2_GDF['latitude'],
            #     bounds_error=False,
            #     fill_value=(ICESat2_GDF['latitude'].iloc[0], ICESat2_GDF['latitude'].iloc[-1])
            # )
            # interp_lon = interp1d(
            #     ICESat2_GDF['delta_time'],
            #     ICESat2_GDF['longitude'],
            #     bounds_error=False,
            #     fill_value=(ICESat2_GDF['longitude'].iloc[0], ICESat2_GDF['longitude'].iloc[-1])
            # )
            # interpolated_lat = interp_lat(bckgrd_time)
            # interpolated_lon = interp_lon(bckgrd_time)

            # # Create a GeoDataFrame for is_land_interpolated with lat/lon
            # is_land_df = pd.DataFrame({
            #     'delta_time': bckgrd_time,
            #     'latitude': interpolated_lat,
            #     'longitude': interpolated_lon,
            #     'is_land': is_land_interpolated
            # })
            # is_land_gdf = gpd.GeoDataFrame(
            #     is_land_df,
            #     geometry=gpd.points_from_xy(is_land_df['longitude'], is_land_df['latitude']),
            #     crs="EPSG:4326"
            # )
            
            # # Save to a GeoPackage file
            # file_dir = os.path.dirname(file)
            # is_land_output_dir = os.path.join(file_dir, "is_land_interpolated")
            # if not os.path.exists(is_land_output_dir):
            #     os.makedirs(is_land_output_dir)
            # output_filename = os.path.join(is_land_output_dir, f"{os.path.basename(file)}_is_land.gpkg")
            # is_land_gdf.to_file(output_filename, driver="GPKG")
            # logger.info(f"Saved is_land_interpolated to {output_filename}")
            ##############################################

            # Compute total background rates for land and ocean
            land_mask = is_land_interpolated == 1
            ocean_mask = is_land_interpolated == 0
            total_background_rate_land = np.mean(bckgrd_rate[land_mask]) if np.sum(land_mask) > 0 else np.nan
            total_background_rate_ocean = np.mean(bckgrd_rate[ocean_mask]) if np.sum(ocean_mask) > 0 else np.nan

            # Solar elevation handling
            FILL_VALUE = 3.4028235e38
            solar_elevations = f[f'{beam}/geolocation/solar_elevation'][:].astype(np.float64)
            valid_solar = solar_elevations[(solar_elevations != FILL_VALUE) & (np.abs(solar_elevations) < 1e30) & np.isfinite(solar_elevations)]
            solar_elevation = np.nan if valid_solar.size == 0 else np.mean(valid_solar)

            mean_photon_height = np.mean(heights) if heights.size > 0 else np.nan

            result = {
                'file': os.path.basename(file),
                'total_background_rate_land': total_background_rate_land,
                'total_background_rate_ocean': total_background_rate_ocean,
                'solar_elevation': solar_elevation,
                'mean_photon_height': mean_photon_height
            }
            logger.info(f"Processed {file}: {result}")
            return result
    except Exception as e:
        logger.error(f"Error processing {file}: {str(e)}")
        return {'file': os.path.basename(file), 'error': str(e)}

def fit_surface_model(df, classification, surface_type, model_func, p0, bounds, x_col='solar_elevation', y_col='solar_background_rate'):
    """Fit a model to a specific classification and surface type."""
    subset = df[(df['classification'] == classification) & (df['surface_type'] == surface_type)].copy()
    if len(subset) < 3:
        print(f"Warning: Insufficient data for {classification} {surface_type} model.")
        return None, None, None

    mask = subset[x_col].notna() & subset[y_col].notna() & np.isfinite(subset[x_col]) & np.isfinite(subset[y_col])
    valid_x = subset.loc[mask, x_col]
    valid_y = subset.loc[mask, y_col]

    if len(valid_x) < 3:
        print(f"Warning: Fewer than 3 valid points for {classification} {surface_type} model.")
        return None, None, None

    try:
        popt, _ = curve_fit(model_func, valid_x, valid_y, p0=p0, bounds=bounds, maxfev=20000)
        y_pred = model_func(valid_x, *popt)
        r2 = 1 - np.sum((valid_y - y_pred)**2) / np.sum((valid_y - np.mean(valid_y))**2)
        return popt, r2, lambda x: model_func(x, *popt)
    except RuntimeError as e:
        print(f"Fit failed for {classification} {surface_type}: {e}")
        return None, None, None

def load_and_process_files(atl03_directory, beam, shoreline_data_path):
    """Load and process ATL03 files in parallel."""
    atl03_files = [os.path.join(atl03_directory, f) for f in os.listdir(atl03_directory) if f.endswith('.h5') and 'ATL03' in f]
    print(f"Found {len(atl03_files)} ATL03 files.")
    
    if not atl03_files:
        raise ValueError("No ATL03 files found in the directory.")

    with Pool() as pool:
        results = pool.map(process_file, [(f, beam, shoreline_data_path) for f in atl03_files])
    
    print("Results from process_file:", results)  # Debug output
    df = pd.DataFrame(results)
    
    
    if 'error' in df.columns:
        error_files = df[df['error'].notna()]['file'].tolist()
        print(f"Errors in {len(error_files)} files: {error_files}")
        df = df[df['error'].isna()].drop(columns=['error'])
    
    if df.empty:
        raise ValueError("No files were successfully processed. Check the error messages above.")
    
    print("DataFrame columns after processing:", df.columns)
    return df

def assign_surface_type(df):
    """Assign surface type based on background rates."""
    required_columns = ['total_background_rate_land', 'total_background_rate_ocean']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in DataFrame: {missing_columns}")

    df['surface_type'] = np.where(
        df['total_background_rate_land'].notna() & (df['total_background_rate_land'] > 0),
        'Land',
        np.where(df['total_background_rate_ocean'].notna() & (df['total_background_rate_ocean'] > 0), 'Ocean', 'Mixed')
    )
    return df

def classify_data(df, twilight_threshold=6.0):
    """Classify data into Daytime, Nighttime, Twilight, or Unknown."""
    df['classification'] = np.where(
        df['solar_elevation'].isna(), 'Unknown',
        np.where(df['solar_elevation'] > twilight_threshold, 'Daytime',
                 np.where(df['solar_elevation'] < -twilight_threshold, 'Nighttime', 'Twilight'))
    )
    daytime_df = df[df['classification'] == 'Daytime'].copy()
    nighttime_df = df[df['classification'] == 'Nighttime'].copy()
    twilight_df = df[df['classification'] == 'Twilight'].copy()
    unknown_df = df[df['classification'] == 'Unknown'].copy()
    print(f"Daytime: {len(daytime_df)}, Nighttime: {len(nighttime_df)}, Twilight: {len(twilight_df)}, Unknown: {len(unknown_df)}")
    return df, daytime_df, nighttime_df, twilight_df, unknown_df

def calculate_non_solar_rate(nighttime_df):
    """Calculate the non-solar background rate from nighttime data."""
    non_solar_rate_land = np.nanmean(nighttime_df['total_background_rate_land'])
    non_solar_rate_ocean = np.nanmean(nighttime_df['total_background_rate_ocean'])
    non_solar_rate = np.nanmean([non_solar_rate_land, non_solar_rate_ocean])
    print(f"Non-solar background rate (land): {non_solar_rate_land:.2e} ph/s")
    print(f"Non-solar background rate (ocean): {non_solar_rate_ocean:.2e} ph/s")
    print(f"Non-solar background rate (combined): {non_solar_rate:.2e} ph/s")
    return non_solar_rate

def compute_solar_background_rates(df, non_solar_rate):
    """Compute solar background rates for land and ocean."""
    df['solar_background_rate_land'] = (df['total_background_rate_land'] - non_solar_rate).clip(lower=0)
    df['solar_background_rate_ocean'] = (df['total_background_rate_ocean'] - non_solar_rate).clip(lower=0)
    df['solar_background_rate'] = np.where(
        df['surface_type'] == 'Land', df['solar_background_rate_land'],
        np.where(df['surface_type'] == 'Ocean', df['solar_background_rate_ocean'], np.nan)
    )
    return df

def fit_daytime_model(df, daytime_df, non_solar_rate):
    """Fit daytime models for land and ocean."""
    daytime_models = {}
    for surface_type in ['Land', 'Ocean']:
        subset = daytime_df[daytime_df['surface_type'] == surface_type].copy()
        if len(subset) < 3:
            print(f"Warning: Insufficient daytime {surface_type} data for modeling.")
            continue

        mask = subset['solar_elevation'].notna() & subset['solar_background_rate'].notna() & \
               np.isfinite(subset['solar_elevation']) & np.isfinite(subset['solar_background_rate'])
        valid_x = subset.loc[mask, 'solar_elevation']
        valid_y = subset.loc[mask, 'solar_background_rate']

        if len(valid_x) < 3:
            print(f"Warning: Fewer than 3 valid points for daytime {surface_type} model.")
            continue

        # Remove outliers
        log_y = np.log10(valid_y + 1)
        mean_log_y = np.mean(log_y)
        std_log_y = np.std(log_y)
        outlier_mask = (log_y > mean_log_y - 3 * std_log_y) & (log_y < mean_log_y + 3 * std_log_y)
        valid_x = valid_x[outlier_mask]
        valid_y = valid_y[outlier_mask]

        print(f"Daytime regression ({surface_type}): {len(valid_x)} valid data points")
        try:
            p0_exp = [np.max(valid_y) - non_solar_rate, 0.05, non_solar_rate]
            bounds_exp = ([0, 0.01, non_solar_rate * 0.9], [1e8, 0.5, non_solar_rate * 1.1])
            popt_exp, _ = curve_fit(exp_model, valid_x, valid_y, p0=p0_exp, bounds=bounds_exp, maxfev=20000)
            r2_exp = 1 - np.sum((valid_y - exp_model(valid_x, *popt_exp))**2) / np.sum((valid_y - np.mean(valid_y))**2)

            x_shifted = valid_x - min(valid_x) + 1
            p0_power = [np.max(valid_y) - non_solar_rate, 0.6, non_solar_rate]
            bounds_power = ([0, 0.1, non_solar_rate * 0.9], [1e8, 1.0, non_solar_rate * 1.1])
            popt_power, _ = curve_fit(power_model, x_shifted, valid_y, p0=p0_power, bounds=bounds_power, maxfev=20000)
            r2_power = 1 - np.sum((valid_y - power_model(x_shifted, *popt_power))**2) / np.sum((valid_y - np.mean(valid_y))**2)

            if r2_power > r2_exp:
                print(f"Daytime model ({surface_type}, power law): rate = {popt_power[0]:.2e} * solar_elevation^{popt_power[1]:.2f} + {popt_power[2]:.2e}, R² = {r2_power:.3f}")
                daytime_models[surface_type] = {'params': popt_power, 'r2': r2_power, 'model': lambda x: power_model(x - min(valid_x) + 1, *popt_power)}
            else:
                print(f"Daytime model ({surface_type}, exponential): rate = {popt_exp[0]:.2e} * exp({popt_exp[1]:.2f} * solar_elevation) + {popt_exp[2]:.2e}, R² = {r2_exp:.3f}")
                daytime_models[surface_type] = {'params': popt_exp, 'r2': r2_exp, 'model': lambda x: exp_model(x, *popt_exp)}

            # Apply model
            daytime_mask = (df['classification'] == 'Daytime') & (df['surface_type'] == surface_type)
            valid_daytime_mask = daytime_mask & df['solar_elevation'].notna() & np.isfinite(df['solar_elevation'])
            x_for_model = df.loc[valid_daytime_mask, 'solar_elevation']
            if r2_power > r2_exp:
                x_for_model_shifted = x_for_model - min(valid_x) + 1
                df.loc[valid_daytime_mask, 'solar_background_rate'] = daytime_models[surface_type]['model'](x_for_model_shifted).clip(lower=0)
            else:
                df.loc[valid_daytime_mask, 'solar_background_rate'] = daytime_models[surface_type]['model'](x_for_model).clip(lower=0)
            df.loc[daytime_mask & ~valid_daytime_mask, 'solar_background_rate'] = non_solar_rate
        except RuntimeError as e:
            print(f"Daytime model fitting failed for {surface_type}: {e}. Using non-solar rate as fallback.")
            df.loc[(df['classification'] == 'Daytime') & (df['surface_type'] == surface_type), 'solar_background_rate'] = non_solar_rate
    return df, daytime_models


def fit_twilight_model(df, twilight_df, non_solar_rate):
    """Fit twilight models for land and ocean."""
    twilight_models = {}
    if len(twilight_df) > 3:
        df.loc[df['classification'] == 'Twilight', 'solar_background_rate'] = (
            df['solar_background_rate'].fillna(0).clip(lower=0)
        )
        for surface_type in ['Land', 'Ocean']:
            subset = twilight_df[twilight_df['surface_type'] == surface_type].copy()
            if len(subset) < 3:
                print(f"Warning: Insufficient twilight {surface_type} data for modeling.")
                continue

            mask = subset['solar_elevation'].notna() & subset['solar_background_rate'].notna() & \
                   np.isfinite(subset['solar_elevation']) & np.isfinite(subset['solar_background_rate'])
            valid_twilight_x = subset.loc[mask, 'solar_elevation']
            valid_twilight_y = subset.loc[mask, 'solar_background_rate']

            if len(valid_twilight_x) < 3:
                print(f"Warning: Fewer than 3 valid points for twilight {surface_type} model.")
                continue

            print(f"Twilight regression ({surface_type}): {len(valid_twilight_x)} valid data points")
            try:
                initial_a = np.max(valid_twilight_y) - non_solar_rate
                initial_b = 0.01  # Smaller b for a flatter Gaussian
                initial_c = non_solar_rate
                p0 = [initial_a, initial_b, initial_c]
                bounds = ([0, 0.001, non_solar_rate * 0.9], [1e6, 1.0, non_solar_rate * 1.1])

                popt, _ = curve_fit(twilight_model, valid_twilight_x, valid_twilight_y, p0=p0, bounds=bounds, maxfev=50000)
                y_pred = twilight_model(valid_twilight_x, *popt)
                r_squared = 1 - np.sum((valid_twilight_y - y_pred)**2) / np.sum((valid_twilight_y - np.mean(valid_twilight_y))**2)

                if r_squared < 0:
                    # Fallback to a constant model (mean of the data)
                    mean_rate = np.mean(valid_twilight_y)
                    twilight_models[surface_type] = {
                        'params': [mean_rate],
                        'r2': 0.0,  # R² of a constant model is 0 by definition
                        'model': lambda x: np.full_like(x, mean_rate)
                    }
                    print(f"Twilight model ({surface_type}) failed with R² = {r_squared:.3f}. Using constant model: {mean_rate:.2e} ph/s")
                    
                    twilight_mask = (df['classification'] == 'Twilight') & (df['surface_type'] == surface_type)
                    valid_twilight_mask = twilight_mask & df['solar_elevation'].notna() & np.isfinite(df['solar_elevation'])
                    df.loc[valid_twilight_mask, 'solar_background_rate'] = mean_rate
                    df.loc[twilight_mask & ~valid_twilight_mask, 'solar_background_rate'] = non_solar_rate
                else:
                    print(f"Twilight model ({surface_type}): rate = {popt[0]:.2e} * exp(-{popt[1]:.2f} * solar_elevation^2) + {popt[2]:.2e}, R² = {r_squared:.3f}")
                    twilight_models[surface_type] = {'params': popt, 'r2': r_squared, 'model': lambda x: twilight_model(x, *popt)}

                    twilight_mask = (df['classification'] == 'Twilight') & (df['surface_type'] == surface_type)
                    valid_twilight_mask = twilight_mask & df['solar_elevation'].notna() & np.isfinite(df['solar_elevation'])
                    x_for_model = df.loc[valid_twilight_mask, 'solar_elevation']
                    df.loc[valid_twilight_mask, 'solar_background_rate'] = twilight_models[surface_type]['model'](x_for_model).clip(lower=0)
                    df.loc[twilight_mask & ~valid_twilight_mask, 'solar_background_rate'] = non_solar_rate
            except RuntimeError as e:
                print(f"Twilight model fitting failed for {surface_type}: {e}. Setting twilight rates to non-solar rate.")
                df.loc[(df['classification'] == 'Twilight') & (df['surface_type'] == surface_type), 'solar_background_rate'] = non_solar_rate
    else:
        print("Too few twilight points for modeling. Setting twilight rates to non-solar rate.")
        df.loc[df['classification'] == 'Twilight', 'solar_background_rate'] = non_solar_rate
    return df, twilight_models

def create_diagnostic_plot(df, daytime_df, twilight_df, nighttime_df, daytime_models, twilight_models, non_solar_rate, output_dir):
    """Create diagnostic plot of background rate vs. solar elevation, including nighttime comparison."""
    plt.figure(figsize=(10, 6))
    
    # Total background rate (all data)
    plt.scatter(df['solar_elevation'], df['total_background_rate'], c='gray', alpha=0.5, label='Total')
    
    # Daytime data
    plt.scatter(daytime_df[daytime_df['surface_type'] == 'Land']['solar_elevation'],
                daytime_df[daytime_df['surface_type'] == 'Land']['solar_background_rate'],
                c='blue', alpha=0.5, label='Daytime Land')
    plt.scatter(daytime_df[daytime_df['surface_type'] == 'Ocean']['solar_elevation'],
                daytime_df[daytime_df['surface_type'] == 'Ocean']['solar_background_rate'],
                c='cyan', alpha=0.5, label='Daytime Ocean')
    
    # Twilight data
    plt.scatter(twilight_df[twilight_df['surface_type'] == 'Land']['solar_elevation'],
                twilight_df[twilight_df['surface_type'] == 'Land']['solar_background_rate'],
                c='orange', alpha=0.5, label='Twilight Land')
    plt.scatter(twilight_df[twilight_df['surface_type'] == 'Ocean']['solar_elevation'],
                twilight_df[twilight_df['surface_type'] == 'Ocean']['solar_background_rate'],
                c='yellow', alpha=0.5, label='Twilight Ocean')
    
    # Nighttime data (using total_background_rate since solar contribution is minimal)
    plt.scatter(nighttime_df[nighttime_df['surface_type'] == 'Land']['solar_elevation'],
                nighttime_df[nighttime_df['surface_type'] == 'Land']['total_background_rate'],
                c='purple', alpha=0.5, label='Nighttime Land')
    plt.scatter(nighttime_df[nighttime_df['surface_type'] == 'Ocean']['solar_elevation'],
                nighttime_df[nighttime_df['surface_type'] == 'Ocean']['total_background_rate'],
                c='lightgreen', alpha=0.5, label='Nighttime Ocean')
    
    # Non-solar background rate line
    plt.axhline(non_solar_rate, color='r', linestyle='--', label=f'Non-Solar: {non_solar_rate:.2e} ph/s')

    # Plot daytime fits
    for surface_type in daytime_models:
        if daytime_models[surface_type]['model'] is not None:
            x_fit = np.linspace(min(daytime_df['solar_elevation']), max(daytime_df['solar_elevation']), 100)
            y_fit = daytime_models[surface_type]['model'](x_fit if 'power' not in daytime_models[surface_type] else x_fit - min(daytime_df['solar_elevation']) + 1)
            plt.plot(x_fit, y_fit, label=f'Daytime {surface_type} Fit (R²={daytime_models[surface_type]["r2"]:.3f})',
                     linestyle='-' if surface_type == 'Land' else '--')

    # Plot twilight fits
    for surface_type in twilight_models:
        if twilight_models[surface_type]['model'] is not None:
            x_twilight = np.linspace(min(twilight_df['solar_elevation']), max(twilight_df['solar_elevation']), 100)
            y_twilight = twilight_models[surface_type]['model'](x_twilight)
            plt.plot(x_twilight, y_twilight, label=f'Twilight {surface_type} Fit (R²={twilight_models[surface_type]["r2"]:.3f})',
                     linestyle='-' if surface_type == 'Land' else '--')

    # Set plot properties
    plt.yscale('log')
    plt.ylim(1e3, 1e8)
    plt.xlabel('Solar Elevation (degrees)')
    plt.ylabel('Background Rate (ph/s)')
    plt.legend()
    plt.title('Background Rate vs. Solar Elevation (Land vs. Ocean) with Nighttime Comparison')
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "background_vs_elevation_with_nighttime_2021.png"))
    plt.close()

def create_calibration_plot(df, daytime_df, output_dir):
    """Create a calibration plot using mean photon height as a proxy."""
    if not daytime_df.empty:
        sample_file = daytime_df.iloc[0]
        heights = sample_file['mean_photon_height']
        if np.isfinite(heights):
            solar_rate = sample_file['solar_background_rate']
            bin_size = 0.1
            # Simulate histogram with 1000 points
            hist, edges = np.histogram(np.array([heights] * 1000), bins=np.arange(min(heights) - 1, max(heights) + 1, bin_size))
            photons_per_bin = solar_rate * (1000 / np.sum(hist)) * bin_size / (max(heights) - min(heights) + 2)
            calibrated_counts = np.maximum(hist - photons_per_bin, 0)
            calibrated_heights = []
            for i, count in enumerate(calibrated_counts):
                if count > 0:
                    calibrated_heights.extend(np.random.uniform(edges[i], edges[i+1], int(count)))

            plt.figure(figsize=(8, 6))
            plt.hist([heights] * 1000, bins=100, alpha=0.5, label='Original (Simulated)')
            plt.hist(calibrated_heights, bins=100, alpha=0.5, label='Calibrated')
            plt.xlabel('Photon Height (m)')
            plt.ylabel('Count')
            plt.legend()
            plt.title(f'Calibration Example: {sample_file["file"]}')
            plt.savefig(os.path.join(output_dir, "calibration_example.png"))
            plt.close()

def save_results(df, output_dir):
    """Save the results to a CSV file."""
    df.to_csv(os.path.join(output_dir, "atl03_analysis_updated.csv"), index=False)

def analyze_icesat2_data(atl03_directory, beam='gt1l', twilight_threshold=6.0, output_dir="output", shoreline_data_path=None):
    """Main analysis function orchestrating the workflow."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = load_and_process_files(atl03_directory, beam, shoreline_data_path)
    if df.empty:
        raise ValueError("No data processed successfully. Check logs for errors.")
    
    df = assign_surface_type(df)
    
    # Compute total_background_rate based on surface_type
    df['total_background_rate'] = np.where(
        df['surface_type'] == 'Land', df['total_background_rate_land'],
        np.where(df['surface_type'] == 'Ocean', df['total_background_rate_ocean'], np.nan)
    )
    
    df, daytime_df, nighttime_df, twilight_df, unknown_df = classify_data(df, twilight_threshold)
    
    non_solar_rate = calculate_non_solar_rate(nighttime_df)
    df = compute_solar_background_rates(df, non_solar_rate)
    
    # Recreate filtered DataFrames to include the updated columns from compute_solar_background_rates
    daytime_df = df[df['classification'] == 'Daytime'].copy()
    nighttime_df = df[df['classification'] == 'Nighttime'].copy()
    twilight_df = df[df['classification'] == 'Twilight'].copy()
    unknown_df = df[df['classification'] == 'Unknown'].copy()
    
    df, daytime_models = fit_daytime_model(df, daytime_df, non_solar_rate)
    df, twilight_models = fit_twilight_model(df, twilight_df, non_solar_rate)
    save_results(df, output_dir)
    
    # Updated call to include nighttime_df
    create_diagnostic_plot(df, daytime_df, twilight_df, nighttime_df, daytime_models, twilight_models, non_solar_rate, output_dir)
    # create_calibration_plot(df, daytime_df, output_dir)
    
    return df, {
        'daytime_models': daytime_models,
        'twilight_models': twilight_models,
        'non_solar_rate': non_solar_rate
    }  

if __name__ == "__main__":
    args = get_args()
    atl03_directory = os.path.join(args.workspace_path, args.atl03_path)
    shoreline_data_path = os.path.join(args.workspace_path, args.shoreline_data)
    output_dir = os.path.join(args.workspace_path, args.output_path)
    df, models = analyze_icesat2_data(atl03_directory, args.beam, output_dir=output_dir, shoreline_data_path=shoreline_data_path)
    print("Analysis complete.")
    


    
  