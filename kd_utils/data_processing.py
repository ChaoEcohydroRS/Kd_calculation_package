# utils/data_processing.py


import re
import os
import io
import time
import math
import h5py
import logging
import netCDF4
import numpy as np
import geopandas as gpd

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial import ConvexHull

import argparse
import subprocess
# import fiona
import utm
import pyproj
from pyproj import Transformer, Proj

# from pyproj import Transformer
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

from sklearn.cluster import DBSCAN
from .interpolation import *


logger = logging.getLogger(__name__)


# Allow-list of HDF5 variables read per beam. Anything not listed here is
# skipped entirely. Pre-slim, this function loaded the full per-beam tree plus
# top-level orbit_info / ancillary_data / atlas_impulse_response / calibrations
# trees — see debugging.md for the audit. The Kd pipeline never accesses any
# of those, so dropping them saves ~half a GB of HDF5 I/O per granule.
_HEIGHTS_KEYS = (
    'h_ph', 'lat_ph', 'lon_ph', 'delta_time', 'dist_ph_along',
    'quality_ph',          # used only when --enable_ir_ap_filter is set
    'dist_ph_across',      # used only by SeaSurfaceFlattening (off in production)
)
_GEOLOCATION_KEYS = (
    'segment_id', 'ph_index_beg', 'segment_ph_cnt', 'segment_dist_x',
    'delta_time', 'reference_photon_lat', 'reference_photon_lon',
    'ref_elev', 'ref_azimuth',
    'solar_elevation',     # used only when --enable_solar_background_filter is set
    'segment_length',      # used only by SeaSurfaceFlattening (off in production)
)
_GEOPHYS_CORR_KEYS = ('geoid',)  # tide_*, dac, dem_h are read in ATL03 but never applied — see debugging.md
_BCKGRD_ATLAS_KEYS = ('bckgrd_rate', 'delta_time', 'bckgrd_int_height')


#this function from icesat2_toolkit
# PURPOSE: read ICESat-2 ATL03 HDF5 data files
def read_granule(FILENAME, ATTRIBUTES=False, **kwargs):
    """
    Reads ICESat-2 ATL03 Global Geolocated Photons data files.

    Flowchart step: 1 — Acquire ATL03 data (download or use cloud services).

    Parameters
    ----------
    FILENAME: str
        full path to ATL03 file
    ATTRIBUTES: bool, default False
        read file, group and variable attributes

    Returns
    -------
    IS2_atl03_mds: dict
        ATL03 variables
    IS2_atl03_attrs: dict
        ATL03 attributes
    IS2_atl03_beams: list
        valid ICESat-2 beams within ATL03 file
    """
    # Open the HDF5 file for reading
    if isinstance(FILENAME, io.IOBase):
        fileID = h5py.File(FILENAME, 'r')
    else:
        fileID = h5py.File(os.path.expanduser(FILENAME), 'r')

    # Output HDF5 file information
    logging.info(fileID.filename)
    logging.info(list(fileID.keys()))

    # allocate python dictionaries for ICESat-2 ATL03 variables and attributes
    IS2_atl03_mds = {}
    IS2_atl03_attrs = {}

    # read each input beam within the file
    IS2_atl03_beams = []
    for gtx in [k for k in fileID.keys() if bool(re.match(r'gt\d[lr]',k))]:
        # check if subsetted beam contains data
        # check in both the geolocation and heights groups
        try:
            fileID[gtx]['geolocation']['segment_id']
            fileID[gtx]['heights']['delta_time']
        except KeyError:
            pass
        else:
            IS2_atl03_beams.append(gtx)

    # for each included beam
    for gtx in IS2_atl03_beams:
        IS2_atl03_attrs.setdefault(gtx, {})
        IS2_atl03_attrs[gtx]['atlas_beam_type'] = fileID[gtx].attrs['atlas_beam_type']

        IS2_atl03_mds[gtx] = {
            'heights': {},
            'geolocation': {},
            'bckgrd_atlas': {},
            'geophys_corr': {},
        }

        # ICESat-2 Measurement Group — explicit allow-list.
        # signal_conf_ph is (N, 5) on disk but only column 1 (ocean) is ever
        # read downstream, so we slice at HDF5 read to cut ~80% of that
        # dataset's I/O.
        h = fileID[gtx]['heights']
        for key in _HEIGHTS_KEYS:
            if key in h:
                IS2_atl03_mds[gtx]['heights'][key] = h[key][:]
        if 'signal_conf_ph' in h:
            IS2_atl03_mds[gtx]['heights']['signal_conf_ph'] = h['signal_conf_ph'][:, 1]

        # ICESat-2 Geolocation Group
        g = fileID[gtx]['geolocation']
        for key in _GEOLOCATION_KEYS:
            if key in g:
                IS2_atl03_mds[gtx]['geolocation'][key] = g[key][:]

        # ICESat-2 Background Photon Rate Group
        b = fileID[gtx]['bckgrd_atlas']
        for key in _BCKGRD_ATLAS_KEYS:
            if key in b:
                IS2_atl03_mds[gtx]['bckgrd_atlas'][key] = b[key][:]

        # ICESat-2 Geophysical Corrections Group — only geoid is applied.
        gc = fileID[gtx]['geophys_corr']
        for key in _GEOPHYS_CORR_KEYS:
            if key in gc:
                IS2_atl03_mds[gtx]['geophys_corr'][key] = gc[key][:]

        if ATTRIBUTES:
            IS2_atl03_attrs[gtx].setdefault('heights', {})
            IS2_atl03_attrs[gtx].setdefault('geolocation', {})
            IS2_atl03_attrs[gtx].setdefault('bckgrd_atlas', {})
            IS2_atl03_attrs[gtx].setdefault('geophys_corr', {})

            for att_name, att_val in fileID[gtx].attrs.items():
                IS2_atl03_attrs[gtx][att_name] = att_val
            for sub, keys in (('heights', _HEIGHTS_KEYS),
                              ('geolocation', _GEOLOCATION_KEYS),
                              ('bckgrd_atlas', _BCKGRD_ATLAS_KEYS),
                              ('geophys_corr', _GEOPHYS_CORR_KEYS)):
                node = fileID[gtx][sub]
                for key in keys:
                    if key in node:
                        IS2_atl03_attrs[gtx][sub][key] = dict(node[key].attrs)
            if 'signal_conf_ph' in fileID[gtx]['heights']:
                IS2_atl03_attrs[gtx]['heights']['signal_conf_ph'] = dict(
                    fileID[gtx]['heights']['signal_conf_ph'].attrs)

    # Global File Attributes
    if ATTRIBUTES:
        for att_name,att_val in fileID.attrs.items():
            IS2_atl03_attrs[att_name] = att_val

    # Closing the HDF5 file
    fileID.close()
    # Return the datasets and variables
    return (IS2_atl03_mds,IS2_atl03_attrs,IS2_atl03_beams)



# convert_wgs_to_utm function, see https://stackoverflow.com/a/40140326/4556479
def convert_wgs_to_utm(lon: float, lat: float):
    """Based on lat and lng, return best utm epsg-code"""
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = 'epsg:326' + utm_band
        return epsg_code
    epsg_code = 'epsg:327' + utm_band
    return epsg_code


def orthometric_correction(lat, lon, Z, epsg):
    # Define the Proj string
    #To transform from WGS84 ellipsoidal height
    # to EGM2008 orthometric height using PyProj
    # proj_string = '+proj=latlong +ellps=WGS84 +datum=WGS84 +vunits=m +no_defs +geoidgrids=egm2008-1.gtx'
    # # Define the Proj string for WGS84 ellipsoidal height
    # wgs84_proj_string = '+proj=latlong +ellps=WGS84 +datum=WGS84 +no_defs'
    
    # # Define the Proj string for EGM2008 orthometric height: egm08_25,egm2008-1
    # egm2008_proj_string = \
    #     '+proj=latlong +ellps=WGS84 +datum=WGS84 +no_defs ' \
    #     '+geoidgrids=C:/Workstation/ICESat2_HLS/Code/Geoids/egm08_25.gtx'

    # transform ellipsoid (WGS84) height to orthometric height
    # transformer = Transformer.from_crs(wgs84_proj_string, egm2008_proj_string, always_xy=True)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    X_egm08, Y_egm08, Z_egm08 = transformer.transform(lon, lat, Z)

    # transform WGS84 proj to local UTM
    myProj = Proj(epsg)
    X_utm, Y_utm = myProj(lon, lat)

    return Y_utm, X_utm, Z_egm08



def load_data(file_path, read_attributes=False):
    """
    Wrapper around read_granule that lets you decide
    whether to pull the full attribute tree.

    Flowchart step: 1 — Acquire ATL03 data (download or use cloud services).

    Parameters
    ----------
    file_path : str
        Path to the ATL03 HDF5 granule.
    read_attributes : bool, optional
        If True, read_granule returns the full IS2_atl03_attrs tree.
        Defaults to False.
    """
    
    
    return read_granule(file_path, ATTRIBUTES = read_attributes)



# requires that the input gdf has ranged index values i
# will need to change if index is changed to time or something
# this currently checks point in polygon for EVERY point
# would be significantly sped up if evaluated at 10m or something similar
# maybe later, fine for now
def isolate_sea_land_photons(shoreline_data_path, ICESat2_GDF):
    # try loading the shoreline data
    try:
        ICESat2_GDF.insert(0, 'lat', ICESat2_GDF.geometry.y, False)
        ICESat2_GDF.insert(0, 'lon', ICESat2_GDF.geometry.x, False)

        # allocation of to be used arrays
        zero_int_array = np.int64(np.zeros_like(ICESat2_GDF.geometry.x))

        # Land flag initialized as -1
        # If shorelines downloaded already, will be set to 0 or 1
        ICESat2_GDF.insert(0, 'is_land',
                           zero_int_array - 1, False)

        # set the projection
        ICESat2_GDF.set_crs("EPSG:4326", inplace=True)

        # load shoreline dataset to include only the features that intersect the bounding box
        # bbox can be GeoDataFrame or GeoSeries | shapely Geometry, default None
        # Filter features by given bounding box, GeoSeries, GeoDataFrame or a shapely geometry.
        # engine str, 'fiona' or 'pyogrio'
        # somtime it gives error if using fiona
        # land_polygon_gdf = gpd.read_file(shoreline_data_path, bbox=ICESat2_GDF, engine='fiona')
        land_polygon_gdf = gpd.read_file(shoreline_data_path, bbox=ICESat2_GDF, engine='pyogrio')

        # continue with getting a new array of 0-or-1 labels for each photon
        land_point_labels = np.zeros_like(ICESat2_GDF.is_land.values)

        # update labels for points in the land polygons
        pts_in_land = gpd.sjoin(ICESat2_GDF, land_polygon_gdf, predicate='within')

        # get land or not bool value
        land_loc = ICESat2_GDF.index.isin(pts_in_land.index)

        # asigned them to new numpy array
        land_point_labels[land_loc] = 1
        land_point_labels[~land_loc] = 0

        return land_point_labels

    except Exception as e:

        print(e)

        print("Error loading shoreline data, returning -1s for is_land flag")

        # if the shoreline data is not available
        # return the original label array

        return -np.ones_like(ICESat2_GDF.is_land.values)


def create_photon_dataframe(lat_ph, lon_ph, ref_elev, ref_azimuth, geoid, h_ph, \
                            quality_ph, is_land_label_interp1d, signal_conf_photon, x_atc, relative_AT_dist,
                            solar_elevation=None, background_rate=None, bckgrd_int_height=None):
    # Apply geoid correction to the photon heights to convert them from ellipsoidal to orthometric heights
    h_ph_geoid_cor = h_ph[:] - geoid[:]
    
    # Determine the EPSG code for the UTM zone based on the first photon's longitude and latitude   
    epsg_code = convert_wgs_to_utm(lon_ph[0], lat_ph[0])
    
    # Perform orthometric correction to obtain UTM coordinates and corrected heights
    lat_utm, lon_utm, h_ph_cor = orthometric_correction(lat_ph, lon_ph, h_ph, epsg_code)
    
    # Put the data into the dataframe
    sea_photon_dataset = pd.DataFrame({
        'latitude': lat_ph,
        'longitude': lon_ph,
        'lat': lat_utm,
        'lon': lon_utm,
        'photon_height': h_ph_geoid_cor,
        'quality_ph': quality_ph,
        'is_land_label': is_land_label_interp1d,
        'photon_conf': signal_conf_photon,
        'ref_elevation': ref_elev,
        'ref_azimuth': ref_azimuth,
        'relative_AT_dist': relative_AT_dist
    }, columns=['latitude', 'longitude', 'lat', 'lon', 'photon_height', 'quality_ph', 'is_land_label', 'photon_conf', 'ref_elevation', 'ref_azimuth', 'relative_AT_dist'])

    if solar_elevation is not None:
        sea_photon_dataset['solar_elevation'] = solar_elevation
    if background_rate is not None:
        sea_photon_dataset['background_rate'] = background_rate
    if bckgrd_int_height is not None:
        sea_photon_dataset['bckgrd_int_height'] = bckgrd_int_height

    return sea_photon_dataset


def interpolate_by_time(source_time, source_values, target_time):
    """Linearly interpolate source_values from source_time to target_time."""
    source_time = np.asarray(source_time)
    source_values = np.asarray(source_values)
    target_time = np.asarray(target_time)

    valid_mask = np.isfinite(source_time) & np.isfinite(source_values)
    if valid_mask.sum() < 2:
        return np.full_like(target_time, np.nan, dtype=float)

    sorted_idx = np.argsort(source_time[valid_mask])
    sorted_time = source_time[valid_mask][sorted_idx]
    sorted_values = source_values[valid_mask][sorted_idx]

    unique_time, unique_idx = np.unique(sorted_time, return_index=True)
    unique_values = sorted_values[unique_idx]
    if unique_time.size < 2:
        return np.full_like(target_time, unique_values[0], dtype=float)

    return np.interp(target_time, unique_time, unique_values, left=unique_values[0], right=unique_values[-1])


def apply_optional_solar_background_filter(
    sea_photon_dataset,
    enabled=False,
    day_threshold=6.0,
    median_window_deg=10.0,
    noise_multiplier=1.5,
    min_signal_conf=2
):
    """
    Optionally filter photons under strong daytime solar background conditions.

    Uses a degree-based rolling median of background_rate over solar_elevation
    to compute the expected background at each elevation. Photons whose actual
    background_rate exceeds expected * noise_multiplier are flagged noisy.
    Noisy photons are removed unless their signal confidence >= min_signal_conf.

    Flowchart step: 5 — Compute solar background from atmospheric signal (using
    selected x and z bin length scales) and subtract from subsurface photon
    histogram data.
    """
    if not enabled:
        return sea_photon_dataset

    required_cols = {'solar_elevation', 'background_rate', 'photon_conf'}
    if not required_cols.issubset(set(sea_photon_dataset.columns)):
        logger.warning("Solar background filter skipped because required columns are missing.")
        return sea_photon_dataset

    filtered_dataset = sea_photon_dataset.copy()
    daytime_mask = filtered_dataset['solar_elevation'] >= day_threshold

    # Select valid daytime photons for rolling median computation
    daytime = filtered_dataset.loc[daytime_mask].copy()
    valid = np.isfinite(daytime['background_rate']) & np.isfinite(daytime['solar_elevation'])
    daytime = daytime.loc[valid]

    if len(daytime) < 10:
        logger.info(
            "Solar background filter enabled, but fewer than 10 valid daytime "
            "photons were found. Skipping filter."
        )
        return filtered_dataset

    # Clamp solar elevations to physical range [DAY_THRESHOLD, 90] to avoid
    # HDF5 fill values (e.g. 3.4e+38) blowing up the bin array.
    daytime = daytime.copy()
    daytime['solar_elevation'] = daytime['solar_elevation'].clip(
        lower=day_threshold, upper=90.0
    )

    # Vectorized degree-based rolling median:
    # Bin elevations into fine steps, compute one median per bin using
    # np.searchsorted on the sorted array, then map back to photons.
    daytime = daytime.sort_values('solar_elevation')
    elevations = daytime['solar_elevation'].values
    bg_rates = daytime['background_rate'].values
    half_window = median_window_deg / 2.0

    bin_step = 0.1  # degrees — fine enough for smooth curve
    bin_centers = np.arange(
        elevations[0], elevations[-1] + bin_step, bin_step
    )
    bin_medians = np.empty(len(bin_centers))
    for i in range(len(bin_centers)):
        lo = np.searchsorted(elevations, bin_centers[i] - half_window, side='left')
        hi = np.searchsorted(elevations, bin_centers[i] + half_window, side='right')
        if lo < hi:
            bin_medians[i] = np.median(bg_rates[lo:hi])
        else:
            bin_medians[i] = np.nan

    # Map bin medians back to each photon via nearest-bin lookup (vectorized)
    bin_indices = np.searchsorted(bin_centers, elevations, side='right') - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_centers) - 1)
    expected_bg = bin_medians[bin_indices]

    # Map expected background onto the full DataFrame. Non-daytime and
    # non-finite daytime rows stay NaN, so they are never flagged noisy.
    filtered_dataset['expected_bg'] = np.nan
    filtered_dataset.loc[daytime.index, 'expected_bg'] = expected_bg

    # Flag noisy daytime photons
    noisy_daytime_mask = daytime_mask & (
        filtered_dataset['background_rate'] >= filtered_dataset['expected_bg'] * noise_multiplier
    )
    keep_mask = (~noisy_daytime_mask) | (filtered_dataset['photon_conf'] >= min_signal_conf)

    # Clean up temporary column
    filtered_dataset.drop(columns=['expected_bg'], inplace=True, errors='ignore')

    before_count = len(filtered_dataset)
    filtered_dataset = filtered_dataset.loc[keep_mask].copy()
    after_count = len(filtered_dataset)
    logger.info(
        "Solar background filter (multiplier=%.1f) removed %s photons (from %s to %s).",
        noise_multiplier, before_count - after_count, before_count, after_count
    )
    return filtered_dataset


def apply_optional_ir_ap_filter(
    sea_photon_dataset,
    enabled=False,
    quality_max=0,
    min_signal_conf=0
):
    """
    Optionally remove photons flagged as afterpulse using the quality_ph bitmask.

    Flowchart step: 2 — Remove photons flagged as afterpulse or impulse response.

    ATL03 v007 quality_ph bit encoding:
      bit 0 (value 1) = possible afterpulse  <- targeted by this filter
      bit 1 (value 2) = possible impulse response effect
      bit 2 (value 4) = possible TEP

    Only photons with bit 0 set (afterpulse) are removed. Photons flagged
    for impulse response or TEP only are kept, avoiding over-filtering on
    turbid high-background sites where many photons carry non-zero flags.
    """
    if not enabled:
        return sea_photon_dataset

    filtered_dataset = sea_photon_dataset.copy()
    keep_mask = np.ones(len(filtered_dataset), dtype=bool)

    if 'quality_ph' in filtered_dataset.columns:
        quality_values = pd.to_numeric(filtered_dataset['quality_ph'], errors='coerce').fillna(0).astype(int)
        afterpulse_mask = (quality_values & 1) == 1   # bit 0 set = afterpulse
        keep_mask &= ~afterpulse_mask
        logger.info(
            "IR/AP filter: %d photons flagged as afterpulse (quality_ph bit 0) will be removed.",
            int(afterpulse_mask.sum())
        )
    else:
        logger.warning("IR/AP filter enabled, but quality_ph is missing.")

    if 'photon_conf' in filtered_dataset.columns:
        conf_values = pd.to_numeric(filtered_dataset['photon_conf'], errors='coerce')
        keep_mask &= conf_values >= min_signal_conf
    else:
        logger.warning("IR/AP filter enabled, but photon_conf is missing.")

    before_count = len(filtered_dataset)
    filtered_dataset = filtered_dataset.loc[keep_mask].copy()
    after_count = len(filtered_dataset)
    logger.info(
        "IR/AP proxy filter removed %s photons (from %s to %s).",
        before_count - after_count, before_count, after_count
    )
    return filtered_dataset


def Extract_sea_photons(IS2_atl03_mds, target_strong_beams, shoreline_data_path):
    Segment_ID = {}
    Segment_Index_begin = {}
    Segment_PE_count = {}
    Equator_Segment_Distance = {}
    Segment_Length = {}
    Segment_Is_Land = {}
    Segment_Lon = {}
    Segment_Lat = {}
    Segment_Elev = {}
    Segment_Time = {}
    Segment_ref_elev = {}
    Segment_ref_azimuth = {}
    background_rate = {}
    background_counts = {}
    
    # Initialize a list to store data for each beam
    beam_datasets = []
    
    
    # Loop over each strong beam in target_strong_beams
    for gtx in target_strong_beams:
        print('Processing strong beam ID:', gtx)
        
        # Access the data for the current beam
        IS2_val = IS2_atl03_mds[gtx]
        
        # Initialize dictionaries to store segment data for the beam
        Segment_ID[gtx] = IS2_val['geolocation']['segment_id']
        n_seg = len(Segment_ID[gtx])
        n_pe, = IS2_val['heights']['delta_time'].shape
        Segment_Index_begin[gtx] = IS2_val['geolocation']['ph_index_beg'] - 1
        Segment_PE_count[gtx] = IS2_val['geolocation']['segment_ph_cnt']
        Equator_Segment_Distance[gtx] = IS2_val['geolocation']['segment_dist_x']
        Segment_Length[gtx] = IS2_val['geolocation']['segment_length']
        delta_time = IS2_val['geolocation']['delta_time']
        segment_lat = IS2_val['geolocation']['reference_photon_lat'][:].copy()
        segment_lon = IS2_val['geolocation']['reference_photon_lon'][:].copy()
        ref_elev = IS2_val['geolocation']['ref_elev'][:].copy()
        ref_azimuth = IS2_val['geolocation']['ref_azimuth'][:].copy()
        geoid = IS2_val['geophys_corr']['geoid'][:].copy()
        h_ph = IS2_val['heights']['h_ph'][:].copy()
        photon_delta_time = IS2_val['heights']['delta_time'][:].copy()
        lat_ph = IS2_val['heights']['lat_ph'][:].copy()
        lon_ph = IS2_val['heights']['lon_ph'][:].copy()
        signal_conf_photon = IS2_val['heights']['signal_conf_ph'].copy()
        x_atc = IS2_val['heights']['dist_ph_along'][:].copy()
        y_atc = IS2_val['heights']['dist_ph_across'][:].copy()
        quality_ph = IS2_val['heights']['quality_ph']

        # Optional variables for solar background sensitivity testing
        photon_solar_elevation = np.full_like(photon_delta_time, np.nan, dtype=float)
        photon_background_rate = np.full_like(photon_delta_time, np.nan, dtype=float)
        if 'solar_elevation' in IS2_val['geolocation']:
            segment_solar_elevation = IS2_val['geolocation']['solar_elevation'][:].copy()
            photon_solar_elevation = interpolate_by_time(
                delta_time, segment_solar_elevation, photon_delta_time
            )
        photon_bckgrd_int_height = np.full_like(photon_delta_time, np.nan, dtype=float)
        if 'bckgrd_atlas' in IS2_val and 'bckgrd_rate' in IS2_val['bckgrd_atlas'] and 'delta_time' in IS2_val['bckgrd_atlas']:
            bckgrd_rate = IS2_val['bckgrd_atlas']['bckgrd_rate'][:].copy()
            bckgrd_time = IS2_val['bckgrd_atlas']['delta_time'][:].copy()
            photon_background_rate = interpolate_by_time(
                bckgrd_time, bckgrd_rate, photon_delta_time
            )
            if 'bckgrd_int_height' in IS2_val['bckgrd_atlas']:
                bckgrd_int_height = IS2_val['bckgrd_atlas']['bckgrd_int_height'][:].copy()
                photon_bckgrd_int_height = interpolate_by_time(
                    bckgrd_time, bckgrd_int_height, photon_delta_time
                )

        # Adjust x_atc based on segment distances        
        for seg_index in range(n_seg):
            idx = Segment_Index_begin[gtx][seg_index]
            cnt = Segment_PE_count[gtx][seg_index]
            x_atc[idx:idx + cnt] += Equator_Segment_Distance[gtx][seg_index]

        # Calculate relative distances        
        relative_AT_dist = (x_atc - x_atc[0]) / 1000
        relative_seg_dist = (Equator_Segment_Distance[gtx] - Equator_Segment_Distance[gtx][0]) / 1000
        
        # Create a GeoDataFrame to hold segment data for shoreline check
        Segment_Is_Land['geometry'] = gpd.points_from_xy(segment_lon, segment_lat)
        ICESat2_GDF = gpd.GeoDataFrame(Segment_Is_Land, crs="EPSG:4326")
        
        # Determine if it is land by the land/sea mask
        Segment_Is_Land_Labels = isolate_sea_land_photons(shoreline_data_path, ICESat2_GDF)
        ICESat2_GDF.loc[:, 'is_land'] = Segment_Is_Land_Labels

        # Apply interpolations for required data
        is_land_label_interp1d = apply_interpolation(interpolate_labels(segment_lat, Segment_Is_Land_Labels), lat_ph)
        ph_ref_elev = apply_interpolation(interpolate_labels(segment_lat, ref_elev), lat_ph)
        ph_ref_azimuth = apply_interpolation(interpolate_labels(segment_lat, ref_azimuth), lat_ph)
        ph_geoid = apply_interpolation(interpolate_labels(segment_lat, geoid), lat_ph)


        # Create photon DataFrame for the current beam 
        sea_photon_dataset = create_photon_dataframe(lat_ph=lat_ph, lon_ph=lon_ph,
                                            ref_elev=ph_ref_elev,ref_azimuth=ph_ref_azimuth,
                                            geoid=ph_geoid,h_ph=h_ph,
                                              quality_ph=quality_ph,
                                              is_land_label_interp1d=is_land_label_interp1d,
                                              signal_conf_photon=signal_conf_photon,
                                              x_atc=x_atc,  # Note: x_atc is not used in the function. Remove if unnecessary.
                                              relative_AT_dist=relative_AT_dist,
                                              solar_elevation=photon_solar_elevation,
                                              background_rate=photon_background_rate,
                                              bckgrd_int_height=photon_bckgrd_int_height)

        # Filter out land photons
        sea_photon_dataset = sea_photon_dataset[sea_photon_dataset['is_land_label'] != 1]
        
        # Add a new column to indicate the beam ID
        sea_photon_dataset['beam_id'] = gtx
        
        # Append the processed dataset for the current beam to the list
        beam_datasets.append(sea_photon_dataset)
        
    # Concatenate all beam data into a single DataFrame
    all_beams_dataset = pd.concat(beam_datasets, ignore_index=True)
    
    return all_beams_dataset


def filter_photon_dataset_by_hull_area(photon_dataset, hull_area_threshold=3000):
    """
    Filters photon dataset based on ConvexHull area threshold and returns the filtered dataset,
    convex hull areas, and convex hull points.
    """
    lat_bins_grouped = photon_dataset.groupby('lat_bins', observed=False)
    filtered_dataset = photon_dataset.copy()
    
    convex_hulls = {}
    convex_hull_areas = {}
    
    for lat_bin, bin_data in lat_bins_grouped:
        if len(bin_data) >= 3:
            points = bin_data[['lat', 'photon_height']].to_numpy()
            hull = ConvexHull(points)
            area = hull.volume
            convex_hull_areas[lat_bin] = area
            
            # Store the ConvexHull points if area meets the threshold
            if area >= hull_area_threshold:
                convex_hulls[lat_bin] = points[hull.vertices]
            else:
                # Remove bins with hull area below the threshold
                filtered_dataset = filtered_dataset[filtered_dataset['lat_bins'] != lat_bin]
                
    return filtered_dataset, convex_hull_areas, convex_hulls





    



##########################
##Discard Functions Below 
##########################


#  Extracts key information from filenames, 
#  which could include processed status, product ID, timestamps, identifiers, or metadata.
def extract_file_params(file_path):
    '''
    Example: input "processed_ATL06_20231115120000_20231115_001_12_data.h5"
            Output: ('processed_', 'ATL06', '2023', '11', '15', '12', '00', '00', 
            '2023', '11', '15', '001', '12', '_data')
    '''
    # Defines a regex pattern to match strings in the file path
    rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})'
                    r'(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
    # Searches for all matches of the regex pattern in the given
    params = rx.findall(file_path).pop()
    return params


def create_mask(binned_data, sea_surface_height, seafloor_height, threshold_height):
    '''
    create a mask for filtering sea surface, sea floor, and threshold_height
    '''
    mask = (binned_data['height'] <= sea_surface_height) & \
           (binned_data['height'] >= seafloor_height) & \
           (binned_data['height'] >= threshold_height)
    return mask


def generate_polygons_from_binned_data(lat, height, mask, lat_interval, height_interval):
    '''
    horizontal_vertical_bin_dataset
    Generate polygons for each bin after masking within a rectangle formed by height and latitude intervals
    '''
    # Create latitude bins based on the specified interval
    lat_bins = np.arange(lat.min(), lat.max() + lat_interval, lat_interval)
    
    # Create height bins within the range [-12, 2] based on the specified interval
    height_bins = np.arange(-12, 2 + height_interval, height_interval)
    
    # Identify which bin each masked latitude and height value belongs to
    lat_bin_indices = np.digitize(lat[mask], lat_bins)
    
    # Determine which height bin each masked point belongs to
    height_bin_indices = np.digitize(height[mask], height_bins)
    
    # Combine latitude and height bin indices for each point
    combined_bins = list(zip(lat_bin_indices, height_bin_indices))
    
    # Find unique bin combinations and count the number of points in each bin    
    unique_bins, counts = np.unique(combined_bins, axis=0, return_counts=True)
    
    # Filter bins to include only those within valid index ranges   
    valid_bins = [(bin[0], bin[1]) for bin in unique_bins if 1 <= bin[0] < len(lat_bins) and 1 <= bin[1] < len(height_bins)]

    polygons = []
    for bin_lat, bin_height in valid_bins:
        if bin_lat > 0 and bin_lat < len(lat_bins) and bin_height > 0 and bin_height < len(height_bins):
            bin_points = np.array([(lat[mask][i], height[mask][i]) for i in range(sum(mask))
                                   if lat_bin_indices[i] == bin_lat and height_bin_indices[i] == bin_height])
            if len(bin_points) > 2:
                hull = ConvexHull(bin_points)
                polygons.append(bin_points[hull.vertices])

    return lat_bins, height_bins, valid_bins, polygons


    
