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





# requires that the input gdf has ranged index values i
# will need to change if index is changed to time or something
# this currently checks point in polygon for EVERY point
# would be significantly sped up if evaluated at 10m or something similar
# maybe later, fine for now
import logging

# Set up logging
logger = logging.getLogger(__name__)

def isolate_sea_land_photons(shoreline_data_path, ICESat2_GDF):
    """
    Classify ICESat-2 points as land (1) or ocean (0) using shoreline data.
    
    Args:
        shoreline_data_path (str): Path to the shoreline dataset (e.g., shapefile).
        ICESat2_GDF (gpd.GeoDataFrame): GeoDataFrame with ICESat-2 points.
    
    Returns:
        np.ndarray: Array of 0s (ocean) or 1s (land) for each point.
    """
    try:
        # Ensure ICESat2_GDF has a valid geometry column and CRS
        if ICESat2_GDF.empty:
            logger.error("ICESat2_GDF is empty, cannot classify land/ocean.")
            return np.zeros(len(ICESat2_GDF), dtype=bool)  # Default to ocean

        # Confirm that ICESat2_GDF already has the correct CRS
        if ICESat2_GDF.crs is None or ICESat2_GDF.crs.to_string() != "EPSG:4326":
            logger.info("Setting CRS of ICESat2_GDF to EPSG:4326")
            ICESat2_GDF.set_crs("EPSG:4326", inplace=True)

        # Load shoreline dataset with bounding box optimization
        logger.info(f"Loading shoreline data from {shoreline_data_path}")
        try:
            land_polygon_gdf = gpd.read_file(shoreline_data_path, bbox=ICESat2_GDF, engine='pyogrio')
        except Exception as e:
            logger.warning(f"Failed to load shoreline data with pyogrio: {e}. Falling back to default engine.")
            land_polygon_gdf = gpd.read_file(shoreline_data_path, bbox=ICESat2_GDF)

        # Ensure the shoreline data is in the same CRS as ICESat2_GDF
        if land_polygon_gdf.crs.to_string() != "EPSG:4326":
            logger.info("Reprojecting shoreline data to EPSG:4326")
            land_polygon_gdf = land_polygon_gdf.to_crs("EPSG:4326")

        # Initialize land labels as 0 (ocean)
        land_point_labels = np.zeros(len(ICESat2_GDF), dtype=int)

        # Perform spatial join to find points within land polygons
        pts_in_land = gpd.sjoin(ICESat2_GDF, land_polygon_gdf, predicate='within')

        # Update labels for points that are within land polygons
        land_loc = ICESat2_GDF.index.isin(pts_in_land.index)
        land_point_labels[land_loc] = 1
        # Points not in land are already 0 (ocean)

        logger.info(f"Classified {np.sum(land_point_labels)} points as land, {len(land_point_labels) - np.sum(land_point_labels)} as ocean")
        return land_point_labels

    except Exception as e:
        logger.error(f"Error in isolate_sea_land_photons: {str(e)}")
        logger.error(f"ICESat2_GDF info: {ICESat2_GDF.head()}")
        logger.error("Returning False (ocean) for all points")
        return np.zeros(len(ICESat2_GDF), dtype=bool)  # Default to ocean


def create_photon_dataframe(lat_ph, lon_ph, ref_elev, ref_azimuth, geoid, h_ph, \
                            quality_ph, is_land_label_interp1d, signal_conf_photon, x_atc, relative_AT_dist):
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
    return sea_photon_dataset


    





    
