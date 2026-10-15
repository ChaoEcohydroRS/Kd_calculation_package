# utils/visualization.py

import numpy as np
import os
import pandas as pd
import time
import geopandas as gpd
from pyproj import Transformer, Proj
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def plot_photon_height(sea_photon_dataset, hlims=[-25, 10]):
    fig, ax = plt.subplots()
    ax.scatter(sea_photon_dataset['latitude'], sea_photon_dataset['photon_height'], s=1, c='k', alpha=0.15, edgecolors='none', label='ATL03 Photons')
    ax.set_xlabel('Relative AT Distance')
    ax.set_ylabel('Height (h_ph)')
    ax.set_title('Scatter Plot of ATL03 Photons')
    ax.set_ylim(hlims)
    ax.legend()
    plt.show()

def plot_filtered_seafloor_photons(filtered_seafloor_subsurface_dataset, sea_photon_dataset, sea_surface_height, output_path):
    """
    Plots the filtered seafloor photon data along with sea surface and seafloor elevation data.

    Parameters:
    - filtered_seafloor_subsurface_dataset: DataFrame containing the filtered subsurface photon data.
    - sea_photon_dataset: DataFrame containing the original sea photon data.    
    - sea_surface_height: Array of sea surface height values.
    - output_path: Path to save the output plot.
    """
    
    # get sea_surface_x_axis_bins
    sea_surface_x_axis_bins = np.linspace(filtered_seafloor_subsurface_dataset['relative_AT_dist'].min(), 
                                          filtered_seafloor_subsurface_dataset['relative_AT_dist'].max(),
                                          len(sea_surface_height))
    
    hlims = [-25, 10]
    fig, ax = plt.subplots()

    # Scatter plot of subsurface photons
    ax.scatter(sea_photon_dataset['relative_AT_dist'], sea_photon_dataset['photon_height'],
               s=1, c='k', alpha=0.15, edgecolors='none', label='Subsurface ATL03 Photons')

    # Overlay the seafloor elevation data as points
    ax.plot(filtered_seafloor_subsurface_dataset['relative_AT_dist'], filtered_seafloor_subsurface_dataset['seafloor_elevation'],
            linewidth=0.8, c='b', alpha=0.4, label='Seafloor Elevation')

    ax.plot(sea_surface_x_axis_bins, [x - 0.5 for x in sea_surface_height], 
            linewidth=0.8, color='#DD571C', alpha=0.4, label='0.5 m Below Surface Peak')

    # Set labels and title
    ax.set_xlabel('Distance From Start of Track (km)')
    ax.set_ylabel('Photon Height (m)')
    ax.set_title('Scatter Plot of Filtered Sea Photon Dataset')
    ax.set_ylim(hlims)

    # Add a legend
    ax.legend()

    # Save the plot
    plt.legend(loc='upper right')
    plt.savefig(output_path, dpi=400, format='jpeg')

    # Show the plot
    plt.show()


def plot_convex_hulls(photon_dataset, target_beam_ids, convex_hulls, convex_hull_areas):
    """
    Plots ConvexHull polygons and annotated areas for each lat_bin group in the dataset.
    """
    #filter beam type
    # photon_dataset
    photon_dataset = photon_dataset[photon_dataset['beam_id'].isin(target_beam_ids)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    hlims = [-10, 2]
    
    for lat_bin, hull_points in convex_hulls.items():
        hull = ConvexHull(hull_points)
        ax.fill(hull_points[hull.vertices, 0], hull_points[hull.vertices, 1], alpha=0.3, label=f'Bin {lat_bin}')
        
        # Calculate centroid to place the label
        centroid = np.mean(hull_points[hull.vertices], axis=0)
        ax.text(centroid[0], centroid[1], f'{convex_hull_areas[lat_bin]:.2f}', 
                horizontalalignment='center', verticalalignment='center', fontsize=10, color='black')
    
    ax.scatter(photon_dataset['lat'], photon_dataset['photon_height'],
               s=1, c='k', alpha=0.15, edgecolors='none', label='Subsurface ATL03 Photons')
    ax.set_ylim(hlims)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Photon Height')
    ax.set_title('ConvexHull of Photons within each Lat Bin')
    plt.show()

# plot photons coloured by quality_ph flag for diagnostic purposes
def plot_photon_quality_flags(output_path, timestamp, sea_photon_dataset, target_beam_ids):
    """
    Scatter plot of photon height vs. along-track distance, coloured by quality_ph value.

    quality_ph bit meanings (ATL03 v007):
      bit 0 (value 1) = possible afterpulse
      bit 1 (value 2) = possible impulse response effect
      bit 2 (value 4) = possible TEP
    """
    dataset = sea_photon_dataset[sea_photon_dataset['beam_id'].isin(target_beam_ids)].copy()
    if dataset.empty:
        return

    quality_colors = {
        0: ('dimgray',  'nominal (0)'),
        1: ('red',      'afterpulse (1)'),
        2: ('orange',   'impulse response (2)'),
        3: ('darkred',  'afterpulse + impulse (3)'),
        4: ('royalblue','TEP (4)'),
        5: ('purple',   'afterpulse + TEP (5)'),
        6: ('cyan',     'impulse + TEP (6)'),
        7: ('black',    'all flags (7)'),
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    unique_flags = sorted(dataset['quality_ph'].dropna().astype(int).unique())
    for flag in unique_flags:
        subset = dataset[dataset['quality_ph'].astype(int) == flag]
        color, label = quality_colors.get(flag, ('magenta', f'unknown ({flag})'))
        ax.scatter(subset['relative_AT_dist'], subset['photon_height'],
                   s=0.5, c=color, alpha=0.3, edgecolors='none', label=label)

    ax.set_xlabel('Relative Along-Track Distance (km)', fontsize=12)
    ax.set_ylabel('Photon Height (m)', fontsize=12)
    ax.set_title('Photon quality_ph flag distribution', fontsize=13)
    ax.set_ylim([-15, 5])
    ax.legend(loc='lower right', markerscale=6, fontsize=9)
    fig.tight_layout()
    save_path = os.path.join(output_path, f'{timestamp}_quality_ph_flags.jpg')
    plt.savefig(save_path, dpi=300, format='jpeg')
    plt.show()
    print(f"Quality flag plot saved: {save_path}")

    # Print summary counts
    print("\nquality_ph flag summary:")
    for flag in unique_flags:
        count = (dataset['quality_ph'].astype(int) == flag).sum()
        _, label = quality_colors.get(flag, ('', f'unknown ({flag})'))
        pct = 100.0 * count / len(dataset)
        print(f"  {label:35s}: {count:>8,d}  ({pct:.1f}%)")


# plot the kd and photon
def plot_kd_photons(OutputPath, timestamp, target_beam_ids, subsurface_photon_dataset, Kd_DF_MergedDistance,
                    all_photon_dataset=None):
    #filter beam type
    subsurface_photon_dataset = subsurface_photon_dataset[subsurface_photon_dataset['beam_id'].isin(target_beam_ids)]
    Kd_DF_MergedDistance = Kd_DF_MergedDistance[Kd_DF_MergedDistance['beam_id'].isin(target_beam_ids)]

    hlims = [-25, 5]
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.set_facecolor('#0d1117')

    # Background: all photons (unfiltered)
    if all_photon_dataset is not None:
        all_beam = all_photon_dataset[all_photon_dataset['beam_id'].isin(target_beam_ids)]
        ax1.scatter(all_beam['relative_AT_dist'], all_beam['photon_height'],
                    s=0.3, c='cyan', alpha=0.05, edgecolors='none', rasterized=True,
                    label='All ATL03 Photons')

    # Foreground: subsurface photons that survived filtering
    ax1.scatter(subsurface_photon_dataset['relative_AT_dist'], subsurface_photon_dataset['photon_height'],
                s=1.5, c='cyan', alpha=0.25, edgecolors='none', label='Subsurface Photons (filtered)')
    ax1.set_xlabel('Relative Along-Track Distance (km)', fontsize=14)
    ax1.set_ylabel('Photon Height (m)', fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)
    if 'seafloor_elevation' in subsurface_photon_dataset.columns:
        ax1.plot(subsurface_photon_dataset['relative_AT_dist'], subsurface_photon_dataset['seafloor_elevation'],
                 linewidth=0.8, c='b', alpha=0.4, label='Seafloor Elevation')
    ax1.set_ylim(hlims)

    ax2 = ax1.twinx()
    ax2.scatter(Kd_DF_MergedDistance['relative_AT_dist'], Kd_DF_MergedDistance['kd'],
                label='Kd values', color='r', alpha=0.6)
    ax2.set_ylabel('Kd (m⁻¹)', color='r', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='r', labelsize=12)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.85, 0.85))
    fig.tight_layout()
    plt.savefig(os.path.join(OutputPath, f'{timestamp}_IS2_subsurface_kd_2.jpg'), dpi=300, format='jpeg')
    plt.close()



def plot_bin_polygon_data(lat, height, seafloor_height, mask, lat_bins, height_bins, valid_bins, polygons, y_min, y_max):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Height Data")
    plt.scatter(lat, height, c=height, cmap='viridis', s=10)
    plt.plot(np.sort(lat), seafloor_height[np.argsort(lat)], 'r-', label='Seafloor')
    plt.colorbar(label='Height')
    plt.xlabel('Latitude')
    plt.ylabel('Height')
    plt.ylim(y_min, y_max)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Masked Region (ROI)")
    plt.scatter(lat[mask], height[mask], c=height[mask], cmap='viridis', s=10)
    plt.plot(np.sort(lat), seafloor_height[np.argsort(lat)], 'r-', label='Seafloor')

    for polygon in polygons:
        plt.gca().add_patch(plt.Polygon(polygon, edgecolor='red', facecolor='none', linewidth=1))

    for lb in lat_bins:
        plt.axvline(x=lb, color='gray', linestyle='--', linewidth=0.5)
    for hb in height_bins:
        plt.axhline(y=hb, color='gray', linestyle='--', linewidth=0.5)

    plt.colorbar(label='Height')
    plt.xlabel('Latitude')
    plt.ylabel('Height')
    plt.ylim(y_min, y_max)
    plt.legend()

    plt.tight_layout()
    plt.show()

#
def produce_figures(binned_data, bath_height, sea_height, solo_sea_surface_label,
                    y_limit_top, y_limit_bottom, percentile, file, geo_df,
                    ref_y, ref_z, beam, epsg_num):
    """Create figures"""

    # Create bins for latitude
    bath_x_axis_bins = np.linspace(binned_data.lat.min(),
                              binned_data.lat.max(), len(bath_height))+20

    sea_surface_x_axis_bins = np.linspace(binned_data.lat.min(),
                              binned_data.lat.max(), len(sea_height))+10

    # Create new dataframes for median values
    bath_median_df = pd.DataFrame({'x': bath_x_axis_bins, 'y': bath_height})

    # Create uniform sea surface based on median sea surface values and filter out surface breaching
    sea_height1 = [np.nanmedian(sea_height) if i == i else np.nan for i in sea_height]
    sea_median_df = pd.DataFrame({'x': sea_surface_x_axis_bins, 'y': sea_height1})

    # Create uniform solo sea surface label
    sea_surface_label = solo_sea_surface_label
    sea_surface_label_df = pd.DataFrame({'x': sea_surface_x_axis_bins, 'y': sea_surface_label})
    idx_1 = np.where(sea_surface_label_df.y == 1)
    idx_0 = np.where(sea_surface_label_df.y == 0)

    # Define figure size
    fig = plt.rcParams["figure.figsize"] = (40, 25)

    # Plot raw points
    #     plt.scatter(x=binned_data.lat,
    #     y = binned_data.photon_height, marker='o', lw=0, s=1, alpha = 0.8,
    #     c = 'yellow', label = 'Raw photon height')
    plt.scatter(ref_y, ref_z, s=0.5, alpha=0.1, c='black')
    plt.scatter(geo_df.lat, geo_df.photon_height, s=0.8, marker = 'o',
                alpha=0.1, c='red', label='Classified Photons')

    # plt.scatter(x=geo_df.lat,
    # y = geo_df.photon_height, marker='o', lw=0, s=0.8,
    # alpha = 0.8, c = 'black', label = 'Corrected photon bin')

    # Plot median values
    plt.scatter(bath_median_df.x, bath_median_df.y,
                marker='o', c='r', alpha=0.8, s=2, label='Median bathymetry')

    plt.scatter(sea_median_df.x, sea_median_df.y,
                marker='o', c='b', alpha=1, s=2, label='Median sea surface')

    plt.scatter(sea_surface_label_df.iloc[idx_1].x, sea_surface_label_df.iloc[idx_1].y,
                marker='o', c='pink', alpha=1, s=3, label='solo_sea_surface')
    plt.scatter(sea_surface_label_df.iloc[idx_0].x, sea_surface_label_df.iloc[idx_0].y,
                marker='o', c='g', alpha=1, s=3, label='non_solo_sea_surface')

    # Insert titles and subtitles
    plt.title('Icesat2 Bathymetry\n' + file)
    plt.xlabel('Latitude', fontsize=25)
    plt.ylabel('Photon Height (m)', fontsize=25)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(loc="upper left", prop={'size': 20})

    # Limit the x and y axes using parameters
    plt.xlim(left=binned_data.lat.min(), right=binned_data.lat.max())
    plt.ylim(top=y_limit_top, bottom=y_limit_bottom)

    timestr = time.strftime("%Y%m%d_%H%M%S")
    file = file.replace('.h5', '')
    # Define where to save file
    plt.tight_layout()
    plt.savefig("C:/Workstation/ICESat2_HLS/" + file + '_gt' +
                str(beam) + '_' + str(percentile) +
                '_EPSG' + str(epsg_num) + '_' + timestr + ".pdf")
    # plt.show()
    # plt.close()

    # convert corrected locations back to wgs84 (useful to contain)
    transformer = Transformer.from_crs("EPSG:" + str(epsg_num),
                                       "EPSG:4326", always_xy=True)
    print(transformer)
    lon_wgs84, lat_wgs84 = transformer.transform(
        geo_df.lon.values, geo_df.lat.values)

    geo_df['lon_wgs84'] = lon_wgs84
    geo_df['lat_wgs84'] = lat_wgs84

    geodf = gpd.GeoDataFrame(geo_df,
                             geometry=gpd.points_from_xy(geo_df.lon_wgs84,
                                                         geo_df.lat_wgs84))

    geodf.set_crs(epsg=4326, inplace=True)

    # geodf.to_file("C:/Workstation/ICESat2_HLS/" +file + '_gt' + '_' + str(percentile) + '_EPSG' +
    #               str(epsg_num) + '_' + timestr + ".gpkg",
    #               driver="GPKG")