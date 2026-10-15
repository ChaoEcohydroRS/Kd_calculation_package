# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:25:53 2024

@author: wayne
"""

from kd_utils import *
import scipy.interpolate


path = 'C:/Workstation/ICESat2_HLS/Kd_ComparisionPaper/Dataset/ATL03_ICESat2/'
ATL03_h5_file = "processed_ATL03_20220122044818_04721407_006_01.h5"
ATL03_h5_file_path = path + ATL03_h5_file


# Set the shoreline data path
Current_Path='C:/Workstation/ICESat2_HLS/Kd_ComparisionPaper/Dataset/'
    
#load the shoreline package    
shoreline_data_path = Current_Path+'Shorelines/GeoPkgGlobalShoreline.gpkg'



# speed of light
c = 299792458.0

# bin data first based on horizontal resolution
horizontal_res=10

# vertical resolution
# vertical_res=0.1
vertical_res=0.25
# vertical_res=1

# sea subsurface thresh
# determine subsurface by setting how depth below the sea surface
subsurface_thresh = 0.5


# define specific rectangular target zone for batch analysis
# i.e., From subsurface depth 0.5 to specific KdMaxDepth depth (e.g, 6)
Kd_max_depth = 6


########################################
#readATL03_github
IS2_atl03_mds,IS2_atl03_attrs,IS2_atl03_beams = read_granule(ATL03_h5_file_path,ATTRIBUTES=True)

# extract parameters from ICESat-2 ATLAS HDF5 file name
rx = re.compile(r'(processed_)?(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})'
    r'(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})(.*?).h5$')
SUB,PRD,YY,MM,DD,HH,MN,SS,TRK,CYCL,GRAN,RL,VERS,AUX = rx.findall(ATL03_h5_file_path).pop()


# initialize data storage variables  
# variables of interest for generating three types classification
Segment_ID = {}
Segment_Index_begin = {}
Segment_PE_count = {}

Equator_Segment_Distance = {}
Segment_Length = {}

Segment_Is_Land={}

# mean geolocation, height and delta time
Segment_Lon = {}
Segment_Lat = {}
Segment_Elev = {}
Segment_Time = {}

Segment_ref_elev= {}
Segment_ref_azimuth= {}

# relative_AT_dist = {}
# relative_seg_dist = {}

# background photon rate
background_rate = {}
background_counts={}



########################################
#proc02_manually_select_water_forGitHub
#remove water surface based on simple 0.5 m


#raw beam-by-beam photon data after cutoff below the max value 0.5 below peak
IS2_atl03Cut50cmBelowPeak={}

# for each input beam within the file
for gtx in sorted(IS2_atl03_beams[0:1]):
# for gtx in sorted(IS2_atl03_beams):
    # data and attributes for beam gtx
    IS2_val = IS2_atl03_mds[gtx]
    IS2_attrs = IS2_atl03_attrs[gtx]
    
    # ATL03 Segment ID
    Segment_ID[gtx] = IS2_val['geolocation']['segment_id']
    # number of valid overlapping ATL03 segments
    n_seg = len(Segment_ID[gtx])
    # number of photon events
    n_pe, = IS2_val['heights']['delta_time'].shape
    
    # first photon ID (1-based) in each segment (convert to 0-based indexing)
    Segment_Index_begin[gtx] = IS2_val['geolocation']['ph_index_beg'] - 1
    
    # number of photon events in the segment
    Segment_PE_count[gtx] = IS2_val['geolocation']['segment_ph_cnt']
    
    # along-track distance from the equator crossing to the start of each ATL03 20 meter geolocation segment
    Equator_Segment_Distance[gtx] = IS2_val['geolocation']['segment_dist_x']
    
    # along-track length for each ATL03 segment
    Segment_Length[gtx] = IS2_val['geolocation']['segment_length']
    
    # Transmit time of the reference photon
    delta_time = IS2_val['geolocation']['delta_time']
    
    # get geolocation lat/lon
    segment_lat = IS2_val['geolocation']['reference_photon_lat'][:].copy()
    segment_lon= IS2_val['geolocation']['reference_photon_lon'][:].copy()
    
    # get parameters ref elev and azimuth for refraction correction
    ref_elev = IS2_val['geolocation']['ref_elev'][:].copy()
    ref_azimuth = IS2_val['geolocation']['ref_azimuth'][:].copy()
    
    #get geoid for converting WGS84 ellipsoid to geoid correction 
    geoid = IS2_val['geophys_corr']['geoid'][:].copy()
        
    # photon event heights
    h_ph = IS2_val['heights']['h_ph'][:].copy()
    lat_ph = IS2_val['heights']['lat_ph'][:].copy()
    lon_ph = IS2_val['heights']['lon_ph'][:].copy()
#     dist_ph_along = IS2_val['heights']['dist_ph_along'][:].copy()
    signal_conf_photon = IS2_val['heights']['signal_conf_ph'].copy()

    # along-track and across-track distance for photon events
    x_atc = IS2_val['heights']['dist_ph_along'][:].copy()
    y_atc = IS2_val['heights']['dist_ph_across'][:].copy()
    
    # photon quality
    quality_ph=IS2_val['heights']['quality_ph']
    
    #calculate along track distance
    for seg_index in range(n_seg):
        # index for 20m segment j
        idx = Segment_Index_begin[gtx][seg_index]
        # number of photons in 20m segment
        cnt = Segment_PE_count[gtx][seg_index]
        # add segment distance to along-track coordinates
        x_atc[idx:idx+cnt] += Equator_Segment_Distance[gtx][seg_index]
        
    # calculate along track distance relative to the beginning of the cut segment
    relative_AT_dist=(x_atc-x_atc[0])/1000
    
    
    relative_seg_dist=(Equator_Segment_Distance[gtx]-Equator_Segment_Distance[gtx][0])/1000
    
    
    # this function is a significant slowdown without pygeos
    Segment_Is_Land['geometry'] = gpd.points_from_xy(segment_lon, segment_lat)
    
    # create a geo dataframe 
    ICESat2_GDF = gpd.GeoDataFrame(Segment_Is_Land,crs="EPSG:4326")
    
    #Step1: isolate water using landmask
    # spatial joint to determine land/sea mask
    Segment_Is_Land_Labels=isolate_sea_land_photons(shoreline_data_path,ICESat2_GDF)
       
#     # set the labels back to ICESat2 geopandas dataframe
    ICESat2_GDF.loc[:, 'is_land'] = Segment_Is_Land_Labels
    
    # interpolate is_land labels based on photon lat  
    island_interp1d_model = scipy.interpolate.interp1d(segment_lat[:],
                                                           Segment_Is_Land_Labels[:],
                                                           fill_value="extrapolate")
    
    # Apply interpid model
    is_land_label_interp1d = island_interp1d_model(lat_ph[:])

    # Ref_elev on a per photon level (assign seg ref_elev to photons)
    ref_elev_interp1d_model = scipy.interpolate.interp1d(segment_lat[:],
                                                         ref_elev[:],
                                                         fill_value="extrapolate")
    # apply interpid model
    ph_ref_elev = ref_elev_interp1d_model(lat_ph[:])

    # Ref_azimuth on a per photon level (assign seg ref_azimuth to photons)
    ref_azimuth_interp1d_model = scipy.interpolate.interp1d(segment_lat[:],
                                                            ref_azimuth[:],
                                                            fill_value="extrapolate")
    
    # apply interpid model
    ph_ref_azimuth = ref_azimuth_interp1d_model(lat_ph[:])
    
    
    # interpolate geoid based on segment lat  
    geoid_interp1d_model = scipy.interpolate.interp1d(segment_lat[:],
                                                           geoid[:],
                                                           fill_value="extrapolate")
    
    # Apply interpid model
    ph_geoid = geoid_interp1d_model(lat_ph[:])
    
    
    #geoid correct
    h_ph_geoid_cor = h_ph[:] - ph_geoid[:]
    
    # Find the epsg code
    epsg_code = convert_wgs_to_utm(lon_ph[0], lat_ph[0])
    epsg_num = int(epsg_code.split(':')[-1])
    
    # Orthometrically correct the data using the epsg code
    lat_utm, lon_utm, h_ph_cor = orthometric_correction(lat_ph, lon_ph, h_ph, epsg_code)

    # Aggregate data into dataframe
    sea_photon_dataset = \
        pd.DataFrame({'latitude': lat_ph, 'longitude': lon_ph,
                      'lat': lat_utm, 'lon': lon_utm,
                      'photon_height': h_ph_geoid_cor,
                      'quality_ph': quality_ph,
                      'is_land_label': is_land_label_interp1d,
                      'photon_conf': signal_conf_photon,
                      'ref_elevation': ph_ref_elev,
                      'ref_azimuth': ph_ref_azimuth,
                     'relative_AT_dist':relative_AT_dist},
                     columns=['latitude', 'longitude',
                              'lat', 'lon',
                              'photon_height', 
                              'quality_ph',
                              'is_land_label', 'photon_conf',
                              'ref_elevation', 'ref_azimuth',
                              'relative_AT_dist'])
    
    # #remove the saturated photons
    # sea_photon_dataset=sea_photon_dataset[sea_photon_dataset['quality_ph'] < 1]
    
    #remove the land photons
    sea_photon_dataset=sea_photon_dataset[sea_photon_dataset['is_land_label'] != 1]
    
    # bin the data by 25 cm
    binned_dataset_sea_surface=horizontal_vertical_bin_dataset(sea_photon_dataset, horizontal_res, vertical_res)
    
    # get sea surface height
    # threshold to determine how much depth below sea surface will be accounted
    # Output:
    # ->final_sea_surface_height
    # ->sea_surface_height_abnormal_label
    # ->sea_surface_dominated_label
    sea_surface_height, sea_surface_label, solo_sea_surface_label, subsurface_photon_dataset = \
        get_sea_surface_height(binned_dataset_sea_surface, subsurface_thresh)
    

sea_surface_height.reverse()





####################################################################
# Tasks: Sea Surface flattening
# I have a list value of sea_surface_height over each 10 m bin, I would like to 
# 1) calculate the mean sea_surface_height value along each big bin with 500 m ;
# 2) Within each 10-m along track increment, 
#   adjust all of the photons (i.e., subsurface_photon_dataset) within that 10-m column either up or down 
#   based on the deviation of the local sea surface height from the mean sea surface height 

######################
# Step 1: Creating a DataFrame for sea_surface_height with a corresponding 10 m small bin index
# each element in sea_surface_height corresponds to a 10 m bin
df_sea_surface = pd.DataFrame({
    'sea_surface_height': sea_surface_height, # sea_surface_height is provided as a list
    'small_bins': np.arange(len(sea_surface_height))  # Index for each 10 m bin
})

# # Map 'small_bins'(i.e., 'lat_bins') to 500 m 'big_bins'
df_sea_surface['big_bins'] = df_sea_surface['small_bins'] // 50  # 50 10 m bins in each 500 m bin


######################
# Step 2: Calculate Mean Sea_surface_height value for Each 500-m Big Bin
mean_sea_surface_elevation_bigBin = df_sea_surface.groupby('big_bins')['sea_surface_height'].mean().reset_index()
mean_sea_surface_elevation_bigBin.rename(columns={'sea_surface_height': 'mean_sea_surface_height_big_bins'}, inplace=True)

# Prepare subsurface_photon_dataset for merging
subsurface_photon_dataset['big_bins'] = subsurface_photon_dataset['lat_bins'].astype(int) // 50

# Merge the mean elevation data with the subsurface_photon_dataset
subsurface_photon_dataset = subsurface_photon_dataset.merge(mean_sea_surface_elevation_bigBin,\
                                                            on='big_bins', how='left')

# Merge sea_surface_height data with subsurface_photon_dataset based on the small_bins
subsurface_photon_dataset = subsurface_photon_dataset.merge(df_sea_surface[['small_bins', 'sea_surface_height']], \
                                                            left_on='lat_bins', right_on='small_bins', how='left')


#######################
# Step 3: Within each 10 m small bin, adjust the photon heights  
#         based on the deviation between their sea surface height at each small bin and the mean sea surface height within each 500-m bin
subsurface_photon_dataset['adjusted_photon_height'] = subsurface_photon_dataset['photon_height'] - \
                                                        (subsurface_photon_dataset['mean_sea_surface_height_big_bins']-\
                                                         subsurface_photon_dataset['sea_surface_height']\
                                                         )

subsurface_photon_dataset['Dif_photon_height'] = (subsurface_photon_dataset['mean_sea_surface_height_big_bins']-\
                                                         subsurface_photon_dataset['sea_surface_height']\
                                                         )
######################################
###################################
######################################
###################################
###Visualization 

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms



sea_surface_x_axis_bins = np.linspace(relative_seg_dist.min(),
                          relative_seg_dist.max(), len(sea_surface_height))



mean_sea_surface_x_axis_bins = np.linspace(relative_seg_dist.min(),
                          relative_seg_dist.max(), len(mean_sea_surface_elevation_bigBin))


# Adjusting the global font size
plt.rcParams['font.size'] = 18  # You can choose your own font size here

# Define your zoom area limits for the x-axis
zoom_xlim = (10.5, 11.5)  # for example, from 10 to 20 on your x-axis

# Filter the data based on your x-axis limits
# Here, we're creating a boolean index that matches your zoom criteria
x_index_within_zoom = (sea_photon_dataset.relative_AT_dist >= zoom_xlim[0]) & (sea_photon_dataset.relative_AT_dist <= zoom_xlim[1])

# Apply the index to your x and y data
zoomed_relative_AT_dist = sea_photon_dataset.relative_AT_dist[x_index_within_zoom]
zoomed_photon_height = sea_photon_dataset.photon_height[x_index_within_zoom]

#get the x index for adjusted subsurface_photon
subsurface_x_index_within_zoom = (subsurface_photon_dataset.relative_AT_dist >= zoom_xlim[0]) & (subsurface_photon_dataset.relative_AT_dist <= zoom_xlim[1])
subsurface_zoomed_relative_AT_dist = subsurface_photon_dataset.relative_AT_dist[subsurface_x_index_within_zoom]
zoomed_photon_height_adjusted = subsurface_photon_dataset.adjusted_photon_height[subsurface_x_index_within_zoom]
zoomed_photon_height_Dif = subsurface_photon_dataset.Dif_photon_height[subsurface_x_index_within_zoom]



# Now proceed with creating your main plot
hlims = [-40, 5]  # height limits
sub_hlims = [-10, 3]  # height limits



#########################
fig = plt.figure(figsize=(16, 8), dpi=300, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)
ax.set_xlabel('Distance From Start of Track (km)')
ax.set_ylabel('Photon Height (m)')

ax.scatter(sea_photon_dataset.relative_AT_dist, sea_photon_dataset.photon_height,
           s=10, c='k', alpha=0.15, edgecolors='none', label='ATL03 Photons')
ax.plot(sea_surface_x_axis_bins, sea_surface_height, linewidth=0.8, color='b', label='Surface Peak')
ax.plot(sea_surface_x_axis_bins, [x - 0.5 for x in sea_surface_height], linewidth=0.8, color='#DD571C', label='0.5 m Below Surface Peak')
ax.plot(sea_surface_x_axis_bins, [x - 1 for x in sea_surface_height], linewidth=0.8, color='#FDA172', label='1 m Below Surface Peak')
ax.plot(sea_surface_x_axis_bins, [x - 2 for x in sea_surface_height], linewidth=0.8, color='#FCAE1E', label='2 m Below Surface Peak')

# Create an inset axis for the zoomed area
# axins = inset_axes(ax, width=5, height=3, loc='lower left')  # adjust the location as needed

# Assuming 'ax' is your main axes
trans = mtransforms.blended_transform_factory(ax.figure.transFigure, ax.transAxes)
axins = inset_axes(ax, width=5, height=3, loc=3,
                   bbox_to_anchor=(0.2, 0.1, 0.4, 0.4), # The numbers are x0, y0, width, and height in percentages of the figure size.
                   bbox_transform=trans,
                   borderpad=0)

# Plot the filtered data on the inset axes
axins.scatter(zoomed_relative_AT_dist, zoomed_photon_height, 
              s=10, c='k', alpha=0.15, edgecolors='none', label='Zoomed ATL03 Photons')

axins.scatter(subsurface_zoomed_relative_AT_dist, zoomed_photon_height_adjusted, 
              s=10, c='r', alpha=0.15, edgecolors='none', label='Zoomed Adjusted ATL03 Photons')


axins.scatter(subsurface_zoomed_relative_AT_dist, zoomed_photon_height_Dif, 
              s=20, c='g', alpha=0.15, edgecolors='none', label='Zoomed Dif ATL03 Photons')

# Add horizontal line at y=0
axins.axhline(y=0, color='blue', alpha=0.15, linestyle='--', linewidth=1)

# # For the lines, you might need to adjust your data arrays to match the zoomed range
# # assuming your sea_surface_x_axis_bins aligns directly with your sea_surface_height data
# zoomed_sea_surface_x = [x for x in sea_surface_x_axis_bins if zoom_xlim[0] <= x <= zoom_xlim[1]]
# zoomed_sea_surface_height = sea_surface_height[:len(zoomed_sea_surface_x)]  # adjust based on your data alignment

# First, we need to find the indices of the items in sea_surface_x_axis_bins that fall within the zoomed range.
zoomed_indices = [index for index, x in enumerate(sea_surface_x_axis_bins) if zoom_xlim[0] <= x <= zoom_xlim[1]]

# Then, use these indices to select the corresponding items from both sea_surface_x_axis_bins and sea_surface_height.
zoomed_sea_surface_x = [sea_surface_x_axis_bins[i] for i in zoomed_indices]
zoomed_sea_surface_height = [sea_surface_height[i] for i in zoomed_indices]


#0197f6,#BE5504,#FA8128
axins.plot(zoomed_sea_surface_x, zoomed_sea_surface_height, linewidth=1,  color='b', label='Zoomed Surface Peak')
axins.plot(zoomed_sea_surface_x, [x - 0.5 for x in zoomed_sea_surface_height], linewidth=1, color='#DD571C', label='0.5 m Below Zoomed Surface Peak')
axins.plot(zoomed_sea_surface_x, [x - 1 for x in zoomed_sea_surface_height], linewidth=1, color='#FDA172', label='1 m Below Zoomed Surface Peak')
axins.plot(zoomed_sea_surface_x, [x - 2 for x in zoomed_sea_surface_height],linewidth=1,  color='#FCAE1E', label='2 m Below Zoomed Surface Peak')

#mean sea surface height at 500m bins
axins.plot(mean_sea_surface_x_axis_bins, mean_sea_surface_elevation_bigBin, linewidth=1,  color='r', label='Zoomed Surface Peak 500m bin')


# Set the x limits for the inset axes; y limits remain unchanged
axins.set_xlim(zoom_xlim)
axins.set_ylim(sub_hlims)

axins.set_title('Zoomed in')


# Adding the rectangle indicating the zoomed area
# Draw a rectangle on the main plot to indicate the zoomed area
# We use the data coordinates for the rectangle, as it needs to correspond to the actual data points on the main axes.
rec_hlims=[-10,3]
rectangle = Rectangle((zoom_xlim[0], rec_hlims[0]), zoom_xlim[1] - zoom_xlim[0], rec_hlims[1] - rec_hlims[0], 
                      linewidth=1, edgecolor='r', facecolor='none', linestyle='--')
ax.add_patch(rectangle)


# Finalize the main plot adjustments
ax.legend(loc='lower right')
ax.set_ylim(hlims)

plt.savefig(path+'/'+YY+MM+DD+'Track'+TRK+'Plot'+gtx+'_With_depth_colored.jpg', dpi=500,bbox_inches='tight')
# Display the plot
plt.show()



