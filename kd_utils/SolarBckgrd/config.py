import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Analyze ICESat-2 ATL03 data for solar background.")

    #task name
    parser.add_argument("--taskID", type=str, default='2022_granules',
                        help="task name")
    # workspace path
    parser.add_argument("--workspace_path", type=str, default='/work/users/w/a/wayne128/ICESat2/IS2_kd_py/',
                        help="Root path for all data processing.")

    
    # parser.add_argument("--atl03_path", type=str, default='Dataset/ATL03_ICESat2/Extracted_HDF_Files/NC/NCCENTRAL/',
    #                 help="Base path for ICESat-2 ATL03 data.")
                    
    # parser.add_argument("--atl03_path", type=str, default='Dataset/ATL03_ICESat2/Extracted_HDF_Files/NC/NCNORTH/',
    #                 help="Base path for ICESat-2 ATL03 data.")
                    
    # parser.add_argument("--atl03_path", type=str, default='Dataset/ATL03_ICESat2/Extracted_HDF_Files/NC/NCSOUTH/',
    #                 help="Base path for ICESat-2 ATL03 data.")
    
    # parser.add_argument("--atl03_path", type=str, default='Dataset/ATL03_ICESat2/Extracted_HDF_Files/CD/2018_granules/h5_files/',
    #                 help="Base path for ICESat-2 ATL03 data.")
                    
    # parser.add_argument("--atl03_path", type=str, default='Dataset/ATL03_ICESat2/Extracted_HDF_Files/CD/2019_granules/h5_files/',
    #                 help="Base path for ICESat-2 ATL03 data.") 
                    
    # parser.add_argument("--atl03_path", type=str, default='Dataset/ATL03_ICESat2/Extracted_HDF_Files/CD/2020_granules/h5_files/',
    #                 help="Base path for ICESat-2 ATL03 data.")
                    
    # parser.add_argument("--atl03_path", type=str, default='Dataset/ATL03_ICESat2/Extracted_HDF_Files/CD/2021_granules/h5_files/',
    #                 help="Base path for ICESat-2 ATL03 data.")
    
    parser.add_argument("--atl03_path", type=str, default='Dataset/ATL03_ICESat2/Extracted_HDF_Files/CD/2020_granules/h5_files/',
                    help="Base path for ICESat-2 ATL03 data.")
                    
    # parser.add_argument("--parallel", action="store_true", help="Process files in parallel")
    parser.add_argument("--beam", default="gt1l", help="Beam to process (e.g., gt1l)")
    

    parser.add_argument("--atl03_file", type=str, default="processed_ATL03_20221001234100_01721701_006_01.h5",
                        help="Name of the ATL03 H5 file to process.")#processed_ATL03_20221001114643_01641707_006_01

    parser.add_argument("--output_path", type=str, default='Dataset/Results/',
                        help="Directory for saving results.")
    
    # Shoreline data
    parser.add_argument("--shoreline_data", type=str, 
                        # default='Shorelines/GeoPkgGlobalShoreline.gpkg',
                        default='Dataset/Shorelines/ne_10m_land/ne_10m_land.shp',                        
                        help="Path to the global shoreline dataset.")
    
    # Bathymetry datasets
    parser.add_argument("--gebco_path", nargs='+', type=str, 
                        default='Dataset/Bathy/gebco_2024_geotiff/',
                        help="Path to GEBCO bathymetry datasets.")
    
    # Resolution settings
    parser.add_argument("--horizontal_res", type=int, default=500,
                        help="Horizontal resolution for the analysis (in meters).")
    parser.add_argument("--vertical_res", type=float, default=0.25,
                        help="Vertical resolution for the analysis.")
    
    # Analysis parameters
    parser.add_argument("--subsurface_thresh", type=float, default=0.5,
                        help="Threshold for photons below the sea surface.")
    parser.add_argument("--ignore_subsurface_height_thres", type=float, default=-1,
                        help="Maximum depth thres for Kd calculations (in meters).")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    # Access parameters like this:
    atl03_file_path = args.workspace_path + args.atl03_file
    
    print("Processing file:", atl03_file_path)
    print("Output path:", args.output_path)