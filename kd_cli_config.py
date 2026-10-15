# # config.py

# path = 'C:/Workstation/ICESat2_HLS/Kd_ComparisionPaper/Dataset/ATL03_ICESat2/'

# #US, Orgon
# # ATL03_h5_file = "processed_ATL03_20181206092124_10500106_005_01.h5"#
# #South America
# # ATL03_h5_file = "processed_ATL03_20220530041141_10391513_005_02.h5"#


# #Alaska Cook Inlet
# # ATL03_h5_file = "processed_ATL03_20210801125753_05941205_005_01.h5"#
# ATL03_20200805175053_06320803_006_01_subsetted
# ATL03_20210828231447_10131203_006_01_subsetted

# # Hawaii line without afterpulses
# # ATL03_h5_file = "processed_ATL03_20220122044818_04721407_006_01.h5"#


# # ChesapeakeBay
# ATL03_h5_file = "processed_ATL03_20230825074121_10102002_006_02.h5"#


# # India
# # ATL03_h5_file = "processed_ATL03_20200331204156_00810707_006_01.h5"#



# #China Bohai
# # ATL03_h5_file = "processed_ATL03_20191128115256_09560502_006_01.h5"#
# # ATL03_h5_file = "processed_ATL03_20190530203316_09560302_006_02.h5"#

# ATL03_h5_file = "processed_ATL03_20190829161305_09560402_006_02.h5"#



# ATL03_h5_file_path = path + ATL03_h5_file

# Current_Path = 'C:/Workstation/ICESat2_HLS/Kd_ComparisionPaper/Dataset/'

# shoreline_data_path = Current_Path + 'Shorelines/GeoPkgGlobalShoreline.gpkg'

# #load the Global bathy dataset
# GEBCO_paths = [
#     Current_Path + "Bathy/gebco_2024_geotiff/gebco_2024_n0.0_s-90.0_w0.0_e90.0.tif",
#     Current_Path + "Bathy/gebco_2024_geotiff/gebco_2024_n0.0_s-90.0_w-90.0_e0.0.tif",
#     Current_Path + "Bathy/gebco_2024_geotiff/gebco_2024_n0.0_s-90.0_w90.0_e180.0.tif",
#     Current_Path + "Bathy/gebco_2024_geotiff/gebco_2024_n0.0_s-90.0_w-180.0_e-90.0.tif",
#     Current_Path + "Bathy/gebco_2024_geotiff/gebco_2024_n90.0_s0.0_w0.0_e90.0.tif",
#     Current_Path + "Bathy/gebco_2024_geotiff/gebco_2024_n90.0_s0.0_w-90.0_e0.0.tif",
#     Current_Path + "Bathy/gebco_2024_geotiff/gebco_2024_n90.0_s0.0_w90.0_e180.0.tif",
#     Current_Path + "Bathy/gebco_2024_geotiff/gebco_2024_n90.0_s0.0_w-180.0_e-90.0.tif",
# ]

# horizontal_res = 500

# vertical_res = 0.25

# # Use calculated sea height to determine photons at 0.5m below peak
# subsurface_thresh = 0.5

# # subsurface distance below the sea surface beginning to account
# Kd_max_depth = -1

# OutputPath='C:/Workstation/ICESat2_HLS/Kd_ComparisionPaper/Results/'

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Configure ICESat-2 HLS analysis script.")
    
    # General paths
    parser.add_argument("--workspace_path", type=str, default='C:/Workstation/ICESat2_HLS/Kd_ComparisionPaper/',
                        help="Root path for all data processing.")
    
    # parser.add_argument("--atl03_path", type=str, default='Dataset/ATL03_ICESat2/New/',
    #                     help="Base path for ICESat-2 ATL03 data.")

    # # Bohai Sea (v006)
    # parser.add_argument("--atl03_path", type=str, default='Dataset/ATL03_ICESat2/',
    #                     help="Base path for ICESat-2 ATL03 data.")

    # Wax Delta (v007) — required for IR/AP filter sensitivity test
    parser.add_argument("--atl03_path", type=str, default='Dataset/ATL03_ICESat2/Wax_Delta/',
                        help="Base path for ICESat-2 ATL03 data.")
    
    
    
    # #ChesapeakeBay
    # parser.add_argument("--atl03_file", type=str, default="processed_ATL03_20230825074121_10102002_006_02.h5",
    #                     help="Name of the ATL03 H5 file to process.")    
   
    # parser.add_argument("--atl03_file", type=str, default="processed_ATL03_20221001113813_01641706_006_01.h5",
    #                     help="Name of the ATL03 H5 file to process.")
    
    # # India: processed_ATL03_20200331204156_00810707_006_01.h5
    # parser.add_argument("--atl03_file", type=str, default="processed_ATL03_20200331204156_00810707_006_01.h5",
    #                     help="Name of the ATL03 H5 file to process.")
    
    # #ATL03_20210628230109_00811207_006_01_subsetted
    
    # parser.add_argument("--atl03_file", type=str, default="processed_ATL03_20210628230109_00811207_006_01.h5",
    #                     help="Name of the ATL03 H5 file to process.")
    
    
    # # WaxDelta
    # parser.add_argument("--atl03_file", type=str, default="ATL03_20231103172707_06982106_006_01_subsetted.h5",
    #                     help="Name of the ATL03 H5 file to process.")
    
    
    # # Bohai (v006) — baseline sensitivity test site
    # parser.add_argument("--atl03_file", type=str, default="ATL03_20190829161305_09560402_006_02_subsetted.h5",
    #                     help="Name of the ATL03 H5 file to process.")

    # Wax Delta (v007) — required for IR/AP filter; use with atl03_path = Dataset/ATL03_ICESat2/Wax_Delta/
    parser.add_argument("--atl03_file", type=str, default="ATL03_20231103172707_06982106_007_01_subsetted.h5",
                        help="Name of the ATL03 H5 file to process.")

    # # Cook Inlet
    # parser.add_argument("--atl03_file", type=str, default="ATL03_20220507112145_06931503_006_01_subsetted.h5",
    #                     help="Name of the ATL03 H5 file to process.")
    
    
    
    # #Core Sound
    # parser.add_argument("--atl03_file", type=str, default="105430185_ATL03_20220423191113_04841506_006_02_subsetted.h5",
    #                     help="Name of the ATL03 H5 file to process.")
    
    # # Pamlico Sound
    # parser.add_argument("--atl03_file", type=str, default="ATL03_20240415083629_04232306_006_01_subsetted.h5",
    #                     help="Name of the ATL03 H5 file to process.")
    
    
    #
    # #Cook Inlet: processed_ATL03_20210801125753_05941205_005_01
    # parser.add_argument("--atl03_file", type=str, default="processed_ATL03_20210801125753_05941205_005_01.h5",
    #                     help="Name of the ATL03 H5 file to process.")
    
    parser.add_argument("--other_data_path", type=str, default='Dataset/',
                        help="Path to the current working directory.")
    
    parser.add_argument("--output_path", type=str, default='Results/',
                        help="Directory for saving results.")
    
    # Shoreline data
    parser.add_argument("--shoreline_data", type=str, 
                        # default='Shorelines/GeoPkgGlobalShoreline.gpkg',
                        default='Shorelines/ne_10m_land/ne_10m_land.shp',                        
                        help="Path to the global shoreline dataset.")
    
    # Bathymetry datasets
    parser.add_argument("--gebco_path", nargs='+', type=str, 
                        default='Bathy/gebco_2024_geotiff/',
                        help="Path to GEBCO bathymetry datasets.")
    
    # Resolution settings
    parser.add_argument("--horizontal_res", type=int, default=500,
                        help="Horizontal resolution for the analysis (in meters).")
    
    parser.add_argument("--vertical_res", type=float, default=0.25,
                        help="Vertical resolution for the analysis.")

    parser.add_argument("--surface_exclusion_floor_m", type=float, default=1.0,
                        help="Minimum depth (m) below the fitted surface peak treated as surface and "
                             "excluded from the Kd fit; boundary = min(mu-2sigma, mu-this). Default 1.0. "
                             "In turbid water (Kd > ~2) the whole decay sits inside the top metre, so the "
                             "field-validation notebooks effectively used 0.5.")
    parser.add_argument("--max_fit_depth_m", type=float, default=0.0,
                        help="Drop subsurface photons deeper than this (m below the per-bin sea surface) "
                             "before the Kd fit. 0 (default) = off. Mirrors the notebooks' hard depth cut "
                             "(10-15 m) that kept afterpulse/TEP clusters at 20-35 m and the seafloor out of "
                             "the fit when no GEBCO filter is used.")
    parser.add_argument("--photon_height_offset_m", type=float, default=0.0,
                        help="Constant subtracted from every photon height right after ingestion "
                             "(both local and SlideRule paths), BEFORE binning. The binner keeps only "
                             "heights in (-70, +5) m relative to the geoid, so an inland lake sitting "
                             "at e.g. 1262 m (Upper Klamath Lake) loses every photon unless its surface "
                             "is shifted to ~0 m first. Pass the approximate lake-surface orthometric "
                             "height; the Gaussian surface detector absorbs the residual. Default 0.")

    parser.add_argument("--atl09_root", type=str, default=None,
                        help="Root directory to recursively search for ATL09 files "
                             "matching each ATL03 (by rgt+cycle token). When set, "
                             "the BMA fitter uses ATL09 backg_c per lat_bin to scale "
                             "the Poisson-floor threshold. Defaults to None (skip).")

    parser.add_argument("--min_noise_floor_photons_per_bin", type=float, default=0.0,
                        help="BMA Poisson-floor gate threshold (photons per depth bin). "
                             "Default 0.0 disables the gate (legacy behaviour). Pass a "
                             "positive value (e.g. 2.0) to enable the daytime solar-"
                             "background patch from the TwoDayNightTime diagnostic: "
                             "depth bins with noise_floor below max(this, "
                             "bkg_hist+2*sqrt(bkg_hist)) are excluded from the fit. "
                             "ATL09 backg_c (via --atl09_root) refines bkg_hist per "
                             "lat_bin; otherwise the gate scales with --horizontal_res.")

    # ------------------------------------------------------------------
    # Input source: local ATL03 .h5 (default) or SlideRule server-side subset.
    # Both feed the IDENTICAL downstream chain (binning -> subsurface filtering ->
    # Kd fit -> coherent model selection), so the only difference is ingestion.
    # SlideRule avoids bulk granule downloads for sparse point matchups and pulls
    # ATL09 backg_c via the atl09_fields ancillary join (no local ATL09 needed).
    # ------------------------------------------------------------------
    parser.add_argument("--input_source", type=str, default="local",
                        choices=["local", "sliderule"],
                        help="Where ATL03 photons come from. 'local' reads "
                             "--atl03_path/--atl03_file. 'sliderule' fetches a "
                             "server-side subset (requires --sr_bbox and --sr_t0/--sr_t1).")
    parser.add_argument("--sr_bbox", type=str, default=None,
                        help="SlideRule subset bbox as 'W,S,E,N' (degrees). "
                             "Required when --input_source sliderule.")
    parser.add_argument("--sr_t0", type=str, default=None,
                        help="SlideRule window start, ISO e.g. 2021-08-31T00:00:00Z.")
    parser.add_argument("--sr_t1", type=str, default=None,
                        help="SlideRule window end, ISO e.g. 2021-09-01T00:00:00Z.")
    parser.add_argument("--sr_rgt", type=int, default=None,
                        help="Restrict the SlideRule request to this RGT (recommended: "
                             "the manifest already knows it, and it avoids pulling "
                             "unrelated passes in the window).")
    parser.add_argument("--sr_atl09_fields", type=str, default="high_rate/backg_c",
                        help="Comma-separated group-qualified ATL09 ancillary fields to "
                             "join onto the photons (default 'high_rate/backg_c', the "
                             "Poisson-floor solar background). Pass '' to disable.")
    parser.add_argument("--sr_url", type=str, default="slideruleearth.io",
                        help="SlideRule service host.")
    parser.add_argument("--sr_clip_geojson", type=str, default=None,
                        help="Optional GeoJSON (Polygon/MultiPolygon features) — after the SlideRule "
                             "fetch keep ONLY photons inside these polygons. Stands in for the shoreline "
                             "land mask the local path applies (Extract_sea_photons), which the SlideRule "
                             "path lacks; use hand-digitized water polygons for lagoons / lakes / rivers.")
    parser.add_argument("--sr_exclude_geojson", type=str, default=None,
                        help="Optional GeoJSON — after the SlideRule fetch DROP photons inside these "
                             "polygons (e.g. a sandbar island inside the water ROI).")
    parser.add_argument("--sr_label", type=str, default=None,
                        help="Output filename stem for SlideRule runs (replaces the "
                             "ATL03 timestamp). Defaults to 'sr<rgt>_<t0compact>'.")

    # Analysis parameters
    parser.add_argument("--subsurface_thresh", type=float, default=1.0,
                        help="Threshold for photons below the sea surface.")
    
    parser.add_argument("--ignore_subsurface_height_thres", type=float, default=-6,
                        help="Maximum depth thres for Kd calculations (in meters).") #5m

    parser.add_argument("--min_seabed_depth", type=float, default=3.0,
                        help="Discard along-track bins where the GEBCO seabed is shallower than "
                             "this (m); Kd is not calculated there (Step 14). Default 3.0 m "
                             "(evidence-backed: decay quality is clean at >=3 m at the shallow "
                             "sites, degrades below; see docs/shallow_seabed_gate_and_decay_quality.md). "
                             "Pass 6 to restore the legacy gate. Supersedes "
                             "abs(--ignore_subsurface_height_thres) for the shallow-seabed cut.")

    # Empty string '' auto-selects the first strong beam. Pass comma-separated IDs to override.
    # Valid IDs: gt1l, gt1r, gt2l, gt2r, gt3l, gt3r — only strong beams for the granule are kept.
    # To process all strong beams, use --target_beams "gt1r,gt2r,gt3r".
    parser.add_argument("--target_beams", type=str, default='',
                        help="Comma-separated beam IDs to process. Empty means auto-select first strong beam.")

    # -- Solar background filter (photon-level, OFF by default) --
    parser.add_argument("--enable_solar_background_filter", action="store_true",
                        help="Enable the legacy photon-level solar background filter. "
                             "OFF by default; the physics-based noise floor is preferred.")
    # Deprecated: kept as no-op so old scripts don't crash
    parser.add_argument("--disable_solar_background_filter", action="store_true",
                        help="Deprecated no-op (solar bg filter is now off by default). "
                             "Use --enable_solar_background_filter to turn it on.")
    parser.add_argument("--solar_elevation_day_threshold", type=float, default=6.0,
                        help="Solar elevation threshold (deg) to define daytime for optional filtering.")
    parser.add_argument("--solar_bg_median_window_deg", type=float, default=10.0,
                        help="Rolling median window width (degrees of solar elevation) for "
                             "the solar-elevation-aware background filter.")
    parser.add_argument("--solar_bg_noise_multiplier", type=float, default=1.5,
                        help="Ratio of actual/expected background rate above which a photon "
                             "is flagged as noisy by the solar background filter.")
    parser.add_argument("--solar_background_min_signal_conf", type=int, default=2,
                        help="Minimum photon signal confidence kept in high daytime background.")

    # -- Physics-based noise floor (ON by default) --
    parser.add_argument("--disable_physics_noise_floor", action="store_true",
                        help="Disable physics-based noise floor from bckgrd_rate. "
                             "Falls back to two-pass self-consistent estimation.")
    parser.add_argument("--iss_velocity", type=float, default=7000.0,
                        help="ISS ground velocity in m/s for dwell time calculation (default 7000).")

    parser.add_argument("--enable_ir_ap_filter", action="store_true",
                        help="Optional: remove photons using IR/afterpulse proxy flags.")
    parser.add_argument("--ir_ap_quality_max", type=int, default=0,
                        help="Maximum allowed quality_ph value when IR/AP filter is enabled.")
    parser.add_argument("--ir_ap_min_signal_conf", type=int, default=0,
                        help="Minimum allowed photon_conf value when IR/AP filter is enabled.")

    parser.add_argument("--enable_gebco_filter", action="store_true",
                        help="Optional: remove photons below GEBCO seafloor and shallow bins.")
    parser.add_argument("--seafloor_buffer", type=float, default=1.0,
                        help="Buffer (metres) above GEBCO/ATL24 seafloor to exclude diffuse "
                             "seafloor return. Default 1.0 m.")
    parser.add_argument("--enable_sea_surface_flattening", action="store_true",
                        help="Optional: flatten small-bin sea-surface variation using larger along-track windows.")
    parser.add_argument("--sea_surface_flattening_fine_res_m", type=float, default=50.0,
                        help="Along-track bin size (m) used to DETECT the local sea surface "
                             "before flattening. Must be finer than --horizontal_res to have "
                             "any effect; the manuscript method calls for much finer than "
                             "100 m. At these sites 50 m leaves ~70-110 surface photons per "
                             "bin, well above the 20-photon detector minimum.")
    parser.add_argument("--sea_surface_flattening_window_m", type=int, default=5000,
                        help="Window size in meters for sea-surface flattening mean level. "
                             "Must be larger than horizontal_res to have any effect.")

    parser.add_argument("--enable_convex_hull_filter", action="store_true",
                        help="Optional: apply convex hull area filtering before Kd fitting.")
    parser.add_argument("--convex_hull_area_threshold", type=float, default=100,
                        help="Minimum convex hull area to keep a horizontal bin when enabled.")

    parser.add_argument("--disable_histogram_quality_filter", action="store_true",
                        help="Disable adaptive Step 6 (enabled by default). Step 6 performs "
                             "per-bin S/N gate (rejects bins with insufficient subsurface signal), "
                             "adaptive depth limit (trims noise bins below signal zone), and "
                             "track-level gate (rejects entire track if >40%% bins insufficient, "
                             "e.g., oligotrophic open ocean).")
    parser.add_argument("--signal_above_noise", type=float, default=1.5,
                        help="Depth-bin photon count must exceed this multiple of the noise floor "
                             "to be included in the adaptive depth limit (Step 6 Level 2). "
                             "Lower values extend the fit deeper into noisy bins. Default 1.5.")

    parser.add_argument("--enable_surface_sigma_filter", action="store_true",
                        help="Optional: discard bins where Gaussian surface sigma is too large.")
    parser.add_argument("--surface_sigma_max", type=float, default=0.5,
                        help="Maximum allowed Gaussian sigma (m) for sea-surface peak.")

    parser.add_argument("--enable_refraction_correction", action="store_true",
                        default=True,
                        help="Apply refraction correction to subsurface photon depths. "
                             "On by default — uncorrected depths have ~25%% systematic error. "
                             "Use --no_refraction_correction to disable.")
    parser.add_argument("--no_refraction_correction", action="store_true",
                        help="Disable refraction correction (not recommended).")
    parser.add_argument("--refraction_water_temp_c", type=float, default=20.0,
                        help="Water temperature (deg C) used for refraction correction.")
    parser.add_argument("--refraction_wavelength_nm", type=float, default=532.0,
                        help="Laser wavelength (nm) used for refraction correction.")

    parser.add_argument("--no_refraction_rebin", action="store_true",
                        help="Disable the post-refraction re-binning of height_bins (step 10). "
                             "On by default whenever refraction correction is on: without it the "
                             "corrected depths never reach the Beer's-law fitter, which reads "
                             "height_bins, and Kd comes out per APPARENT metre (~1.45x low, "
                             "measured on BohaiSea). Use this flag only to reproduce pre-v5 runs.")

    parser.add_argument("--enable_post_refraction_refit", action="store_true",
                        help="LEGACY, not recommended: rebuild histograms AND re-fit the surface "
                             "from the full pre-refraction dataset. Discards the step-6 histogram "
                             "quality filter, re-keys lat_bins, and re-fits the surface Gaussian on "
                             "refracted surface photons (biasing sigma / H_s). Prefer the default "
                             "re-binning path.")

    parser.add_argument("--enable_atl24_filter", action="store_true",
                        help="Optional: use ATL24 bathymetry matching before GEBCO fallback.")
    parser.add_argument("--atl24_file", type=str, default='',
                        help="Path to ATL24 point dataset (CSV/Parquet/GPKG/SHP) for bathymetry matching.")
    parser.add_argument("--atl24_max_match_distance_deg", type=float, default=0.01,
                        help="Maximum lon/lat degree distance for ATL24 nearest-point matching.")

    parser.add_argument("--decay_zone_threshold", type=float, default=0.0,
                        help="Fraction of peak photon count below which depth bins are excluded "
                             "in Step 16 decay zone detection. 0.0 = original behaviour (only "
                             "zero-count bins removed). E.g. 0.01 removes bins with < 1%% of peak.")

    parser.add_argument("--kd_fit_method", type=str, default="bma",
                        choices=["log_linear", "bg_subtract", "breakpoint", "nonlinear",
                                 "hybrid", "bma", "bma_daysub"],
                        help="Beer's Law fitting strategy for Kd calculation. "
                             "bma = Bayesian Model-Averaged Kd across all candidate breakpoints (default); "
                             "bma_daysub = bma, then re-fit the decay zone on background-subtracted "
                             "counts in DAYLIGHT ONLY (see --solar_subtract_threshold); "
                             "hybrid = two-pass breakpoint with BIC single best breakpoint; "
                             "log_linear = original log-space linear regression; "
                             "bg_subtract = subtract noise floor then log-linear (NOT the same as "
                             "bma_daysub — it ignores the breakpoint and uses the deepest 20%% of bins); "
                             "breakpoint = segmented regression with breakpoint detection; "
                             "nonlinear = fit C(z) = A*exp(-Kd*z) + N directly.")

    parser.add_argument("--solar_subtract_threshold", type=float, default=-6.0,
                        help="bma_daysub only: subtract the detected noise floor from the decay "
                             "zone only where the bin-median solar elevation exceeds this, in "
                             "degrees (default -6.0, civil twilight). At night the fitted "
                             "'asymptote' is largely a counting artefact — noise_floor cannot "
                             "report below 1 photon/bin — so subtracting it there removes real "
                             "signal. Results are insensitive to the exact cut (any value from "
                             "-18 to +10 gives a day/night Kd ratio of 0.97-0.99).")

    parser.add_argument("--enable_wave_adaptive_fit", action="store_true",
                        help="Optional: trim shallow depth bins by wave amplitude before Beer's Law fit. "
                             "Uses per-bin surface sigma to skip the top k*sigma metres in wavy bins.")
    parser.add_argument("--wave_exclusion_multiplier", type=float, default=4.0,
                        help="Number of surface sigmas to skip from the actual water surface when "
                             "wave-adaptive fit is enabled. The adaptive surface detection already "
                             "removes 3*sigma; this adds (k-3)*sigma additional margin for bubble "
                             "injection zone. Default 4.0 adds 1*sigma beyond the 3-sigma cutoff.")
    parser.add_argument("--wave_sigma_calm_threshold", type=float, default=0.1,
                        help="Surface sigma (m) below which a bin is considered calm and no "
                             "wave trimming is applied. Default 0.1 m.")
    parser.add_argument("--extreme_wave_sigma_threshold", type=float, default=0.43,
                        help="Surface sigma (m) above which a bin gets the EXTREME_WAVE quality "
                             "flag (bit 5). Reporting only - does NOT change the fit (that is "
                             "--wave_sigma_calm_threshold). Default 0.43 m is the sigma matching "
                             "U10 = 10 m/s, the ATL13 ATBD (r007 sec 4.2.4) wind above which foam "
                             "scattering approaches the volume scattering Kd is fitted to; the "
                             "mapping is empirical (pooled granule-median sigma at 10 m/s = 0.433 "
                             "m over 813 granules). The former 0.3 m fired at the East-coast "
                             "median sea state and flagged half the data.")

    # Kd fit data sufficiency constraints (data-quality gates)
    parser.add_argument("--min_breakpoint_depth_m", type=float, default=2.0,
                        help="BMA: shallowest depth (m below the fit origin) a decay->noise-floor breakpoint "
                             "may sit at. Default 2.0. In very turbid water (Kd > ~2) the decay is over inside "
                             "the top metre, so 2.0 forces the 'decay' segment to include noise; lower it "
                             "(e.g. 0.5) together with --min_decay_range_m/--min_decay_bins for such sites.")
    parser.add_argument("--min_decay_range_m", type=float, default=5.0,
                        help="Minimum decay-zone range (m) required for Kd fit. "
                             "Smaller values increase sensitivity but reduce reliability.")
    parser.add_argument("--min_fit_span_m", type=float, default=2.0,
                        help="Minimum vertical extent (m) the photon histogram must span "
                             "before a Kd is reported; bins spanning less are returned as "
                             "NaN with reason 'fit_span_too_short'. A decay rate cannot be "
                             "measured over a range shorter than the window it is measured "
                             "in: at Bohai every Kd > 1.0 came from a ~1 m span with 5 bins. "
                             "Default 2.0 keeps 89%% of bins and cuts the maximum by 28%% "
                             "while leaving the median unchanged. Set 0 to disable.")
    parser.add_argument("--min_decay_photons", type=int, default=30,
                        help="Minimum photon count in decay zone.")
    parser.add_argument("--min_decay_bins", type=int, default=8,
                        help="Minimum number of non-empty depth bins in decay zone.")
    parser.add_argument("--min_total_photons", type=int, default=30,
                        help="Minimum total photons in bin (reject bin entirely if below).")
    parser.add_argument("--min_total_range_m", type=float, default=5.0,
                        help="Minimum total depth range in bin (reject bin entirely if below).")

    # Bootstrap uncertainty estimation
    parser.add_argument("--bootstrap_iterations", type=int, default=200,
                        help="Residual-bootstrap iterations per bin for kd_std/kd_ci estimation. "
                             "Set to 0 to skip bootstrap (faster runtime). "
                             "CI half-width scales as 1/sqrt(N); 200 gives ~1.58x wider CIs "
                             "than 500 in exchange for ~2.5x faster bootstrap.")
    parser.add_argument("--poisson_weighted", action="store_true",
                        help="BMA only: weight log-space fits by sqrt(photon count) so "
                             "var(log c) ~ 1/c is respected and the sparse deep tail stops "
                             "over-voting the decay slope. Default off (unweighted, legacy). "
                             "Tends to raise and stabilise low-SNR/clear-water Kd.")
    parser.add_argument("--spatial_regularize", action="store_true",
                        help="Add region-adaptive along-track Kd smoothing (kd_smooth + "
                             "kd_smooth_lo/hi band + kd_regularized cols). Where a window is "
                             "majority-unstable (collapse / wide CI), the spiky per-bin Kd is "
                             "replaced by the local rolling median with a 16-84 pct scatter "
                             "band; stable bins are left untouched. Default off; raw kd/kd_std "
                             "columns are always preserved.")
    parser.add_argument("--spatial_window", type=int, default=13,
                        help="Centred window size (in bins) for --spatial_regularize. "
                             "Default 13 (~6.5 km at 0.5 km binning). A 5-bin window "
                             "under-smooths low-SNR zones where the decay-vs-flat flip "
                             "wavelength is several bins (residual bumps survive); 13 "
                             "matches it. Only affects regularized (unstable) bins. "
                             "Ignored when --spatial_adaptive is set.")
    parser.add_argument("--spatial_adaptive", action="store_true",
                        help="Adaptive smoothing window: scale per-bin to the local "
                             "unstable-run length (clipped to "
                             "[--spatial_min_window, --spatial_max_window]) so long "
                             "sparse-flip zones self-select heavy smoothing and short "
                             "blips a light touch — no global window to tune per site.")
    parser.add_argument("--spatial_min_window", type=int, default=7,
                        help="Min adaptive window (bins) for --spatial_adaptive. Default 7.")
    parser.add_argument("--spatial_max_window", type=int, default=25,
                        help="Max adaptive window (bins) for --spatial_adaptive. Default 25.")
    parser.add_argument("--coherent_model_selection", action="store_true",
                        help="Resolve the decay-vs-flat BMA bistability at the region level "
                             "instead of smoothing the bimodal output: re-enumerate each bin's "
                             "candidates, pool the model preference over a window, commit each "
                             "region to its BIC-majority model, then smooth within-region. "
                             "Supersedes --spatial_regularize (it includes the smoothing). Adds "
                             "kd_single/kd_2seg/kd_coherent + band cols. Honours --spatial_window/"
                             "--spatial_adaptive/--spatial_central/--poisson_weighted.")
    parser.add_argument("--spatial_central", choices=("median", "mean"), default="median",
                        help="Central estimator for --spatial_regularize smoothing. "
                             "'median' (default) is robust to the bimodal collapse/breakpoint "
                             "split in low-SNR zones (a mean is dragged up toward the high "
                             "breakpoint cluster). The kd_smooth_lo/hi band brackets both modes "
                             "either way. Use 'mean' only for near-balanced sawtooths.")

    parser.add_argument("--no_plot", action="store_true",
                        help="Suppress all plot generation (useful for batch runs).")

    # switch on consider the coastal water or inland water
    # if Inland water, we should use one mask and for coastal water, it should be another mask
    
    # consider bathymetry or not manual setting
    
    
    #night vs daytime
    
    #solar elevation detemine remove or not for solar background
    
    
    # 1-2 hours 
    
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    # Access parameters like this:
    atl03_file_path = args.workspace_path + args.atl03_file
    
    print("Processing file:", atl03_file_path)
    print("Output path:", args.output_path)
