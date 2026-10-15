import glob
import logging
import os
import re
import sys

import numpy as np
import pandas as pd

from kd_cli_config import get_args
from kd_utils.Kd_analysis import compute_physics_noise_floor, process_kd_calculation
from kd_utils.bathy_processing import process_subsurface_photon_filtering
from kd_utils.quality_flags import compute_quality_flags
from kd_utils.data_processing import (
    Extract_sea_photons,
    apply_optional_ir_ap_filter,
    apply_optional_solar_background_filter,
    filter_photon_dataset_by_hull_area,
    load_data,
)
from kd_utils.sea_photons_analysis import process_sea_photon_binning
from kd_utils.SolarBckgrd.pair_atl09 import (
    beam_to_profile, find_atl09_file, load_backg_c_table, read_atl03_sc_orient,
)
from kd_utils.visualization import plot_convex_hulls, plot_kd_photons, plot_photon_quality_flags


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_target_beams(all_beams, beam_attrs, target_beams_arg):
    strong_beams = [
        gtx for gtx in all_beams
        if beam_attrs[gtx]['atlas_beam_type'].decode('utf-8') == 'strong'
    ]
    if not target_beams_arg:
        return strong_beams[:1]  # auto-select first strong beam

    requested = [b.strip() for b in target_beams_arg.split(',') if b.strip()]
    return [b for b in requested if b in strong_beams]


def _sliderule_region(bbox_str):
    """'W,S,E,N' -> the 5-point closed polygon SlideRule expects."""
    w, s, e, n = [float(v) for v in bbox_str.split(",")]
    return [{"lon": w, "lat": s}, {"lon": e, "lat": s}, {"lon": e, "lat": n},
            {"lon": w, "lat": n}, {"lon": w, "lat": s}]


def _sliderule_label(args):
    """Output filename stem for a SlideRule run (stands in for the ATL03 timestamp)."""
    if getattr(args, "sr_label", None):
        return args.sr_label
    t0 = re.sub(r"[^0-9]", "", args.sr_t0 or "")[:14]
    rgt = args.sr_rgt if args.sr_rgt is not None else "NA"
    return f"sr_{rgt}_{t0}" if t0 else f"sr_{rgt}"


def _sliderule_atl03_version(args, region):
    """ATL03 processing version for a SlideRule run, resolved via CMR.

    Without this the version-gated quality flags (afterpulse / TEP, v007+) and the
    IR/AP filter would silently behave as pre-v007 just because there is no local
    filename to parse. Returns None if it cannot be determined.
    """
    try:
        from sliderule import sliderule as _sr, earthdata as _ed
        _sr.init(args.sr_url, verbose=False)
        granules = _ed.cmr(short_name="ATL03", polygon=region,
                           time_start=args.sr_t0, time_end=args.sr_t1)
        for name in (granules or []):
            m = re.search(r"ATL03_\d{14}_\d{8}_(\d{3})_\d{2}", str(name))
            if m:
                logger.info("SlideRule ATL03 granule %s -> version %03d", name, int(m.group(1)))
                return int(m.group(1))
    except Exception as e:
        logger.warning("Could not resolve ATL03 version via CMR (%s); "
                       "version-gated filters/flags will stay off.", e)
    return None


def _clip_photons_to_geojson(df, include_path=None, exclude_path=None):
    """Keep photons inside `include_path` polygons and outside `exclude_path` polygons.

    The SlideRule path has no shoreline mask, so land/bank photons reach the binner and
    the surface detector unless the caller clips them. Mirrors the per-site hand-digitized
    water ROIs (and sandbar exclusion) used in the JupyterHub field-validation notebooks.
    """
    if not include_path and not exclude_path:
        return df
    import geopandas as gpd
    import shapely
    from shapely.ops import unary_union
    # Vectorized prepared-geometry test (shapely >= 2). The previous
    # GeoSeries.within(<one complex polygon>) was O(points x vertices) with no
    # spatial index: 36 min for 1.1 M photons against a coastline-derived water
    # ROI (Baja, 2026-09-17), i.e. the clip, not the Kd fit, dominated runtime.
    # contains_xy on a prepared polygon does the same 1.1 M points in seconds.
    x = df["longitude"].to_numpy(float)
    y = df["latitude"].to_numpy(float)
    keep = pd.Series(True, index=df.index)
    if include_path:
        inc = unary_union(gpd.read_file(include_path).geometry.values)
        shapely.prepare(inc)
        keep &= shapely.contains_xy(inc, x, y)
        logger.info("SlideRule clip: %d/%d photons inside %s", int(keep.sum()), len(df),
                    os.path.basename(include_path))
    if exclude_path:
        exc = unary_union(gpd.read_file(exclude_path).geometry.values)
        shapely.prepare(exc)
        drop = shapely.contains_xy(exc, x, y)
        keep &= ~drop
        logger.info("SlideRule exclude: dropped %d photons inside %s", int(drop.sum()),
                    os.path.basename(exclude_path))
    return df[keep].copy()


def _load_sliderule_photons(args):
    """Fetch a SlideRule server-side ATL03 subset (+ joined ATL09 backg_c).

    Returns the framework-schema sea-photon dataset and the beam list, matching
    what load_data + get_target_beams + Extract_sea_photons produce for a local
    granule. NOTE: the shoreline land mask applied by Extract_sea_photons has no
    SlideRule equivalent (the adapter sets is_land_label=0), so near-shore land
    photons are not masked on this path.
    """
    from kd_utils.sliderule_adapter import (
        fetch_sliderule_atl03, sliderule_to_framework_dataset,
        fetch_atl09_backg_c_tables,
    )

    if not args.sr_bbox:
        logger.error("--input_source sliderule requires --sr_bbox 'W,S,E,N'")
        sys.exit(5)
    if not (args.sr_t0 and args.sr_t1):
        logger.error("--input_source sliderule requires --sr_t0 and --sr_t1")
        sys.exit(5)

    region = _sliderule_region(args.sr_bbox)
    # Two-step (rather than fetch_and_prepare_*) so we can read sc_orient off the
    # raw return — it drives the ATL03 beam -> ATL09 profile mapping below.
    gdf = fetch_sliderule_atl03(region, args.sr_t0, args.sr_t1, rgt=args.sr_rgt,
                                sliderule_url=args.sr_url)
    if gdf is None or len(gdf) == 0:
        logger.error("SlideRule returned no photons for the requested region/window.")
        sys.exit(2)
    sc_orient = None
    if "sc_orient" in gdf.columns:
        vals = pd.Series(np.asarray(gdf["sc_orient"])).dropna()
        if len(vals):
            sc_orient = int(vals.iloc[0])

    df = sliderule_to_framework_dataset(gdf, strong_beams_only=True)
    if df is None or df.empty:
        logger.error("SlideRule photons did not convert to any usable sea photons.")
        sys.exit(2)

    beams = sorted(df["beam_id"].astype(str).unique().tolist())
    if args.target_beams:
        requested = [b.strip() for b in args.target_beams.split(",") if b.strip()]
        keep = [b for b in beams if b in requested]
        if keep:
            df = df[df["beam_id"].isin(keep)].copy()
            beams = keep
    logger.info("SlideRule fetched %d photons across beams %s (sc_orient=%s)",
                len(df), beams, sc_orient)

    df = _clip_photons_to_geojson(df, getattr(args, "sr_clip_geojson", None),
                                  getattr(args, "sr_exclude_geojson", None))
    if df.empty:
        logger.error("No SlideRule photons left after --sr_clip_geojson/--sr_exclude_geojson.")
        sys.exit(2)
    beams = [b for b in beams if b in set(df["beam_id"].astype(str))]
    if getattr(args, "sr_clip_geojson", None) or getattr(args, "sr_exclude_geojson", None):
        logger.info("After ROI clip: %d photons, beams %s", len(df), beams)

    # ATL09 solar background, remote-read (no download). The documented
    # atl09_fields ancillary join is silently ignored server-side, so this goes
    # through CMR + h5 instead — see fetch_atl09_backg_c_tables.
    atl09_tables = {}
    if sc_orient is not None and (args.sr_atl09_fields or "").strip():
        atl09_tables = fetch_atl09_backg_c_tables(
            region=region, t0=args.sr_t0, t1=args.sr_t1, beams=beams,
            sc_orient=sc_orient,
            lat_range=(float(df.latitude.min()), float(df.latitude.max())),
            sliderule_url=args.sr_url,
        )
    return df, beams, atl09_tables


def _granule_solar_elevation(use_sliderule, sea_photon_dataset, atl03_h5_file_path, beams):
    """Median solar elevation for the granule (drives the day-only Poisson-floor gate).

    SlideRule supplies solar_elevation as a photon field, so no local .h5 is needed.
    """
    if use_sliderule:
        if "solar_elevation" in sea_photon_dataset.columns:
            vals = pd.to_numeric(sea_photon_dataset["solar_elevation"], errors="coerce").dropna()
            return float(np.median(vals)) if len(vals) else float("nan")
        return float("nan")

    import h5py
    elev_medians = []
    with h5py.File(atl03_h5_file_path, "r") as f:
        for beam in beams:
            key = f"{beam}/geolocation/solar_elevation"
            if key in f:
                elev_medians.append(float(np.median(f[key][:])))
    return float(np.median(elev_medians)) if elev_medians else float("nan")


def run_pipeline(args):
    """
    Top-level pipeline orchestrator. Executes all 19 flowchart steps in sequence.

    Flowchart steps directly handled here:
      1  — Load ATL03 data (load_data / Extract_sea_photons).
      2  — Remove afterpulse/impulse-response photons (apply_optional_ir_ap_filter).
      3  — Bin sizes set via args.horizontal_res / args.vertical_res.
      4  — Build histograms (process_sea_photon_binning).
      5  — Legacy solar background filter (OFF by default, use --enable_solar_background_filter).
      5b — Physics-based noise floor from bckgrd_rate (ON by default, feeds into Step 16-17).
      6–14 — Subsurface filtering chain (process_subsurface_photon_filtering).
      16–17 — Beer's Law Kd fit (process_kd_calculation).
      18 — Save Kd CSV output (subsurface_photon_df_added_kd.to_csv).
      19 — Check data for reasonableness (filter_photon_dataset_by_hull_area +
           plot_kd_photons).
    """
    use_sliderule = getattr(args, 'input_source', 'local') == 'sliderule'
    shoreline_data_path = os.path.join(args.workspace_path, args.other_data_path, args.shoreline_data)
    gebco_path_str = args.gebco_path[0] if isinstance(args.gebco_path, list) else args.gebco_path
    gebco_full_path = os.path.join(args.workspace_path, args.other_data_path, gebco_path_str)
    atl24_file_path = args.atl24_file
    if atl24_file_path and (not os.path.isabs(atl24_file_path)):
        atl24_file_path = os.path.join(args.workspace_path, atl24_file_path)
    output_path = os.path.join(args.workspace_path, args.output_path)
    os.makedirs(output_path, exist_ok=True)

    if use_sliderule:
        # No local granule: SlideRule serves a server-side subset. Version is not
        # recoverable from a filename, so the IR/AP guard below disables that filter.
        atl03_h5_file_path = None
        timestamp = _sliderule_label(args)
        atl03_version = _sliderule_atl03_version(args, _sliderule_region(args.sr_bbox)) \
            if args.sr_bbox else None
        logger.info("Input source: SlideRule server-side subset (output stem: %s)", timestamp)
    else:
        atl03_h5_file_path = os.path.join(args.workspace_path, args.atl03_path, args.atl03_file)
        match = re.search(r"_(\d{14})_", atl03_h5_file_path)
        timestamp = match.group(1) if match else "unknown"

        version_match = re.search(r"ATL03_\d{14}_\d{8}_(\d{3})_\d{2}", os.path.basename(atl03_h5_file_path))
        atl03_version = int(version_match.group(1)) if version_match else None
        logger.info("ATL03 version detected: %s", f"{atl03_version:03d}" if atl03_version else "unknown")

    if args.enable_ir_ap_filter:
        if atl03_version is None or atl03_version < 7:
            logger.error(
                "IR/AP filter requires ATL03 version 007 or later. "
                "Detected version: %s. "
                "Skipping IR/AP filter for this run. "
                "Switch to a version 007 file (e.g., Wax Delta dataset) to enable this step.",
                f"{atl03_version:03d}" if atl03_version else "unknown",
            )
            args.enable_ir_ap_filter = False

    gebco_pattern = os.path.join(gebco_full_path, "gebco_*.tif")
    gebco_file_path_lists = [p for p in glob.glob(gebco_pattern)]

    # Step 1 — ingest photons. Both sources produce the same framework schema and
    # feed the identical downstream chain; only acquisition differs.
    sr_backg_c_tables = {}
    if use_sliderule:
        sea_photon_dataset, target_strong_beams, sr_backg_c_tables = \
            _load_sliderule_photons(args)
    else:
        is2_mds, is2_attrs, is2_beams = load_data(atl03_h5_file_path, False)
        target_strong_beams = get_target_beams(is2_beams, is2_attrs, args.target_beams)
        if not target_strong_beams:
            logger.error("No target strong beams found. Check input file and --target_beams.")
            sys.exit(4)
        sea_photon_dataset = Extract_sea_photons(is2_mds, target_strong_beams, shoreline_data_path)
    # Optional constant height shift (inland lakes above the (-70, +5) m binning window).
    _h_off = float(getattr(args, 'photon_height_offset_m', 0.0) or 0.0)
    if _h_off != 0.0:
        sea_photon_dataset = sea_photon_dataset.copy()
        sea_photon_dataset['photon_height'] = sea_photon_dataset['photon_height'] - _h_off
        logger.info("Applied --photon_height_offset_m: subtracted %.2f m from all photon heights "
                    "(new median %.2f m)", _h_off, float(sea_photon_dataset['photon_height'].median()))
    logger.info("Strong beams selected: %s", target_strong_beams)
    plot_target_beam = [target_strong_beams[0]]

    # Day-only gate for the BMA Poisson noise-floor patch
    # (--min_noise_floor_photons_per_bin). At night the actual solar background
    # is near zero, so enforcing a positive floor would over-correct and pull
    # Kd estimates below their true value (verified BohaiSea 2026-05-28:
    # kd_median dropped from 0.21 → 0.087 with the patch unconditionally on).
    # Per project_solar_background_atl09 memory ("Wire B with day-only gate").
    effective_min_noise_floor = float(args.min_noise_floor_photons_per_bin)
    if effective_min_noise_floor > 0:
        granule_solar_elev = _granule_solar_elevation(
            use_sliderule, sea_photon_dataset, atl03_h5_file_path, target_strong_beams
        )
        is_daytime = (
            (not np.isnan(granule_solar_elev))
            and granule_solar_elev > args.solar_elevation_day_threshold
        )
        if not is_daytime:
            logger.info(
                "BMA Poisson-floor patch DISABLED for this granule "
                "(median solar_elev=%.2f deg <= day threshold %.2f). "
                "User passed --min_noise_floor_photons_per_bin=%.2f.",
                granule_solar_elev, args.solar_elevation_day_threshold,
                effective_min_noise_floor,
            )
            effective_min_noise_floor = 0.0
        else:
            logger.info(
                "BMA Poisson-floor patch ACTIVE "
                "(median solar_elev=%.2f deg > day threshold %.2f, "
                "min_noise_floor_photons_per_bin=%.2f).",
                granule_solar_elev, args.solar_elevation_day_threshold,
                effective_min_noise_floor,
            )

    if args.enable_ir_ap_filter and not args.no_plot:
        plot_photon_quality_flags(output_path, timestamp, sea_photon_dataset, plot_target_beam)

    sea_photon_dataset = apply_optional_ir_ap_filter(
        sea_photon_dataset,
        enabled=args.enable_ir_ap_filter,
        quality_max=args.ir_ap_quality_max,
        min_signal_conf=args.ir_ap_min_signal_conf,
    )

    # Step 5 — Legacy photon-level solar background filter (OFF by default).
    # The physics-based noise floor (computed later from bckgrd_rate) is now
    # the default approach. Use --enable_solar_background_filter to re-enable.
    solar_bg_enabled = args.enable_solar_background_filter
    sea_photon_dataset = apply_optional_solar_background_filter(
        sea_photon_dataset,
        enabled=solar_bg_enabled,
        day_threshold=args.solar_elevation_day_threshold,
        median_window_deg=args.solar_bg_median_window_deg,
        noise_multiplier=args.solar_bg_noise_multiplier,
        min_signal_conf=args.solar_background_min_signal_conf,
    )

    binned_dataset_sea_surface = process_sea_photon_binning(
        sea_photon_dataset, horizontal_res=args.horizontal_res, vertical_res=args.vertical_res
    )
    if binned_dataset_sea_surface.empty:
        logger.error("No sea photons after binning — strong-beam track has no usable surface photons in bbox.")
        sys.exit(2)

    # Handle refraction: on by default, --no_refraction_correction disables
    if args.no_refraction_correction:
        args.enable_refraction_correction = False

    post_refraction_refit_enabled = args.enable_post_refraction_refit
    if post_refraction_refit_enabled and (not args.enable_refraction_correction):
        logger.warning(
            "--enable_post_refraction_refit requested without --enable_refraction_correction. "
            "The post-refraction refit step will be skipped."
        )
        post_refraction_refit_enabled = False

    # Step 10 (narrow form): re-cut height_bins on the corrected depths so the
    # Beer's-law fit runs in true metres. On by default with refraction; the
    # legacy full refit supersedes it (it rebuilds the bins itself).
    refraction_rebin_enabled = (args.enable_refraction_correction
                                and not args.no_refraction_rebin
                                and not post_refraction_refit_enabled)
    if args.enable_refraction_correction and args.no_refraction_rebin:
        logger.warning(
            "--no_refraction_rebin set: the Beer's-law fit will use the PRE-refraction "
            "depth axis, so Kd is reported per apparent metre (~1.45x low). "
            "Use only to reproduce pre-v5 runs."
        )
    logger.info("Post-refraction re-binning (step 10): %s",
                "ENABLED" if refraction_rebin_enabled else "disabled")

    # Shallow-seabed gate (Step 14): bins where GEBCO depth < this are discarded.
    # --min_seabed_depth controls it (default 3 m, evidence-backed). Falls back to
    # the legacy abs(--ignore_subsurface_height_thres) only if explicitly unset
    # (None). Passed downstream as a positive value (the chain takes abs()).
    _min_seabed = (args.min_seabed_depth if getattr(args, 'min_seabed_depth', None) is not None
                   else abs(args.ignore_subsurface_height_thres))
    logging.info("Shallow-seabed gate (Step 14): discard bins with GEBCO depth < %.1f m", _min_seabed)

    sea_surface_height, sea_surface_label, filtered_seafloor_subsurface_photon_dataset, \
        adaptive_depth_limits, snr_per_bin, surface_photon_counts, \
        surface_diagnostics_per_beam = \
        process_subsurface_photon_filtering(
            binned_dataset_sea_surface,
            gebco_file_path_lists,
            args.subsurface_thresh,
            _min_seabed,
            use_atl24_filter=args.enable_atl24_filter,
            atl24_file_path=atl24_file_path,
            atl24_max_match_distance_deg=args.atl24_max_match_distance_deg,
            use_gebco_filter=args.enable_gebco_filter,
            seafloor_buffer_m=args.seafloor_buffer,
            apply_histogram_quality_filter=not args.disable_histogram_quality_filter,
            signal_above_noise=args.signal_above_noise,
            apply_surface_sigma_filter=args.enable_surface_sigma_filter,
            surface_sigma_max=args.surface_sigma_max,
            apply_refraction_correction=args.enable_refraction_correction,
            refraction_water_temp_c=args.refraction_water_temp_c,
            refraction_wavelength_nm=args.refraction_wavelength_nm,
            apply_post_refraction_refit=post_refraction_refit_enabled,
            apply_refraction_rebin=refraction_rebin_enabled,
            apply_flattening=args.enable_sea_surface_flattening,
            flattening_window_m=args.sea_surface_flattening_window_m,
            flattening_fine_res_m=args.sea_surface_flattening_fine_res_m,
            surface_floor_m=args.surface_exclusion_floor_m,
            horizontal_res=args.horizontal_res,
            vertical_res=args.vertical_res,
        )

    if filtered_seafloor_subsurface_photon_dataset.empty or \
            'beam_id' not in filtered_seafloor_subsurface_photon_dataset.columns:
        logger.error("No subsurface photons after filtering — strong-beam track has no usable subsurface photons.")
        sys.exit(3)

    final_filtered_subsurface_photon_dataset = filtered_seafloor_subsurface_photon_dataset[
        filtered_seafloor_subsurface_photon_dataset['beam_id'].isin(target_strong_beams)
    ].copy()

    if args.enable_convex_hull_filter:
        final_filtered_subsurface_photon_dataset, convex_hull_areas, convex_hulls = \
            filter_photon_dataset_by_hull_area(
                final_filtered_subsurface_photon_dataset,
                hull_area_threshold=args.convex_hull_area_threshold
            )
        if plot_target_beam and not args.no_plot:
            plot_convex_hulls(
                final_filtered_subsurface_photon_dataset,
                plot_target_beam,
                convex_hulls,
                convex_hull_areas
            )

    subsurface_output_path = os.path.join(
        output_path,
        f"{timestamp}_strongBeam_{'_'.join(target_strong_beams)}_subsurface_photons.csv",
    )
    final_filtered_subsurface_photon_dataset.to_csv(subsurface_output_path, index=False)

    kd_input_dataset = final_filtered_subsurface_photon_dataset
    _max_fit_depth = float(getattr(args, 'max_fit_depth_m', 0.0) or 0.0)
    if _max_fit_depth > 0 and 'sea_surface_height' in kd_input_dataset.columns:
        _depth = kd_input_dataset['sea_surface_height'] - kd_input_dataset['photon_height']
        _keep = _depth <= _max_fit_depth
        logger.info("--max_fit_depth_m %.1f: keeping %d/%d subsurface photons (dropped deeper than %.1f m below surface)",
                    _max_fit_depth, int(_keep.sum()), len(kd_input_dataset), _max_fit_depth)
        kd_input_dataset = kd_input_dataset[_keep].copy()
        if 'height_bins' in kd_input_dataset.columns:
            kd_input_dataset['height_bins'] = kd_input_dataset['height_bins'].cat.remove_unused_categories() \
                if hasattr(kd_input_dataset['height_bins'], 'cat') else kd_input_dataset['height_bins']
    wave_mult = args.wave_exclusion_multiplier if args.enable_wave_adaptive_fit else 0.0

    # Build per-beam ATL09 backg_c lookup for the BMA Poisson-floor gate.
    # When --atl09_root is set and a matching ATL09 file is found, the
    # fitter uses backg_c-derived bkg_hist per lat_bin. Otherwise it falls
    # back to the horizontal_res scaling, which is calibrated at 500 m.
    backg_c_lookup_per_beam = {}
    if use_sliderule:
        # ATL09 backg_c remote-read via CMR + h5 (no granule download). Same
        # ('latitude','backg_c') per-beam contract as load_backg_c_table.
        backg_c_lookup_per_beam = sr_backg_c_tables
        if backg_c_lookup_per_beam:
            logger.info("ATL09 backg_c via SlideRule remote read -> %d beams",
                        len(backg_c_lookup_per_beam))
        else:
            logger.warning(
                "No ATL09 backg_c from SlideRule; falling back to "
                "horizontal_res-scaled threshold."
            )
    elif getattr(args, "atl09_root", None):
        atl03_stem = os.path.basename(atl03_h5_file_path).replace(".h5", "")
        atl09_path = find_atl09_file(atl03_stem, args.atl09_root)
        if atl09_path is None:
            logger.warning(
                "No ATL09 file found for %s under %s; falling back to "
                "horizontal_res-scaled threshold.",
                atl03_stem, args.atl09_root,
            )
        else:
            sc_orient = read_atl03_sc_orient(atl03_h5_file_path)
            for beam in target_strong_beams:
                profile = beam_to_profile(beam, sc_orient)
                if profile is None:
                    continue
                try:
                    backg_c_lookup_per_beam[beam] = load_backg_c_table(atl09_path, profile)
                except KeyError:
                    logger.warning("ATL09 %s missing profile %s for beam %s",
                                   atl09_path.name, profile, beam)
            logger.info(
                "ATL09 backg_c loaded: %s -> %d beams",
                atl09_path.name, len(backg_c_lookup_per_beam),
            )

    subsurface_photon_df_added_kd = process_kd_calculation(
        kd_input_dataset, decay_zone_threshold=args.decay_zone_threshold,
        kd_fit_method=args.kd_fit_method,
        poisson_weighted=args.poisson_weighted,
        solar_subtract_threshold=args.solar_subtract_threshold,
        wave_exclusion_multiplier=wave_mult,
        wave_sigma_calm_threshold=args.wave_sigma_calm_threshold,
        min_decay_range_m=args.min_decay_range_m,
        min_fit_span_m=args.min_fit_span_m,
        min_decay_photons=args.min_decay_photons,
        min_decay_bins=args.min_decay_bins,
        min_total_photons=args.min_total_photons,
        min_total_range_m=args.min_total_range_m,
        min_breakpoint_depth=args.min_breakpoint_depth_m,
        horizontal_res=args.horizontal_res,
        backg_c_lookup_per_beam=backg_c_lookup_per_beam if backg_c_lookup_per_beam else None,
        min_noise_floor_photons_per_bin=effective_min_noise_floor,
        vertical_res=args.vertical_res,
    )

    # Compute physics-based noise floor as diagnostic column (not used in fit)
    if not args.disable_physics_noise_floor:
        physics_nf = compute_physics_noise_floor(
            kd_input_dataset,
            horizontal_res=args.horizontal_res,
            iss_velocity=args.iss_velocity,
            vertical_res=args.vertical_res,
        )
        if physics_nf is not None and len(physics_nf) > 0:
            subsurface_photon_df_added_kd = subsurface_photon_df_added_kd.merge(
                physics_nf.rename('physics_nf').reset_index(),
                on='lat_bins', how='left'
            )
            valid = physics_nf.dropna()
            logger.info(
                "Physics NF diagnostic: %d/%d bins (median=%.3f)",
                len(valid), len(physics_nf),
                float(valid.median()) if len(valid) > 0 else 0.0,
            )

    # Attach per-(beam, lat_bin) sea surface height as diagnostic columns.
    # `sea_surface_height` and `sea_surface_label` are lists-of-lists (one entry
    # per beam, in beam_id-sorted order), aligned to the lat_bins ordering used
    # inside get_sea_surface_height (groupby('lat_bins', observed=False)).
    sea_surface_dfs = []
    for (beam_id, beam_data), beam_heights, beam_labels in zip(
        binned_dataset_sea_surface.groupby('beam_id'),
        sea_surface_height,
        sea_surface_label,
    ):
        lat_bin_keys = list(beam_data.groupby('lat_bins', observed=False).groups.keys())
        if len(lat_bin_keys) == len(beam_heights):
            sea_surface_dfs.append(pd.DataFrame({
                'beam_id': beam_id,
                'lat_bins': lat_bin_keys,
                'sea_surface_height': list(beam_heights),
                'sea_surface_abnormal': list(beam_labels),
            }))
        else:
            logger.warning(
                "Sea surface height alignment mismatch for beam %s "
                "(%d lat_bins vs %d heights); skipping sea_surface merge for this beam.",
                beam_id, len(lat_bin_keys), len(beam_heights),
            )

    if sea_surface_dfs and 'beam_id' in subsurface_photon_df_added_kd.columns:
        sea_surface_df = pd.concat(sea_surface_dfs, ignore_index=True)
        subsurface_photon_df_added_kd = subsurface_photon_df_added_kd.merge(
            sea_surface_df, on=['beam_id', 'lat_bins'], how='left'
        )
        valid_sh = sea_surface_df['sea_surface_height'].dropna()
        logger.info(
            "Sea surface height diagnostic: %d/%d bins valid (median=%.3f m)",
            len(valid_sh), len(sea_surface_df),
            float(valid_sh.median()) if len(valid_sh) > 0 else 0.0,
        )

    # Attach per-(beam, lat_bin) surface-peak diagnostics as bin-level columns
    # on AddedKd: surface_peak_count, surface_n_within_3sigma,
    # surface_bin_sum_3sigma, surface_fit_quality. surface_sigma is intentionally
    # NOT re-merged here — the existing per-photon broadcast column (set inside
    # get_subsurface_photon and load-bearing for surface/subsurface separation)
    # already carries the same per-bin value on every photon row. With those
    # four new columns plus the existing surface_sigma, AddedKd has all five
    # surface-peak diagnostics together at the bin level.
    new_diag_cols = [
        'surface_peak_count',
        'surface_n_within_3sigma',
        'surface_bin_sum_3sigma',
        'surface_fit_quality',
    ]
    diag_dfs = []
    for (beam_id, _beam_data), beam_diag in zip(
        binned_dataset_sea_surface.groupby('beam_id'),
        surface_diagnostics_per_beam,
    ):
        if not beam_diag:
            continue
        rows = [
            {'beam_id': beam_id, 'lat_bins': lat_bin,
             **{c: diag.get(c, float('nan')) for c in new_diag_cols}}
            for lat_bin, diag in beam_diag.items()
        ]
        if rows:
            diag_dfs.append(pd.DataFrame(rows))

    if diag_dfs and 'beam_id' in subsurface_photon_df_added_kd.columns:
        diag_df = pd.concat(diag_dfs, ignore_index=True)
        subsurface_photon_df_added_kd = subsurface_photon_df_added_kd.merge(
            diag_df, on=['beam_id', 'lat_bins'], how='left'
        )
        valid_q = diag_df['surface_fit_quality'].dropna()
        valid_pc = diag_df['surface_peak_count'].dropna()
        logger.info(
            "Surface diagnostics: %d/%d bins with peak_count (median=%.0f), "
            "%d with R² (median=%.3f)",
            len(valid_pc), len(diag_df),
            float(valid_pc.median()) if len(valid_pc) > 0 else 0.0,
            len(valid_q),
            float(valid_q.median()) if len(valid_q) > 0 else 0.0,
        )

    # Bootstrap Kd uncertainty (kd_std, kd_ci_low, kd_ci_high, n_boot_success)
    if args.bootstrap_iterations > 0:
        from kd_utils.Kd_analysis import bootstrap_kd_uncertainty
        # Forward the SAME fit gates the main fit used (previously the bootstrap
        # silently ran with the 5 m / 30 / 8 defaults whatever the CLI said).
        bootstrap_df = bootstrap_kd_uncertainty(
            kd_input_dataset,
            n_boot=args.bootstrap_iterations,
            fit_method=args.kd_fit_method,
            min_decay_range_m=args.min_decay_range_m,
            min_decay_photons=args.min_decay_photons,
            min_decay_bins=args.min_decay_bins,
            min_total_photons=args.min_total_photons,
            min_total_range_m=args.min_total_range_m,
            min_breakpoint_depth=args.min_breakpoint_depth_m,
            min_noise_floor_photons_per_bin=effective_min_noise_floor,
            poisson_weighted=args.poisson_weighted,
        )
        if len(bootstrap_df) > 0:
            merge_cols = (
                ['beam_id', 'lat_bins']
                if 'beam_id' in bootstrap_df.columns else ['lat_bins']
            )
            keep_cols = merge_cols + ['kd_std', 'kd_ci_low', 'kd_ci_high', 'n_boot_success']
            subsurface_photon_df_added_kd = subsurface_photon_df_added_kd.merge(
                bootstrap_df[keep_cols],
                on=merge_cols, how='left'
            )
            valid_std = bootstrap_df['kd_std'].dropna()
            logger.info("Bootstrap uncertainty: %d/%d bins, median kd_std=%.4f",
                        len(valid_std), len(bootstrap_df),
                        float(valid_std.median()) if len(valid_std) > 0 else 0.0)
    else:
        bootstrap_df = None

    # Compute per-bin quality flags
    quality_flags = compute_quality_flags(
        subsurface_photon_df_added_kd,
        kd_input_dataset,
        context={
            'atl03_version': atl03_version,
            'gebco_enabled': args.enable_gebco_filter,
            'active_filters': [name for name, on in [
                ('ir_ap', args.enable_ir_ap_filter),
                ('surface_sigma', args.enable_surface_sigma_filter),
                ('solar_bg', args.enable_solar_background_filter),
            ] if on],
            'adaptive_depth_limits': adaptive_depth_limits,
            'snr_per_bin': snr_per_bin,
            'surface_photon_counts': surface_photon_counts,
            'bootstrap_df': bootstrap_df,
            'n_boot': args.bootstrap_iterations,
            'extreme_wave_sigma_threshold': args.extreme_wave_sigma_threshold,
        }
    )
    subsurface_photon_df_added_kd['quality_flags'] = quality_flags

    # Optional offshore-Kd post-processing. Coherent model selection supersedes
    # plain spatial regularization (it resolves the decay-vs-flat bistability at
    # the region level, then smooths). Both keep raw kd/kd_std.
    if getattr(args, 'coherent_model_selection', False):
        from kd_utils.Kd_analysis import coherent_model_selection
        subsurface_photon_df_added_kd = coherent_model_selection(
            subsurface_photon_df_added_kd, kd_input_dataset,
            window=args.spatial_window, adaptive=args.spatial_adaptive,
            min_window=args.spatial_min_window, max_window=args.spatial_max_window,
            central_stat=args.spatial_central, poisson_weighted=args.poisson_weighted,
            min_decay_range_m=args.min_decay_range_m, min_decay_photons=args.min_decay_photons,
            min_fit_span_m=args.min_fit_span_m,
            min_decay_bins=args.min_decay_bins, min_total_photons=args.min_total_photons,
            min_total_range_m=args.min_total_range_m,
            min_breakpoint_depth=args.min_breakpoint_depth_m,
        )
        n_reg = int(subsurface_photon_df_added_kd['coherent_regularized'].sum())
        logger.info("Coherent model selection: %d/%d bins region-selected+smoothed",
                    n_reg, len(subsurface_photon_df_added_kd))
    elif getattr(args, 'spatial_regularize', False):
        from kd_utils.Kd_analysis import spatial_regularize_kd
        subsurface_photon_df_added_kd = spatial_regularize_kd(
            subsurface_photon_df_added_kd, window=args.spatial_window,
            adaptive=args.spatial_adaptive,
            min_window=args.spatial_min_window, max_window=args.spatial_max_window,
            central_stat=args.spatial_central,
        )
        n_reg = int(subsurface_photon_df_added_kd['kd_regularized'].sum())
        logger.info("Spatial regularization (%s): %d/%d bins regularized",
                    f"adaptive {args.spatial_min_window}-{args.spatial_max_window}"
                    if args.spatial_adaptive else f"window={args.spatial_window}",
                    n_reg, len(subsurface_photon_df_added_kd))

    kd_output_path = os.path.join(output_path, f"{timestamp}_AddedKdDataset_strongBeams_Further.csv")
    subsurface_photon_df_added_kd.to_csv(kd_output_path, index=False)

    if not args.no_plot and 'relative_AT_dist' in kd_input_dataset.columns:
        unique_photon_dataset = kd_input_dataset[
            ['relative_AT_dist', 'lat_bins', 'photon_height']
        ].drop_duplicates()
        unique_photon_dataset['relative_AT_dist_center'] = unique_photon_dataset.groupby(
            'lat_bins', observed=False
        )['relative_AT_dist'].transform('mean')

        _closest_rows = []
        for _lat_bin, _group in unique_photon_dataset.groupby('lat_bins', observed=False):
            if _group.empty:
                continue
            _center = _group['relative_AT_dist_center'].iloc[0]
            _group = _group.copy()
            _group['dist_to_center'] = abs(_group['relative_AT_dist'] - _center)
            _closest_rows.append(_group.loc[[_group['dist_to_center'].idxmin()]])
        if not _closest_rows:
            logger.warning("No along-track bins survived filtering. Skipping plot.")
        else:
            closest_to_center = pd.concat(_closest_rows).reset_index(drop=True)
            kd_df_merged_distance = closest_to_center.merge(
                subsurface_photon_df_added_kd,
                on='lat_bins',
                how='left'
            ).drop(columns=['relative_AT_dist_center', 'dist_to_center'], errors='ignore')

            plot_kd_photons(
                output_path,
                timestamp,
                plot_target_beam,
                kd_input_dataset,
                kd_df_merged_distance,
                all_photon_dataset=binned_dataset_sea_surface,
            )

    logger.info("SUCCESS! Kd output: %s", kd_output_path)


if __name__ == '__main__':
    cli_args = get_args()
    try:
        run_pipeline(cli_args)
    except Exception as e:
        logger.error("An error occurred: %s", e)
        raise
