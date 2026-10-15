import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pyproj import Geod, Proj

from kd_utils.data_processing import convert_wgs_to_utm

logger = logging.getLogger(__name__)


def _get_beam_id_from_track_pair(df: pd.DataFrame) -> pd.Series:
    if 'gt' in df.columns:
        return df['gt'].astype(str)
    if 'track' in df.columns and 'pair' in df.columns:
        suffix = df['pair'].map({0: 'l', 1: 'r'}).fillna('x')
        return 'gt' + df['track'].astype(int).astype(str) + suffix
    return pd.Series(['unknown'] * len(df), index=df.index, dtype='object')


def _infer_strong_beam_suffix(df: pd.DataFrame) -> Optional[str]:
    if 'sc_orient' not in df.columns or df.empty:
        return None
    orient = df['sc_orient'].dropna()
    if orient.empty:
        return None
    # Same logic as your Florida script.
    return 'l' if int(orient.iloc[0]) == 0 else 'r'


# ATL09 ancillary fields are requested group-qualified (e.g. "high_rate/backg_c").
# SlideRule may return the column under the qualified name or the bare leaf name,
# so probe both. backg_c is the Method-1 atmospheric-histogram background that the
# BMA Poisson-floor gate wants (see kd_utils/SolarBckgrd/atl09_backg_c.py).
ATL09_BACKG_C_FIELD = "high_rate/backg_c"
_BACKG_C_CANDIDATES = ("high_rate/backg_c", "backg_c", "atl09_high_rate_backg_c")
_INT_HEIGHT_CANDIDATES = ("bckgrd_atlas/bckgrd_int_height", "bckgrd_int_height")


def _resolve_first(src: pd.DataFrame, candidates) -> Optional[pd.Series]:
    """Return the first present column among `candidates`, coerced to numeric."""
    for col in candidates:
        if col in src.columns:
            return pd.to_numeric(src[col], errors="coerce")
    return None


def _normalize_signal_conf(values: pd.Series) -> pd.Series:
    def to_scalar(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            if len(v) == 0:
                return np.nan
            # Index 1 = ocean confidence (0=land, 1=ocean, 2=sea_ice, 3=land_ice, 4=inland_water)
            return float(v[1]) if len(v) > 1 else float(v[0])
        try:
            return float(v)
        except Exception:
            return np.nan
    return values.apply(to_scalar)


def _relative_distance_km_per_beam(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for beam_id, beam_df in df.groupby('beam_id'):
        beam_df = beam_df.sort_values('delta_time')
        if 'segment_dist' in beam_df.columns:
            rel_km = (beam_df['segment_dist'] - beam_df['segment_dist'].min()) / 1000.0
            out.loc[beam_df.index] = rel_km.values
            continue

        lons = beam_df['longitude'].to_numpy(dtype=float)
        lats = beam_df['latitude'].to_numpy(dtype=float)
        if len(lons) == 0:
            continue
        geod = Geod(ellps='WGS84')
        cumulative_m = np.zeros(len(lons), dtype=float)
        for i in range(1, len(lons)):
            _, _, dist_m = geod.inv(lons[i - 1], lats[i - 1], lons[i], lats[i])
            cumulative_m[i] = cumulative_m[i - 1] + max(dist_m, 0.0)
        out.loc[beam_df.index] = cumulative_m / 1000.0
    return out


def _project_to_utm(df: pd.DataFrame) -> pd.DataFrame:
    projected = df.copy()
    projected['lon'] = np.nan
    projected['lat'] = np.nan
    for beam_id, beam_df in projected.groupby('beam_id'):
        if beam_df.empty:
            continue
        lon0 = float(beam_df['longitude'].iloc[0])
        lat0 = float(beam_df['latitude'].iloc[0])
        epsg = convert_wgs_to_utm(lon0, lat0)
        proj = Proj(epsg)
        x, y = proj(beam_df['longitude'].to_numpy(dtype=float), beam_df['latitude'].to_numpy(dtype=float))
        projected.loc[beam_df.index, 'lon'] = x
        projected.loc[beam_df.index, 'lat'] = y
    return projected


def sliderule_to_framework_dataset(
    atl03_gdf,
    strong_beams_only: bool = True,
) -> pd.DataFrame:
    """
    Convert SlideRule ATL03 photon GeoDataFrame to the framework's sea_photon_dataset schema.
    """
    if atl03_gdf is None or len(atl03_gdf) == 0:
        return pd.DataFrame(columns=[
            'beam_id', 'latitude', 'longitude', 'lat', 'lon', 'photon_height',
            'quality_ph', 'photon_conf', 'ref_elevation', 'ref_azimuth',
            'relative_AT_dist', 'solar_elevation', 'background_rate', 'is_land_label'
        ])

    src = pd.DataFrame(atl03_gdf).copy()
    src['beam_id'] = _get_beam_id_from_track_pair(src)
    if strong_beams_only:
        strong_suffix = _infer_strong_beam_suffix(src)
        if strong_suffix in ('l', 'r'):
            src = src[src['beam_id'].str.endswith(strong_suffix)]

    # Resolve latitude/longitude from either custom fields or GeoDataFrame geometry
    if 'lat_ph' in src.columns:
        lat_col, lon_col = 'lat_ph', 'lon_ph'
    elif 'geometry' in src.columns:
        try:
            import geopandas as gpd
            geo = gpd.GeoSeries(src['geometry'])
            src['_lat'] = geo.y
            src['_lon'] = geo.x
        except Exception:
            src['_lat'] = src['geometry'].apply(lambda g: g.y if hasattr(g, 'y') else np.nan)
            src['_lon'] = src['geometry'].apply(lambda g: g.x if hasattr(g, 'x') else np.nan)
        lat_col, lon_col = '_lat', '_lon'
    else:
        lat_col, lon_col = 'latitude', 'longitude'

    # Resolve height column — apply geoid correction if available
    # SlideRule default returns WGS84 ellipsoidal height; subtract geoid to get ~MSL
    height_col = 'h_ph' if 'h_ph' in src.columns else 'height'
    if 'geoid' in src.columns and height_col == 'height':
        src['height'] = src['height'] - src['geoid']
        logger.info("Applied geoid correction (%.1f m avg) to SlideRule heights",
                     float(src['geoid'].mean()))

    out = pd.DataFrame({
        'beam_id': src['beam_id'].astype(str),
        'latitude': pd.to_numeric(src[lat_col], errors='coerce'),
        'longitude': pd.to_numeric(src[lon_col], errors='coerce'),
        'photon_height': pd.to_numeric(src[height_col], errors='coerce'),
        'quality_ph': pd.to_numeric(src.get('quality_ph', np.nan), errors='coerce'),
        'ref_elevation': pd.to_numeric(src.get('ref_elev', np.nan), errors='coerce'),
        'ref_azimuth': pd.to_numeric(src.get('ref_azimuth', np.nan), errors='coerce'),
        'solar_elevation': pd.to_numeric(src.get('solar_elevation', np.nan), errors='coerce'),
        'background_rate': pd.to_numeric(
            src.get('bckgrd_rate', src.get('background_rate', np.nan)), errors='coerce'),
        'delta_time': pd.to_numeric(src.get('delta_time', np.nan), errors='coerce'),
        'segment_dist': pd.to_numeric(src.get('segment_dist', np.nan), errors='coerce'),
    })

    # Resolve signal confidence column
    if 'signal_conf_ph' in src.columns:
        out['photon_conf'] = _normalize_signal_conf(src['signal_conf_ph'])
    elif 'atl03_cnf' in src.columns:
        out['photon_conf'] = _normalize_signal_conf(src['atl03_cnf'])
    else:
        out['photon_conf'] = np.nan

    # ATL09 solar background (joined per-photon by SlideRule via atl09_fields).
    # Carried through so the caller can build the per-(beam, lat_bin) backg_c
    # table that the BMA Poisson-floor gate consumes, without a local ATL09 file.
    backg_c = _resolve_first(src, _BACKG_C_CANDIDATES)
    if backg_c is not None:
        out['backg_c'] = backg_c
        logger.info("ATL09 backg_c joined from SlideRule: %d/%d photons finite",
                    int(np.isfinite(backg_c).sum()), len(backg_c))
    else:
        logger.warning("No ATL09 backg_c column in SlideRule return "
                       "(requested atl09_fields?); Poisson-floor gate will fall back.")

    int_height = _resolve_first(src, _INT_HEIGHT_CANDIDATES)
    if int_height is not None:
        out['bckgrd_int_height'] = int_height

    out = out.dropna(subset=['latitude', 'longitude', 'photon_height']).copy()
    out = out.reset_index(drop=True)
    out = _project_to_utm(out)
    out['relative_AT_dist'] = _relative_distance_km_per_beam(out)
    out['is_land_label'] = 0

    return out.drop(columns=['delta_time', 'segment_dist'], errors='ignore')


def fetch_sliderule_atl03(
    region: List[Dict[str, float]],
    t0: str,
    t1: str,
    rgt: Optional[int] = None,
    ph_fields: Optional[List[str]] = None,
    atl09_fields: Optional[List[str]] = None,
    srt: int = 1,
    cnf: int = -2,
    sliderule_url: str = "slideruleearth.io",
    verbose: bool = False,
):
    """
    Fetch ATL03 photons from SlideRule.

    ``srt`` / ``cnf`` control SlideRule's signal-confidence prefilter and MUST stay
    inclusive for Kd work. The local pipeline (``Extract_sea_photons``) keeps ALL
    photons (only land is dropped), but SlideRule's DEFAULT ``cnf`` discards
    low-confidence photons — and the subsurface water-column decay photons are exactly
    those. Verified 2026-07-24 (OR coast): the default dropped the entire subsurface
    column on gt3l (frac below −3 m: 0.000 vs local 0.054) → NaN Kd on gt2l/gt3l.
    ``cnf=-2`` (CNF_POSSIBLE_TEP, the lowest) keeps everything and reproduces the local
    frac-below-−3 m to 3 decimals on all beams. ``srt=1`` (ocean) sets which surface
    type's confidence is used; moot once cnf=-2 keeps all, but correct for our target.

    ``atl09_fields`` requests ATL09 ancillary fields (from the ATL09
    "bckgrd_atlas"/"high_rate"/"low_rate" groups, group-qualified). NOTE: this join is
    silently ignored server-side (verified 5.3.1 & 5.4.4) — ATL09 backg_c is fetched
    separately via ``fetch_atl09_backg_c_tables`` (CMR + h5). Kept for forward-compat;
    pass ``atl09_fields=[]`` to omit.
    """
    try:
        from sliderule import sliderule, icesat2
    except Exception as e:
        raise ImportError("sliderule package is required for SlideRule ingestion.") from e

    sliderule.init(sliderule_url, verbose=verbose)
    # ATL03 group layout — a field requested against the WRONG group makes H5Coro
    # fail the whole track ("read failure on /gt2r/heights/ref_azimuth") and return
    # zero photons, so keep these straight:
    #   /gtx/heights/      delta_time lat_ph lon_ph h_ph quality_ph signal_conf_ph
    #   /gtx/geolocation/  ref_elev ref_azimuth solar_elevation segment_dist_x geoid
    #   /gtx/bckgrd_atlas/ bckgrd_rate bckgrd_int_height   (NOT reachable via
    #                      atl03_ph_fields / atl03_geo_fields)
    # atl03sp already returns natively: height, geometry(lat/lon), quality_ph,
    # atl03_cnf, solar_elevation, background_rate, segment_dist, segment_id,
    # sc_orient, x_atc/y_atc — so only genuinely-missing fields are requested here.
    fields = ph_fields if ph_fields is not None else ["signal_conf_ph"]
    if atl09_fields is None:
        atl09_fields = [ATL09_BACKG_C_FIELD]
    params = {"poly": region, "t0": t0, "t1": t1, "atl03_ph_fields": fields,
              "atl03_geo_fields": ["ref_elev", "ref_azimuth", "geoid"],
              "srt": int(srt), "cnf": int(cnf)}
    if atl09_fields:
        params["atl09_fields"] = list(atl09_fields)
    if rgt is not None:
        params["rgt"] = int(rgt)
    logger.info("SlideRule atl03sp: rgt=%s t0=%s t1=%s srt=%s cnf=%s",
                rgt, t0, t1, srt, cnf)
    return icesat2.atl03sp(params)


def fetch_and_prepare_sliderule_dataset(
    region: List[Dict[str, float]],
    t0: str,
    t1: str,
    rgt: Optional[int] = None,
    strong_beams_only: bool = True,
    atl09_fields: Optional[List[str]] = None,
    sliderule_url: str = "slideruleearth.io",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    One-call helper:
    fetch ATL03 photons (+ ATL09 ancillary background) from SlideRule and convert
    to the framework dataset schema.
    """
    atl03_gdf = fetch_sliderule_atl03(
        region=region,
        t0=t0,
        t1=t1,
        rgt=rgt,
        atl09_fields=atl09_fields,
        sliderule_url=sliderule_url,
        verbose=verbose,
    )
    return sliderule_to_framework_dataset(atl03_gdf, strong_beams_only=strong_beams_only)


_ATL09_CACHE: Dict[tuple, pd.DataFrame] = {}


def fetch_atl09_backg_c_tables(
    region: List[Dict[str, float]],
    t0: str,
    t1: str,
    beams: List[str],
    sc_orient: int,
    lat_range: Optional[tuple] = None,
    sliderule_url: str = "slideruleearth.io",
) -> Dict[str, pd.DataFrame]:
    """Per-beam ATL09 ('latitude','backg_c') tables, fetched WITHOUT downloading.

    Why not the documented ``atl09_fields`` ancillary join? It is silently ignored
    by atl03sp (verified 2026-07-23 on client 5.3.1 AND 5.4.4 against server
    5.4.4): a bogus atl03_geo_field errors loudly, while a bogus atl09_field is a
    no-op and no column ever appears. So we resolve the ATL09 granule via CMR and
    remote-read just the two 1-D datasets we need.

    NOTE: ATL09 is one granule per *orbit*, so its timestamp differs from the
    ATL03 *region* granule (e.g. ATL03 ..._083629_04232306 pairs with
    ATL09 ..._080457_04232301). Never construct the ATL09 name from the ATL03
    timestamp — resolve it, exactly as pair_atl09.find_atl09_file globs for it.

    Returns the same contract as pair_atl09.load_backg_c_table: finite,
    non-negative backg_c sorted by latitude, keyed by beam id.
    """
    from sliderule import sliderule, earthdata, h5
    from kd_utils.SolarBckgrd.pair_atl09 import beam_to_profile

    sliderule.init(sliderule_url, verbose=False)
    try:
        granules = earthdata.cmr(short_name="ATL09", polygon=region,
                                 time_start=t0, time_end=t1)
    except Exception as e:
        logger.warning("ATL09 CMR lookup failed (%s); Poisson-floor gate will fall back.", e)
        return {}
    if not granules:
        logger.warning("No ATL09 granule found for the requested window; "
                       "Poisson-floor gate will fall back.")
        return {}
    resource = granules[0]
    logger.info("ATL09 granule resolved via CMR: %s", resource)

    out: Dict[str, pd.DataFrame] = {}
    for beam in beams:
        profile = beam_to_profile(beam, sc_orient)
        if profile is None:
            logger.warning("No ATL09 profile maps to beam %s (sc_orient=%s)", beam, sc_orient)
            continue
        key = (resource, profile)
        if key not in _ATL09_CACHE:
            try:
                backg = np.asarray(h5.h5(f"/{profile}/high_rate/backg_c", resource, "icesat2"))
                lat = np.asarray(h5.h5(f"/{profile}/high_rate/latitude", resource, "icesat2"))
            except Exception as e:
                logger.warning("ATL09 remote read failed for %s %s: %s", resource, profile, e)
                continue
            if backg.size == 0 or lat.size != backg.size:
                logger.warning("ATL09 %s %s: empty/mismatched (backg=%d lat=%d)",
                               resource, profile, backg.size, lat.size)
                continue
            tbl = pd.DataFrame({"latitude": lat.astype(float),
                                "backg_c": backg.astype(float)})
            tbl = tbl[np.isfinite(tbl.backg_c) & (tbl.backg_c >= 0)
                      & np.isfinite(tbl.latitude)].sort_values("latitude").reset_index(drop=True)
            _ATL09_CACHE[key] = tbl
        tbl = _ATL09_CACHE[key]
        if lat_range is not None:
            lo, hi = min(lat_range), max(lat_range)
            pad = 0.5  # keep a margin so per-bin median lookups near the edges resolve
            tbl = tbl[(tbl.latitude >= lo - pad) & (tbl.latitude <= hi + pad)]
        if tbl.empty:
            continue
        out[beam] = tbl.reset_index(drop=True)
        logger.info("ATL09 backg_c %s->%s: %d rows (median %.4g)",
                    beam, profile, len(tbl), float(tbl.backg_c.median()))
    return out


def backg_c_table_from_photons(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build the per-beam ('latitude','backg_c') table that the Poisson-floor gate
    consumes, from SlideRule-joined per-photon backg_c.

    Mirrors kd_utils.SolarBckgrd.pair_atl09.load_backg_c_table (same columns, same
    filtering: finite and non-negative, sorted by latitude) so the downstream
    `backg_c_lookup_per_beam` contract is identical for local-ATL09 and SlideRule.
    """
    if df is None or df.empty or 'backg_c' not in df.columns:
        return {}
    out: Dict[str, pd.DataFrame] = {}
    for beam_id, beam_df in df.groupby('beam_id'):
        tbl = pd.DataFrame({
            'latitude': pd.to_numeric(beam_df['latitude'], errors='coerce'),
            'backg_c': pd.to_numeric(beam_df['backg_c'], errors='coerce'),
        })
        tbl = tbl[np.isfinite(tbl.backg_c) & (tbl.backg_c >= 0) & np.isfinite(tbl.latitude)]
        if tbl.empty:
            continue
        # Collapse duplicate per-photon repeats of the same ATL09 high_rate value.
        tbl = (tbl.groupby('latitude', as_index=False)['backg_c'].median()
                  .sort_values('latitude').reset_index(drop=True))
        out[str(beam_id)] = tbl
    return out
