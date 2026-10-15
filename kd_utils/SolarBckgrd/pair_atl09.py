"""Pair an ATL03 stem with its matching ATL09 file.

ATL03 stem:  ``ATL03_YYYYMMDDhhmmss_RRRRCCSS_VVV_RR.h5``
ATL09 stem:  ``ATL09_YYYYMMDDhhmmss_RRRRCC01_VVV_RR.h5``

where RRRR=rgt, CC=cycle, SS=region (varies for ATL03; ATL09 always has 01),
VVV=release, RR=version. The timestamp differs slightly between ATL03 and
the corresponding ATL09 (ATL09 starts a few minutes earlier in the granule),
so we cannot match on timestamp. Instead we match on the ``RRRRCC01`` token
plus VVV+RR.

ATL03 beam → ATL09 profile mapping depends on the ``orbit_info/sc_orient``
flag (0 = backward, 1 = forward):
  sc_orient=1 (forward):  gt1r→profile_1, gt2r→profile_2, gt3r→profile_3
  sc_orient=0 (backward): gt1l→profile_1, gt2l→profile_2, gt3l→profile_3
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple
import h5py
import numpy as np
import pandas as pd


ATL03_STEM_RE = re.compile(
    r"^(?:processed_)?ATL03_(?P<ts>\d{14})_(?P<rgt>\d{4})(?P<cycle>\d{2})(?P<region>\d{2})_(?P<release>\d{3})_(?P<version>\d{2})$"
)


BEAM_TO_PROFILE_SC1 = {"gt1r": "profile_1", "gt2r": "profile_2", "gt3r": "profile_3"}
BEAM_TO_PROFILE_SC0 = {"gt1l": "profile_1", "gt2l": "profile_2", "gt3l": "profile_3"}


def parse_atl03_stem(stem: str) -> Optional[dict]:
    """Parse an ATL03 stem (with or without ``.h5`` and ``processed_`` prefix)."""
    stem = stem.replace(".h5", "")
    m = ATL03_STEM_RE.match(stem)
    if not m:
        return None
    return {k: m.group(k) for k in ("ts", "rgt", "cycle", "region", "release", "version")}


def atl09_token_for_atl03(parts: dict) -> str:
    """Build the ATL09 RRRRCC01 segment token from parsed ATL03 parts."""
    return f"{parts['rgt']}{parts['cycle']}01"


def find_atl09_file(atl03_stem: str, atl09_root: Path) -> Optional[Path]:
    """Search ``atl09_root`` recursively for the ATL09 file matching this ATL03.

    Matching is on the ``RRRRCC01`` rgt+cycle token only — release and
    version are intentionally NOT required to match. In practice, CD's
    ATL03 release 006 is paired with ATL09 release 007 on this disk.
    Returns the first matching path, or None if not found.
    """
    parts = parse_atl03_stem(atl03_stem)
    if parts is None:
        return None
    token = atl09_token_for_atl03(parts)
    pattern = f"ATL09_*_{token}_*_*.h5"
    matches = list(Path(atl09_root).rglob(pattern))
    return matches[0] if matches else None


def read_atl03_sc_orient(atl03_h5: Path) -> int:
    """Read /orbit_info/sc_orient[0] from an ATL03 file."""
    with h5py.File(atl03_h5, "r") as f:
        return int(f["orbit_info/sc_orient"][:][0])


def beam_to_profile(beam: str, sc_orient: int) -> Optional[str]:
    """Map an ATL03 strong-beam id (gt1r/gt2r/gt3r or gt1l/gt2l/gt3l) to ATL09 profile_N."""
    if sc_orient == 1:
        return BEAM_TO_PROFILE_SC1.get(beam)
    elif sc_orient == 0:
        return BEAM_TO_PROFILE_SC0.get(beam)
    return None


def load_backg_c_table(
    atl09_h5: Path, profile: str
) -> pd.DataFrame:
    """Return DataFrame with columns ('latitude', 'backg_c') for the profile.

    Filters out non-finite and negative backg_c. Sorted by latitude.
    """
    from kd_utils.SolarBckgrd.atl09_backg_c import read_atl09_backg_c
    raw = read_atl09_backg_c(str(atl09_h5), profile=profile)
    df = pd.DataFrame({"latitude": raw["latitude"], "backg_c": raw["backg_c"]})
    df = df[np.isfinite(df.backg_c) & (df.backg_c >= 0)].sort_values("latitude").reset_index(drop=True)
    return df


def median_backg_c_in_lat_range(
    backg_df: pd.DataFrame, lat_lo: float, lat_hi: float
) -> float:
    """Median backg_c in a [lat_lo, lat_hi] interval; nearest-neighbour fallback if empty."""
    if backg_df.empty:
        return float("nan")
    inside = backg_df[(backg_df.latitude >= lat_lo) & (backg_df.latitude <= lat_hi)]
    if not inside.empty:
        return float(inside.backg_c.median())
    mid = 0.5 * (lat_lo + lat_hi)
    idx = (backg_df.latitude - mid).abs().idxmin()
    return float(backg_df.loc[idx, "backg_c"])
