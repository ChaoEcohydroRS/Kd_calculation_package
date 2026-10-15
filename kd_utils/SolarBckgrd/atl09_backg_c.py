"""ATL09 atmospheric-histogram background (backg_c) helpers.

The conclusion from the email exchange with Almut Pingel and Aimee Gibbons
(NASA/GSFC, May 2026) is that the appropriate solar background quantity for
ICESat-2 Kd retrieval is the Method-1 atmospheric-histogram background
``/profile_x/high_rate/backg_c`` from ATL09.

Units of ``backg_c``: photons per ATM histogram bin, where one ATM bin is
29.9792 m in the vertical (z) by 280 m along-track (x), summed over
400 laser shots.

For a custom histogram with bin sizes ``bin_z_m`` (vertical) and ``bin_x_m``
(along-track), the expected number of background photons per bin is:

    bkg_per_bin = backg_c * (bin_z_m / 29.9792) * (bin_x_m / 280.0)

That is the formula Emily Eidam closed the thread with.
"""
from __future__ import annotations

from typing import Dict
import h5py
import numpy as np

ATM_BIN_Z_M = 29.9792
ATM_BIN_X_M = 280.0

ATL09_STRONG_PROFILES = ("profile_1", "profile_2", "profile_3")


def bkg_per_hist_bin(
    backg_c: np.ndarray | float,
    bin_z_m: float,
    bin_x_m: float,
    atm_bin_z_m: float = ATM_BIN_Z_M,
    atm_bin_x_m: float = ATM_BIN_X_M,
) -> np.ndarray | float:
    """Scale ATL09 backg_c (photons/ATM-bin) to a histogram bin of bin_z_m x bin_x_m."""
    scale = (bin_z_m / atm_bin_z_m) * (bin_x_m / atm_bin_x_m)
    return np.asarray(backg_c, dtype=np.float64) * scale


def read_atl09_backg_c(h5_path: str, profile: str = "profile_1") -> Dict[str, np.ndarray]:
    """Read backg_c and the matching time/segment indices from one ATL09 profile."""
    group = f"/{profile}/high_rate"
    with h5py.File(h5_path, "r") as f:
        if group not in f:
            raise KeyError(f"{h5_path} has no group {group}")
        out = {"backg_c": f[f"{group}/backg_c"][:].astype(np.float64)}
        for opt in ("delta_time", "latitude", "longitude", "segment_id"):
            ds = f.get(f"{group}/{opt}")
            if ds is not None:
                out[opt] = ds[:]
    return out
