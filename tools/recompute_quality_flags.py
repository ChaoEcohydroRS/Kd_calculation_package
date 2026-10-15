#!/usr/bin/env python
"""Rebuild the per-beam ``quality_flags`` column of existing AddedKd CSVs.

Why (2026-10-15): until the per-beam fix in kd_utils/quality_flags.py, the
pipeline keyed every flag by lat_bin alone. On a multi-beam table that meant
row-based flags kept the last beam processed, photon-based flags pooled all
beams' photons, and the result was broadcast onto every beam's row (all 18 bits
identical across beams in 100% of shared bins). Kd itself never reads the
bitmask, so the retrievals in those CSVs are unaffected; only the flag column
needs rebuilding. This script does that from the stored per-beam columns rather
than re-running the retrieval.

Two modes
---------
--photons <subsurface_photons.csv>  (figure-site runs; photon file kept)
    Calls the fixed compute_quality_flags() per beam. Populated bits:
      exact from stored columns      5, 10, 13, 14, 15, 16
      exact from the photon file     0, 1, 3, 4, 9, 12  (0/1/12 are version-
                                     gated and stay off for v006 input)
      bit 8  FEW_PHOTONS             APPROXIMATE: depth limit := deepest retained
                                     photon of the (beam, bin). Equal to the
                                     pipeline's limit wherever no seafloor cut
                                     truncated the column above it.
      bit 17 BOOTSTRAP_UNSTABLE      PARTIAL: low-success criterion only
                                     (bp_std is not stored).
    Cleared (0) because the pipeline state they need is not stored and no proxy
    reproduces them (checked on the last-processed beam, whose stored values
    were correct): 2 SEAFLOOR_SHALLOW, 6 LOW_SNR, 7 SHORT_WATER_COLUMN,
    11 SURFACE_PEAK_DOMINATED. A pipeline re-run restores them per beam.

(no --photons)  (per-pass validation runs; photon files were deleted)
    Only the row-based bits are populated: 5, 13, 14, 15, 16, 17(low-success).
    All other bits are 0. Say so wherever the column is redistributed.

Usage
-----
  python 03_analysis/recompute_quality_flags.py KD.csv [--photons PH.csv]
         [--atl03-version 6] [--gebco-enabled] [--n-boot 200]
         [--wave-sigma 0.43] [--backup-dir DIR] [--dry-run]

Writes KD.csv in place (after copying the original to --backup-dir) and prints
per-beam before/after bit rates. --dry-run only prints.
"""
from __future__ import annotations
import argparse, os, shutil, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from kd_utils.quality_flags import (QualityFlag, compute_quality_flags,      # noqa: E402
                                    _flag_extreme_wave, _flag_fit_insufficient_photons,
                                    _flag_fit_insufficient_range, _flag_fit_insufficient_bins,
                                    _flag_fit_no_breakpoint, _flag_bootstrap_unstable)

# Bits whose inputs (adaptive depth limit, S/N, surface photon count dict) are
# not stored; cleared in the post-hoc rebuild rather than approximated.
CLEARED_BITS = (QualityFlag.SEAFLOOR_SHALLOW, QualityFlag.LOW_SNR,
                QualityFlag.SHORT_WATER_COLUMN, QualityFlag.SURFACE_PEAK_DOMINATED)
ROW_BITS = (QualityFlag.EXTREME_WAVE, QualityFlag.FIT_INSUFFICIENT_PHOTONS,
            QualityFlag.FIT_INSUFFICIENT_RANGE, QualityFlag.FIT_INSUFFICIENT_BINS,
            QualityFlag.FIT_NO_BREAKPOINT, QualityFlag.BOOTSTRAP_UNSTABLE)
PH_COLS = ["beam_id", "lat_bins", "photon_height", "sea_surface_height", "seafloor_elevation",
           "quality_ph", "is_land_label", "background_rate", "solar_elevation"]


def context_from_photons(kd, ph, atl03_version, gebco_enabled, n_boot, wave_sigma):
    """Rebuild the pipeline context (per-(beam, bin) dicts) from stored data."""
    ph = ph.copy()
    ph["depth"] = ph["sea_surface_height"] - ph["photon_height"]
    depth_limits = {}
    for (beam, lb), g in ph.groupby(["beam_id", "lat_bins"]):
        d = g["depth"].to_numpy()
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            depth_limits[(beam, lb)] = float(d.max())      # proxy for bit 8 only
    boot = kd[["beam_id", "lat_bins", "n_boot_success"]].copy() if "n_boot_success" in kd.columns else None
    return {"atl03_version": atl03_version, "gebco_enabled": gebco_enabled,
            "adaptive_depth_limits": depth_limits, "snr_per_bin": {},
            "surface_photon_counts": {}, "bootstrap_df": boot, "n_boot": n_boot,
            "extreme_wave_sigma_threshold": wave_sigma}


def row_only_flags(kd, n_boot, wave_sigma):
    """Bits derivable from the Kd rows alone, computed one beam at a time."""
    out = pd.Series(0, index=kd.index, dtype="int64")
    for beam, kd_b in kd.groupby("beam_id"):
        boot = kd_b[["lat_bins", "n_boot_success"]] if "n_boot_success" in kd_b.columns else None
        res = {QualityFlag.EXTREME_WAVE: _flag_extreme_wave(kd_b, threshold=wave_sigma),
               QualityFlag.FIT_INSUFFICIENT_PHOTONS: _flag_fit_insufficient_photons(kd_b),
               QualityFlag.FIT_INSUFFICIENT_RANGE: _flag_fit_insufficient_range(kd_b),
               QualityFlag.FIT_INSUFFICIENT_BINS: _flag_fit_insufficient_bins(kd_b),
               QualityFlag.FIT_NO_BREAKPOINT: _flag_fit_no_breakpoint(kd_b),
               QualityFlag.BOOTSTRAP_UNSTABLE: _flag_bootstrap_unstable(kd_b, boot, n_boot_expected=n_boot)}
        masks = []
        for lb in kd_b["lat_bins"].values:
            m = 0
            for flag, dct in res.items():
                if dct.get(lb, False):
                    m |= (1 << flag)
            masks.append(m)
        out.loc[kd_b.index] = masks
    return out


def bit_table(kd, col_old, col_new):
    rows = []
    for beam, g in kd.groupby("beam_id"):
        o = g[col_old].fillna(0).astype(np.int64).to_numpy()
        n = g[col_new].astype(np.int64).to_numpy()
        for f in QualityFlag:
            ro, rn = ((o >> f) & 1).mean() * 100, ((n >> f) & 1).mean() * 100
            if ro > 0 or rn > 0:
                rows.append((beam, f.name, round(ro, 1), round(rn, 1)))
    return pd.DataFrame(rows, columns=["beam", "flag", "old_pct", "new_pct"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kd_csv")
    ap.add_argument("--photons", default=None)
    ap.add_argument("--atl03-version", type=int, default=6)
    ap.add_argument("--gebco-enabled", action="store_true")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--wave-sigma", type=float, default=0.43)
    ap.add_argument("--backup-dir", default=None, help="where the original CSV is copied (default: <dir>/_flags_backup_20261015)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    kd = pd.read_csv(a.kd_csv)
    if "beam_id" not in kd.columns:
        sys.exit("kd CSV has no beam_id column; nothing to fix")
    kd["quality_flags_old"] = kd["quality_flags"] if "quality_flags" in kd.columns else 0
    if a.photons:
        ph = pd.read_csv(a.photons, usecols=lambda c: c in PH_COLS)
        missing = [c for c in PH_COLS if c not in ph.columns]
        if missing:
            print(f"  note: photon file lacks {missing}; dependent bits stay 0")
            for c in missing:
                ph[c] = np.nan
        ctx = context_from_photons(kd, ph, a.atl03_version, a.gebco_enabled, a.n_boot, a.wave_sigma)
        new = compute_quality_flags(kd, ph, ctx)
        clear = 0
        for b in CLEARED_BITS:
            clear |= (1 << b)
        new = new & ~clear
        mode = "photons"
    else:
        new = row_only_flags(kd, a.n_boot, a.wave_sigma)
        mode = "row-only"
    kd["quality_flags"] = new.astype("int64").values
    tab = bit_table(kd, "quality_flags_old", "quality_flags")
    same = (kd["quality_flags_old"].fillna(0).astype(np.int64) == kd["quality_flags"]).mean() * 100
    if not a.quiet:
        print(f"{os.path.basename(a.kd_csv)} [{mode}] rows {len(kd)}, beams {kd.beam_id.nunique()}, "
              f"bitmask unchanged on {same:.0f}% of rows")
        print(tab.to_string(index=False))
    else:
        print(f"{os.path.basename(os.path.dirname(a.kd_csv))}: {mode}, {len(kd)} rows, unchanged {same:.0f}%")
    if a.dry_run:
        return
    bdir = a.backup_dir or os.path.join(os.path.dirname(os.path.abspath(a.kd_csv)), "_flags_backup_20261015")
    os.makedirs(bdir, exist_ok=True)
    dst = os.path.join(bdir, os.path.basename(a.kd_csv))
    if not os.path.exists(dst):
        shutil.copy2(a.kd_csv, dst)
    kd.drop(columns=["quality_flags_old"]).to_csv(a.kd_csv, index=False)


if __name__ == "__main__":
    main()
