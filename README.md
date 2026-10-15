# Kd_calculation_package

Photon-profile diffuse attenuation coefficient (K<sub>dph</sub>) from ICESat-2 ATLAS
photon returns in coastal and river-mouth waters.

Given one ATL03 granule (a local HDF5 subset or a SlideRule request), the
pipeline detects the sea surface, extracts sub-surface photons, builds a photon
count versus depth histogram for every 500 m along-track bin and every beam,
fits Beer's law to the exponential-decay part of each histogram, and writes one
row per bin and beam with K<sub>dph</sub>, its bootstrap uncertainty, the fitted
breakpoint and background floor, surface diagnostics and a quality bitmask.

This is the code behind the ICESat-2 K<sub>dph</sub> manuscript. It is the retrieval only; the figure and validation scripts live in
separate project folders.

## Processing steps

1. **Load photons.** Local ATL03 subset (`--input_source local`) or SlideRule
   (`--input_source sliderule`, `--sr_bbox`, `--sr_t0/--sr_t1`). Heights are
   referenced to the geoid at load time.
2. **Bin.** 500 m along track (`--horizontal_res`) by 0.25 m vertical
   (`--vertical_res`), per beam.
3. **Sea surface.** Per bin, a Gaussian is fitted to the dominant surface peak;
   its mean is the surface height, its width σ the surface roughness. A zone of
   max(2σ, 1 m) below the surface is excluded from the fit.
4. **Sub-surface photons and optional masks.** GEBCO seafloor mask
   (`--enable_gebco_filter`, bins shallower than `--min_seabed_depth` dropped),
   ATL24 seafloor labels (`--enable_atl24_filter`), surface-σ filter,
   IR afterpulse filter.
5. **Refraction.** Snell correction of photon depth and along-track position
   (n from `--refraction_water_temp_c`, `--refraction_wavelength_nm`), followed
   by re-binning so that K<sub>dph</sub> is per true metre
   (`--no_refraction_rebin` reverts to apparent metres).
6. **Adaptive histogram quality filter.** Per bin: S/N of the 1 to 2 m layer
   against the deep background must exceed 3; an adaptive depth limit is set at
   the deepest 1 m layer exceeding `--signal_above_noise` × background, plus a
   10 m tail margin.
7. **Background floor.** From the ATL03 background rate, or from ATL09
   `backg_c` when `--atl09_root` is given.
8. **Beer's-law fit per histogram** (`--kd_fit_method bma`, default). Every
   populated 0.25 m depth bin at least 2 m below the surface exclusion is a
   candidate breakpoint, provided the decay segment above it spans at least 5 m
   with at least 8 populated bins and 30 photons and has a negative slope, and
   the segment below it has at least 4 populated bins whose mean lies at least
   two residual standard deviations off the extrapolated decay line. Each
   admissible candidate is fitted and scored by BIC (four parameters, the
   breakpoint location counted) against the single-segment fit (two
   parameters). The per-histogram estimate combines the candidates with
   weights ∝ exp(−½ ΔBIC); when the single-segment fit holds more than 95% of
   the weight it is returned unchanged (`fit_failure_reason = no_noise_floor`).
   Histograms with fewer than 30 photons, fewer than 6 populated bins or less
   than 2.5 m of populated depth range are rejected.
9. **Uncertainty.** Residual bootstrap of the per-histogram fit
   (`--bootstrap_iterations`, 200 in the paper), resampling within segments and
   repeating the full candidate search; `kd_std`, `kd_ci_low`, `kd_ci_high`
   are the standard deviation and 2.5 to 97.5 percentiles.
10. **Coherent model selection** (`--coherent_model_selection`). For each
    histogram the minimum-BIC two-segment fit (`kd_2seg`) and the single-segment
    fit (`kd_single`) are kept; the choice between the two models is made by
    majority over a centred window of `--spatial_window` histograms (13 in the
    paper) per beam; a running median is applied where neighbouring fits are
    unstable. The result is `kd_coherent`, the along-track K<sub>dph</sub>
    reported in the paper. No BIC weighting enters this value.
11. **Quality flags.** An 18-bit mask per bin and beam (`kd_utils/quality_flags.py`).
    Diagnostic only; nothing in the retrieval reads it.

## Installation

Python 3.9. Pinned versions in `requirements.txt` are the ones used for the
paper.

```bash
conda create -n kdph python=3.9
conda activate kdph
pip install -r requirements.txt
python -m pytest -q tests        # 97 tests, about 30 s
```

`rasterio`, `Rtree`, `hdbscan` and `pyproj` are easier from conda-forge if pip
wheels fail. `sliderule` and `earthaccess` are needed only for the SlideRule
input path and for ATL09 retrieval through CMR.

## Running

Inputs: an ATL03 granule (or a SlideRule bounding box), a land/shoreline
shapefile (Natural Earth `ne_10m_land`), GEBCO tiles if the seafloor mask is on,
and optionally ATL09 granules for the background rate. The command that
produced the paper's site retrievals (see `examples/run_four_sites.sbatch`):

```bash
python -u main.py \
  --workspace_path  /path/to/workspace \
  --atl03_path      /path/to/workspace/Dataset/ATL03 \
  --atl03_file      ATL03_20231103172707_06982106_006_01_subsetted.h5 \
  --other_data_path Dataset/ \
  --shoreline_data  Shorelines/ne_10m_land/ne_10m_land.shp \
  --atl09_root      /path/to/workspace/Dataset/ATL09 \
  --output_path     /path/to/output \
  --target_beams    gt1l,gt2l,gt3l,gt1r,gt2r,gt3r \
  --enable_gebco_filter --poisson_weighted --coherent_model_selection \
  --min_seabed_depth 3 --no_plot
```

`python main.py --help` lists every option with its default.

## Outputs

`<granule>_AddedKdDataset_strongBeams_Further.csv`, one row per beam and 500 m bin:

| column | meaning |
|---|---|
| `beam_id`, `lat_bins`, `latitude`, `longitude` | beam and bin identity and centre |
| `kd`, `e0` | per-histogram Beer's-law estimate (BIC-weighted) and its surface intercept, m⁻¹ |
| `kd_std`, `kd_ci_low`, `kd_ci_high`, `n_boot_success` | residual-bootstrap uncertainty of `kd` |
| `kd_single`, `bic_single`, `kd_2seg`, `bic_2seg`, `has_breakpoint` | the two candidate models kept for the regional vote |
| `coherent_2seg`, `kd_coherent`, `kd_coherent_lo`, `kd_coherent_hi`, `coherent_regularized` | regionally voted and smoothed K<sub>dph</sub> and its window band |
| `noise_floor`, `physics_nf` | fitted and physics-based background floor, photons per bin |
| `sea_surface_height`, `surface_sigma`, `surface_fit_quality`, `surface_peak_count` | surface diagnostics |
| `fit_failure_reason` | `None`, `insufficient_photons`, `too_few_bins`, `insufficient_range`, `no_valid_candidates`, `negative_kd`, `no_noise_floor` |
| `quality_flags` | 18-bit diagnostic mask, see `kd_utils/quality_flags.py` |

`<granule>_strongBeam_<beams>_subsurface_photons.csv` holds the sub-surface
photons after all masks, with per-photon depth, bin labels and ATL03 quality
fields.

## Notes on provenance

- The retrievals in the manuscript were produced with this code on
  2026-08-13 with the flags above. The sea-surface flattening option
  (`--enable_sea_surface_flattening`, Appendix C of the manuscript) was added
  afterwards and is off by default.
- **Quality-flag fix, 2026-10-15.** Earlier versions keyed the quality bitmask
  by along-track bin alone, so on multi-beam runs every beam carried the last
  beam's (or the pooled) flags. K<sub>dph</sub> never read the bitmask and is
  unaffected. The flags are now computed per beam (regression test
  `tests/test_quality_flags_per_beam.py`). `tools/recompute_quality_flags.py`
  rebuilds the column of existing output CSVs without re-running the retrieval;
  its docstring lists which bits can be rebuilt exactly.
- The figure gate on `surface_fit_quality` used in earlier drafts was removed
  from all figure scripts on 2026-10-12 and is not part of this package.

## Layout

```
main.py                 pipeline driver
kd_cli_config.py        all command-line options and defaults
kd_utils/               surface detection, masks, refraction, Beer's-law fitting,
                        bootstrap, coherent selection, quality flags, SlideRule adapter
kd_utils/SolarBckgrd/   ATL09 background-rate pairing and solar-background analysis
tests/                  pytest suite (fit snapshots, bootstrap RNG, BIC equivalence,
                        refraction re-binning, flattening, per-beam flags)
tools/                  post-hoc utilities
examples/               SLURM array script used for the four paper sites
```

## License

MIT, see `LICENSE`.
