#!/bin/bash
#SBATCH -J kdph_sites
#SBATCH --array=1-4
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16g
#SBATCH -t 02:00:00
#SBATCH -p general
#SBATCH -o logs/kdph_%A_%a.out
#SBATCH -e logs/kdph_%A_%a.err

# SLURM array that produced the four site retrievals in the manuscript
# (one local ATL03 subset per task). Set the three paths below for your system.
#
# WORKSPACE/Dataset is expected to hold:
#   ATL03/<granule>.h5                 ATL03 granules (subset to the study box)
#   ATL09/                             ATL09 granules for the background rate (optional)
#   GEBCO tiles referenced by --gebco_path (default location under Dataset/)
# and WORKSPACE/Shorelines/ne_10m_land/ne_10m_land.shp (Natural Earth land polygons).

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
WORKSPACE=${WORKSPACE:-/path/to/workspace}
OUTROOT=${OUTROOT:-/path/to/output}
PY=${PY:-python}

case "${SLURM_ARRAY_TASK_ID}" in
  1) SITE=BohaiSea;  ATL03=ATL03_20190829161305_09560402_006_02_subsetted.h5 ;;
  2) SITE=CookInlet; ATL03=ATL03_20220507112145_06931503_006_01_subsetted.h5 ;;
  3) SITE=NC_Coast;  ATL03=ATL03_20240415083629_04232306_006_01_subsetted.h5 ;;
  4) SITE=WaxDelta;  ATL03=ATL03_20231103172707_06982106_006_01_subsetted.h5 ;;
  *) echo "unknown task id"; exit 1 ;;
esac

OUT="$OUTROOT/$SITE"
mkdir -p "$OUT" logs
echo "Host $(hostname)  $(date)  SITE=$SITE  OUT=$OUT"

cd "$REPO"
$PY -u main.py \
  --workspace_path "$WORKSPACE" \
  --atl03_path "$WORKSPACE/Dataset/ATL03" \
  --atl03_file "$ATL03" \
  --other_data_path Dataset/ \
  --shoreline_data Shorelines/ne_10m_land/ne_10m_land.shp \
  --output_path "$OUT" \
  --target_beams gt1l,gt2l,gt3l,gt1r,gt2r,gt3r \
  --atl09_root "$WORKSPACE/Dataset/ATL09" \
  --no_plot \
  --enable_gebco_filter --poisson_weighted --coherent_model_selection \
  --min_seabed_depth 3

echo "Finished $(date)  SITE=$SITE"
