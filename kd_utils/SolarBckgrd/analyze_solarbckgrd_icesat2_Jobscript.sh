#!/bin/bash

#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem 32g
#SBATCH -t 02:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=wayne128@email.unc.edu

module load anaconda
source activate Diffusion
cd /work/users/w/a/wayne128/ICESat2/IS2_kd_py/SolarBckgrd/

# python analyze_solarbckgrd_icesat2.py
# python analyze_nighttime_BG_rate_icesat2.py
python IS2_BG_Rate_Land_Ocean_plot.py