#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --job-name=extraction
#SBATCH --partition=short
#SBATCH -M carrington
#SBATCH --mem-per-cpu=128G

source /../proj/ivanzait/analysator/.venv/bin/activate

module load SciPy-bundle
module load matplotlib/3.8.2-gfbf-2023b

export PYTHONPATH=$PYTHONPATH:$HOME/analysator

#python3 flux_function.py -o x_points
#python3 extract_data.py -o frames
python prepare_states.py -o series
