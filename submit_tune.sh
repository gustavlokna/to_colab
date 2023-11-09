#!/bin/bash

#SBATCH --job-name=data
#SBATCH -n 24
#SBATCH --time=4:00:00
#SBACTH --mem-per-cpu=16000
#SBATCH --output=outputs/tuning.txt
#SBATCH --error=errors/tuning.txt
#SBATCH --mail-type=ALL

python tuning.py