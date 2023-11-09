#!/bin/bash

#SBATCH --job-name=data2
#SBATCH -n 24
#SBATCH --time=4:00:00
#SBACTH --mem-per-cpu=16000
#SBATCH --output=outputs/tuning2.txt
#SBATCH --error=errors/tuning2.txt
#SBATCH --mail-type=ALL

python tuning.py
