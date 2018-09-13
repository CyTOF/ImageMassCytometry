#!/bin/bash#
#SBATCH --mem 80000 # memory pool for all cores
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
python cl_script_on_cluster.py
