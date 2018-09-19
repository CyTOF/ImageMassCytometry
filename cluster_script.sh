#!/bin/bash
#SBATCH --mem 80000 # memory pool for all cores
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
python clustering.py -s ./ims_2018_09_13.py --tissue_id Tonsil_D1 --ward --nb_clusters 40 
