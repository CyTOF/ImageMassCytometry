#!/bin/bash
#SBATCH --mem 80000 # memory pool for all cores
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR

export CLUSTER_ENV=True

python clustering.py -s ./ims_2018_09_13.py --tissue_id Tonsil_120628_D1-2A_4MAY2017_1 --hierarchical_clustering --nb_clusters 60  --distance euclidean --method ward
python clustering.py -s ./ims_2018_09_13.py --tissue_id Tonsil_120828_D18-2A_5MAY2017_1 --hierarchical_clustering --nb_clusters 60  --distance euclidean --method ward

#python clustering.py -s ./ims_2018_09_13.py --tissue_id Tonsil_D2 --hierarchical_clustering --nb_clusters 60  --distance euclidean --method ward
#python clustering.py -s ./ims_2018_09_13.py --tissue_id Tonsil_D1 --hierarchical_clustering --nb_clusters 60 --cluster_galleries --cluster_maps --distance euclidean --method ward

#python clustering.py -s ./ims_2018_09_13.py --tissue_id Tonsil_D2 --hierarchical_clustering --nb_clusters 60  --distance correlation --method average
#python clustering.py -s ./ims_2018_09_13.py --tissue_id Tonsil_D1 --hierarchical_clustering --nb_clusters 60 --cluster_galleries --cluster_maps --distance correlation --method average

#python clustering.py -s ./ims_2018_09_13.py --tissue_id Tonsil_D2 --hierarchical_clustering --nb_clusters 60  --distance cosine --method average
#python clustering.py -s ./ims_2018_09_13.py --tissue_id Tonsil_D1 --hierarchical_clustering --nb_clusters 60 --cluster_galleries --cluster_maps --distance cosine --method average

