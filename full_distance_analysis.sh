# full processing from cluster results to maps + distance analysis
#python clustering.py -s ./ims_2018_08_23.py --tissue_id Tonsil_D1 --load_clustering --nb_clusters 60  --distance euclidean --method ward --cluster_maps --cluster_galleries --cluster_fusion
#python clustering.py -s ./ims_2018_08_23.py --tissue_id Tonsil_D2 --load_clustering --nb_clusters 60  --distance euclidean --method ward --cluster_maps --cluster_galleries --cluster_fusion
#python clustering.py -s ./ims_2018_08_23.py --tissue_id Tonsil_120828_D18-2A_5MAY2017_1 --load_clustering --nb_clusters 60  --distance euclidean --method ward --cluster_maps --cluster_fusion --cluster_galleries

#python clustering.py -s ./ims_2018_08_23.py --tissue_id Tonsil_D1 --load_clustering --nb_clusters 60  --distance euclidean --method ward --cluster_maps 
#python clustering.py -s ./ims_2018_08_23.py --tissue_id Tonsil_D2 --load_clustering --nb_clusters 60  --distance euclidean --method ward --cluster_maps 
#python clustering.py -s ./ims_2018_08_23.py --tissue_id Tonsil_120828_D18-2A_5MAY2017_1 --load_clustering --nb_clusters 60  --distance euclidean --method ward --cluster_maps 

#zip -r Tonsil_D1_distance_analysis.zip /Users/twalter/data/Elodie/results/Tonsil_D1/plots/distance_histograms 
#zip -r Tonsil_D2_distance_analysis.zip /Users/twalter/data/Elodie/results/Tonsil_D2/plots/distance_histograms 
#zip -r Tonsil_120828_D18-2A_5MAY2017_1_distance_analysis.zip /Users/twalter/data/Elodie/results/Tonsil_120828_D18-2A_5MAY2017_1/plots/distance_histograms 

cd /Users/twalter/data/Elodie/results/Tonsil_D1/clustering
zip -r Tonsil_D1_clustermaps.zip clusterfusion/ clustergalleries/ clustermaps/
mv Tonsil_D1_clustermaps.zip /Users/twalter/Dropbox/Public/

cd /Users/twalter/data/Elodie/results/Tonsil_D2/clustering
zip -r Tonsil_D2_clustermaps.zip clusterfusion/ clustergalleries/ clustermaps/
mv Tonsil_D2_clustermaps.zip /Users/twalter/Dropbox/Public/

cd /Users/twalter/data/Elodie/results/Tonsil_120828_D18-2A_5MAY2017_1/clustering
zip -r Tonsil_120828_D18-2A_5MAY2017_1_clustermaps.zip clusterfusion/ clustergalleries/ clustermaps/
mv Tonsil_120828_D18-2A_5MAY2017_1_clustermaps.zip /Users/twalter/Dropbox/Public/

cd /Users/twalter/data/Elodie/results/Tonsil_D1/plots 
zip -r Tonsil_D1_distance_analysis.zip distance_histograms/
mv Tonsil_D1_distance_analysis.zip /Users/twalter/Dropbox/Public/

cd /Users/twalter/data/Elodie/results/Tonsil_D2/plots 
zip -r Tonsil_D2_distance_analysis.zip distance_histograms/
mv Tonsil_D2_distance_analysis.zip /Users/twalter/Dropbox/Public/

cd /Users/twalter/data/Elodie/results/Tonsil_120828_D18-2A_5MAY2017_1/plots 
zip -r Tonsil_120828_D18-2A_5MAY2017_1_distance_analysis.zip distance_histograms/
mv Tonsil_120828_D18-2A_5MAY2017_1_distance_analysis.zip /Users/twalter/Dropbox/Public/

#zip -r Tonsil_D2_clustermaps.zip /Users/twalter/data/Elodie/results/Tonsil_D2/clustering/clusterfusion /Users/twalter/data/Elodie/results/Tonsil_D2/clustering/clustergalleries /Users/twalter/data/Elodie/results/Tonsil_D2/clustering/clustermaps
#zip -r Tonsil_120828_D18-2A_5MAY2017_1_clustermaps.zip /Users/twalter/data/Elodie/results/Tonsil_120828_D18-2A_5MAY2017_1/clustering/clusterfusion /Users/twalter/data/Elodie/results/Tonsil_120828_D18-2A_5MAY2017_1/clustering/clustergalleries /Users/twalter/data/Elodie/results/Tonsil_120828_D18-2A_5MAY2017_1/clustering/clustermaps

#mv *.zip /Users/twalter/Dropbox/Public
#python distance_analysis.py -s ./ims_2018_08_23.py --tissue_id Tonsil_D1
#python distance_analysis.py -s ./ims_2018_08_23.py --tissue_id Tonsil_D2
#python distance_analysis.py -s ./ims_2018_08_23.py --tissue_id Tonsil_120828_D18-2A_5MAY2017_1
