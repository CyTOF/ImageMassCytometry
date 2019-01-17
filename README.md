# Imaging Mass Cytometry 

Tools to analyze imaging mass cytometry data.
These software tools have been developed in a collaboration between Thomas Walter and Elodie Segura. 

## Preparation of data for processing

We assume that we have data extracted with MCD viewer as OME tiff. 
The first script will copy and rename these files such that they can be easily interpreted by 
the program. 

For this, we need to edit the file copy_data.py and launch it with 
`python copy_data.py`

## Preparation of data for Ilastik

To prepare data for ilastik, launch `process_ilastik.py`. For postfiltering
nothing is to be done (this is done on the fly during reading):
`python process_ilastik.py --settings_file ims_2019_01_16.py --tissue_id Tonsil_D1 --prefilter`

Then make the classification with ilastik to find the B- and T-regions.

Finally, we should start the post-processing:
`python process_ilastik.py --settings_file ./ims_2019_01_16.py --tissue_id Tonsil_D2 --post`

## Fissure detection

Fissures can only be detected from the channels `avanti` and `Rutenium Red`. 
If these channels are not in the data set, fissure detection will not work properly. 
In this case, it will copy an empty image to the corresponding folder.

`python fissure_detection.py --settings_file ./ims_2019_01_16.py --tissue_id Tonsil_D2 --segmentation`

This is deprecated for Tonsil_D1 and Tonsil_D2 (not the same channels).
Finally, the segmentation is best done manually or with thresholds on suitable channels. 
The final image has to be stored in <base_folder>/results/<dataset>/fissure

## Normalization (for visualization)

Here we normalize all images individually in order to be able to browse them and visualize them in any
viewer without adjusting the contrast etc.

`python visualization.py --settings_file ./ims_2019_01_16.py --tissue_id Tonsil_D2 --normalize`

## Cell Segmentation

Cell segmentation (blob detection and watershed). This takes a few seconds.

`python cell_detection.py --settings_file ./ims_2019_01_16.py --tissue_id Tonsil_D2 --make_segmentation`


## Clustering

`python clustering.py -s ./ims_2018_08_23.py --tissue_id Tonsil_D2 --load_clustering --nb_clusters 60  --distance euclidean --method ward --cluster_maps --cluster_galleries --cluster_fusion`

## Distance analysis

`python distance_analysis.py -s ./ims_2018_08_23.py --tissue_id Tonsil_D2`