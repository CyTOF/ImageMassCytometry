# Imaging Mass Cytometry 

Tools to analyze imaging mass cytometry data. These software tools have been developed in a collaboration between [Thomas Walter](http://members.cbio.mines-paristech.fr/~twalter/index.html) (Centre for Computational Biology, Mines ParisTech) and Elodie Segura (Institut Curie). This is a first very preliminary version; this README shows how to produce the results in [1]. 

## Imaging Technique

The images have been produced by [Fluidigm Canada Inc](https://www.fluidigm.com). Each channel corresponds to one protein (like in immunofluorescence), but we have typically tens of channels. In order to import the data into our software, we first need to export it with the MCD viewer provided by Fluidigm as OME tiff (each channel is one image). 

## Preparation of data for processing

We assume that we have a stack of tiff files, each channel correponding to one protein. 
For instance, the data might have been exported via the MCD viewer as OME tiff.

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

Test whether everything works fine (important: use downsample to reduce the number of cells.)

`python clustering.py -s ./ims_2019_01_16.py --tissue_id Tonsil_D2 --nb_clusters 60  --distance euclidean --method ward --cluster_maps --cluster_galleries --cluster_fusion --downsample 1000`

Then, move to the cluster and write a cluster script (see for instance cluster_script_revision.sh)
It is important that the paths are set correctly in the settings file.
Run the script on the cluster:

`sbatch cluster_script_revision.sh`
`squeue -u twalter`

Then, you can recover the results, and you can generate the cluster maps.

`python clustering.py -s ./ims_2019_01_27_revision.py --tissue_id Tonsil_D2 --load_clustering --nb_clusters 60  --distance euclidean --method ward --cluster_maps --cluster_galleries --cluster_fusion`


## Distance analysis

`python distance_analysis.py -s ./ims_2018_08_23.py --tissue_id Tonsil_D2`

## References

1. Durand et al. Human lymphoid organ cDC2 and macrophages play complementary roles in the induction of T follicular helper responses. Journal of Experimental Medicine (in press).
