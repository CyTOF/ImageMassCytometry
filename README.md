# Imaging Mass Cytometry 

Tools to analyze imaging mass cytometry data. These software tools have been developed in a collaboration between [Thomas Walter](http://members.cbio.mines-paristech.fr/~twalter/index.html) (Centre for Computational Biology, Mines ParisTech) in collaboration with [Elodie Segura](https://science.curie.fr/members/elodie-segura/) (Institut Curie). This is a first very preliminary version; this README shows how to produce the results published in [1]. 

## Imaging Technique

The images in [1] have been produced by [Fluidigm Canada Inc](https://www.fluidigm.com). Each channel shows the spatial distribution of one protein in the same section of a human tonsil. In these data, we have typically tens of channels. In order to import the data into our software, we first need to export individual images with the MCD viewer provided by Fluidigm as OME tiff (each channel is one image). 

## Installation

Currently, our software is a set of python scripts. 

Here, we give a series of installation commands to make this work in a conda virtual environment. The dependencies are listed [here](requirements.txt). In this file, we have written the absolute requirements obtained by `pip freeze > requirements.txt`. It is likely that the scripts will also work with more recent versions of the packages. 

```conda create -n improt python=3.6 anaconda
source activate improt
pip install -r requirements.txt```

## Preparation of data for processing

We assume that we have a stack of tiff files, each channel correponding to one protein. 
For instance, the data might have been exported via the MCD viewer as OME tiff.

We suppose that the data is named in the following way: 
`<sample_id>_<metal>_<protein>.tif`

The `sample_id` is the identifier (name) of the tissue section. 
`metal` is the metal that has been used in acquisition and `protein` is the labeled protein. 

We suppose that these data live in :
`<project_data_folder>/<sample_id>/in_data`

In the following example we assume that 
`sample_id = Tonsil_D2`

Importantly, all the settings are defined in a settings file. 
An example is shown in `ims_2019_04_29.py`
For a new project, we recommend editing this settings file to define paths, the sample_id, etc. 

## Preparation of data for Ilastik

To prepare data for ilastik, launch `process_ilastik.py` : 
`python process_ilastik.py --settings_file ims_2019_04_29.py --tissue_id Tonsil_D2 --prepare`

This will generate a downsampled RGB image that can be read by [Ilastik](https://www.ilastik.org) for region annotation and classification. Here, we use Ilastik to segment the B-region, T-region, the crypt and the background. Ilastik outputs an image with these four labeled regions. You will need to copy and rename this image; it should have the following form: 

`<project_data_folder>/<sample_id>/Ilastik/result/rgb_<sample_id>_Simple Segmentation.png`

No postprocessing is required (it is done on the fly when calling the reader of the Ilastik-class). 

## Fissure detection

Fissures can be detected easily from channels `avanti` and/or `Rutenium Red`. For this, we have developed a small script in `fissure_detection.py` which can be launched in the following way: 

`python fissure_detection.py --settings_file ./ims_2019_04_29.py --tissue_id Tonsil_D2 --segmentation`

The problem was that for different data sets, we had different channels (and sometimes `avanti` and `Rutenium Red` were not present. In this case, the script just copies an empty image to the fissure segmentation folder. 

We therefore recommend to segment the fissure from a suitable channel with another software (such as FIJI), which is usually farely easy. The resulting binary image should have the following name : 

`<project_data_folder>/<sample_id>/results/fissure/fissure_segmentation.png`

## Normalization (for visualization; optional)

In order to better visualize the images, they can be normalized with the following command: 

`python visualization.py --settings_file ./ims_2019_04_29.py --tissue_id Tonsil_D2 --normalize`

The script generates the folder: 

`<project_data_folder>/<sample_id>/results/normalized_images`

## Cell Segmentation

Cell segmentation (blob detection and watershed). This takes a few seconds.

`python cell_detection.py --settings_file ./ims_2019_04_29.py --tissue_id Tonsil_D2 --make_segmentation`

This will generate the following files:
`<project_data_folder>/<sample_id>/results/cell_segmentation/dna_cell_segmentation_random_colors.png`
`<project_data_folder>/<sample_id>/results/cell_segmentation/dna_cell_segmentation.tiff`

The most relevant file is the tiff-file, as it is a labeled image with all cells detected. The other file is for visual inspection. 

Optionally, we can also run the segmentation in debug mode which is basically working on a reasonably sized crop. 

`python cell_detection.py --settings_file ./ims_2019_04_29.py --tissue_id Tonsil_D2 --test_segmentation`

In this case, the results are stored in:
`<project_data_folder>/<sample_id>/debug`

## Clustering

Test whether everything works fine (important: use downsample to reduce the number of cells.)

`python clustering.py -s ./ims_2019_04_29.py --tissue_id Tonsil_D2 --hierarchical_clustering --nb_clusters 60  --distance euclidean --method ward --downsample 1000`

This script will generate a pickle file and a heatmap:
`<project_data_folder>/<sample_id>/results/clustering/cluster_assignment_normalization_percentile_metric_<distance>_method_<clustering method>.pickle`
`<project_data_folder>/<sample_id>/results/clustering/clustering_normalization_percentile_metric_<distance>_method_<clustering method>.pickle`

If you rerun the clustering algorithm (for instance without downsampling), this file will be overwritten. 

If this runs fine, you can make the clustering for the full data set. For this, you need a computer with sufficient RAM. The command is then:

`python clustering.py -s ./ims_2019_04_29.py --tissue_id Tonsil_D2 --hierarchical_clustering --nb_clusters 60  --distance euclidean --method ward`

In our case, this was done on the compute cluster. Be aware that if the machine has no display associated to it, there can be a problem with matplotlib. For this, we have added a script that also sets the environmental variable: `cluster_script.sh`. This script can be submitted to the cluster. With `slurm` this can look like: 

`sbatch cluster_script.sh`
`squeue -u <your username>`

Then, you can recover the results, and you can generate the cluster maps and cluster galleries.

`python clustering.py -s ./ims_2019_04_29.py --tissue_id Tonsil_D2 --load_clustering --nb_clusters 60  --distance euclidean --method ward --cluster_maps --cluster_galleries`

In our workflow, we used the clustermaps, clustergalleries and the heatmap generated for the full data set in order to decide which clusters were to be joined (according to our biological priors on the cell types). These fusions of clusters were then defined in the settings file `ims_2019_04_29.py`. The structure to hold these fusion data is called `cluster_fusion` and is a dictionary: `{<cluster name>: <list of cluster labels in the heatmap>}`. In order to produce the final result, we need to rerun the method with the `cluster_fusion` argument: 

`python clustering.py -s ./ims_2019_04_29.py --tissue_id Tonsil_D2 --load_clustering --nb_clusters 60  --distance euclidean --method ward --cluster_maps --cluster_galleries --cluster_fusion`	

This will output the final maps and galleries (the heatmap remains unchanged). 

## Distance analysis

Finally, you will want to analyse the distance distributions. This can be easily done: 

`python distance_analysis.py -s ./ims_2019_04_29.py --tissue_id Tonsil_D2`

## References

1. Durand et al. Human lymphoid organ cDC2 and macrophages play complementary roles in the induction of T follicular helper responses. Journal of Experimental Medicine (in press).
