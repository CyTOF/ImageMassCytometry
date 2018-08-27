# ImageProteomics

Tools to analyze image mass spectrometry data.
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
`python process_ilastik.py --tissue_id Tonsil_D1 --prefilter`

Then make the classification with ilastik to find the B- and T-regions.

Finally, we should start the post-processing:
`python process_ilastik.py --tissue_id Tonsil_D1 --post`

## Fissure detection

Fissures can only be detected from the channels `avanti` and `Rutenium Red`. 
If these channels are not in the data set, fissure detection will not work properly. 
In this case, it will copy an empty image to the corresponding folder.

## make the distance analysis for the sub-populations

