import os
import argparse
import pdb 

# seaborn for plotting
import seaborn as sns

# numpy
import numpy as np 

# skimage imports
from skimage.measure import label, regionprops

# project imports
from settings import Settings, overwrite_settings
from sequence_importer import SequenceImporter
from fissure_detection import FissureDetection
from cell_detection import SpotDetector, CellDetection
from visualization import Overlays

class ClusterAnalysis(object):
    
    def __init__(self, settings_filename=None, settings=None, 
                 tissue_id=None):
        if settings is None and settings_filename is None:
            raise ValueError("Either a settings object or a settings filename has to be given.")
        if not settings is None:
            self.settings = settings
        elif not settings_filename is None:
            self.settings = Settings(os.path.abspath(settings_filename), dctGlobals=globals())

        if not tissue_id is None:
            print('Overwriting settings ... ')
            self.settings = overwrite_settings(self.settings, tissue_id)
            print('tissue id: %s' % tissue_id)
            print('output_folder: %s' % self.settings.output_folder)

        for folder in self.settings.makefolders:
            if not os.path.exists(folder):
                os.makedirs(folder)

        self.sp = SpotDetector(settings=self.settings)
        self.fissure = FissureDetection(settings=self.settings)
        self.cell_detector = CellDetection(settings=self.settings)
 
    def get_data(self, force=False):
        
        if not force:
            print('no')
            
        print('Reading data ... ')
        markers = ['Bcl-6', 'CD279(PD-1)', 'CD3',
                   'CD45', 'CD14',
                   'CD370', 'CD141',
                   'CD11c',  'CD1c-biotin-NA',  'HLA-DR',
                   'CD123', 'CD303(BDCA2)']
        si = SequenceImporter(markers)
        img, channel_names = si(self.settings.input_folder)

        # get the individual cells.
        ws = self.cell_detector.get_image(False)
        
        # number of samples
        N = ws.max()
        
        # number of features
        P = len(channel_names)

        X = np.zeros((N, P))

        for j in range(P):
            props = regionprops(ws, img[:,:,j])
            intensities = np.array([props[k]['mean_intensity'] for k in range(len(props))])
            # centers = np.array([props[k]['centroid'] for k in range(len(props))])
            X[:,j] = intensities
        
        return X 
    
    def pca(self):
        X = self.get_data()
        pdb.set_trace()
        return
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser( \
        description=('Run post filter on Ilastik results in order to'
                     'get a smoother output and to assign grey levels'
                     'according to what is required for the rest.'))

    parser.add_argument('-s', '--settings_file', dest='settings_file', required=True,
                        type=str,
                        help='settings file for the analysis. Often the settings file is in the same folder.')
    parser.add_argument('-t', '--tissue_id', dest='tissue_id', required=False,
                        type=str, default=None, 
                        help='Tissue id (optional). If not specificied, the tissue id from the settings file is taken.')

    parser.add_argument('--pca', dest='pca', required=False,
                        action='store_true',
                        help='Make PCA.')

    args = parser.parse_args()
    
    ca = ClusterAnalysis(args.settings_file, tissue_id=args.tissue_id)
    if args.pca:
        print(' *** Perform principal component analysis ***')
        ca.pca()
        
