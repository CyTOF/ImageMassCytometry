import numpy as np
import os
import shutil
import argparse
import pdb 

from settings import Settings, overwrite_settings
from visualization import Overlays

from sequence_importer import SequenceImporter

import skimage.io

from skimage.filters import median 
from skimage.morphology import opening, closing, reconstruction, disk
from skimage.transform import rescale
from skimage.morphology import remove_small_holes


class Ilastik(object):
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
    
    def adjust(self, img, nb_rows, nb_cols, fill_value=0):
        imout = np.zeros((nb_rows, nb_cols), dtype=img.dtype) + fill_value
        
        if img.shape[0] > nb_rows:
            img = img[:nb_rows,:]
        if img.shape[1] > nb_cols:
            img = img[:,:nb_cols]
        delta_rows = (img.shape[0] - nb_rows) // 2
        delta_cols = (img.shape[1] - nb_cols) // 2
        imout[delta_rows:,delta_cols:] = img
        return imout
    
    def read_region_images(self):
        filename = self.settings.ilastik_filename
            
        im_small = skimage.io.imread(filename)
        img = rescale(im_small, 4, preserve_range=True)

        # get the right size
        si = SequenceImporter(['CD3'])
        original_img, channel_names = si(self.settings.input_folder)
        nb_rows = original_img.shape[0]
        nb_cols = original_img.shape[1]
        img = self.adjust(img, nb_rows, nb_cols, fill_value=255)
        
        background = np.zeros(img.shape, dtype=np.uint8)
        background[img>250] = 255
        background = remove_small_holes(background.astype(np.bool), 
                                        400, connectivity=2)

        bcell = np.zeros(img.shape, dtype=np.uint8)
        bcell[img>124] = 255
        bcell[background>0] = 0
        bcell = remove_small_holes(bcell.astype(np.bool), 
                                   400, connectivity=2)
        
        tcell = np.zeros(img.shape, dtype=np.uint8)
        tcell[img<5] = 255
        tcell = remove_small_holes(tcell.astype(np.bool), 
                                   400, connectivity=2)

        
        return background, bcell, tcell
    
    # DEPRECATED
    # postprocessing is done during reading (see function above).
    def _post_filter(self):
        raw_filename = os.path.join(self.settings.ilastik_folder,
                                    'raw_ilastik_segmentations',
                                    'rgb_%s_Simple Segmentation.png' % self.settings.dataset)
        if not os.path.isfile(raw_filename):
            copy_filename = os.path.join(self.settings.ilastik_folder,
                                         'rgb_%s_Simple Segmentation.png' % self.settings.dataset)
            if not os.path.isfile(copy_filename):
                raise ValueError("please run Ilastik first.")
            shutil.copyfile(copy_filename, raw_filename)

        img = skimage.io.imread(raw_filename)

        values = np.unique(img)
        target_values = [0, 128, 255]

        # reattribution of values
        for k, value in enumerate(values.tolist()):
            img[img==value] = target_values[min(k, len(target_values) - 1)]

        im_pref1 = opening(img, disk(2)).astype(np.uint8)
        im_pref = median(im_pref1, disk(4))
        im_open = reconstruction(opening(im_pref, disk(5)).astype(np.uint8),
                                 im_pref.astype(np.uint8), method='dilation').astype(np.uint8)
        #im_close = closing(im_open, disk(3))
        im_close = reconstruction(closing(im_open, disk(6)).astype(np.uint8), 
                                  im_open.astype(np.uint8), method='erosion').astype(np.uint8)
        
        new_filename = os.path.join(self.settings.ilastik_folder, 
                                    'rgb_%s_Simple Segmentation.png' % self.settings.dataset)
        skimage.io.imsave(new_filename, im_close)
        return

    def post_processing(self):
        filename = self.settings.ilastik_filename
        img = skimage.io.imread(filename)
        img_normalized = 255.0 * (img - img.min()) / (img.max() - img.min())
        skimage.io.imsave(filename, img_normalized.astype(np.uint8))
        return
    
    def prepare(self):
        rgb_folder = self.settings.ilastik_input_rgb_folder
        prep_folder = self.settings.ilastik_input_folder
        for folder in [rgb_folder, prep_folder]: 
            if not os.path.isdir(folder): 
                os.makedirs(folder)

        si = SequenceImporter(['CD3', 'CD19', 'E-Cadherin'])
        img, channel_names = si(self.settings.input_folder)
        img_downscale = rescale(img, .25)
        rgb_image = np.zeros((img_downscale.shape[0], img_downscale.shape[1], 3), dtype=np.float64)
        for i in range(img.shape[2]):
            perc = np.percentile(img_downscale[:,:,i], [10, 99.9])
            minval = perc[0]; maxval = perc[1]
            normalized = (img_downscale[:,:,i] - minval) / (maxval - minval)
            normalized[normalized > 1.0] = 1.0
            normalized[normalized < 0.0] = 0.0
            rgb_image[:,:,i] = 255 * normalized
        skimage.io.imsave(self.settings.ilastik_input_rgb_filename, rgb_image.astype(np.uint8))
        return

    def save_overlay(self):
        ov = Overlays()
        segmentation = skimage.io.imread(self.settings.ilastik_filename)
        segmentation[segmentation<10] = 1
        segmentation[(segmentation>120) * (segmentation<135)] = 2
        segmentation[segmentation>245] = 0
        rgb_image = skimage.io.imread(self.settings.downsample_image)
        colors = {
            1: np.array([255, 0, 0]), # B-cells
            2: np.array([0, 255, 0])  # T-cells
            }
        overlay_image = ov.overlay_rgb_img(rgb_image, segmentation, colors, True)
        filename = os.path.join(self.settings.output_folder, 
                                'segmentation_overlay_ilastik_%s.png' % self.settings.dataset)
        skimage.io.imsave(filename,
                          overlay_image)
        print('Saved: %s' % filename)
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

    parser.add_argument('--prepare', dest='prepare', required=False,
                        action='store_true',
                        help='The script will prepare an RGB file for Ilastik processing.')
    parser.add_argument('--post', dest='post', required=False,
                        action='store_true',
                        help='The script will adapt the output to the required format (no filter).')
    parser.add_argument('--save_overlay', dest='save_overlay', required=False,
                        action='store_true',
                        help='To save the overlay of B-region and T-region on top of the image.')

    args = parser.parse_args()
    il = Ilastik(args.settings_file, tissue_id=args.tissue_id)
        
    if args.prepare:
        print(' *** Preparation for Ilastik ***')
        il.prepare()
        
    if args.post:
        print(' *** Postprocessing ***')
        il.post_processing()

    if args.save_overlay:
        print(' *** Saving overlay to ***')
        il.save_overlay()
        

    