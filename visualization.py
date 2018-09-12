import os
import argparse

import numpy as np

from skimage.color import grey2rgb
import skimage.io
from skimage.morphology import erosion, disk

from settings import Settings, overwrite_settings
from sequence_importer import SequenceImporter

class QuickView(object):
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

    def normalize_images(self):
        si = SequenceImporter()
        img, channel_names = si(self.settings.input_folder)
        for i in range(len(channel_names)):
            image = img[:,:,i]
            perc = np.percentile(image, [1, 99])
            image_norm = 255.0 * (image - perc[0]) / (perc[1] - perc[0])
            image_norm[image_norm>255.0] = 255.0
            image_norm[image_norm<0] = 0.0
            out_folder = os.path.join(self.settings.output_folder, 'normalized_images')
            if not os.path.isdir(out_folder):
                print('made %s' % out_folder)
                os.makedirs(out_folder)
            
            filename = os.path.join(out_folder, 'normalized_%s.png' % channel_names[i])
            print('writing %s' % filename)
            skimage.io.imsave(filename, image_norm.astype(np.uint8))
        return
    
class Overlays(object):
    def overlay_grey_img(self, img, segmentation, colors, contour=True):
        overlay_img = grey2rgb(img)
        values = sorted(np.unique(segmentation))
        
        if contour:
            display_img = np.zeros(segmentation.shape, dtype=np.uint8)
            
            for value in values:
                temp = np.zeros(segmentation.shape, dtype=np.uint8)
                temp[segmentation==value] = value
                temp = temp - erosion(temp, disk(1))
                display_img = temp + display_img
        else:
            display_img = segmentation

        for value in values:
            if not value in colors:
                continue
            if value==0:
                print('attention: 0 is not an appropriate label, as it is reserved for background.')
            overlay_img[display_img==value] = colors[value]

        return overlay_img 
    
    def overlay_rgb_img(self, rgb_img, segmentation, colors, contour=True):
        overlay_img = rgb_img.copy()
        values = sorted(np.unique(segmentation))
        
        if contour:
            display_img = np.zeros(segmentation.shape, dtype=np.uint8)
            
            for value in values:
                temp = np.zeros(segmentation.shape, dtype=np.uint8)
                temp[segmentation==value] = value
                temp = temp - erosion(temp, disk(1))
                display_img = temp + display_img
        else:
            display_img = segmentation

        for value in values:
            if not value in colors:
                continue
            if value==0:
                print('attention: 0 is not an appropriate label, as it is reserved for background.')
            overlay_img[display_img==value] = colors[value]

        return overlay_img 

if __name__ == '__main__':

    parser = argparse.ArgumentParser( \
        description=('different visualization tools'))

    parser.add_argument('-s', '--settings_file', dest='settings_file', required=True,
                        type=str,
                        help='settings file for the analysis. Often the settings file is in the same folder.')
    parser.add_argument('-t', '--tissue_id', dest='tissue_id', required=False,
                        type=str, default=None, 
                        help='Tissue id (optional). If not specificied, the tissue id from the settings file is taken.')

    parser.add_argument('--normalize', dest='normalize', required=False,
                        action='store_true',
                        help='Save normalized images for a quick inspection.')

    args = parser.parse_args()

    qv = QuickView(args.settings_file, tissue_id=args.tissue_id)    
    if args.normalize:
        print(' *** Save normalized images ***')
        qv.normalize_images()

