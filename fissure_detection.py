import numpy as np
import os
import shutil
import argparse
import pdb 
import matplotlib.pyplot as plt

from settings import Settings, overwrite_settings
from visualization import Overlays

from sequence_importer import SequenceImporter

import skimage.io

from skimage.filters import gaussian, median
from skimage.transform import rescale
from skimage.filters import threshold_isodata
from skimage.morphology import closing, disk, square, dilation
from skimage.morphology import remove_small_objects

class FissureDetection(object):
    def __init__(self, settings_filename=None, settings=None):
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

        self.image_result_folder = os.path.join(self.settings.output_folder, 
                                                'fissure')
        if not os.path.isdir(self.image_result_folder):
            os.makedirs(self.image_result_folder)
        

    def plot_histogram(self, image, filename):
        out_folder = os.path.join(self.settings.output_folder, 'plots', 'fissure_histogram')
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        #image_downscale = rescale(image, .25)
        #vec = image_downscale.ravel()
        perc = np.percentile(image, [90])
        medval = perc[0]
        vec = image[image > medval]

        full_filename = os.path.join(out_folder, filename)
        fig = plt.figure()
        
        plt.hist(vec, bins=100, normed=True)
        plt.title('Histogram for Fissure detection (upper 20%)')
        plt.xlabel('Grey levels')
        plt.ylabel('Frequency')
        plt.savefig(full_filename)
                             
        return
    
    def write_image(self, image):
        filename = os.path.join(self.image_result_folder, 
                                'fissure_segmentation.png')
        skimage.io.imsave(filename, image)
        return

    def read_image(self):
        filename = os.path.join(self.image_result_folder, 
                                'fissure_segmentation.png')
        image = skimage.io.imread(filename)
        return image

    def get_image(self, process_image=False):
        if not process_image:
            try:
                image = self.read_image()
            except:
                image = self.__call__()
                self.write_image(image)
        else:
            image = self.__call__()
            self.write_image(image)

        return image

    def __call__(self, downscale=False):

        markers = ['avanti', 'Rutenium Red']
        si = SequenceImporter(markers)
        img, channel_names = si(self.settings.input_folder)

        image = np.mean(img, axis=2)
        if downscale:
            image_downscale = rescale(image, .25)
        else:
            image_downscale = image
    
        perc = np.percentile(image_downscale, [80, 99])
        low_val = perc[0]
        high_val = perc[1]
        
        image = 255 * (image_downscale - low_val) / (high_val - low_val)
        image[image>255] = 255
        image[image<0] = 0
        image = image.astype(np.uint8)

        pref1 = median(image, disk(3))
        pref2 = gaussian(pref1, .5)

        thresh = threshold_isodata(pref2)
        binary = pref2 > thresh
        binary_closed = closing(binary, square(8))
        clean_binary = remove_small_objects(binary_closed, 30)
        temp = dilation(clean_binary, disk(2))
        output_downscale = closing(temp, square(15))
        if downscale:
            output = rescale(output_downscale, 4)
        else:
            output = output_downscale

        return output

    def test_routine(self):
        markers = ['avanti', 'Rutenium Red']
        si = SequenceImporter(markers)
        img, channel_names = si(self.settings.input_folder)

        image = np.mean(img, axis=2)
        image_downscale = rescale(image, .25)

        perc = np.percentile(image_downscale, [80, 99])
        low_val = perc[0]
        high_val = perc[1]
        
        image = 255 * (image_downscale - low_val) / (high_val - low_val)
        image[image>255] = 255
        image[image<0] = 0
        image = image.astype(np.uint8)

        pref1 = median(image, disk(3))
        pref2 = gaussian(pref1, .5)

        thresh = threshold_isodata(pref2)
        binary = pref2 > thresh
        binary_closed = closing(binary, square(8))
        clean_binary = remove_small_objects(binary_closed, 30)
        temp = dilation(clean_binary, disk(2))
        output = closing(temp, square(15))
        
        if self.settings.debug:
            out_folder = os.path.join(self.settings.debug_folder, 'fissure')
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder)

            filename = os.path.join(out_folder, 'debug_fissure_median.png')
            skimage.io.imsave(filename, pref1)
            
            filename = os.path.join(out_folder, 'debug_fissure_gauss.png')
            skimage.io.imsave(filename, pref2)

            filename = os.path.join(out_folder, 'debug_fissure_binary_closed.png')
            skimage.io.imsave(filename, 255 * binary_closed)

            filename = os.path.join(out_folder, 'debug_fissure_normalized.png')
            skimage.io.imsave(filename, image)

            filename = os.path.join(out_folder, 'debug_thresh_isodata.png')
            skimage.io.imsave(filename, 255 * binary)

            filename = os.path.join(out_folder, 'debug_thresh_clean.png')
            skimage.io.imsave(filename, 255 * clean_binary)

            filename = os.path.join(out_folder, 'debug_thresh_dilated.png')
            skimage.io.imsave(filename, 255 * temp)

            filename = os.path.join(out_folder, 'debug_thresh_output.png')
            skimage.io.imsave(filename, 255 * output)

        return output


if __name__ == '__main__':

    parser = argparse.ArgumentParser( \
        description=('Fissure detection: applies simple fissure detection to the chosen tissue.'
                     'If there is no avanti-channel, an empty image is copied.'))
    
    parser.add_argument('-s', '--settings_file', dest='settings_file', required=True,
                        type=str,
                        help='settings file for the analysis. Often the settings file is in the same folder.')
    parser.add_argument('-t', '--tissue_id', dest='tissue_id', required=False,
                        type=str, default=None, 
                        help='Tissue id (optional). If not specificied, the tissue id from the settings file is taken.')


    parser.add_argument('--histo', dest='histo', required=False,
                        type=bool, default=False,
                        help='Plots the histogram of the channels.')
    parser.add_argument('--segmentation', dest='segmentation', required=False,
                        type=bool, default=False,
                        help='Segmentation of the fissure.')
    parser.add_argument('--save_overlay', dest='save_overlay', required=False,
                        type=bool, default=False,
                        help='To save the overlay')

    args = parser.parse_args()
    il = FissureDetection(args.settings_file, tissue_id=args.tissue_id)
        
    if args.histo:
        print(' *** Fissure Detection Histogram Plotting ***')
        il.plot_histogram()
        
    if args.segmentation:
        print(' *** Postprocessing ***')
        il.post_processing()

    if args.save_overlay:
        print(' *** Saving overlay to ***')
        il.save_overlay()
