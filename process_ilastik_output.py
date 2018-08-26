import numpy as np
import os
import shutil
import skimage.io

from settings import Settings

from skimage.filters import median 
from skimage.morphology import opening, closing, reconstruction, disk
from skimage.transform import rescale

class Ilastik(object):
    def __init__(self, settings_filename=None, settings=None):
        if settings is None and settings_filename is None:
            raise ValueError("Either a settings object or a settings filename has to be given.")
        if not settings is None:
            self.settings = settings
        elif not settings_filename is None:
            self.settings = Settings(os.path.abspath(settings_filename), dctGlobals=globals())

        for folder in self.settings.makefolders:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def post_filter(self):
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
            perc = np.percentile(img_downscale[:,:,i], [5, 99.9])
            minval = perc[0]; maxval = perc[1]
            normalized = (img_downscale[:,:,i] - minval) / (maxval - minval)
            normalized[normalized > 1.0] = 1.0
            normalized[normalized < 0.0] = 0.0
            rgb_image[:,:,i] = 255 * normalized
        skimage.io.imsave(self.settings.ilastik_input_rgb_filename, rgb_image.astype(np.uint8))
        return
