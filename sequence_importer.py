import os
import skimage.io
import skimage.transform
import numpy as np

import pdb

class SequenceImporter(object):
    def __init__(self, marker_list=None):
        
        self.marker_list = marker_list
        return

    def __call__(self, folder):
        filenames = list(filter(lambda x: os.path.splitext(x)[-1] in ['.tif', '.tiff', '.png'],
                                os.listdir(folder)))
        if not self.marker_list is None:
            filtered_filenames = []
            for marker in self.marker_list:
                filtered_filenames.extend(list(filter(lambda x: x.split('.')[0].split('_')[-1] == marker,
                                                      filenames)))
            filenames = filtered_filenames

        nb_channels = len(filenames)
        if nb_channels==0:
            raise ValueError("No image found for markers: %s" % str(self.marker_list))
        temp = skimage.io.imread(os.path.join(folder, filenames[0]))
        width = temp.shape[0]
        height = temp.shape[1]
        img = np.zeros((width, height, nb_channels))
        for i in range(nb_channels):
            
            img[:,:,i] = skimage.io.imread(os.path.join(folder, filenames[i]))

        channel_names = dict(zip(range(nb_channels), [os.path.splitext(x)[0].split('_')[-1] for x in filenames]))
        return img, channel_names

class RGB_Generator(object):
    def __init__(self, downscale=4, perc_low=10, perc_high=99.9, channel_order=None):
        self.downscale = downscale
        self.perc_low = perc_low
        self.perc_high = perc_high
        self.channel_order = channel_order
        
    def get_rgb_image(self, in_folder):
        filenames = list(filter(lambda x: os.path.splitext(x)[-1].lower() in ['.tif', '.tiff', '.png'],
                                os.listdir(in_folder)))
        filenames.sort()
        if self.channel_order:
            channel_dict = dict(zip([os.path.splitext(x)[0].split('_')[-1] for x in filenames], filenames))
            filenames = [channel_dict[x] for x in self.channel_order]
            
        if len(filenames) > 3: 
            raise ValueError('more than 3 channels: unclear color assignment')
        rgb_img = None
        image_shape = (1, 1, 3)
        print(filenames)
        for i, filename in enumerate(filenames):
            print(i, filename)
            channel = skimage.io.imread(os.path.join(in_folder, filename))
            if channel.max() == channel.min():
                channel_ds = np.zeros(channel.shape) - 1
            else:
                channel_ds = skimage.transform.pyramid_reduce(2 * (channel - channel.min()) / (channel.max() - channel.min()) - 1,
                                                              downscale=self.downscale)
            if rgb_img is None:
                image_shape = (channel_ds.shape[0], channel_ds.shape[1], 3)
                rgb_img = np.zeros(image_shape, dtype=np.float64)
            rgb_img[:,:,i] = channel_ds.copy()
    
        return rgb_img
    
    def normalize_rgb_image(self, rgb_image):
        rgb_norm = rgb_image.copy()
        for i in range(rgb_image.shape[2]):
            low, high = np.percentile(rgb_image[:,:,i], [self.perc_low, self.perc_high])
            rgb_norm[:,:,i] = 255.0 * (rgb_image[:,:,i] - low) / (high - low)
        rgb_norm[rgb_norm>255.0] = 255.0
        rgb_norm[rgb_norm<0.0] = 0
        return rgb_norm.astype(np.uint8)
    
    def batch_processing(self, top_folder, out_folder, tissue_ids=None):
        if tissue_ids is None:
            in_folders = list(filter(lambda x: os.path.isdir(os.path.join(top_folder, x)),
                                     os.listdir(top_folder)))
        else:
            in_folders = tissue_ids
            
        print(in_folders)
        for in_folder in in_folders:
            rgb_img = self.get_rgb_image(os.path.join(top_folder, in_folder))
            rgb_img_norm = self.normalize_rgb_image(rgb_img)
            skimage.io.imsave(os.path.join(out_folder, 'rgb_%s.tif' % in_folder),
                              rgb_img_norm)
        return

