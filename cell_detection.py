import os, sys, re, time
import numpy as np
import pdb 

from sequence_importer import SequenceImporter
from skimage.morphology import reconstruction
from scipy.ndimage import gaussian_laplace
from scipy.ndimage.morphology import distance_transform_edt

# skimage imports
import skimage.io

from skimage.filters import gaussian, median, laplace
from skimage.morphology import area_closing
from skimage.morphology import opening, closing, disk, square, dilation, erosion, watershed
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.feature import blob_log, blob_doh, peak_local_max
from skimage.measure import label, regionprops
from skimage.color import grey2rgb
from skimage.morphology import h_maxima
from skimage.draw import circle_perimeter, circle_perimeter_aa, circle
from skimage.filters import threshold_otsu
from skimage.filters import threshold_isodata
from skimage.morphology import diamond, reconstruction

from settings import Settings

from skimage.transform import rescale, resize, downscale_local_mean

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, binom

from numpy.random import permutation
import pickle

from colour import Color

import shutil

from fissure_detection import FissureDetection
from process_ilastik import Ilastik
from visualization import Overlays


class SpotDetector(object):
    
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

    # here we remove small components from the image. 
    # they are isolated local maxima of small size (defined by area_threshold)
    # area_threshold = 1 --> remove single pixels
    # area_threshold = 2 --> remove chunks of 2 neighboring pixels (8-connectivity)
    def remove_individual_pixels_old(self, img, threshold=0.1, 
                                 area_threshold=2):
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask[img >= threshold] = 255
        
        bigger_objects = remove_small_objects(mask.astype(np.bool),
                                              min_size=area_threshold + 1,
                                              connectivity=2)
        mask[bigger_objects] = 0
        
        ring_se = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        dil_img = dilation(img, ring_se)
        out = img.copy()
        out[mask > 0] = dil_img[mask>0]

        return out

    def remove_individual_pixels(self, img, threshold=0.1, 
                                 area_threshold=2):
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask[img >= threshold] = 255
        
        bigger_objects = remove_small_objects(mask.astype(np.bool),
                                              min_size=area_threshold + 1,
                                              connectivity=2)
        mask[bigger_objects] = 0

        out = img.copy()
        out[mask > 0] = 0

        return out

    def remove_salt_noise(self, image):
        se = np.ones((3,3), dtype=np.uint8)
        se[1,1] = 0
        dil = dilation(image, se)
        output = image.copy()
        indices = dil<image
        output[indices] = dil[indices]
        return output


    def get_pixel_noise(self, img, threshold=1.0):
        # get a mask
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask[img >= threshold] = 255
        
        bigger_objects = remove_small_objects(mask.astype(np.bool), 2, connectivity=2)
        mask[bigger_objects] = 0
        
        out = img.copy()
        out[mask==0] = 0

        return out

    def plot_distribution_pixel_noise(self, img, threshold=1.0, channel_name='channel'):
        img_single_pixels = self.get_pixel_noise(img, threshold)
        bins_imposed = np.hstack([np.arange(0, 6, 0.1),
                                  max(6.0, 1.1*img_single_pixels.max())])
        try:
            histo, bins = np.histogram(img_single_pixels, bins=bins_imposed)
        except:
            pdb.set_trace()
        
        # output: descriptors
        select = img_single_pixels[img_single_pixels > 0]
        mean_val = np.mean(select)
        std = np.std(select)
        min_val = select.min()
        max_val = select.max()
        
        print()
        print('CHANNEL: %s' % channel_name)
        print('Image:\tmin: %.5f\tmax: %.5f\tmean: %.5f\tmean+2sigma: %.5f' % (img.min(), img.max(), 
                                                                               img.mean(), (img.mean() + 2 *np.std(img))))
        print('PNoise:\tmin: %.5f\tmax: %.5f\tmean: %.5f\tmean+2sigma: %.5f' % (min_val, max_val, 
                                                                                     mean_val, (mean_val + 2 *std)))
        plot_folder = os.path.join(self.settings.plot_folder, 'pixel_noise')
        if not os.path.isdir(plot_folder):
            os.makedirs(plot_folder)
        filename = os.path.join(plot_folder, 'pixel_noise_distribution_%s.pdf' % channel_name)

        fig = plt.figure()
        
        delta_bin = bins[1] - bins[0]
        
        plt.bar(bins[1:-1],histo[1:],width=delta_bin)
        plt.axvline(x=mean_val, color=(1.0, 0, 0))
        plt.axvline(x=mean_val + 2*std, color=(1.0, 0.8, 0))
        plt.title('pixel noise: %s (%.2f, %.2f)' % (channel_name, mean_val, mean_val + 2 * std))
        plt.xlabel('value of single pixels')
        plt.ylabel('number of occurences')
        #plt.show()
        fig.savefig(filename)
        
        # image histograms
        plot_folder = os.path.join(self.settings.plot_folder, 'img_histograms')
        if not os.path.isdir(plot_folder):
            os.makedirs(plot_folder)
        filename = os.path.join(plot_folder, 'image_histogram_%s.pdf' % channel_name)

        fig = plt.figure()
        
        select = img[img > threshold]
        #select = img[img>0]
        mean_val = np.mean(select)
        std = np.std(select)
        bins = np.hstack([np.arange(0, 40, 0.5),
                          max(40, np.percentile(select, [99.9])[0])])
        plt.hist(select, bins=bins)
        plt.xlim(0, 40)
        plt.axvline(x = mean_val, color=(1.0, 0, 0))
        plt.axvline(x = mean_val + 2 * std, color=(1.0, 0.8, 0))
        plt.title('image histogram: %s (%.2f, %.2f)' % (channel_name, mean_val, mean_val + 2 * std))
        plt.xlabel('grey level')
        plt.ylabel('number of occurences')
        #plt.show()
        fig.savefig(filename)

        
        return

    def make_overlay_img(self, img, blobs, color, perimeter=True):
        if len(img.shape) <= 2 or img.shape[-1] < 2:
            rgb_image = grey2rgb(img)
        else:
            rgb_image = img.copy()
        
        x_coord = []
        y_coord = []
        for y, x in blobs:
            if perimeter:
                xx, yy = circle_perimeter(int(x), int(y), 5, shape=(rgb_image.shape[1], rgb_image.shape[0]))
            else:
                xx, yy = circle(int(x), int(y), 5, shape=(rgb_image.shape[1], rgb_image.shape[0]))
            x_coord.extend(xx)
            y_coord.extend(yy)
            #x_coord.append(x)
            #y_coord.append(y)
        rgb_image[(y_coord,x_coord)] = color
        
        return rgb_image

    def export_spot_detection(self, img, blobs, filename):
        #fig = plt.figure()
        #ax = fig.gca()

        fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True,
                                 subplot_kw={'adjustable': 'box-forced'})
        ax = axes.ravel()
        if len(img.shape) <= 2:
            ax[0].imshow(img, cmap=plt.cm.gray)
            ax[1].imshow(img, cmap=plt.cm.gray)
        elif img.shape[-1] < 2:
            ax[0].imshow(img, cmap=plt.cm.gray)
            ax[1].imshow(img, cmap=plt.cm.gray)
        else:
            ax[0].imshow(img)
            ax[1].imshow(img)

        ax[0].axis('off')
        ax[0].set_title('Original')

        ax[1].axis('off')
        ax[1].set_title('Spot Detections')

        #plt.imshow(img, cmap="Greys_r")

        r = 4
        
        for y, x in blobs:
            c = plt.Circle((x,y), r, color="red", linewidth=2, fill=False)
            ax[1].add_artist(c)
        fig.savefig(filename)
        return

    def export_spot_detection2(self, img, blobs, filename):
        #fig = plt.figure()
        #ax = fig.gca()

        fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True,
                                 subplot_kw={'adjustable': 'box-forced'})
        ax = axes.ravel()
        if len(img.shape) <= 2:
            ax[0].imshow(img, cmap=plt.cm.gray)
            ax[1].imshow(img, cmap=plt.cm.gray)
        elif img.shape[-1] < 2:
            ax[0].imshow(img, cmap=plt.cm.gray)
            ax[1].imshow(img, cmap=plt.cm.gray)
        else:
            ax[0].imshow(img)
            ax[1].imshow(img)

        ax[0].axis('off')
        ax[0].set_title('Original')

        ax[1].axis('off')
        ax[1].set_title('Spot Detections')

        #plt.imshow(img, cmap="Greys_r")

        #r = 5
        
        for y, x, r in blobs:
            c = plt.Circle((x,y), r, color="red", linewidth=2, fill=False)
            ax[1].add_artist(c)
        fig.savefig(filename)
        return
    
    def blob_detection(self, img):
        #image, min_sigma=1, max_sigma=50, num_sigma=10, threshold=0.2, overlap=0.5, log_scale=False
        coordinates = blob_log(img, min_sigma = 4, max_sigma = 10, num_sigma = 3,
                               threshold = 0.05, log_scale = False)
        #coordinates = blob_doh(img, min_sigma=4, max_sigma=10, num_sigma=3, threshold=.01)
        coordinates[:, 2] = coordinates[:, 2] * np.sqrt(2)
        return coordinates

    def save_debug_img(self, img, filename, normalized = True, alpha = 1.0,
                       low_perc = 10, high_perc = 99):

        if normalized:
            if img.max() > img.min():
                low, high = np.percentile(img, [low_perc, high_perc])
                temp_img = 255.0 / (high - low) * (img - low)
            else:
                temp_img = np.zeros(img.shape)
        else:
            temp_img = alpha * img

        temp_img[temp_img > 255] = 255
        temp_img[temp_img  < 0] = 0
        temp_img = temp_img.astype(np.uint8)
        skimage.io.imsave(os.path.join(self.settings.debug_folder,
                                       filename), 
                          temp_img.astype(np.uint8))
        return

    def save_debug_img_float(self, img, filename):
        skimage.io.imsave(os.path.join(self.settings.debug_folder, filename),
                          img)
        return

    def clean_log(self, img, prefix='debug', eps=0.008, 
                  method='local_max', h=0.004, sigma=2.4, k=3,
                  gauss_threshold=30.0):

        gauss_img = gaussian(img, sigma = sigma)

        log = laplace(gauss_img, ksize=k)

        # just threshold
        if method == 'threshold': 
            spots = np.zeros(log.shape, dtype=np.uint8)
            spots[log>eps] = 1
            spots[gauss_img<=gauss_threshold] = 0
        elif method == 'local_max':
            spots = h_maxima(log, h=h)
            spots[log<=eps] = 0
            spots[gauss_img<=gauss_threshold] = 0

        lab = label(spots)
        properties = regionprops(lab)
        coordinates = [properties[i]['centroid'] for i in range(len(properties))]

        return coordinates

    def simple_log(self, img, prefix='debug', eps=0.008, 
                   method='local_max', h=0.004, sigma=2.4, k=3,
                   gauss_threshold=30.0):
        #img_normalized = img.astype(np.float32) / 100.0
        #img_normalized[img_normalized > 1.0] = 1.0
        #img_normalized[img_normalized < 0.0] = 0.0
        #pref = median(img.astype(np.uint8), disk(1))

        gauss_img = gaussian(img, sigma = sigma)
        if self.settings.debug:
            #self.save_debug_img_float(img_normalized, 'debug_spot_detection_original.tif')
            #self.save_debug_img_float(temp, 'debug_spot_detection_gaussian.tif')
            #self.save_debug_img(pref, '%s_spot_detection_median.png' % prefix, False, alpha=1.0)

            self.save_debug_img(img, '%s_spot_detection_original.png' % prefix, False, alpha=1.0)
            self.save_debug_img(gauss_img, '%s_spot_detection_gaussian.png' % prefix, False, alpha=255)

        log = laplace(gauss_img, ksize=k)
        if self.settings.debug:
            print('min, max: ', log.min(), log.max())
            #histo = np.histogram(log)
            #print histo
            #self.save_debug_img_float(log, 'debug_spot_detection_laplacian_unnormalized.tif')
            temp = 0.5 * (10 * log + 1.0)
            temp[temp > 1.0] = 1.0
            temp[temp < 0.0] = 0.0
            self.save_debug_img(temp, '%s_spot_detection_laplacian.png' % prefix, False, alpha=255.0)
            temp = log.copy()
            temp[temp <= eps] = 0
            temp = temp * 1000
            temp[temp > 255.0] = 255
            temp = temp.astype(np.uint8)
            self.save_debug_img(temp, '%s_spot_detection_laplacian_eps_to_zero.png' % prefix, False, alpha=1)
            #skimage.io.imsave(os.path.join(self.settings.debug_folder, 
            #                               'spot_detection_laplacian_unnormalized.tif'),
            #                  log)

        #log = 2 * (log - log.min()) / (log.max() - log.min()) - 1
        #if self.settings.debug:
        #    print 'after normalization: min, max: ', log.min(), log.max()
        #    #self.save_debug_img_float(log, 'debug_spot_detection_after_normalization.tif')
        #    self.save_debug_img(log, 'debug_spot_detection_laplacian_normalized.png', False, alpha=255)

        #coordinates = peak_local_max(log, min_distance=3, indices=True, threshold_abs=0.00000001)

        # just threshold
        if method == 'threshold': 
            spots = np.zeros(log.shape, dtype=np.uint8)
            spots[log>eps] = 1
            spots[gauss_img<=gauss_threshold] = 0
        elif method == 'local_max':
            spots = h_maxima(log, h=h)
            spots[log<=eps] = 0
            spots[gauss_img<=gauss_threshold] = 0
            
        lab = label(spots)
        properties = regionprops(lab)
        coordinates = [(int(properties[i]['centroid'][0]), int(properties[i]['centroid'][1]))
                       for i in range(len(properties))]
        if self.settings.debug:
            ov = Overlays()
            rgb_img = ov.overlay_grey_img(img, spots, {1: (255, 0, 0)}, True)
            filename = os.path.join(self.settings.debug_folder,
                                    '%s_spot_detection_result_overlay.tif' % prefix)
            skimage.io.imsave(filename, rgb_img)
            
            print('number of maxima: %i' % len(coordinates))
            filename = os.path.join(self.settings.debug_folder,
                                    '%s_spot_detection_result.tif' % prefix)
            self.export_spot_detection(img, coordinates, filename)

            filename = os.path.join(self.settings.debug_folder,
                                    '%s_spot_detection_result_gauss.tif' % prefix)
            gauss_max_val = gauss_img.max()
            if gauss_max_val == 0:
                gauss_max_val = 1.0
            gauss_filter_export = 255.0 * gauss_img / gauss_max_val
            gauss_filter_export = gauss_filter_export.astype(np.uint8)
            self.export_spot_detection(gauss_filter_export, coordinates, filename)

        return coordinates

    def normalize(self, img, lower_bound_of_max=5.0,
                  lower_bound_of_value=0.1, percentile=99.9):
        perc = np.percentile(img, [percentile])
        max_val = max(lower_bound_of_max, perc[-1])
        img_norm = (img - lower_bound_of_value) / (max_val - lower_bound_of_value)
        img_norm[img_norm > 1.0] = 1.0
        img_norm[img_norm < 0.0] = 0.0
        return img_norm

    def normalize_minmax(self, img, percentile=99.9,
                         min_val=None, max_val=None):

        if min_val is None:
            min_val = np.min(img)
        if max_val is None:        
            perc = np.percentile(img, [percentile])
            max_val = perc[0]

        img_norm = (img - min_val) / (max_val - min_val)
        img_norm[img_norm > 1.0] = 1.0
        img_norm[img_norm < 0.0] = 0.0
        return img_norm

    def normalize_filtered_image(self, img, fissure_image):
        prefiltered = self.remove_salt_noise(img)
        gauss_img = gaussian(image, sigma=1.5)
        max_val = np.max(gauss_img)
        min_val = np.min(gauss_img)
        
        img_norm = (img - min_val) / (max_val - min_val)
        img_norm[img_norm > 1.0] = 1.0
        img_norm[img_norm < 0.0] = 0.0

        laplace_img[:,:,i] = laplace(gauss_img[:,:,i], 3)
        laplace_img[fissure_image>0] = -1
        laplace_max = np.max(laplace_img)
        
        spots = h_maxima(log, h=h)
        spots[log<=eps] = 0
        spots[gauss_img<=gauss_threshold] = 0
            
        lab = label(spots)
        properties = regionprops(lab)
        coordinates = [properties[i]['centroid'] for i in range(len(properties))]

        #gauss_img[:,:,i] = gaussian(image, sigma=1.0)
        props = regionprops(ws, gauss_img[:,:,i])
        intensities['gauss'][channel] = np.array([props[k]['mean_intensity'] for k in range(len(props))])
        #perc = np.percentile(intensities['gauss'][channel],perc_for_vis)
        perc = np.percentile(gauss_img[:,:,i], perc_for_vis)
        maxvals['gauss'][channel] = perc[1]
        minvals['gauss'][channel] = perc[0]

            
        return

    def __call__(self, img):
        return self.simple_log(img)


class CellDetection(object):

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

        self.sp = SpotDetector(settings=self.settings)
        self.fissure = FissureDetection(settings=self.settings)
        self.image_result_folder = os.path.join(self.settings.output_folder, 
                                                'cell_segmentation')
        if not os.path.isdir(self.image_result_folder):
            os.makedirs(self.image_result_folder)


    def __call__(self):
        
        print('Reading data ... ')
        si = SequenceImporter(['DNA1'])
        img, channel_names = si(self.settings.input_folder)
        image = img[:,:,0]

        print('Detecting spots ... ')
        start_time = time.time()
        coordinates = self.spot_detection_dna_raw(image)
        diff_time = time.time() - start_time
        print('\ttime elapsed, spot detection: %.2f s' % diff_time)

        print('Fissure image ... ')
        start_time = time.time()
        fissure_image = self.fissure.get_image(False)
        filtered_coordinates = list(filter(lambda x: fissure_image[x] == 0, coordinates))
        print(len(coordinates), ' --> ', len(filtered_coordinates))
        coordinates = filtered_coordinates
        print('\ttime elapsed, fissure exclusion: %.2f s' % diff_time)

        print('Voronoi diagram ... ')
        start_time = time.time()
        ws = self.get_voronoi_regions(image, coordinates, radius=8)
        diff_time = time.time() - start_time
        print('\ttime elapsed, voronoi: %.2f s' % diff_time)

        print('Writing result image ... ')
        self.write_image(ws)
        
        return ws

    def get_voronoi_regions(self, image, coordinates, radius=5):
        marker = np.ones(image.shape, dtype=np.uint8)
        yvec = [coord[0] for coord in coordinates]
        xvec = [coord[1] for coord in coordinates]
        marker[yvec, xvec] = 0
        mask = 1 - erosion(marker, disk(radius))
        distance_map = distance_transform_edt(marker)
        marker = 1 - marker

        marker_label = label(marker)
        ws = watershed(distance_map, marker_label, mask=mask,
                       watershed_line=False)
        
        return ws

    def make_random_colors(self, ws):
        max_label = ws.max()
        colors = 255 * np.random.rand(max_label+1,3)
        colors[0] = np.zeros(3)
        colors = colors.astype(np.uint8)
        
        output = colors[ws]
        return output

    def remove_salt_noise(self, image):
        se = np.ones((3,3), dtype=np.uint8)
        se[1,1] = 0
        dil = dilation(image, se)
        output = image.copy()
        indices = dil<image
        output[indices] = dil[indices]
        return output

    def spot_detection_dna_raw(self, image):
        sigma = self.settings.dna_spot_detection['sigma']
        h = self.settings.dna_spot_detection['h']
        eps = self.settings.dna_spot_detection['eps']
        gauss_threshold = self.settings.dna_spot_detection['gauss_threshold']
        area_threshold = self.settings.dna_spot_detection['area_threshold']
        norm_low = self.settings.dna_spot_detection['normalization_low']
        norm_high = self.settings.dna_spot_detection['normalization_high']

        # prefiltering and normalization
        img_filtered = self.remove_salt_noise(image)
        img_norm = self.sp.normalize(img_filtered, lower_bound_of_max=norm_high,
                                     lower_bound_of_value=norm_low, percentile=99.5)
        img_in = img_norm * 255.0
        img_in = img_in.astype(np.uint8)
        
        # get coordinates
        coordinates = self.sp.simple_log(img_in,
                                         method='local_max',
                                         prefix='debug_dna',
                                         eps=eps, h=h, sigma=sigma, k=3,
                                         gauss_threshold=gauss_threshold)
        
        return coordinates

    def write_image(self, image):
        filename = os.path.join(self.image_result_folder, 
                                'dna_cell_segmentation.tiff')
        skimage.external.tifffile.imsave(filename, image.astype('uint32'))
        # pdb.set_trace()
        # bug: this does not work ! skimage.io.imsave(filename, image)
        # fp = open(filename, image)
        
        rgb_image = self.make_random_colors(image)
        filename = os.path.join(self.image_result_folder, 
                                'dna_cell_segmentation_random_colors.png')
        skimage.io.imsave(filename, rgb_image)

        return

    def read_image(self):
        filename = os.path.join(self.image_result_folder, 
                                'dna_cell_segmentation.tiff')
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





