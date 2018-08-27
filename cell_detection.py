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
                                'dna_cell_segmentation.png')
        skimage.io.imsave(filename, image)

        rgb_image = self.make_random_colors(image)
        filename = os.path.join(self.image_result_folder, 
                                'dna_cell_segmentation_random_colors.png')
        skimage.io.imsave(filename, rgb_image)

        return

    def read_image(self):
        filename = os.path.join(self.image_result_folder, 
                                'dna_cell_segmentation.png')
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

class IlastikPreparation(object):
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
        #histo, bins = np.histogram(img, bins=img.max())
        #values = np.where(histo>0)[0]
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

    def __call__(self):
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

class FissureDetection(object):
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

#         threshold1 = np.mean(pref2) + 1.5 * np.std(pref2)
#         threshold2 = np.mean(pref2) + 3 * np.std(pref2)
#         
#         #threshold = threshold_otsu(pref2)
#         seed = np.zeros(pref2.shape, dtype=np.uint8)
#         mask = np.zeros(pref2.shape, dtype=np.uint8)
#         mask[pref2 > threshold1] = 255
#         seed[pref2 > threshold2] = 255
#         output = reconstruction(seed, mask, method='dilation')
#         
#         output = output.astype(np.uint8)
#         print 'OUTPUT: ', output.min(), output.max()
#         print len(output[output>0])
# 
#         output = remove_small_objects(output, 400)
#         
#         #output = rescale(output_s, 4)
#         #output = output_s
#         
#         output_folder = os.path.join(self.settings.output_folder, 'fissure')
#         if not os.path.isdir(output_folder):
#             os.makedirs(output_folder)
#         filename = os.path.join(output_folder, '%s_fissure.png' % 
#                                 self.settings.dataset)
#         #ov = Overlays()
#         #rgb_image = ov.overlay_grey_img(image, output, {1: (255, 0, 0)}, True)
#         #skimage.io.imsave(filename, rgb_image)
#         skimage.io.imsave(filename, output)
#         print len(output[output>0])
        
        return output



class SubPopulations(object):
    
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
        self.cell_detector = CellDetection(settings=self.settings)
        
    def get_fissure(self, process_image=False):
        if not process_image:
            try:
                fissure_image = self.fissure.read_image()
            except:
                fissure_image = self.fissure()
                self.fissure.write_image(fissure_image)
        else:
            fissure_image = self.fissure()
            self.fissure.write_image(fissure_image)        
        return fissure_image
    
    def _get_fissure(self):
        markers = ['avanti']
        si = SequenceImporter(markers)
        img, channel_names = si(self.settings.input_folder)
        perc = np.percentile(img[:,:,0], [1, 99])
        low_val = perc[0]
        high_val = perc[1]
        image = 255 * (img[:,:,0] - low_val) / (high_val - low_val)
        image = image.astype(np.uint8)

        
        
        avanti = self.remove_salt_noise(image)
        #avanti = rescale(temp, .25)
        pref1 = median(avanti, disk(6))
        pref2 = gaussian(pref1, 4)
        
        threshold1 = np.mean(pref2) + 1.5 * np.std(pref2)
        threshold2 = np.mean(pref2) + 3 * np.std(pref2)
        
        #threshold = threshold_otsu(pref2)
        seed = np.zeros(pref2.shape, dtype=np.uint8)
        mask = np.zeros(pref2.shape, dtype=np.uint8)
        mask[pref2 > threshold1] = 255
        seed[pref2 > threshold2] = 255
        output = reconstruction(seed, mask, method='dilation')
        
        output = output.astype(np.uint8)
        print('OUTPUT: ', output.min(), output.max())
        print(len(output[output>0]))

        output = remove_small_objects(output, 400)
        
        #output = rescale(output_s, 4)
        #output = output_s
        
        output_folder = os.path.join(self.settings.output_folder, 'fissure')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        filename = os.path.join(output_folder, '%s_fissure.png' % 
                                self.settings.dataset)
        #ov = Overlays()
        #rgb_image = ov.overlay_grey_img(image, output, {1: (255, 0, 0)}, True)
        #skimage.io.imsave(filename, rgb_image)
        skimage.io.imsave(filename, output)
        print(len(output[output>0]))
        
        return output


    def multiply_and_clip(self, img, alpha):
        out = img.copy()
        out = out.astype(np.uint16)
        out = alpha * out
        out[out > 255] = 255
        return out.astype(np.uint8)

    def spot_detection(self, img=None, channel_names=None, population=1):
        sigma = self.settings.spot_detection['sigma']
        h = self.settings.spot_detection['h']
        eps = self.settings.spot_detection['eps']
        gauss_threshold = self.settings.spot_detection['gauss_threshold']
        area_threshold = self.settings.spot_detection['area_threshold']
        norm_low = self.settings.spot_detection['normalization_low']
        norm_high = self.settings.spot_detection['normalization_high']

        report = ''
        if img is None or channel_names is None:
            print('reading population %i' % population)
            img, channel_names = self.get_population(population)

        report += 'REPORT : population: %i\n' % population
        report += '-----------------------\n'
        report += 'dataset: %s\n' % self.settings.dataset
        report += 'channels: %s\n' % str(channel_names)
        pref = img.copy()
        for i in range(img.shape[-1]):
            # prefiltering and normalization
            img_filtered = self.sp.remove_individual_pixels(img[:,:,i], threshold=norm_low,
                                                       area_threshold=area_threshold)
            img_norm = self.sp.normalize(img_filtered, lower_bound_of_max=norm_high,
                                    lower_bound_of_value=norm_low)
            img_in = img_norm * 255.0
            img_in = img_in.astype(np.uint8)

            pref[:,:,i] = img_in
            report += '%s: %i %i\t-->\t %.4f %.4f\t-->\t %.4f %.4f\n' % (channel_names[i],
                                                                         img[:,:,i].min(), img[:,:,i].max(),
                                                                         img_norm.min(), img_norm.max(), 
                                                                         pref[:,:,i].min(), pref[:,:,i].max())
            if self.settings.debug:
                self.save_debug_img(img_in, '%s_spot_detection_%s_population_%i.png' % 
                                    (self.settings.dataset, channel_names[i], population), 
                                    False, 1)

        # the minimum over all channels (we are seeking a subpopulation
        # expressing ALL markers
        min_img = np.min(pref, axis=2)
        img_filtered = self.sp.remove_individual_pixels(min_img, threshold=1,
                                                        area_threshold=area_threshold)
        report += 'min-image [shape: %s]: %i %i\tmean of non-zero pixels: %f\n' % (str(img_filtered.shape),
                                                                                   img_filtered.min(), 
                                                                                   img_filtered.max(),
                                                                                   np.mean(img_filtered[img_filtered > 0]))

        if self.settings.debug:
            self.save_debug_img(min_img, '%s_min_img_population_%i.png' % 
                                (self.settings.dataset, population), 
                                False, 1.0)
            self.save_debug_img(img_filtered, '%s_min_img_filtered_population_%i.png' % 
                                (self.settings.dataset, population), 
                                False, 1.0)
        
        self.sp.settings.debug = False
        coordinates = self.sp.simple_log(img_filtered, 
                                         prefix='%s_debug_threshold_min_img_population_%i' % 
                                         (self.settings.dataset, population),
                                         method='local_max',
                                         eps=eps, h=h, sigma=sigma, k=3,
                                         gauss_threshold=gauss_threshold)
        
        report += 'number of spots: %i\n' % len(coordinates)

        filename = os.path.join(self.settings.output_folder,
                                '%s_spot_detection_population_%i.png' % 
                                (self.settings.dataset, population))
        print(report) 

        self.sp.export_spot_detection(img_filtered, coordinates, filename)

        # output : color image
        rgb_image = np.zeros((img_filtered.shape[0], img_filtered.shape[1], 3),
                             dtype=np.uint8)
        nb_of_channels = img.shape[-1]
        alpha = 2
        for k in range(min(nb_of_channels, 3)):
            rgb_image[:,:,k] = self.multiply_and_clip(pref[:,:,k], alpha)
        if nb_of_channels > 3:
            colors = {3: [0, 1], 
                      4: [1, 2], 
                      5: [0, 2]}
            rgb_temp = rgb_image.copy()
            rgb_temp = rgb_temp.astype(np.uint16)
            for k in range(3, min(nb_of_channels, 6)):
                for c in colors[k]: 
                    rgb_temp[:,:,c] += self.multiply_and_clip(pref[:,:,k], alpha)
            rgb_temp[rgb_temp > 255] = 255
            rgb_image = rgb_temp.astype(np.uint8)

        filename = os.path.join(self.settings.output_folder,
                                '%s_composite_original_population_%i.png' % 
                                (self.settings.dataset, population))
        skimage.io.imsave(filename, rgb_image)
        
        # todo: write color image with detected spots
        filename = os.path.join(self.settings.output_folder,
                                '%s_spot_detection_population_rgb_%i.png' % 
                                (self.settings.dataset, population))
        self.sp.export_spot_detection(rgb_image, coordinates, filename)

        if nb_of_channels <= 2:
            color = (0,0,255)
        else:
            color = (255, 255, 255)
        filename = os.path.join(self.settings.output_folder,
                                '%s_spot_detection_population_rgb_overlay_%i.png' % 
                                (self.settings.dataset, population))
        overlay_img = self.sp.make_overlay_img(rgb_image, coordinates, color)
        skimage.io.imsave(filename, overlay_img)

        filename = os.path.join(self.settings.output_folder,
                                '%s_spot_detection_population_min_img_overlay_%i.png' % 
                                (self.settings.dataset, population))
        overlay_img = self.sp.make_overlay_img(img_filtered.astype(np.uint8), coordinates, color)
        
        skimage.io.imsave(filename, overlay_img)
        
        return coordinates

    def spot_detection_raw(self, img=None, channel_names=None, population=1):
        sigma = self.settings.spot_detection['sigma']
        h = self.settings.spot_detection['h']
        eps = self.settings.spot_detection['eps']
        gauss_threshold = self.settings.spot_detection['gauss_threshold']
        area_threshold = self.settings.spot_detection['area_threshold']
        norm_low = self.settings.spot_detection['normalization_low']
        norm_high = self.settings.spot_detection['normalization_high']

        report = ''
        if img is None or channel_names is None:
            print('reading population %i' % population)
            img, channel_names = self.get_population(population)

        report += 'REPORT : population: %i\n' % population
        report += '-----------------------\n'
        report += 'dataset: %s\n' % self.settings.dataset
        report += 'channels: %s\n' % str(channel_names)
        pref = img.copy()
        for i in range(img.shape[-1]):
            # prefiltering and normalization
            img_filtered = self.sp.remove_individual_pixels(img[:,:,i], threshold=norm_low,
                                                       area_threshold=area_threshold)
            img_norm = self.sp.normalize(img_filtered, lower_bound_of_max=norm_high,
                                    lower_bound_of_value=norm_low)
            img_in = img_norm * 255.0
            img_in = img_in.astype(np.uint8)

            pref[:,:,i] = img_in
            report += '%s: %i %i\t-->\t %.4f %.4f\t-->\t %.4f %.4f\n' % (channel_names[i],
                                                                         img[:,:,i].min(), img[:,:,i].max(),
                                                                         img_norm.min(), img_norm.max(), 
                                                                         pref[:,:,i].min(), pref[:,:,i].max())

        # the minimum over all channels (we are seeking a subpopulation
        # expressing ALL markers
        min_img = np.min(pref, axis=2)
        img_filtered = self.sp.remove_individual_pixels(min_img, threshold=1,
                                                        area_threshold=area_threshold)
        report += 'min-image [shape: %s]: %i %i\tmean of non-zero pixels: %f\n' % (str(img_filtered.shape),
                                                                                   img_filtered.min(), 
                                                                                   img_filtered.max(),
                                                                                   np.mean(img_filtered[img_filtered > 0]))
        
        self.sp.settings.debug = False
        coordinates = self.sp.simple_log(img_filtered, 
                                         prefix='%s_debug_threshold_min_img_population_%i' % 
                                         (self.settings.dataset, population),
                                         method='local_max',
                                         eps=eps, h=h, sigma=sigma, k=3,
                                         gauss_threshold=gauss_threshold)
        
        report += 'number of spots: %i\n' % len(coordinates)
        print(report)

        return coordinates



    def make_composite_image(self, image, population=1, filename=None):
        temp_image = image.copy()
        for i in range(image.shape[-1]):
            channel = image[:,:,i].astype(np.float64)
            if np.percentile(channel, 99.5) > 0:
                alpha = 255.0 / np.percentile(channel, 99.5)
            elif np.max(channel) > 0:
                alpha = 255.0 / np.max(channel)
            else:
                alpha = 1.0
            temp_image[:,:,i] = alpha * channel
        
#         pdb.set_trace()
#         if image.shape[-1] > 3:
#             for j in range(image.shape[-1] - 3):
#                 channel = temp_image[:,:,j]
#                 for i in range(3): 
#                     temp_image[:,:,i] = np.maximum(temp_image[:,:,i], channel)
#         
#         temp_image = temp_image[:,:,range(3)].astype(np.uint8)
        #pdb.set_trace()
        temp_image = temp_image.astype(np.uint8)
        
        rgb_image = np.zeros((temp_image.shape[0], temp_image.shape[1], 3))
        
        for i in range(min(temp_image.shape[-1], 3)):
            rgb_image[:,:,i] = temp_image[:,:,i].copy()
        
        if self.settings.debug:
            if filename is None:
                filename = 'population_%i.png' % population
            self.save_debug_img(rgb_image, filename, False)

        return rgb_image

    def plot_composite(self):
        
        pop1, channel_names = self.get_population(1)
        print('population 1: ', channel_names)
        self.make_composite_image(pop1, 1)
        
        pop2, channel_names = self.get_population(2)
        print('population 2: ', channel_names)
        self.make_composite_image(pop2, 2)
        
        pop3, channel_names = self.get_population(3)
        print('population : ', channel_names)
        self.make_composite_image(pop3, 3)
        return 

    def save_debug_img(self, img, filename, normalized = True, alpha = 1.0):

        if normalized and (img.max() > img.min()):
            temp_img = 255.0 / (img.max() - img.min()) * (img - img.min())
        else:
            temp_img = alpha * img.astype(np.float64)

        temp_img[temp_img>255.0] = 255.0
        temp_img[temp_img<0.0] = 0.0
        temp_img = temp_img.astype(np.uint8)
        skimage.io.imsave(os.path.join(self.settings.debug_folder,
                                       filename), 
                          temp_img.astype(np.uint8))
        return

    def get_dna_images(self):
        markers = ['DNA1', 'DNA3']
        si = SequenceImporter(markers)
        img, channel_names = si(self.settings.input_folder)
        return img, channel_names

    def remove_salt_noise(self, image):
        se = np.ones((3,3), dtype=np.uint8)
        se[1,1] = 0
        dil = dilation(image, se)
        output = image.copy()
        indices = dil<image
        output[indices] = dil[indices]
        return output

    def make_random_colors(self, ws):
        max_label = ws.max()
        colvec = 255 * np.random.rand(max_label+1,3)
        colvec[0] = np.zeros(3)
        colvec = colvec.astype(np.uint8)
        
        output = colvec[ws]
        return output

    def make_linear_colors(self, ws, image, N=100):
        max_label = ws.max()
        green = Color("green")
        black = Color("black")
        color_values = np.array([x.get_rgb() for x in black.range_to(green, N)])

        props = regionprops(ws, image)
        intensities = np.array([props[k]['mean_intensity'] for k in range(len(props))])

        #histo, bin_edges = np.histogram(intensities, N-1)
        perc = np.percentile(image, [1, 95])
        minval = perc[0]
        maxval = perc[1]
        bins = np.linspace(minval, maxval, N-1)
        bin_vec = np.digitize(intensities, bins)

        colvec = 255 * color_values[bin_vec]
        colvec = np.vstack([np.zeros(3), colvec])
        colvec = colvec.astype(np.uint8)
        
        output = colvec[ws]
        return output
    
    def get_voronoi_regions(self, image, coordinates, radius=5, 
                            prefix='debug', out_sub_folder='dna_detection'):
        marker = np.ones(image.shape, dtype=np.uint8)
        yvec = [coord[0] for coord in coordinates]
        xvec = [coord[1] for coord in coordinates]
        marker[yvec, xvec] = 0
        mask = 1 - erosion(marker, disk(radius))
        distance_map = distance_transform_edt(marker)
        marker = 1 - marker

        marker_label = label(marker)
        ws = watershed(distance_map, marker_label, mask=mask,
                       watershed_line=True)
        
        output_folder = os.path.join(self.settings.output_folder, out_sub_folder)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        if self.settings.debug:
            filename = os.path.join(output_folder, 
                                    '%s_dna_detection_watershed.png' % prefix)
            skimage.io.imsave(filename, ws)
                        
            filename = os.path.join(output_folder, 
                                    '%s_dna_detection_watershed_color.png' % prefix)
            rgb_out = self.make_random_colors(ws)
            skimage.io.imsave(filename, rgb_out)
            
            filename = os.path.join(output_folder, 
                                    '%s_dna_detection_watershed_wsl.png' % prefix)
            norm_img = 255 * self.sp.normalize(image, percentile=92)
            ov = Overlays()
            wsl = np.zeros((image.shape), dtype=np.uint8)
            wsl[ws==0] = 1
            ov_image = ov.overlay_grey_img(norm_img.astype(np.uint8), wsl, {1: (255, 0, 0)}, contour=False)
            skimage.io.imsave(filename, ov_image)

        return


    def spot_detection_dna_raw(self, image, filename_addon='', 
                               out_sub_folder='dna_detection'):
        sigma = self.settings.dna_spot_detection['sigma']
        h = self.settings.dna_spot_detection['h']
        eps = self.settings.dna_spot_detection['eps']
        gauss_threshold = self.settings.dna_spot_detection['gauss_threshold']
        area_threshold = self.settings.dna_spot_detection['area_threshold']
        norm_low = self.settings.dna_spot_detection['normalization_low']
        norm_high = self.settings.dna_spot_detection['normalization_high']

        img_filtered = self.remove_salt_noise(image)
        img_norm = self.sp.normalize(img_filtered, lower_bound_of_max=norm_high,
                                     lower_bound_of_value=norm_low, percentile=99.5)
        img_in = img_norm * 255.0
        img_in = img_in.astype(np.uint8)
        
        coordinates = self.sp.simple_log(img_in,
                                         method='local_max',
                                         prefix='debug_dna',
                                         eps=eps, h=h, sigma=sigma, k=3,
                                         gauss_threshold=gauss_threshold)

        print('number of spots: %i' % len(coordinates))

        if self.settings.debug:
            output_folder = os.path.join(self.settings.output_folder, out_sub_folder)
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
    
            filename = os.path.join(output_folder, 'simple_log_%s_both.png' % filename_addon)
            
            self.sp.export_spot_detection(img_in, coordinates, filename)
            filename = os.path.join(output_folder, 'simple_log_%s_original.png' % filename_addon)
            skimage.io.imsave(filename, img_in)
    
            filename = os.path.join(output_folder, 'simple_log_%s_overlay.png' % filename_addon)
            color = (255, 0, 0)
            overlay_img = self.sp.make_overlay_img(img_in, coordinates, color)
            skimage.io.imsave(filename, overlay_img)

        return coordinates

    def _detect_cells_from_dna(self, img=None, channel_names=None, 
                              filename_addon='', method='simple_log'):
        if img is None or channel_names is None: 
            img, channel_names = self.get_dna_images()

        output_folder = os.path.join(self.settings.output_folder, 'dna_detection')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        for i in range(len(channel_names)):
            channel = channel_names[i]
            image = 255 * self.sp.normalize(img[:,:,i], percentile=95)
            image = image.astype(np.uint8)
            prefiltered = median(image, disk(1))
            
            print(prefiltered.min(), prefiltered.max())
            #if method=='blob_log':
            blobs = skimage.feature.blob_log(prefiltered, min_sigma=1.5, max_sigma=5.0, 
                                             num_sigma=16, threshold=0.1, overlap=0.6)

            print('%s\tnumber of blobs: %i' % (channel, len(blobs)))
            filename = os.path.join(output_folder, 'blob_log_%s%s.png' % (filename_addon, 
                                                                          channel))
            self.sp.export_spot_detection2(image, blobs, filename)
            filename = os.path.join(output_folder, 'original_%s%s.png' % (filename_addon, 
                                                                          channel))
            skimage.io.imsave(filename, image)
        return

    def read_region_images(self):
        input_folder = os.path.join(self.settings.output_folder, 'ilastik')
        filename = os.path.join(input_folder, 
                                'rgb_%s_Simple Segmentation.png' % self.settings.dataset)
        if not os.path.isfile(filename): 
            raise ValueError("file %s does not exist" % filename)
        
        im_small = skimage.io.imread(filename)
        img = rescale(im_small, 4, preserve_range=True)
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

    def normalize_noise(self, img, channel_names):
        for i, channel in channel_names.iteritems():
            image = img[:,:,i]
            filtered = self.remove_salt_noise(image)
            noise_image = image - filtered
            noise_level = np.mean(noise_image)
            noise_level_nonzero = np.mean(noise_image[noise_image>0])
            noise_level_median_nonzero = np.mean(noise_image[noise_image>0])
            print('%s\t%f\t%f\t%f' % (channel, noise_level, noise_level_nonzero,
                                      noise_level_median_nonzero))
        return

    def get_salt_noise_image(self, image):
        filtered = self.remove_salt_noise(image)
        noise_image = image - filtered
        return noise_image

    def get_pixel_noise(self, img, threshold=1.0):
        # get a mask
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask[img >= threshold] = 255

        bigger_objects = remove_small_objects(mask.astype(np.bool), 2, connectivity=2)
        mask[bigger_objects] = 0

        vec = img[mask>0]

        return vec

    def plot_intensity_pixel_noise(self, vec1, img, channel_name,
                                   threshold=0.1):
        
        plot_folder = os.path.join(self.settings.plot_folder, 'pixel_noise')
        if not os.path.isdir(plot_folder):
            os.makedirs(plot_folder)
        filename = os.path.join(plot_folder, 'pixel_noise_histogram_intensities_%s.png' % 
                                channel_name)
        
        vec2 = self.get_pixel_noise(img, threshold)
        all_vec = np.concatenate((vec1, vec2))
        perc = np.percentile(all_vec, [0.1, 99.9])
        minval = perc[0]
        maxval = perc[1]
        vec1 = vec1[vec1 < maxval]
        vec1 = vec1[vec1 > minval]
        vec2 = vec2[vec2 < maxval]
        vec2 = vec2[vec2 > minval]
        
        X = [vec1, vec2]
        mean_val1 = np.mean(vec1)
        std1 = np.std(vec1)
        mean_val2 = np.mean(vec2)
        std2 = np.std(vec2)
        #width = 0.7 * (bins[1] - bins[0])
        #center = (bins[:-1] + bins[1:]) / 2
        
        fig = plt.figure()
        
        #plt.bar(center, hist, align='center', width=width)
        labels = ['intensities', 'pixel noise']
        plt.hist(X, normed=True, histtype='bar', 
                 color=['steelblue', 'darkorange'],
                 edgecolor='none',
                 label=labels,
                 bins=60)
        plt.title('intensity distribution: %s' % (channel_name))

        plt.axvline(x=mean_val1, ymin=0, ls='-', lw=2, color='blue')
        plt.axvline(x=mean_val1 + 2*std1, 
                    ymin=0, ls='--', lw=2, color='blue')
        plt.axvline(x=mean_val2, ymin=0, ls='-', lw=2, color='red')
        plt.axvline(x=mean_val2 + 2*std2, 
                    ymin=0, ls='--', lw=2, color='red')

        plt.legend(prop={'size': 10})
        plt.xlabel('intensities')
        plt.ylabel('')
        plt.grid(b=True, which='major', color='darkgrey', linewidth=1)
        fig.savefig(filename)
        plt.close()
        return
    
    def plot_intensity_histogram(self, vec1, vec2,
                                 channel_name,
                                 filename_addon='raw'):
        
        plot_folder = os.path.join(self.settings.plot_folder, 'scatter_intensity')
        if not os.path.isdir(plot_folder):
            os.makedirs(plot_folder)
        filename = os.path.join(plot_folder, 'histogram_intensities_%s_%s.png' % 
                                (channel_name, filename_addon))
        
        #hist, bins = np.histogram(vec, bins=20, normed=1)
        perc = np.percentile(vec1, [0.1, 99.9])
        minval = perc[0]
        maxval = perc[1]
        vec1 = vec1[vec1 < maxval]
        vec1 = vec1[vec1 > minval]
        vec2 = vec2[vec2 < maxval]
        vec2 = vec2[vec2 > minval]
        
        X = [vec1, vec2]
        mean_val1 = np.mean(vec1)
        std1 = np.std(vec1)
        mean_val2 = np.mean(vec2)
        std2 = np.std(vec2)
        #width = 0.7 * (bins[1] - bins[0])
        #center = (bins[:-1] + bins[1:]) / 2
        
        fig = plt.figure()
        
        #plt.bar(center, hist, align='center', width=width)
        labels = ['%s intensities' % filename_addon, 'randomized']
        plt.hist(X, normed=False, histtype='bar', 
                 color=['steelblue', 'darkorange'],
                 edgecolor='none',
                 label=labels,
                 bins=60)
        plt.title('intensity distribution: %s' % (channel_name))

        plt.axvline(x=mean_val1, ymin=0, ls='-', lw=2, color='blue')
        plt.axvline(x=mean_val1 + 2*std1, 
                    ymin=0, ls='--', lw=2, color='blue')
        plt.axvline(x=mean_val2, ymin=0, ls='-', lw=2, color='red')
        plt.axvline(x=mean_val2 + 2*std2, 
                    ymin=0, ls='--', lw=2, color='red')

        plt.legend(prop={'size': 10})
        plt.xlabel('intensities')
        plt.ylabel('')
        plt.grid(b=True, which='major', color='darkgrey', linewidth=1)
        fig.savefig(filename)
        plt.close()
        
        # normal histogram
        filename = os.path.join(plot_folder, 'simple_histogram_intensities_%s_%s.png' % 
                                (channel_name, filename_addon))
        fig = plt.figure()
        plt.hist(vec1, normed=False, histtype='bar', 
                 color='steelblue',
                 edgecolor='none',
                 bins=60)
        plt.xlabel('intensities')
        plt.ylabel('')
        plt.grid(b=True, which='major', color='darkgrey', linewidth=1)
        plt.title('intensity distribution: %s' % (channel_name))

        plt.axvline(x=mean_val1, ymin=0, ls='-', lw=2, color='blue')
        plt.axvline(x=mean_val1 + 2*std1, 
                    ymin=0, ls='--', lw=2, color='blue')

        fig.savefig(filename)
        plt.close()

        # log histogram
        filename = os.path.join(plot_folder, 'log_histogram_intensities_%s_%s.png' % 
                                (channel_name, filename_addon))
        fig = plt.figure()
        log_vec = np.log(vec1 - np.min(vec1) + 0.00001)
        mean_val1 = np.mean(log_vec)
        std1 = np.std(log_vec)
        plt.hist(log_vec, normed=False, histtype='bar', 
                 color='steelblue',
                 edgecolor='none',
                 bins=60)
        plt.xlabel('intensities')
        plt.ylabel('')
        plt.grid(b=True, which='major', color='darkgrey', linewidth=1)
        plt.title('intensity distribution: %s' % (channel_name))

        plt.axvline(x=mean_val1, ymin=0, ls='-', lw=2, color='blue')
        plt.axvline(x=mean_val1 + 2*std1, 
                    ymin=0, ls='--', lw=2, color='blue')

        fig.savefig(filename)
        plt.close()

        return

    def plot_distance_histogram(self, vec, img_values, filename,
                                region_name='B-cell region', 
                                subset_name='1'):
        
        plot_folder = os.path.join(self.settings.plot_folder, 'distance_histograms')
        if not os.path.isdir(plot_folder):
            os.makedirs(plot_folder)
        full_filename = os.path.join(plot_folder, filename)
        
        #hist, bins = np.histogram(vec, bins=20, normed=1)
        X = [vec, img_values]
        mean_val = np.mean(vec)
        std = np.std(vec)
        #width = 0.7 * (bins[1] - bins[0])
        #center = (bins[:-1] + bins[1:]) / 2
        
        fig = plt.figure()

        #plt.bar(center, hist, align='center', width=width)
        colors = ['red', 'blue']
        labels = [subset_name, 'CSR']
        plt.hist(X, normed=1, bins=32, histtype='bar', color=colors, label=labels)
        plt.title('distances: %s (%.2f +/- %.2f)' % (region_name, 
                                                     mean_val, 
                                                     mean_val + 2 * std))
        plt.legend(prop={'size': 10})
        plt.xlabel('distance in pixels')
        plt.ylabel('frequency')
        fig.savefig(full_filename)
        plt.close()
        
        fig = plt.figure()
        flierprops = dict(marker='o', markerfacecolor='red', markersize=6,
                          linestyle='none')
        plt.boxplot(X, sym='o', flierprops=flierprops, labels=labels)
        plt.title('Distance boxplots: %s in %s' % (subset_name, region_name))

        full_filename = os.path.join(plot_folder, 'boxplot_%s' % filename)
        fig.savefig(full_filename)
        
        return


    def make_intensity_scatter_plot(self, x, y, filename, x_name, y_name,
                                    plot_grid=True, plot_lines=True, 
                                    randomize=False):
        perc = np.percentile(x, [0.1, 50, 99.9])
        xmin = perc[0]; xmax = perc[-1]; xmed = perc[1]
        xmean = np.mean(x)
        xstd = np.std(x)
        
        perc = np.percentile(y, [0.1, 50, 99.9])
        ymin = perc[0]; ymax = perc[-1]; ymed = perc[1]
        ymean = np.mean(y)
        ystd = np.std(y)

        if randomize:
            x = permutation(x)
            y = permutation(y)

        fig = plt.figure(1)
        ax = plt.subplot(1,1,1)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title('Scatterplot: %s vs. %s' % (x_name, y_name))  

        plt.scatter(x, y, color = 'red', marker='.', edgecolor='none')
        axis = [xmin - (xmax - xmin) * 0.05,
                xmax + (xmax - xmin) * 0.05,
                ymin - (ymax - ymin) * 0.05,
                ymax + (ymax - ymin) * 0.05]
        plt.axis(axis)
        
        if plot_grid:
            plt.grid(b=True, which='major', linewidth=1)
        if plot_lines:
            added_lines = {}
            added_lines[0] = plt.axvline(#ymin=ymin, ymax=ymax,
                        x=xmean,
                        color=(0, 0.2, 0.5),
                        linewidth=2, ls='-')
            added_lines[1] = plt.axvline(#ymin=ymin, ymax=ymax,
                        x=xmean + 2 * xstd,
                        color=(0, 0.5, 1),
                        linewidth=2, ls='-')

            plt.axhline(#xmin=xmin, xmax=xmax,
                        y=ymean,
                        color=(0, 0.2, 0.5),
                        linewidth=2, ls='-')
            plt.axhline(#xmin=xmin, xmax=xmax,
                        y=ymean + 2 * ystd,
                        color=(0, 0.5, 1),
                        linewidth=2, ls='-')

            added_lines[2] = plt.axvline(#ymin=ymin, ymax=ymax,
                        x=xmed,
                        color=(0.8, 0.5, 0),
                        linewidth=2, ls='-')
            plt.axhline(#xmin=xmin, xmax=xmax,
                        y=ymed,
                        color=(0.8, 0.5, 0),
                        linewidth=2, ls='-')
            leg = ax.legend([added_lines[i] for i in range(3)], 
                            ['mean', 'mean + 2std', 'median'], 
                            loc='upper right')

        plt.savefig(filename)
        plt.close(1)
        return


    def make_intensity_density_plot(self, x, y, filename, x_name, y_name, 
                                    reduce_perc=None, 
                                    plot_grid=True, plot_lines=True, 
                                    additional_stats=None):

        if not reduce_perc is None:
            th_x = np.percentile(x, [reduce_perc])[0]
            th_y = np.percentile(y, [reduce_perc])[0]
            ind_x = x > th_x
            ind_y = y > th_y
            indices = np.multiply(ind_x, ind_y)
            x = x[indices]
            y = y[indices]
        
        perc = np.percentile(x, [0.1, 50, 99.9])
        xmin = perc[0]; xmax = perc[-1]; xmed = perc[1]
        xmean = np.mean(x)
        xstd = np.std(x)
        
        perc = np.percentile(y, [0.1, 50, 99.9])
        ymin = perc[0]; ymax = perc[-1]; ymed = perc[1]
        ymean = np.mean(y)
        ystd = np.std(y)
        
        # Calculate the point density
        xy = np.vstack([x,y])
        z = np.log(gaussian_kde(xy)(xy))

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        fig = plt.figure(1)
        ax = plt.subplot(1,1,1)

        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title('Density plot: %s vs. %s' % (x_name, y_name))
        cax = ax.scatter(x, y, c=z, s=10, edgecolor='')
        cbar = plt.colorbar(cax)
        
        axis = [xmin - (xmax - xmin) * 0.05,
                xmax + (xmax - xmin) * 0.05,
                ymin - (ymax - ymin) * 0.05,
                ymax + (ymax - ymin) * 0.05]
        plt.axis(axis)

        #print '%s:\t%.2f, %.2f\t%.2f,%.2f' % (filename, xmin, xmax, ymin, ymax)
        #pdb.set_trace()

        #plt.colorbar()
        if plot_grid:
            plt.grid(b=True, which='major', linewidth=1)
        if plot_lines:
            added_lines = {}
            added_lines[0] = plt.axvline(#ymin=ymin, ymax=ymax,
                        x=xmean,
                        color=(0, 0.2, 0.5),
                        linewidth=2, ls='-')
            added_lines[1] = plt.axvline(#ymin=ymin, ymax=ymax,
                        x=xmean + 2 * xstd,
                        color=(0, 0.5, 1),
                        linewidth=2, ls='-')

            plt.axhline(#xmin=xmin, xmax=xmax,
                        y=ymean,
                        color=(0, 0.2, 0.5),
                        linewidth=2, ls='-')
            plt.axhline(#xmin=xmin, xmax=xmax,
                        y=ymean + 2 * ystd,
                        color=(0, 0.5, 1),
                        linewidth=2, ls='-')

            added_lines[2] = plt.axvline(#ymin=ymin, ymax=ymax,
                        x=xmed,
                        color=(0.8, 0.5, 0),
                        linewidth=2, ls='-')
            plt.axhline(#xmin=xmin, xmax=xmax,
                        y=ymed,
                        color=(0.8, 0.5, 0),
                        linewidth=2, ls='-')
            legend_lines = ['mean', 'mean + 2std', 'median']
            
            if not additional_stats is None:
                added_lines[3] = plt.axhline(#xmin=xmin, xmax=xmax,
                            y=additional_stats[y_name]['mean'],
                            color=(0.5, 0, 0.5),
                            linewidth=2, ls='-')
                added_lines[4] = plt.axhline(#xmin=xmin, xmax=xmax,
                            y=additional_stats[y_name]['mean'] + 
                            2 * additional_stats[y_name]['std'],
                            color=(0.8, 0, 0.4),
                            linewidth=2, ls='-')
                added_lines[5] = plt.axhline(#xmin=xmin, xmax=xmax,
                            y=additional_stats[y_name]['median'],
                            color=(1, 0, 0.2),
                            linewidth=2, ls='-')

                added_lines[3] = plt.axvline(#ymin=ymin, ymax=ymax,
                            x=additional_stats[x_name]['mean'],
                            color=(0.5, 0, 0.5),
                            linewidth=2, ls='-')
                added_lines[4] = plt.axvline(#ymin=ymin, ymax=ymax,
                            x=additional_stats[x_name]['mean'] + 
                            2 * additional_stats[x_name]['std'],
                            color=(0.8, 0, 0.4),
                            linewidth=2, ls='-')
                added_lines[5] = plt.axvline(#ymin=ymin, ymax=ymax,
                            x=additional_stats[x_name]['median'],
                            color=(1, 0, 0.2),
                            linewidth=2, ls='-')
                legend_lines += ['mean', 'mean + 2std', 'median']
            leg = ax.legend([added_lines[i] for i in range(len(added_lines))], 
                            legend_lines, 
                            loc='upper right')


        plt.savefig(filename)
        plt.close(1)
        return

    def get_background_mask(self, ws=None, fissure_image=None):
        if ws is None:
            ws = self.cell_detector.get_image(False)
        if fissure_image is None:
            fissure_image = self.fissure.get_image(False)
        background_signal = ws == 0
        background_signal = opening(background_signal, diamond(1))
        background_signal[fissure_image > 0] = 0
        return background_signal
    
    def get_bakground_signal(self, image, ws=None, fissure_image=None):
        background_mask = self.get_background_mask(ws, fissure_image)
        xvec = image[background_mask]
        background_stats = {
            'mean': np.mean(xvec),
            'std': np.std(xvec),
            'median': np.median(xvec), 
            }
        return background_stats
    
    def randomize_image(self, image, method, 
                        fissure_image, ws=None):
        img = image.copy()
        img[fissure_image>0] = 0
        if method=='flip':
            image_randomized = np.flip(img, axis=0)
        elif method=='reorder':
            image_randomized = permutation(img)
        elif method=='mask_reorder':
            if ws is None:
                raise ValueError("For method %s, I need a mask image (ws)" % method)
            mask = ws > 0
            values = image[mask]
            perm_values = permutation(values)
            image_randomized = np.zeros(image.shape,
                                        dtype=image.dtype)
            image_randomized[mask] = perm_values
        else:
            raise ValueError('Method %s is not implemented' % method)
        
        return image_randomized
    
    def cut_for_method(self, population=3, method='spot_detection'):
        coord_out_folder = os.path.join(self.settings.output_folder,
                                        'coordinates')
        coord_filename = os.path.join(coord_out_folder, 
                                      'coordinates__%s__%s__population%i.pickle' % 
                                            (self.settings.dataset, method, population))
        output_folder = os.path.join(self.settings.output_folder, 
                                     'cutouts_%s' % method,
                                     'population_%i' % population)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        self.cut_from_file(coord_filename, output_folder, population)
        return

    def make_galery_images(self, population, start_threshold, nb_rows=40, nb_cols=10, 
                           window_size=61, method='raw'):
        output_folder = os.path.join(self.settings.output_folder, 
                                     'threshold_galleries')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
            
        nb_rows_max = nb_rows
        
        perc_for_vis = [10, 99.9]
        ws = self.cell_detector.get_image(False)
        ws_color = self.make_random_colors(ws)
        if window_size % 2 == 0:
            window_size = window_size + 1 
        radius = (window_size - 1) / 2
        nb_samples_max = nb_rows * nb_cols
        
        print('population %i' % population)
        
        img, channel_names = self.get_population(population)
        
        mean_fraction = 0.8 # meaning that we take only 1 - mean_fraction for calculating the mean
        for i in channel_names:
            channel_name = channel_names[i]

            if method=='raw':
                image = img[:,:,i]
            elif method=='median':
                image = median(img[:,:,i], disk(1))
            elif method=='pixel_noise':
                image = self.sp.remove_individual_pixels(img[:,:,i])
            elif method=='salt_noise':
                image = self.sp.remove_salt_noise(img[:,:,i])
            
#                image = img[:,:,i]
            percentiles = np.percentile(image, perc_for_vis)
            min_val = percentiles[0]
            max_val = percentiles[1]
            
            
            #background_stats[channel_name] = self.get_bakground_signal(image, ws, fissure_image)
            props = regionprops(ws, image)
            intensities = np.array([props[k]['mean_intensity'] for k in range(len(props))])
            centers = np.array([props[k]['centroid'] for k in range(len(props))])
            areas = np.array([props[k]['area'] for k in range(len(props))])
            #intensities_raw = np.array([np.mean(sorted(prop.intensity_image[prop.image])[(np.ceil(mean_fraction*prop.area)).astype(np.uint):])
            #               for prop in props])
            
            idx = intensities.argsort()
            intensities = intensities[idx]
            centers = centers[idx,:].astype(np.int64)
            areas = areas[idx]
            #intensities_raw = intensities_raw[idx]

            try:
                start_index = np.where(intensities>start_threshold[channel_name])[0][0]
                k = 0
            except:
                print('a problem occurred in channel %s' % channel_name)
                print('threshold: %f' % start_threshold[channel_name])
                print('number of cells exceeding this threshold: %i' % (np.sum(intensities > start_threshold[channel_name])))
                raise ValueError("no intensity exceeded the threshold.")
            
                        # find nb_samples and nb_rows
            nb_samples = min(nb_samples_max, len(intensities) - start_index)
            nb_rows = min(nb_rows_max, nb_samples / nb_cols)
            rgb_image = 255 * np.ones((nb_rows * window_size, nb_cols * window_size, 3))

            for t_row in range(nb_rows):
                for t_col in range(nb_cols):
                    
                    index = start_index + k
                    if index > len(intensities):
                        break

                    row_, col_ = centers[index]
                    row = np.rint(row_).astype(np.int)
                    col = np.rint(col_).astype(np.int)
                    ws_label = ws[row, col]
                    
                    y1 = max(0, row-radius)
                    y2 = min(row+radius, ws.shape[0])
                    x1 = max(0, col-radius)
                    x2 = min(col+radius, ws.shape[1])
                    height = np.rint(y2-y1).astype(np.int)
                    width= np.rint(x2-x1).astype(np.int)
                    
                    
                    sample_image = image[y1:y2, x1:x2]
                    offset_y = t_row*window_size
                    offset_x = t_col*window_size
                    #pdb.set_trace()
                    rgb_image[offset_y:(offset_y + height),
                              offset_x:(offset_x + width),
                              :] = grey2rgb(255.0 * self.sp.normalize_minmax(sample_image, min_val=min_val, max_val=max_val))
                    

                    mask_img = ws[y1:y2,x1:x2] == ws_label
                    
                    mask_img = mask_img.astype(np.uint8)
                    mask_img = dilation(mask_img, disk(1)) - mask_img
                    indices = np.where(mask_img)

                    new_indices = (indices[0] + offset_y, indices[1] + offset_x)
                    rgb_image[new_indices] = (255, 0, 0)

                    k += 1

            # writing the output
            filename = os.path.join(output_folder, 
                                    'gallery_population%i_%s.png' % (population, channel_name))
            skimage.io.imsave(filename, rgb_image.astype(np.uint8))

            # write with matplotlib
            new_filename = filename.replace('gallery_population', 'matplotlib_gallery_population').replace('.png', '.pdf')
            my_dpi = 200
            h = rgb_image.shape[0]
            w = rgb_image.shape[1]
            h = np.ceil(h / my_dpi).astype(np.int)
            w = np.ceil(w / my_dpi).astype(np.int)

            fig = plt.figure(frameon=False)
            fig.set_size_inches(w, h)
            #fig.set_size_inches(w,h)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            start_index = np.where(intensities>start_threshold[channel_name])[0][0]
            ax.imshow(rgb_image.astype(np.uint8), aspect='auto', interpolation='none')
            k = 0
            nb_cells = len(intensities)
            for t_row in range(nb_rows):
                for t_col in range(nb_cols):
                    index = start_index + k
                    offset_y = t_row*window_size + 0.12 * window_size
                    offset_x = t_col*window_size + 0.02 * window_size
                    
                    plt.text(offset_x, offset_y, "%.3f   (%i)" % (intensities[index], 
                                                                  nb_cells-index), color='orange',
                             fontsize=2)
                    #rgb_color[offset_y, offset_x]
                    k += 1
            fig.savefig(new_filename, dpi=my_dpi)
            

        return 

    def cut_from_file(self, filename, output_folder, population,
                      radius=30, intensity_channel_name='CXCR5', 
                      plot_outline=True, grid=False, 
                      write_color_maps=True, 
                      use_gauss=False):
        perc_for_vis = [10, 99.9]
        ws = self.cell_detector.get_image(False)
        ws_color = self.make_random_colors(ws)
        
        fp = open(filename, 'r')
        blobs = pickle.load(fp)
        fp.close()
        
        img, channel_names = self.get_population(population)
        img_dna, dna_channels = self.get_dna_images()
        
        props = regionprops(ws)
        centers = np.array([props[k]['centroid'] for k in range(len(props))])
        intensities = {'original': {},
                       'prefilter': {},
                       'gauss': {},
                       'laplace': {}}
        maxvals = {'original': {},
                   'prefilter': {},
                   'gauss': {},
                   'laplace': {}}
        minvals = {'original': {},
                   'prefilter': {},
                   'gauss': {},
                   'laplace': {}}
        gauss_img = np.zeros(img.shape)
        laplace_img = np.zeros(img.shape)
        pref_img = np.zeros(img.shape)
        color_maps = {}
        for i, channel in channel_names.iteritems():
            image = img[:,:,i]
            props = regionprops(ws, image)
            intensities['original'][channel] = np.array([props[k]['mean_intensity'] for k in range(len(props))])
            #perc = np.percentile(intensities['original'][channel],perc_for_vis)
            perc = np.percentile(image,perc_for_vis)
            maxvals['original'][channel] = perc[1]
            minvals['original'][channel] = perc[0]

            pref_img[:,:,i] = self.remove_salt_noise(image)
            props = regionprops(ws, pref_img[:,:,i])
            intensities['prefilter'][channel] = np.array([props[k]['mean_intensity'] for k in range(len(props))])
            #perc = np.percentile(intensities['prefilter'][channel],perc_for_vis)
            perc = np.percentile(pref_img[:,:,i],perc_for_vis)
            maxvals['prefilter'][channel] = perc[1]
            minvals['prefilter'][channel] = perc[0]

            gauss_img[:,:,i] = gaussian(image, sigma=self.settings.spot_detection['sigma'])
            #gauss_img[:,:,i] = gaussian(image, sigma=1.0)
            props = regionprops(ws, gauss_img[:,:,i])
            intensities['gauss'][channel] = np.array([props[k]['mean_intensity'] for k in range(len(props))])
            #perc = np.percentile(intensities['gauss'][channel],perc_for_vis)
            perc = np.percentile(gauss_img[:,:,i], perc_for_vis)
            maxvals['gauss'][channel] = perc[1]
            minvals['gauss'][channel] = perc[0]

            laplace_img[:,:,i] = laplace(gauss_img[:,:,i], 3)
            props = regionprops(ws, laplace_img[:,:,i])
            intensities['laplace'][channel] = np.array([props[k]['mean_intensity'] for k in range(len(props))])
            #perc = np.percentile(intensities['laplace'][channel],perc_for_vis)
            perc = np.percentile(laplace_img[:,:,i], perc_for_vis)
            maxvals['laplace'][channel] = perc[1]
            minvals['laplace'][channel] = perc[0]

            #color_maps[channel] = self.make_linear_colors(ws, image, N=100)
            if use_gauss:
                color_maps[channel] = self.make_linear_colors(ws, gauss_img[:,:,i], N=100)
            else:
                color_maps[channel] = self.make_linear_colors(ws, image, N=100)
            
            if write_color_maps:
                color_map_filename = os.path.join(output_folder, 'color_map_%s_%s_%i.png' % (self.settings.dataset,
                                                                                channel,
                                                                                population))
                skimage.io.imsave(color_map_filename, color_maps[channel])
                print('wrote color map for %s: %s' % (channel, color_map_filename))
        
        # for DNA image
        dna_channel = dna_channels[0]
        props = regionprops(ws, img_dna[:,:,0])
        intensities_dna = np.array([props[k]['mean_intensity'] for k in range(len(props))])
        perc = np.percentile(intensities_dna, perc_for_vis)
        maxvals['original'][dna_channel] = perc[1]
        minvals['original'][dna_channel] = perc[0]

        unlabeled_blobs = []
        K = len(channel_names)
        rescale_factor = 1

        for row_, col_ in blobs:
            row = np.rint(row_).astype(np.int)
            col = np.rint(col_).astype(np.int)
            ws_label = ws[row, col]
            if ws_label==0:
                y1 = max(0, row-radius)
                y2 = min(row+radius, ws.shape[0])
                x1 = max(0, col-radius)
                x2 = min(col+radius, ws.shape[1])
                height = np.rint(rescale_factor*(y2-y1)).astype(np.int)
                width= np.rint(rescale_factor*(x2-x1)).astype(np.int)
                nb_ch = len(channel_names)
                nb_tiles = nb_ch + 2
                rgb_image = np.zeros( (height, nb_tiles * width, 3) )

                xx, yy = circle_perimeter(int(radius), int(radius), 10, shape=(rgb_image.shape[1], rgb_image.shape[0]))

                for i, channel in channel_names.iteritems():
                    image = img[y1:y2,x1:x2,i]
                    rgb_image[:,i*width:(i+1)*width,:] = grey2rgb(255.0 * 
                                                                  self.sp.normalize_minmax(image,
                                                                                       min_val=minvals['original'][channel],
                                                                                       max_val=maxvals['original'][channel]))
                    
                    rgb_image[(yy, xx + i*width)] = (255.0, 0.0, 0.0)
                image = img_dna[y1:y2,x1:x2,0]
                rgb_image[:,nb_ch*width:(nb_ch+1)*width,:] = grey2rgb(255.0 *
                                                                  self.sp.normalize_minmax(image,
                                                                                       min_val=minvals['original'][dna_channel],
                                                                                       max_val=maxvals['original'][dna_channel]))
                rgb_image[(yy, xx + nb_ch*width)] = (255.0, 0.0, 0.0)
                rgb_image[:,(nb_ch+1)*width:(nb_ch+2)*width,:] = 255.0 * ws_color[y1:y2,x1:x2,:]
                rgb_image[radius, (nb_ch+1)*width + radius, :] = (255.0, 255.0, 255.0)
                unlabeled_blobs.append((row, col))
                image_filename = os.path.join(output_folder, 'unlabeled__%s__row%i__col%i.png' % 
                                              (os.path.splitext(os.path.basename(filename))[0], row, col))
                skimage.io.imsave(image_filename, rgb_image.astype(np.uint8))
                continue
            c_row, c_col = centers[ws_label-1]
            c_row = np.rint(c_row).astype(np.int)
            c_col = np.rint(c_col).astype(np.int)
            y1 = max(0, c_row-radius)
            y2 = min(c_row+radius, ws.shape[0])
            x1 = max(0, c_col-radius)
            x2 = min(c_col+radius, ws.shape[1])
            height = np.rint(rescale_factor*(y2-y1)).astype(np.int)
            width= np.rint(rescale_factor*(x2-x1)).astype(np.int)
            
            # definition of the output image
            rgb_image = np.zeros( ( (K +1)* height, 5 * width, 3))
            
            if rescale > 1:
                img_sample = rescale(img[y1:y2,x1:x2,:], rescale_factor, order=0)
                gauss_sample = rescale(gauss_img[y1:y2,x1:x2,:], rescale_factor, order=0)
                laplace_sample = rescale(laplace_img[y1:y2,x1:x2,:], rescale_factor, order=0)
                pref_sample = rescale(pref_img[y1:y2,x1:x2,:], rescale_factor, order=0)
                dna_sample = rescale(img_dna[y1:y2,x1:x2,0], rescale_factor, order=0)
                ws_color_sample = rescale(ws_color[y1:y2,x1:x2,:], rescale_factor, order=0)
            else:
                img_sample = img[y1:y2,x1:x2,:]
                gauss_sample = gauss_img[y1:y2,x1:x2,:]
                laplace_sample = laplace_img[y1:y2,x1:x2,:]
                pref_sample = pref_img[y1:y2,x1:x2,:]
                dna_sample = img_dna[y1:y2,x1:x2,0]
                ws_color_sample = ws_color[y1:y2,x1:x2,:]
                
            for i, channel in channel_names.iteritems():
                if rescale > 1:
                    color_maps_sample = rescale(color_maps[channel][y1:y2,x1:x2,:], rescale_factor, order=0)
                else:
                    color_maps_sample = color_maps[channel][y1:y2,x1:x2,:]

                image = img_sample[:,:,i]
                image_gauss = gauss_sample[:,:,i]
                image_laplace = laplace_sample[:,:,i]
                image_pref = pref_sample[:,:,i]
                
                #gaussian(image, sigma=self.settings.spot_detection['sigma']*rescale_factor)
                #laplace_image = #laplace(gauss_img, 3)

                rgb_image[i*height:(i+1)*height,0:width,:] = grey2rgb(255.0 * 
                                                                      self.sp.normalize_minmax(image,
                                                                                               min_val=minvals['original'][channel],
                                                                                               max_val=maxvals['original'][channel]))
                rgb_image[i*height:(i+1)*height,width:2*width,:] = grey2rgb(255.0 * 
                                                                            self.sp.normalize_minmax(image_pref,
                                                                                                     min_val=minvals['prefilter'][channel],
                                                                                                     max_val=maxvals['prefilter'][channel]))

                rgb_image[i*height:(i+1)*height,2*width:3*width,:] = grey2rgb(255.0 * 
                                                                            self.sp.normalize_minmax(image_gauss,
                                                                                                     min_val=minvals['gauss'][channel],
                                                                                                     max_val=maxvals['gauss'][channel]))
                rgb_image[i*height:(i+1)*height,3*width:4*width,:] = grey2rgb(255.0 * 
                                                                              self.sp.normalize_minmax(image_laplace,
                                                                                                       min_val=minvals['laplace'][channel],
                                                                                                       max_val=maxvals['laplace'][channel]))
                rgb_image[i*height:(i+1)*height,4*width:5*width,:] = 255 * color_maps_sample
                
                rgb_image[K*height:(K+1)*height, 0:width, i] += 255.0 * self.sp.normalize_minmax(image,
                                                                                                 min_val=minvals['original'][channel],
                                                                                                 max_val=maxvals['original'][channel])
            rgb_image[K*height:(K+1)*height, width:2*width, :] = grey2rgb(255.0 * self.sp.normalize_minmax(dna_sample,
                                                                                                 min_val=minvals['original'][dna_channels[0]],
                                                                                                 max_val=maxvals['original'][dna_channels[0]]))
            rgb_image[K*height:(K+1)*height, 2*width:3*width, :] = grey2rgb(255.0 * self.sp.normalize_minmax(dna_sample,
                                                                                                 min_val=minvals['original'][dna_channels[0]],
                                                                                                 max_val=maxvals['original'][dna_channels[0]]))
            rgb_image[K*height:(K+1)*height, 4*width:5*width, :] = 255 * ws_color_sample
            
                #background_stats[channel_name] = self.get_bakground_signal(image, ws, fissure_image)
                #props = regionprops(ws, image)
                #intensities = np.array([props[k]['mean_intensity'] for k in range(len(props))])
                #centers = np.array([props[k]['centroid'] for k in range(len(props))])
                #labels = np.array(range(1, len(intensities) + 1))

            if plot_outline:
                mask_image = ws[y1:y2,x1:x2] == ws_label
                if rescale > 1:
                    mask_img = rescale(mask_image, rescale_factor, order=0)
                else:
                    mask_img = mask_image
                mask_img = mask_img.astype(np.uint8)
                mask_img = mask_img - erosion(mask_img, disk(1))
                indices = np.where(mask_img)
                #pdb.set_trace()
                for i in range(len(channel_names)):
                    for j in range(4):
                        new_indices = (indices[0] + i * height, indices[1] + j * width)
                        rgb_image[new_indices] = (255, 0, 0)

                # for the last row
                new_indices = (indices[0] + K * height, indices[1] + 2 * width)
                rgb_image[new_indices] = (255, 0, 0)

            #if grid:
                
            if intensity_channel_name in intensities['original']:
                if use_gauss:
                    mean_intensity = 100 * intensities['gauss'][intensity_channel_name][ws_label-1]
                else:
                    mean_intensity = 100 * intensities['original'][intensity_channel_name][ws_label-1]
                    print(mean_intensity)
            else:
                mean_intensity = 0
            mean_intensity = np.int(mean_intensity)
            image_filename = os.path.join(output_folder, '%s__%s__%i__row%i__col%i.png' % 
                                          (os.path.splitext(os.path.basename(filename))[0], 
                                           intensity_channel_name, mean_intensity, c_row, c_col))
            print(os.path.basename(image_filename))
            skimage.io.imsave(image_filename, rgb_image.astype(np.uint8))
        
        # output the unlabeled
        if len(unlabeled_blobs) > 0:
            rgb_image = self.sp.make_overlay_img(img_dna[:,:,0], unlabeled_blobs, color=(255, 0, 0))
            skimage.io.imsave(os.path.join(output_folder, 'unlabeled_whole_image.png'), 
                              rgb_image.astype(np.uint8))
        print('made cutouts for: %i / %i' % ((len(blobs) - len(unlabeled_blobs)), len(blobs)))
        print('DONE !')
        return

    def cut_examples(self, radius=30, N=10):
        ws = self.cell_detector.get_image(False)
        fissure_image = self.fissure.get_image(False)

        out_folder = os.path.join(self.settings.output_folder, 
                                  'cutouts')
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)

        processed = []
        vor_radius = 5
        for population in range(1,5):
            print('population %i' % population)
            img, channel_names = self.get_population(population)
            for i, channel in channel_names.iteritems():
                if channel in processed:
                    continue
                channel_folder = os.path.join(out_folder, channel)
                if not os.path.isdir(channel_folder):
                    os.makedirs(channel_folder)

                processed.append(channel)
                
                image = img[:,:,i]
                gauss_img = gaussian(image, sigma=self.settings.spot_detection['sigma'])
                laplace_image = laplace(gauss_img, 3)

                #background_stats[channel_name] = self.get_bakground_signal(image, ws, fissure_image)
                props = regionprops(ws, image)
                intensities = np.array([props[k]['mean_intensity'] for k in range(len(props))])
                centers = np.array([props[k]['centroid'] for k in range(len(props))])
                labels = np.array(range(1, len(intensities) + 1))
                
                idx = intensities.argsort()
                intensities = intensities[idx]
                centers = centers[idx,:].astype(np.int64)
                labels = labels[idx]
                
                for level in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 5.0]:

                    indices = intensities > level
                    if indices.sum() == 0:
                        print('stopped at level %f' % level)
                        break
                    centers = centers[indices]
                    labels = labels[indices]
                    intensities = intensities[indices]
                    for k in range(min(N, labels.shape[0])): 
                        row, col = centers[k]
                        x1 = max(0, row-radius)
                        x2 = min(row+radius, image.shape[0])
                        y1 = max(0, col-radius)
                        y2 = min(col+radius, image.shape[1])
                        cutout_img = image[x1:x2,y1:y2]
                        cutout_img = rescale(cutout_img, 4, order=0)

                        cutout_img = 20 * cutout_img
                        cutout_img[cutout_img > 255] = 255
                        cutout_img[cutout_img < 0] = 0
                        cutout_img = cutout_img.astype(np.uint8)

                        rgb_image = grey2rgb(cutout_img)
                        
                        mask_image = ws[x1:x2,y1:y2] == labels[k]
                        mask_img = rescale(mask_image, 4, order=0)
                        mask_img = mask_img.astype(np.uint8)
                        mask_img = mask_img - erosion(mask_img, disk(1))
                        #xx, yy = circle_perimeter(radius * 4, radius * 4,
                        #                          vor_radius * 4, 
                        #                          shape=(rgb_image.shape[1], rgb_image.shape[0]))
                        
                        #mask_img = grey2rgb(mask_img)
                        rgb_image[np.where(mask_img)] = (255, 0, 0)
                        #try:
                        #    rgb_image[yy, xx] = (1, 0, 0)
                        #except:
                        #    pdb.set_trace()
                        filename = os.path.join(channel_folder, 
                                                '%s_%03i_%i_%i_%i_%i_%i_%.2f.png' % (channel, 
                                                                                int(level * 10),
                                                                                labels[k],
                                                                                x1, x2, y1, y2,
                                                                                intensities[k]
                                                                                ))
                        skimage.io.imsave(filename, rgb_image)
        return
    
    # probability of observing nb_hits or more by pure chance
    # out of N samples with probabilities defined by nb_marginal_hits (vector).
    def calculate_probability(self, nb_marginal_hits, nb_hits, N):
        pvec = nb_marginal_hits / float(N)
        p = np.prod(pvec)
        pval = 1.0 - binom.cdf(nb_hits, N, p)
        return pval
 
    def all_combinations(self, arrays):
        arrays = [np.asarray(a) for a in arrays]
        shape = (len(x) for x in arrays)

        ix = np.indices(shape, dtype=int)
        ix = ix.reshape(len(arrays), -1).T

        res = np.zeros(ix.shape, dtype=np.float)
        for n, arr in enumerate(arrays):
            res[:, n] = arrays[n][ix[:, n]]

        return res

    def calc_pval_thresholds(self, population=1, ws=None, fissure_image=None):
        if ws is None:
            ws = self.cell_detector.get_image(False)
        if fissure_image is None:
            fissure_image = self.fissure.get_image(False)
      
        img, channel_names = self.get_population(population)
        channel_results = {}
  
        # percentiles: tested thresholds
        N = ws.max()
        marginal_number_vec = np.hstack((np.arange(200, 525, 25),
                                         np.arange(500, 3000, 100),
                                         np.arange(3000, 8000, 200)))
        perc_vec = 100 * (1 - marginal_number_vec.astype(np.float) / N)
        nb_thresholds = len(perc_vec)
        
        intensities = np.zeros((len(channel_names), N))
        random_intensities = np.zeros((len(channel_names), N))
        thresholds = np.zeros((len(channel_names), nb_thresholds))
        for i in channel_names:
            channel_name = channel_names[i]
            image = img[:,:,i]

            props = regionprops(ws, image)
            intensities[i] = np.array([props[k]['mean_intensity'] for k in range(len(props))])
            thresholds[i] = np.percentile(intensities[i], perc_vec)
            random_intensities[i] = np.random.permutation(intensities[i])
            
        res = {}
        threshold_combinations = self.all_combinations(thresholds)
        for i in range(len(threshold_combinations)):
            
            tt = threshold_combinations[i]
            tt_expanded = np.tile(tt, (intensities.shape[1], 1) ).T
            res_matrix = intensities>tt_expanded

            res_random_matrix = random_intensities>tt_expanded
            nb_joint_hits_random = np.sum(np.prod(res_random_matrix, axis=0))
            
            vec_marginal_hits = np.sum(res_matrix, axis=1)
            nb_joint_hits = np.sum(np.prod(res_matrix, axis=0))
            pval = self.calculate_probability(vec_marginal_hits, nb_joint_hits, N)
            res[tuple(tt.tolist())] = {
                'marginal': vec_marginal_hits,
                'joint': nb_joint_hits,
                'pval': pval,
                'random_joint_hits': nb_joint_hits_random,
                'hit_ratio': float(nb_joint_hits) / max(nb_joint_hits_random, 1),
                }
            sys.stdout.write("Calculating: %i / %i (%i %%)   \r" % (i, len(threshold_combinations), ((100 * i) / len(threshold_combinations)) ))
            sys.stdout.flush()
        print
        return res

    def detect_cell_population(self, population=1, ws=None, fissure_image=None, 
                               method='raw', th_method='px_noise', optimal_thresholds=None):
        if ws is None:
            ws = self.cell_detector.get_image(False)
        if fissure_image is None:
            fissure_image = self.fissure.get_image(False)
    
        img, channel_names = self.get_population(population)
        channel_results = {}
        
        if optimal_thresholds is None:
            optimal_thresholds = {
                'CD1c': 0.65,
                'CD3': 2.3,
                'CD11c': 1.0,
                'CD206': 0.7,
                'CD370': 0.8,
                'CXCR5': 0.83,
                'PD1': 0.75
                }
        
        print(optimal_thresholds)
        
        for i in channel_names:
            channel_name = channel_names[i]
            if method == 'raw': 
                image = img[:,:,i]
            elif method == 'median':
                maxval = img[:,:,i].max()
                if maxval > 0:
                    alpha = 255.0 / maxval
                else:
                    alpha = 1.0
                help_image = alpha * img[:,:,i]
                help_image[help_image<0] = 0
                help_image[help_image>255] = 255
                help_image = help_image.astype(np.uint8)
                image = median(help_image, disk(1))
                print('img ', img[:,:,i].max(), img[:,:,i].mean())
                print('help_img ', help_image.max(), help_image.mean())
                print('image ', image.max(), image.mean())
                image = image.astype(np.float64)
                image = 1 / alpha * image
                print('1 / alpha * image ', image.max(), image.mean())

            props = regionprops(ws, image)
            intensities = np.array([props[k]['mean_intensity'] for k in range(len(props))])
            centers = np.array([props[k]['centroid'] for k in range(len(props))])

            #coordinates = [properties[i]['centroid'] for i in range(len(properties))]

            if th_method=='manual':
                threshold = optimal_thresholds[channel_name]
            elif th_method=='px_noise':
                pixel_noise = self.get_salt_noise_image(img[:,:,i])
                mean_px_noise = np.mean(pixel_noise)
                std_px_noise = np.std(pixel_noise)
                threshold_px_noise = mean_px_noise + 2 * std_px_noise
                threshold = threshold_px_noise
            #threshold_px_noise = optimal_thresholds[channel_name]

            channel_results[channel_name] = {
                'above_threshold': intensities>threshold,
                'coordinates': centers}

        all_indices = np.array([channel_results[channel_name]['above_threshold'] 
                                for channel_name in channel_results])
        selected_indices = all_indices.prod(axis=0)
        
        # calculation of p-value
        marginal_vec = np.array([channel_results[ch]['above_threshold'].sum() for ch in channel_results.keys()])
        marginal_vec = marginal_vec
        nb_hits = selected_indices.sum()
        N = len(selected_indices)
        pval = self.calculate_probability(marginal_vec, nb_hits, N)
        print()
        print('*********************')
        print('p-value of selection:')
        print('number of cells: %i' % N)
        print('intersection: %i' % nb_hits)
        print('marginals (number of selected cells): %s' % str(marginal_vec))
        print('p-value = %f' % pval)
        print('*********************')
        print() 
        
        selected_centers = channel_results[channel_results.keys()[0]]['coordinates'][selected_indices>0] 
        rows = selected_centers[:,0].tolist()
        cols = selected_centers[:,1].tolist()
        coordinates = zip(rows, cols)
        return coordinates
        
    def test_thresholds(self, method='raw'):
        ws = self.cell_detector.get_image(False)
        fissure_image = self.fissure.get_image(False)
        
        out_folder = os.path.join(self.settings.output_folder, 
                                  'threshold_test')
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        
        channel_result = {}
        population_result = {}
        for population in range(1,5):
            print('population %i' % population)
            
            img, channel_names = self.get_population(population)

            for i in channel_names:
                channel_name = channel_names[i]
                #if channel_name in intensities:
                #    continue
                if method=='raw':
                    image = img[:,:,i]
                    image_randomized = self.randomize_image(img[:,:,i], 'mask_reorder', fissure_image, ws)
                    filename_addon = 'raw'
                elif method=='gauss':
                    image = gaussian(img[:,:,i], sigma=self.settings.spot_detection['sigma'])
                    temp = self.randomize_image(img[:,:,i], 'mask_reorder', fissure_image, ws)
                    image_randomized = gaussian(temp, sigma=self.settings.spot_detection['sigma'])
                    filename_addon = 'gauss'
                elif method=='laplace': 
                    gauss_img = gaussian(img[:,:,i], sigma=self.settings.spot_detection['sigma'])
                    image = laplace(gauss_img, 3)
                    temp = self.randomize_image(img[:,:,i], 'mask_reorder', fissure_image, ws)
                    gauss_img = gaussian(temp, sigma=self.settings.spot_detection['sigma'])
                    image_randomized = laplace(gauss_img, 3)
                    filename_addon = 'laplace'

                #background_stats[channel_name] = self.get_bakground_signal(image, ws, fissure_image)
                props = regionprops(ws, image)
                intensities = np.array([props[k]['mean_intensity'] for k in range(len(props))])
 
                props_rand = regionprops(ws, image_randomized)
                intensities_random = np.array([props_rand[k]['mean_intensity'] 
                                               for k in range(len(props_rand))])

                # thresholds
                # original distribution
                mean_val = np.mean(intensities)
                std = np.std(intensities)
                threshold = mean_val + 2 * std
                
                # log distribution
                log_vec = np.log(intensities - np.min(intensities) + 0.0001)
                mean_val_log = np.mean(log_vec)
                std_log_vec = np.std(log_vec)
                threshold_log = np.exp(mean_val_log + 2 * std_log_vec) + np.min(intensities) - 0.0001
                
                # randomized image
                mean_val_random = np.mean(intensities_random)
                std_random = np.std(intensities_random)
                threshold_random = mean_val_random + 2 * std_random

                # pixel noise
                pixel_noise = self.get_salt_noise_image(image)
                mean_px_noise = np.mean(pixel_noise)
                std_px_noise = np.std(pixel_noise)
                threshold_px_noise = mean_px_noise + 2 * std_px_noise
                
                # non-zero pixel noise
                vec = pixel_noise[pixel_noise > 0]
                mean_px_noise_nz = np.mean(vec)
                std_px_noise_nz = np.std(vec)
                threshold_px_noise_nz = mean_px_noise_nz + 2 * std_px_noise_nz

                thresholds = {
                    'standard': threshold, 
                    'log': threshold_log,
                    'random': threshold_random,
                    'px_noise': threshold_px_noise,
                    'px_noise_nz': threshold_px_noise_nz
                    }
                
                nb_selected = {}
                above_threshold = {}
                for th in thresholds:
                    nb_selected[th] = (intensities > thresholds[th]).sum()
                    above_threshold[th] = intensities > thresholds[th]
                
                channel_result[channel_name] = {'mean': mean_val,
                                                'std': std,
                                                'thresholds': thresholds,
                                                'nb_selected': nb_selected,
                                                'indices_above': above_threshold}
            
            optimal_thresholds = {
                'CD1c': 0.65,
                'CD3': 2.3,
                'CD11c': 1.0,
                'CD206': 0.7,
                'CD370': 0.8,
                'CXCR5': 0.83,
                'PD1': 0.75
                }
            for channel_name in channel_result:
                thresholds = sorted(channel_result[channel_name]['thresholds'].keys())
                values = [channel_result[channel_name]['thresholds'][x] for x in thresholds]
                tempStr = channel_name + '\t' + '\t'.join([str(optimal_thresholds[channel_name])] + 
                          ['%.2f' % x for x in values])
                print(tempStr)
            print(thresholds)
#             all_indices = np.array([channel_result[channel_name]['indices_above'] 
#                                     for channel_name in channel_result])
#             all_indices = all_indices.prod(axis=0)
#             number_of_selected = all_indices.sum()
#             population_result[population] = {'channels': channel_names.values(), 
#                                              'nb': number_of_selected}
#             
#         print
#         print 'CHANNEL LEVEL'
#         print '============='
#         for channel_name in channel_result.keys():
#             print '%s\t%i' % (channel_name, channel_result[channel_name]['nb_selected'])
#         print
#         print 'POPULATION LEVEL'
#         print '================'
#         
#         for population in population_result:
#             print '%i\t%s\t%i' % (population, ', '.join(population_result[population]['channels']),
#                                   population_result[population]['nb'])
        return

    
    def prepare_threshold_panels(self, method='raw', populations=None):
        if populations is None:
            populations = [1, 2, 3, 4]

        self.plot_thresholds(method=method, populations=populations)
        
        for population in populations: 
            start_threshold = self.settings.optimal_thresholds[self.settings.dataset]
#             start_threshold = {
#                 'CD11c': 1.5,
#                 'CD370': 0.4,
#                 'CD206': 0.3,
#                 'CD3': 5, 
#                 'CXCR5': 0.45,
#                 'PD1': 0.8,
#                 'CD1c': 0.3,
#                 'CD14': 1.2,
#                 'Bcl6': 0.3,
#                 'CD45': 13,
#                 }
            self.make_galery_images(population=population, nb_rows=800,
                                  start_threshold = start_threshold)
            
        return
    
    
    def plot_thresholds(self, method='raw', populations=None):
        ws = self.cell_detector.get_image(False)
        fissure_image = self.fissure.get_image(False)
        
        out_folder = os.path.join(self.settings.output_folder, 
                                  'plots', 'thresholds')
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)

        if populations is None:
            populations = [1, 2, 3, 4]
        channel_result = {}
        population_result = {}
        for population in populations:
            print('population %i' % population)
            
            img, channel_names = self.get_population(population)

            for i in channel_names:
                channel_name = channel_names[i]
                if method=='raw':
                    image = img[:,:,i]
                elif method=='median':
                    image = median(img[:,:,i], disk(1))
                elif method=='pixel_noise':
                    image = self.sp.remove_individual_pixels(img[:,:,i])
                elif method=='salt_noise':
                    image = self.sp.remove_salt_noise(img[:,:,i])
                    
                #background_stats[channel_name] = self.get_bakground_signal(image, ws, fissure_image)
                props = regionprops(ws, image)
                intensities = np.array([props[k]['mean_intensity'] for k in range(len(props))])
 
                K = 400
                maxval = np.max(intensities)
                minval = np.min(intensities)
                delta = (maxval - minval) / K
                bins = np.arange(minval, maxval + delta, delta)
                histo, limits = np.histogram(intensities, bins=bins)
                nb_cells = np.concatenate([np.flip(np.cumsum(np.flip(histo, 0)), 0), [0]])

                fig = plt.figure()
                plt.plot(limits, nb_cells, color='red')
                plt.title('number of selected cells: %s' % channel_name)
                plt.xlabel('mean value of %s' % channel_name)
                plt.ylabel('number of selected cells')
                #plt.axis([0, maxval + delta, 0, 2000])
                plt.grid(b=True, which='major', linewidth=1)
                
                filename = os.path.join(out_folder, 
                                        'number_selected_cells_population%i_%s_method_%s.png' % (population, channel_name, method))
                plt.savefig(filename)
                plt.close()
        return

    
    def plot_population_intensities(self,
                                    method='raw', 
                                    scatter_plots=False, 
                                    plot_histograms=True):
        ws = self.cell_detector.get_image(False)
        fissure_image = self.fissure.get_image(False)
        
        print('cells: %i' % ws.max())
        
        out_folder = os.path.join(self.settings.output_folder, 
                                  'plots', 'scatter_intensity')
        if not os.path.isdir(out_folder):
            print('making %s' % out_folder)
            os.makedirs(out_folder)
            
        for population in range(1,5):
            print('population %i' % population)
            intensities = {}
            additional_stats = {}
            img, channel_names = self.get_population(population)

            for i in range(len(channel_names)):
                channel_name = channel_names[i]
                if channel_name in intensities:
                    continue
                if method=='raw':
                    image = img[:,:,i]
                    image_randomized = self.randomize_image(img[:,:,i], 'mask_reorder', fissure_image, ws)
                    filename_addon = 'raw'
                elif method=='gauss':
                    image = gaussian(img[:,:,i], sigma=self.settings.spot_detection['sigma'])
                    temp = self.randomize_image(img[:,:,i], 'mask_reorder', fissure_image, ws)
                    image_randomized = gaussian(temp, sigma=self.settings.spot_detection['sigma'])
                    filename_addon = 'gauss'
                elif method=='laplace': 
                    gauss_img = gaussian(img[:,:,i], sigma=self.settings.spot_detection['sigma'])
                    image = laplace(gauss_img, 3)
                    temp = self.randomize_image(img[:,:,i], 'mask_reorder', fissure_image, ws)
                    gauss_img = gaussian(temp, sigma=self.settings.spot_detection['sigma'])
                    image_randomized = laplace(gauss_img, 3)

                    filename_addon = 'laplace'
                    
                #background_stats[channel_name] = self.get_bakground_signal(image, ws, fissure_image)
                props = regionprops(ws, image)
                intensities[channel_name] = np.array([props[k]['mean_intensity'] for k in range(len(props))])

                props_rand = regionprops(ws, image_randomized)
                intensities_random = np.array([props_rand[k]['mean_intensity'] 
                                               for k in range(len(props_rand))])
                additional_stats[channel_name] = {
                    'mean': np.mean(intensities_random), 
                    'std': np.std(intensities_random),
                    'median': np.median(intensities_random)
                    }
                print('plotting intensities for %s %s' % (method, channel_name))
                if plot_histograms:
                    self.plot_intensity_histogram(intensities[channel_name], 
                                                  intensities_random,
                                                  channel_name,
                                                  filename_addon=method)
                self.plot_intensity_pixel_noise(intensities[channel_name],
                                                image, channel_name)
            
            if scatter_plots:
                for i in range(len(channel_names) - 1):
                    for j in range(i+1, len(channel_names)):
                        filename = os.path.join(out_folder, 'intensity_distribution_population%i_%s_%s_%s_scatterplot.png' % 
                                                (population, channel_names[i], channel_names[j], filename_addon))
                        print('generating ... %s' % filename)
                        self.make_intensity_scatter_plot(intensities[channel_names[i]], 
                                                         intensities[channel_names[j]], 
                                                         filename, channel_names[i], channel_names[j],
                                                         plot_grid=True, plot_lines=True)#,
                                                         #additional_stats=additional_stats)
    
    
                        filename = os.path.join(out_folder, 'intensity_distribution_population%i_%s_%s_%s_density.png' % 
                                                (population, channel_names[i], channel_names[j], filename_addon))
                        print('generating ... %s' % filename)
                        self.make_intensity_density_plot(intensities[channel_names[i]], 
                                                         intensities[channel_names[j]], 
                                                         filename, channel_names[i], channel_names[j], 
                                                         plot_grid=True, plot_lines=False)#,
                                                         #additional_stats=additional_stats)
                    #raise ValueError('finished log')
                
            
        return

    def _analyze_all_populations(self, method='spot_detection'):
        res = {}
        for population in range(1,5):
            res[population] = self.analyze_population_distances(population, method=method)
        print('Population\t number of detected cells\t B-region\t T-region\t BG')
        for population in res:
            tempStr = 'population %i\t %i\t %.1f\t %.1f\t %.1f' % (population, 
                                                               res[population]['nb_cells'], 
                                                               res[population]['perc_B'] * 100, 
                                                               res[population]['perc_T'] * 100, 
                                                               res[population]['perc_BG'] * 100)
            print(tempStr)
        return

    def analyze_all_populations(self, method='intensity'):
        optimal_thresholds = self.settings.optimal_thresholds
        res = {}
        for population in range(1, 6):
            res[population] = self.analyze_population_distances(population, method=method,
                                                                optimal_thresholds=optimal_thresholds[self.settings.dataset],
                                                                th_method='manual')
        print('Population\t number of detected cells\t B-region\t T-region\t BG')
        for population in res:
            tempStr = 'population %i\t %i\t %.1f\t %.1f\t %.1f' % (population, 
                                                               res[population]['nb_cells'], 
                                                               res[population]['perc_B'] * 100, 
                                                               res[population]['perc_T'] * 100, 
                                                               res[population]['perc_BG'] * 100)
            print(tempStr)
            
        return


    def analyze_population_distances(self, population=1, 
                                     method='spot_detection', 
                                     write_coordinates=False,
                                     optimal_thresholds=None,
                                     th_method='manual'):
        background, b_region, t_region = self.read_region_images()
        fissure = self.get_fissure()

        b_dist = distance_transform_edt(b_region)
        t_dist = distance_transform_edt(t_region)
        bg_dist = distance_transform_edt(background)
        
        img, channel_names = self.get_population(population)
        if method=='spot_detection':
            coordinates = self.spot_detection_raw(img, channel_names, population)
        elif method=='intensity':
            coordinates = self.detect_cell_population(population, 
                                                      th_method=th_method,
                                                      optimal_thresholds=optimal_thresholds)
            # we need to cast this to integers. 
            coordinates = [(np.rint(x).astype(np.uint16), np.rint(y).astype(np.uint16)) for x,y in coordinates]
        else:
            raise ValueError('method not implemented')
    
        if write_coordinates:
            coord_out_folder = os.path.join(self.settings.output_folder,
                                            'coordinates')
            if not os.path.isdir(coord_out_folder):
                os.makedirs(coord_out_folder)
            coord_filename = os.path.join(coord_out_folder, 
                                          'coordinates__%s__%s__population%i.txt' % 
                                          (self.settings.dataset, method, population))
            fp = open(coord_filename, 'w')
            for row, col in coordinates:
                fp.write('%i, %i\n' % (row, col))
            fp.close()
            coord_filename = os.path.join(coord_out_folder, 
                                          'coordinates__%s__%s__population%i.pickle' % 
                                          (self.settings.dataset, method, population))
            fp = open(coord_filename, 'w')
            pickle.dump(coordinates, fp)
            fp.close()

        # remove coordinates inside the fissure
        filtered_coordinates = list(filter(lambda x: fissure[x] == 0, coordinates))
        print(len(coordinates), ' --> ', len(filtered_coordinates))
        coordinates = filtered_coordinates
        res = {
            'nb_cells': len(filtered_coordinates),
            }

        y = [int(point[0]) for point in coordinates]
        x = [int(point[1]) for point in coordinates]
        
        b_dist_small = rescale(b_dist, .25)
        b_dist_vec = b_dist_small[b_dist_small>0]

        t_dist_small = rescale(t_dist, .25)
        t_dist_vec = t_dist_small[t_dist_small>0]
        distances = b_dist[y, x]
        print('population detection method: ', method)

        nb_distances = max(len(distances), 1)
        print('percentage of cells in B-region: %.3f' % (float(len(distances[distances>0])) / nb_distances))
        self.plot_distance_histogram(distances[distances>0], b_dist_vec,
                                     filename='%s_population_%i_B_cell_region_distances_%s.png' % (self.settings.dataset, 
                                                                                                   population, 
                                                                                                   method),
                                     region_name='B-cell-region-pop%i' % population,
                                     subset_name='-'.join(channel_names.values()))
        res['perc_B'] = float(len(distances[distances>0])) / nb_distances
        
        distances = t_dist[y, x]
        nb_distances = max(len(distances), 1)
        print('percentage of cells in T-region: %.3f' % (float(len(distances[distances>0])) / nb_distances))
        self.plot_distance_histogram(distances[distances>0], t_dist_vec,
                                     filename='%s_population_%i_T_cell_region_distances_%s.png' % (self.settings.dataset,
                                                                                                   population,
                                                                                                   method),
                                     region_name='T-cell-region-pop%i' % population,
                                     subset_name='-'.join(channel_names.values()))
        res['perc_T'] = float(len(distances[distances>0])) / nb_distances
        
        output_folder = os.path.join(self.settings.output_folder, 'distance_maps_%s' % method)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        color = (255, 0, 0)
        region_image = np.zeros(b_region.shape, dtype=np.uint8)
        region_image[b_region>0] = 100
        region_image[t_region>0] = 200
        filename_addon = '-'.join(channel_names.values())
        filename = os.path.join(output_folder, 'region_population_%i_%s_overlay.png' % (population, filename_addon))
        overlay_img = self.sp.make_overlay_img(region_image, coordinates, color,
                                               perimeter=False)
        skimage.io.imsave(filename, overlay_img)

        
        distances = bg_dist[y, x]
        nb_distances = max(len(distances), 1)
        print('percentage of cells in background: %.3f' % (float(len(distances[distances>0])) / nb_distances))
        res['perc_BG'] = float(len(distances[distances>0])) / nb_distances
        
        #self.plot_distance_histogram(distances[distances>0], 
        #                             filename='%s_background_region_distances.png' % self.settings.dataset,
        #                             region_name='background-region',
                                     #subset_name='-'.join(channel_names.values())
        
        return res

    def get_population(self, population=1):
        if population == 1: # DC1 : CD370 + CD141 + HLADR
            #markers = ['CD11c', 'CD370']
            markers = ['CD370', 'CD141', 'HLADR']
            si = SequenceImporter(markers)
            img, channel_names = si(self.settings.input_folder)
        elif population == 2: # Macrophage : CD45 + CD14 + CD11c
            # there will be either CD14 or CD206 missing. 
            markers = ['CD11c', 'CD14', 'CD206', 'CD45']
            si = SequenceImporter(markers)
            img, channel_names = si(self.settings.input_folder)
        elif population == 3: # Tfh : Bcl6 + PD1 + CD3 
            # CD279 is replaced by PD1
            #markers = ['CD4', 'IL21', 'PD1', 'CD3']
            #markers = ['CD3', 'PD1', 'CXCR5']
            #markers = ['CD4', 'IL21', 'PD1', 'CD3']
            markers = ['CD3', 'PD1', 'Bcl6']
            si = SequenceImporter(markers)
            img, channel_names = si(self.settings.input_folder)
        elif population == 4: # DC2 : CD11c + CD1c + HLADR
            #markers = ['CD1c', 'CD11c']
            markers = ['CD1c', 'CD11c', 'HLDAR']
            si = SequenceImporter(markers)
            img, channel_names = si(self.settings.input_folder)
        if population == 5: # pDC : CD123 + CD303 + HLADR
            #markers = ['CD11c', 'CD370']
            markers = ['CD123', 'CD303', 'HLADR']
            si = SequenceImporter(markers)
            img, channel_names = si(self.settings.input_folder)

        else:
            img = None
            channel_names = []
        return img, channel_names

