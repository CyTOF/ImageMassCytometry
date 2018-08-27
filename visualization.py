import numpy as np

from skimage.color import grey2rgb
from skimage.morphology import erosion, disk

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
