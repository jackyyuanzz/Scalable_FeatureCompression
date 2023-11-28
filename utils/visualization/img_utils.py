import numpy as np

def normalize(im, im_min = None, im_max = None):
    if im_min == None or im_max == None:
        im_max = np.max(im)
        im_max = np.percentile(im, 99)
        im_min = np.min(im)
    im_norm = (im-(im_min))/(im_max-im_min)
    return im_norm