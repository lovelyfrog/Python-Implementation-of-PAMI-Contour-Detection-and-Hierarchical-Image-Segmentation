import scipy.io
import numpy as np
from PIL import Image

def contours2ucm(gPb):
    im = np.max(gPb, 2)
    im = (im - np.min(im)) / (np.max(im) - np.min(im))


# gPb_mat['gPb_orient'] has size of [481, 321, 8]
gPb_mat = scipy.io.loadmat("101087_gPb.mat")
gPb = gPb_mat['gPb_orient']
im0 = Image.fromarray(gPb[:, :, 0] * 255)
im5 = Image.fromarray(gPb[:, :, 5] * 255)
# im0.show()
# im5.show()
contours2ucm(gPb)