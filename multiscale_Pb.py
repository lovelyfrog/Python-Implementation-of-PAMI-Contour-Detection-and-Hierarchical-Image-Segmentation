from PIL import Image
import numpy as np
from gradient_mPb import *

def multiscale_Pb(im, rsz):
    """
    compute local contour cues of an image.
    im: 3 channel image arrays with value scaled between (0,1].
    rsz: resize factor.
    """
    [bg1, bg2, bg3, cga1, cga2, cga3, cgb1, cgb2, cgb3, tg1, tg2, tg3, textons] = mPb(im)
    

im = Image.open('101087.jpg')

im = np.array(im)

# scale im from 0-255 to 0-1
im = im / 255
tx, ty, nchan = im.shape
orig_sz = [tx, ty] 

# default feature weights
weights = [0, 0, 0.0039, 0.0050, 0.0058, 0.0069, 0.0040, 0.0044, 0.0049, 0.0024, 0.0027, 0.0170, 0.0074]

# resize factor in (0, 1], to speed up eigenvector computation
rsz = 1

