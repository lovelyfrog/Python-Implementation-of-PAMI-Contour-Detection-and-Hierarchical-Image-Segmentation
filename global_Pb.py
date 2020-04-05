from multiscale_Pb import *
from spectral_Pb import *
import numpy as np
from PIL import Image

im = Image.open('101087.jpg')

im = np.array(im)

# scale im from 0-255 to 0-1
im = im / 255
tx, ty, nchan = im.shape
orig_sz = [tx, ty] 

# resize factor in (0, 1], to speed up eigenvector computation
rsz = 1

# default feature weights
weights = [0, 0, 0.0039, 0.0050, 0.0058, 0.0069, 0.0040, 0.0044, 0.0049, 0.0024, 0.0027, 0.0170, 0.0074]

# multiscale Pb
mPb_nmax, bg1, bg2, bg3, cga1, cga2, cga3, cgb1, cgb2, cgb3, tg1, tg2, tg3, textons = multiscale_Pb(im, rsz)

# spectral Pb
sPb = spectral_Pb(mPb_nmax, [tx, ty])

# global Pb
gPb = np.zeros_like(sPb)
for o in range(len(bg1)):
    l1 = weights[0] * bg1[o]
    l2 = weights[1] * bg2[o]
    l3 = weights[2] * bg3[o]

    a1 = weights[3] * cga1[o]
    a2 = weights[4] * cga2[o]
    a3 = weights[5] * cga3[o]

    b1 = weights[6] * cgb1[o]
    b2 = weights[7] * cgb2[o]
    b3 = weights[8] * cgb3[o]

    t1 = weights[9] * tg1[o]
    t2 = weights[10] * tg2[o]
    t3 = weights[11] * tg3[o]

    sc = weights[12] * sPb[o]

    gPb[o] = l1 + a1 + b1 + t1 + l2 + a2 + b2 + t2 + l3 + a3 + b3 + t3 + sc

return gPb