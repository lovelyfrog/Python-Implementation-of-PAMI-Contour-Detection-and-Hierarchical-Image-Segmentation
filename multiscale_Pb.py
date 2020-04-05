from PIL import Image
import numpy as np
from gradient_mPb import *
from scipy.signal import convolve2d

def multiscale_Pb(im, rsz):
    """
    compute local contour cues of an image.
    im: 3 channel image arrays with value scaled between (0,1].
    rsz: resize factor.
    """
    [textons, bg1, bg2, bg3, cga1, cga2, cga3, cgb1, cgb2, cgb3, tg1, tg2, tg3] = gradient_mPb(im)

    # default feature weights
    weights = [0, 0, 0.0039, 0.0050, 0.0058, 0.0069, 0.0040, 0.0044, 0.0049, 0.0024, 0.0027, 0.0170, 0.0074]

    # smooth cues
    gtheta = [1.5708, 1.1781, 0.7854, 0.3927, 0, 2.7489, 2.3562, 1.9635]
    filters = make_filters([3, 5, 10, 20], gtheta)

    # compute mPb at full scale
    mPb_all = []
    for o in range(len(tg1)):
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

        mPb_all.append(l1 + a1 + b1 + t1 + l2 + a2 + b2 + t2 + l3 + a3 + b3 + t3)
    
    mPb_all = np.array(mPb_all)
    # non-maximum suppression
    mPb_nmax = np.max(mPb_all, 0)
    # compute mPb_nmax resized if necessary


    return mPb_nmax, bg1, bg2, bg3, cga1, cga2, cga3, cgb1, cgb2, cgb3, tg1, tg2, tg3, textons



def make_filters(radius_list, gtheta):

    d = 2
    # initialiaze filters 
    filters = [[] for i in radius_list]
    for ft in filters:
        for i in gtheta:
            ft.append(0)
    
    # compute filters
    for r in range(len(radius_list)):
        for t in range(len(gtheta)):
            ra = radius_list[r]
            theta = gtheta[t]
            rb = ra / 4
            ra = max(1.5, ra)
            rb = max(1.5, rb)
            ira2 = 1 / ra**2
            irb2 = 1 / rb**2
            wr = np.floor(max(ra, rb))
            wd = 2*wr + 1
            sint = np.sin(theta)
            cost = np.cos(theta)

            # compute linear filters for coefficients
            ft = np.zeros([wd, wd, d+1])
            xx = np.zeros([2*d+1, 1])

            for u in range(-wr, wr+1):
                for v in range(-wr, wr+1):
                    ai = -u*sint + v*cost
                    bi = u*cost + v*sint
                    if ai*ai*ira2 + bi*bi*irb2 > 1:
                        continue
                    xx = xx + np.cumprod(np.append(1, ai+np.zeros(2*d)))

            A = np.zero(d+1, d+1)
            for i in range(0, d+1):
                A[:, i] = xx[i:i+d+1]

            for u in range(-wr, wr+1):
                for v in range(-wr, wr+1):
                    ai = -u*sint + v*cost
                    bi = u*cost + v*sint
                    if (ai*ai*ira2 + bi*bi*irb2 > 1):
                        continue
                    yy = np.cumprod(np.append(1, ai+np.zeros(d)))
                    ft[v+wr, v+wr, :] = np.linalg.solve(A, yy)

            filters[r, t] = ft

    return filters

def fitparab(z, ra, rb, theta, ft):
    """
    Fit cylindrical parabolas to elliptical patches of z at each pixel
    """
    a = convolve2d(z, ft[:, :, 0], 'same')
    a = savagol_border(a, z, ra, rb, theta)
    return a

def savagol_border(a, z, ra, rb, theta):
    return None



