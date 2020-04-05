from buildW import *
import numpy as np
from scipy.linalg import eig
import lib_image
from scipy import ndimage
from PIL import Image

def spectral_Pb(mPb, orig_sz):
    """
    Global contour cue from local mPb
    """
    n_vec = 16
    num_orient = 8
    tx, ty = mPb.shape

    # build the pairwise affinity matrix, whose size will be [tx*ty, tx*ty]
    W = buildW(mPb)

    # compute D, must use sparse matrix (to be done)
    sum_W = np.sum(W, 1)
    D = np.zeros_like(W)
    for i in range(len(sum_W)):
        D[i, i] = sum_W[i]

    eigvals, eigvecs = eig(D - W, W)

    # reshape eigvecs
    vect = np.zeros([16, tx, ty])
    for i in range(n_vec):
        vect[i] = np.reshape(eigvecs[i], [tx, ty])
    
    # compute filters
    filters = lib_image.texton_filters(num_orient, 1)

    # compute sPb
    sPb = np.zeros([num_orient, tx, ty])
    for o in range(len(filters)):
        ft = filters[o]
        for v in range(n_vec):
            if eigvals[v] > 0:
                vec = vect[v] / np.sqrt(eigvals[i])
                sPb[o] += ndimage.filters.convolve(vec, ft)

    return sPb


im = Image.open('101087.jpg')
im = np.array(im)
im = im / 255
tx, ty, nchan = im.shape
orig_sz = [tx, ty] 

sPb = spectral_Pb(im[:, :, 0], [tx, ty])
print(sPb)

