import buildW
def spectral_Pb(mPb, orig_sz):
    """
    Global contour cue from local mPb
    """
    n_vec = 17
    tx, ty = mPb.shape
    l1 = np.zeros([tx+1, ty])
    l1[1:, :] = mPb
    l2 = np.zeros([tx, ty+1])
    l2[:, 1:] = mPb

    # build the pairwise affinity matrix
    val, I, J = buildW(l1, l2)
    
