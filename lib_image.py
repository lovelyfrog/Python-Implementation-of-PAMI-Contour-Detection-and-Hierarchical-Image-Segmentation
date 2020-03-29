import numpy as np
from scipy.signal import hilbert
from scipy import ndimage
from sklearn import cluster

def gaussian(sigma=1, deriv=0, hlbrt=False): 
    """
    Gaussian kernel(1D)
    The length of the returned vector is 2*support+1.
    The support defaults to 3*sigma.
    The kernel is normalized to have unit L1 norm.
    If returning a 1st or 2nd derivative, the kernel has zero mean
    """
    support = np.ceil(3*sigma).astype(int)
    support = support.astype(int)
    
    return gaussian_support(sigma, deriv, hlbrt, support)

def gaussian_support(sigma, deriv, hlbrt, support):
    # Enlarge support so that hilbert transform can be done efficiently
    support_big = support
    if (hlbrt):
        support_big = 1
        temp = support
        while (temp > 0):
            support_big *= 2
            temp //= 2 
    
    # compute constants
    sigma2_inv = 1 / (sigma*sigma)
    
    # compute gaussian (or gaussian derivative)
    size = 2*support_big + 1
    m = np.zeros([size, 1])
    x = -support_big
    if (deriv == 0):
        for n in range(0, size):
            m[n][0] = np.exp(x*x*(-0.5*sigma2_inv))
            x += 1
    elif (deriv == 1):
        for n in range(0, size):
            m[n][0] = np.exp(x*x*(-0.5*sigma2_inv)) * (-x)
            x += 1
    elif (deriv == 2):
        for n in range(0, size):
            m[n][0] = np.exp(x*x*(-0.5*sigma2_inv)) * (x*x*sigma2_inv - 1)
            x += 1
    else:
        raise ValueError("Only derivatives 0, 1, 2 supported")
        
    # take hilbert transform (if requested)
    if (hlbrt):
        # grab power of two sized submatrix (ignore last element)
        m_ignore = m[:-1]
        # grab desired submatrix after hilbert transform
        start = support_big - support
        end = start + 2*support
        m = hilbert(m)[start:end+1]
    
    # make zero mean (if returning derivative)
    if (deriv > 0):
        m -= np.mean(m)
        
    # make unit L1 norm
    m /= np.sum(np.abs(m))
    return m

def gaussian_2D(sigma_x, sigma_y, orient, deriv, hlbrt):
    """
    Gaussian kernel (2D)
    Specify the standard deviation along each axis, and the orientation.
    In addition, optionally specify the support. The support defaults to 3*sigma (automatically adjusted for rotation)
    The 1st and 2nd deriavative and/or the hilbert transform can optionally be taken along the y-axis prior to rotation.
    The kernel is normalized to have unit L1 norm
    If returning a 1st or 2nd derivative, the kernel has zero mean
    """
    # compute support from rotated corners of original support box
    support_x = np.ceil(3*sigma_x).astype(int)
    support_y = np.ceil(3*sigma_y).astype(int)
    return gaussian_2D_support(sigma_x, sigma_y, orient, deriv, hlbrt, support_x, support_y)

def gaussian_2D_support(sigma_x, sigma_y, orient, deriv, hlbrt, support_x, support_y):

    support_x_rot = support_x_rotated(support_x, support_y, orient)
    support_y_rot = support_y_rorated(support_x, support_y, orient)
    support_x_rot = support_x_rotated(support_x_rot, support_y_rot, -orient)
    support_y_rot = support_y_rorated(support_x_rot, support_y_rot, -orient)

    # compute 1D kernels
    mx = gaussian_support(sigma_x, 0, False, support_x_rot)
    my = gaussian_support(sigma_y, deriv, hlbrt, support_y_rot)

    # compute 2D kernel from product of 1D kernels
    m = mx * my.reshape(my.shape[0])

    # rotate 2D kernel by orientation
    m = rotate_2D_crop(m, orient, 4*support_x_rot+2, 4*support_y_rot+2)
    
    # make zero mean (if returning derivative)
    if (deriv > 0):
        m -= np.mean(m)
        
    # make unit L1 norm
    m /= np.sum(np.abs(m))
    return m

def rotate_2D_crop(m, orient, size_x, size_y):
    """
    rotate matrix by orientation
    """

    m_rotate = np.zeros([size_x, size_y])
    # center of m
    ox, oy = m.shape[1] // 2, m.shape[0] // 2

    # center of m_rotate
    ox_r, oy_r = size_y // 2, size_x // 2

    # compute the matrix after rotation
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            x_coord = j - ox
            y_coord = oy - i
            dist = np.sqrt((x_coord)**2 + (y_coord)**2)
            # caution! np.arctan2(y, x)
            theta = np.arctan2(y_coord, x_coord) + orient
            new_y = np.round(dist * np.cos(theta)).astype(int)
            new_y += ox_r
            new_x = -np.round(dist * np.sin(theta)).astype(int)
            new_x += oy_r
            m_rotate[new_x][new_y] = m[i][j]

    x_start = (m_rotate.shape[0] - m.shape[0]) // 2
    y_start = (m_rotate.shape[1] - m.shape[1]) // 2
    m_rotate = m_rotate[x_start: x_start+m.shape[0], y_start: y_start+m.shape[1]]

    return m_rotate
    
def gaussian_cs_2D(sigma_x, sigma_y, orient, scale_factor):
    """
    Gaussian center-surround kernel (2D)

    The center-surround kernel is the difference of a Gaussian with the 
    specified standard deviation and one with a standard deviation scaled by the specified factor

    The kernel is normalized to have unit L1 norm and zero mean.
    """
    support_x = np.ceil(3*sigma_x).astype(int)
    support_y = np.ceil(3*sigma_y).astype(int)
    support_x_rot = support_x_rotated(support_x, support_y, orient)
    support_y_rot = support_y_rorated(support_x, support_y, orient)

    return gaussian_cs_2D_support(sigma_x. sigma_y, scale_factor, support_x_rot, support_y_rot)

def gaussian_cs_2D_support(sigma_x, sigma_y, orient, scale_factor, support_x, support_y):
    sigma_x_c = sigma_x / scale_factor
    sigma_y_c = sigma_y / scale_factor

    m_center = gaussian_2D_support(sigma_x_c, sigma_y_c, orient, 0, False, support_x, support_y)
    m_surround = gaussian_2D_support(sigma_x, sigma_y, orient, 0, False, support_x, support_y)

    m = m_surround - m_center

    # make zero mean and unit L1 norm
    m -=np.mean(m)
    m /= np.sum(np.abs(m))
    return m

def support_x_rotated(support_x, support_y, orient):
    """
    Compute x-support for ratated 2D matrix
    """
    sx_cos_ori = support_x * np.cos(orient)
    sy_sin_ori = support_y * np.sin(orient)
    x0_mag = np.abs(sx_cos_ori - sy_sin_ori).astype(int)
    x1_mag = np.abs(sx_cos_ori + sy_sin_ori).astype(int)
    return x0_mag if x0_mag > x1_mag else x1_mag

def support_y_rorated(support_x, support_y, orient):
    """
    Compute y-support for rotated 2D matrix
    """
    sx_sin_ori = support_x * np.sin(orient)
    sy_cos_ori = support_y * np.cos(orient)
    y0_mag = np.abs(sx_sin_ori - sy_cos_ori).astype(int)
    y1_mag = np.abs(sx_sin_ori + sy_cos_ori).astype(int)
    return y0_mag if y0_mag > y1_mag else y1_mag



def border_mirror_2D(m, size):
    """
    Expand the border of the 2D matrix by the specified sizes in the X and Y dimensions.
    The expand border is filled with the mirror image of interior data.
    """
    # check that matrix is 2D
    if (len(m.shape) != 2):
        raise ValueError("matrix must be 2D")
        
    # get matrix size
    size_x, size_y = m.shape
    
    # compute border of X and Y
    border_x, border_y = size, size
    
    # check that interior can be mirrored
    if (border_x > size_x or border_y > size_x):
        raise ValueError("cannot create mirrored border larger than matrix interior dimensions")
        
    # mirror border
    m_dst = compute_border_mirror_2D(m, border_x, border_y)
    
    return m_dst

def compute_border_mirror_2D(m, border_x, border_y):
    """
    compute border mirrored matrix
    """
    return np.pad(m, [border_x, border_y], 'reflect')

def border_trim_2D(m, size):
    """
    Trim the specified border size off all sides of the 2D matrix
    """
    # check that matrix is 2D
    if (len(m.shape) != 2):
        raise ValueError("matrix must be 2D")
        
    # get matrix size
    size_x, size_y = m.shape
    
    # compute border of X and Y
    border_x, border_y = size, size

    # compute size of result matrix
    size_x_dst = size_x - 2*border_x if (size_x > 2*border_x) else 0
    size_y_dst = size_y - 2*border_y if (size_y > 2*border_y) else 0

    # trim border
    m_dst = m[border_x: border_x+size_x_dst, border_y:border_y+size_y_dst]
    return m_dst

def grayscale(r, g, b):
    """
    compute a grayscale image from an RGB image
    """
    # check arguments
    assert r.shape == g.shape
    assert g.shape == b.shape

    # convert to grayscale
    m = 0.29894 * r + 0.58704 * g + 0.11402 * b

    return m

def rgb_gamma_correct(r, g, b, gamma):
    """
    Gamma correct the RGB image using the given correction value
    """
    # check arguments
    assert r.shape == g.shape
    assert g.shape == b.shape

    # gamma correct image
    r = np.power(r, gamma)
    g = np.power(g, gamma)
    b = np.power(b, gamma)

    return r, g, b

def rgb_to_lab(r, g, b):
    """
    Convert from RGB color space to Lab color space
    """
    x, y, z = rgb_to_xyz(r, g, b)
    L, a, b = xyz_to_lab(r, g, b)

    return L, a, b
def rgb_to_xyz(r, g, b):
    """
    Convert from RGB color space to XYZ color space
    """
    # check arguments
    assert r.shape == g.shape
    assert g.shape == b.shape

    # convert RGB to XYZ
    x = (0.412453 * r) +  (0.357580 * g) + (0.180423 * b)
    y = (0.212671 * r) +  (0.715160 * g) + (0.072169 * b)
    z = (0.019334 * r) +  (0.119193 * g) + (0.950227 * b)
            
    return x, y, z
def xyz_to_lab(x, y, z):
    """
    Convert from XYZ color space to Lab color space
    """
    # check arguments
    assert x.shape == y.shape
    assert y.shape == z.shape

    # white point reference
    x_ref = 0.950456
    y_ref = 1.000000
    z_ref = 1.088754

    # threshold value 
    threshold = 0.008856

    # convert XYZ to Lab
    # normalize by reference point
    x = x / x_ref
    y = y / y_ref
    z = z / z_ref

    # compute fx, fy, fz, L
    fx = np.zeros_like(x)
    fy = np.zeros_like(y)
    fz = np.zeros_like(z)
    L = np.zeros_like(y)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            fx[i][j] = np.power(x[i][j], 1/3) if x[i][j] > threshold else 7.787*x[i][j] + (16.0/116.0)
            fy[i][j] = np.power(y[i][j], 1/3) if y[i][j] > threshold else 7.787*y[i][j] + (16.0/116.0)
            fz[i][j] = np.power(z[i][j], 1/3) if z[i][j] > threshold else 7.787*z[i][j] + (16.0/116.0)
            L[i][j] = 116*np.power(y[i][j], 1/3) - 16 if y[i][j] > threshold else 903.3*y[i][j]

    # compute Lab color value
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return L, a, b
    
def lab_normalize(L, a, b):
    """
    Normalize an Lab image so that values for each channel lie in [0,1].
    """
    # check arguments
    assert L.shape == a.shape
    assert a.shape == b.shape

    # range for a, b channels
    ab_min = -73
    ab_max = 95
    ab_range = ab_max - ab_min

    # normalize Lab image
    L = L / 100
    a = (a - ab_min) / ab_range
    b = (b - ab_min) / ab_range

    L = np.clip(L, 0, 1)
    a = np.clip(a, 0, 1)
    b = np.clip(b, 0, 1)

    return L, a, b

def quantize_values(m, n_bins):
    """
    Quantize image values into uniformly spaced bins in [0,1].
    """
    # check arguments
    if (n_bins == 0):
        raise ValueError('n_bins must be > 0')

    # compute assignments
    assign = np.zeros_like(m)
    for i in range(0, m.shape[0]):
        for j in range(0, m.shape[1]):
            bin = np.floor(m[i][j] * n_bins).astype(int)
            if (bin == n_bins):
                bin = n_bins - 1
            assign[i][j] = bin
    return assign

def texton_filters(n_orient, sigma):
    """
    Filters for computing textons
    """
    filters_even = oe_filters(n_orient, sigma, False)
    filters_odd = oe_filters(n_orient, sigma, True)

    support = np.ceil(3*sigma).astype(int)
    filter_cs = gaussian_cs_2D_support(sigma, sigma, 0, np.sqrt(2), support, support)
    filters = filters_even + filters_odd
    filters.append(filter_cs)
    return filters

def oe_filters(n_orient, sigma, odd):
    """
    Even and odd-symmetric filters for computing oriented edge energy.

    The even-symmetric filter is a Gaussian second-derivative and the odd-
    symmetric is its Hilbert transform. 

    Each returned filter is an (s+1) x (s+1) matrix where s = 3*sigma and
    sigma is the specified standard deviation
    """
    if (odd == False):
        return gaussian_filters(n_orient, sigma, 2, False, 3)
    else:
        return gaussian_filters(n_orient, sigma, 2, True, 3)

def gaussian_filters(n_orient, sigma, deriv, hlbrt, elongation):
    """
    Oriented Gaussian derivative filters

    Create a filter set consisting of rotated versions of the Gaussian
    derivative filter with the specified parameters.

    Specify the standard deviation (sigma) along the longer principle axis.
    The standard deviation along the other principle axis is sigma/r where r is the elongation ratio

    Each returned filter is an (s+1) x (s+1) matrix where s = 3*sigma.
    """
    # compute support from sigma
    support = np.ceil(3*sigma).astype(int)

    # compute oritations
    orientations = filter_orientations(n_orient)
    
    sigma_x = sigma
    sigma_y = sigma_x / elongation
    filters = []
    for orient in orientations:
        filters.append(gaussian_2D_support(sigma_x, sigma_y, orient, deriv, hlbrt, support, support))
    return filters

def filter_orientations(n_orient):
    """
    Get the filter orientations.
    """
    return np.array(range(0, n_orient)) * np.pi / n_orient

def textons(m, filters, K):
    """
    Compute textons using the given filters

    Convolve the image with each filter and cluster the
    response vectors into textons. Return the texton assignment for each pixel

    Cluster textons using K-means clusterer with L2 distance metric
    """
    responses = []
    for filter in filters:
        responses.append(ndimage.filters.convolve(m, filter))

    # compute 17-vector
    responses_17vector = []
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            responses_17vector.append([])
        
    for response in responses:
        k = 0
        for i in range(response.shape[0]):
            for j in range(response.shape[1]):
                responses_17vector[k].append(response[i][j])
                k += 1

    # K-means cluster
    kmeans = cluster.KMeans(K).fit(responses_17vector)
    t_assign = kmeans.labels_

    # reshape t_assign
    t_assign = t_assign.reshape(m.shape[0], m.shape[1])

    return t_assign

def hist_gradient_2D(labels, radius, n_orient, smoothing_kernel=None):
    """
    Compute the distance between histograms of label values in oriented half-dics
    of the specified radius centered at each location in the 2D matrix. Return one
    distance matrix per orientation.

    The half-disc orientations are k*pi/n for k in [0,n) where n is the number
    of orientation requested. 

    The user may opptionally specify a nonempty 1D smoothing kernel to convolve with
    histograms prior to computing the distance between them.
    """
    # compute weight matrix for circular disc
    weights = weight_matrix_disc(radius)

    # check arguments: labels 
    if (len(labels.shape) != 2):
        raise ValueError("labels matrix must be 2D")

    # check orientations
    if (n_orient == 0):
        raise ValueError("n_orient must > 0")

    # build orientation slice lookup map
    slice_map = orientation_slice_map(weights.shape[0], weights.shape[1], n_orient)

    # compute histograms and histogram difference at each location
    gradients = compute_hist_gradient_2D(labels, weights, n_orient, slice_map, smoothing_kernel, x2_distance)
    return gradients

def compute_hist_gradient_2D(labels, weights, n_orient, slice_map, smoothing_kernel, f_dist):
    """
    compute histograms and histogram difference at each location
    """
    # initialize gradients
    gradients = []
    for i in range(n_orient):
        gradients.append(np.zeros_like(labels))

    labels = labels.astype(int)

    # histograms of each slice
    hist_length = np.max(labels.astype(int)) + 1
    slice_hist_dims = np.zeros([1, hist_length])
    slice_hist = []
    for i in range(2*n_orient):
        slice_hist.append(np.zeros_like(slice_hist_dims))

    temp_conv = np.zeros_like(slice_hist_dims)
    
    # allocate half disc histograms
    hist_left = np.zeros_like(slice_hist_dims)
    hist_right = np.zeros_like(slice_hist_dims)

    # get labels matrix size
    size_labels_x, size_labels_y = labels.shape

    # get weights size
    size_weights_x, size_weights_y = weights.shape

    # set start position for gradient matrices
    pos_start_x = size_weights_x // 2
    pos_start_y = size_weights_y // 2
    pos_bound_y = pos_start_y + size_labels_y

    # initialize position in result
    pos_x = pos_start_x
    pos_y = pos_start_y

    # compute initial range of offset_x
    offset_min_x = pos_x + 1 - size_labels_x if (pos_x + 1 - size_labels_x > 0) else 0
    offset_max_x = pos_x if (pos_x - size_weights_x < 0) else size_weights_x - 1
    ind_labels_start_x = (pos_x - offset_min_x) * size_labels_y
    ind_weights_start_x = (offset_min_x) * size_weights_y

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            # compute range of offset_y
            offset_min_y = pos_y + 1 - size_labels_y if (pos_y + 1 - size_labels_y > 0) else 0
            offset_max_y = pos_y if (pos_y < size_weights_y) else size_weights_y - 1
            offset_range_y = offset_max_y - offset_min_y

            # intialize indices
            index_labels = ind_labels_start_x + (pos_y - offset_min_y)
            index_weights = ind_weights_start_x + offset_min_y

            # clear histograms
            for m in range(2*n_orient):
                slice_hist[m].fill(0)

            # update histograms
            for o_x in range(offset_min_x, offset_max_x+1):
                for o_y in range(offset_min_y, offset_max_y):
                    (slice_hist[get_value_by_index(slice_map, index_weights)])[0, get_value_by_index(labels, index_labels)] += get_value_by_index(weights, index_weights)

                    index_labels -= 1
                    index_weights += 1
                # update last histogram value
                (slice_hist[get_value_by_index(slice_map, index_weights)])[0, get_value_by_index(labels, index_labels)] += get_value_by_index(weights, index_weights)
                index_labels = index_labels + offset_range_y - size_labels_y
                index_weights = index_weights - offset_range_y + size_weights_y
                
            # smooth bins                 

            # L1 normalize bins
            for o in range(0, 2*n_orient):
                sum_slice_hist = np.sum(slice_hist[o]).astype(int)
                if (sum_slice_hist != 0):
                    slice_hist[o] /= sum_slice_hist

            # compute circular gradients: initialize histograms
            hist_left.fill(0)
            hist_right.fill(0)
            for o in range(0, n_orient):
                hist_left += slice_hist[o]
                hist_right += slice_hist[o+n_orient]

            # compute circular gradients: spin the disc
            for o in range(0, n_orient):
                (gradients[o])[i][j] = f_dist(hist_left, hist_right)
                hist_left -= slice_hist[o]
                hist_left += slice_hist[o+n_orient]
                hist_right += slice_hist[o]
                hist_right -= slice_hist[o+n_orient]

            # update position
            pos_y += 1
            if (pos_y == pos_bound_y):
                # reset y position, increment x position
                pos_y = pos_start_y
                pos_x += 1
                # update range of offset_x
                offset_min_x = pos_x + 1 - size_labels_x if (pos_x + 1 - size_labels_x > 0) else 0
                offset_max_x = pos_x if (pos_x < size_weights_x) else size_weights_x - 1
                ind_labels_start_x = (pos_x - offset_min_x) * size_labels_y
                ind_weights_start_x = (offset_min_x) * size_weights_y

    return gradients

def x2_distance(hist_left, hist_right):
    """
    Compute histogram distance
    """
    dist = 0
    for i in range(hist_left.shape[1]):
        diff = hist_left[0, i] - hist_right[0, i]
        summ = hist_left[0, i] + hist_right[0, i]
        if (diff != 0):
            dist += diff**2 / summ
    return dist / 2

def get_value_by_index(array, index):
    size_x, size_y = array.shape
    x_index = index // size_y
    y_index = index % size_y
    return array[x_index][y_index]

def orientation_slice_map(size_x, size_y, n_orient):
    """
    Construct orientation slice lookup map
    """
    slice_map = np.zeros([size_x, size_y])

    # compute orientation of each element from center
    x = -size_x / 2
    for nx in range(size_x):
        y = -size_y / 2
        for ny in range(size_y):
            # compute orientation index
            orient = np.arctan2(y, x) + np.pi
            index = np.floor(orient / np.pi * n_orient).astype(int)
            if (index >= 2*n_orient):
                index = 2*n_orient - 1
            slice_map[nx][ny] = index
            y += 1
        x += 1

    slice_map = slice_map.astype(int)
    return slice_map

def weight_matrix_disc(radius):
    """
    Construct weight matrix for circular disc of the given radius.
    """
    # initialize matrix
    size = 2*radius + 1
    weights = np.zeros([size, size])
    center = radius
    r_sq = radius**2

    # set values in disc to 1
    for i in range(size):
        for j in range(size):
            if (i - center)**2 + (j - center)**2 <= r_sq:
                weights[i][j] = 1
    
    return weights
    
