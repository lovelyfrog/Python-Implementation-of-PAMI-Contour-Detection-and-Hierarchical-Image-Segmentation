from PIL import Image
import numpy as np
import lib_image
def gradient_mPb(im):
    """
    compute the gradient of L, a, b, texton.
    im: 3 channel image arrays with value scaled between (0,1].
    """
    # parameters - bin and smooth
    num_orient = 8
    num_L_bins = 25
    num_a_bins = 25
    num_b_bins = 25
    bg_smooth_sigma = 0.1
    cg_smooth_sigma = 0.05
    border = 30
    sigma_tg_filter_sm = 2
    sigma_tg_filter_lg = np.sqrt(2) * 2
    
    # parameters - radius
    num_bg_radius = 3
    num_cg_radius = 3
    num_tg_radius = 3
    bg_radius = [3, 5, 10]
    cg_radius = [5, 10, 20]
    tg_radius = [5, 10, 20]
    
    # compute bg histogram smoothing kernel
    bg_smooth_kernel = lib_image.gaussian(bg_smooth_sigma*num_L_bins)
    cga_smooth_kernel = lib_image.gaussian(cg_smooth_sigma*num_a_bins)
    cgb_smooth_kernel = lib_image.gaussian(cg_smooth_sigma*num_b_bins)
    
    # get image
    L = im[:,:,0]
    a = im[:,:,1]
    b = im[:,:,2]
    
    # mirror border
    L = lib_image.border_mirror_2D(L, border)
    a = lib_image.border_mirror_2D(a, border)
    b = lib_image.border_mirror_2D(b, border)
    
    # convert to grayscale
    print("converting RGB to grayscale ")
    gray = lib_image.grayscale(L, a, b)

    # gamma correct
    L, a, b = lib_image.rgb_gamma_correct(L, a, b, 2.5)

    # convert to Lab
    print("converting RGB to Lab")
    L, a, b = lib_image.rgb_to_lab(L, a, b)
    L, a, b = lib_image.lab_normalize(L, a, b)

    # quantize color channels
    print("quantize color channels")
    Lq = lib_image.quantize_values(L, num_L_bins)
    aq = lib_image.quantize_values(a, num_a_bins)
    bq = lib_image.quantize_values(b, num_b_bins)

    # compute texton filter set
    print("computing filter set for textons")
    filters_small = lib_image.texton_filters(num_orient, sigma_tg_filter_sm)
    filters_large = lib_image.texton_filters(num_orient, sigma_tg_filter_lg)
    filters = filters_small + filters_large

    # compute textons
    print("computing textons")
    t_assign = lib_image.textons(gray, filters, 64)
    t_assign = lib_image.border_mirror_2D(lib_image.border_trim_2D(t_assign, border), border)

    # return textons
    plhs = []
    plhs.append(lib_image.border_trim_2D(t_assign, border))

    # compute bg at each radius
    for radius in bg_radius:
        # compute bg
        print("computing bg, r=", radius)
        bgs = lib_image.hist_gradient_2D(Lq, radius, num_orient, bg_smooth_kernel)
        plhs.append(bgs)

    # compute cga at each radius
    for radius in cg_radius:
        # compute cga
        print("computing cga, r=", radius)
        cgs_a = lib_image.hist_gradient_2D(aq, radius, num_orient, cga_smooth_kernel)
        plhs.append(cgs_a)

    # compute cgb at each radius
    for radius in cg_radius:
        # compute cgb
        print("computing cgb, r=", radius)
        cgs_b = lib_image.hist_gradient_2D(bq, radius, num_orient, cgb_smooth_kernel)
        plhs.append(cgs_b)
    # compute tg at each radius
    for radius in tg_radius:
        # compute tg
        print("computing tg, r=", radius)
        tgs = lib_image.hist_gradient_2D(t_assign, radius, num_orient)
        plhs.append(tgs)
        
    return plhs
    
im = Image.open('101087.jpg')
im = np.array(im)
im = im / 255
print(im.shape)

# plhs = gradients_mPb(im)
# print(plhs[0].shape)
