import numpy as np
def buildW(mPb):
    """
    build sparse affinity matrix from mPb.
    """
    def max_pixel(x, y, u, v):
        """
        get the maximal pixel between (x, y) and (u, v) (line segment)
        """
        max_p = mPb[x, y]
        # ascent not exist
        if (x == u):
            l, r = np.min([y, v]), np.max([y, v])
            for i in range(l ,r+1):
                if mPb[x, i] > max_p:
                    max_p = mPb[x, i]
        else:
            k = (v - y) / (u - x)
            b = y - k * x
            xl, xr = np.min([x, u]), np.max([x, u])
            for i in range(xl, xr+1):
                j = k * i + b
                if (np.abs(np.round(j) - j) < epsilon):
                    j = np.round(j).astype(int)
                    if mPb[i, j] > max_p:
                        max_p = mPb[i, j]
        return max_p

    def index(x, y):
        """
        get the index of (x, y)
        """
        return x * height + y

    radius = 5
    sigma = 0.1
    thresh = 1
    rho = 0.1
    epsilon = 1e-8

    width, height = mPb.shape
    W = np.zeros([width*height, width*height])
    for x in range(width):
        for y in range(height):
            i = index(x, y)
            rxa = np.max([0, x - radius])
            rxb = np.min([width - 1, x + radius])
            rya = np.max([0, y - radius])
            ryb = np.min([height - 1, y + radius])

            for u in range(rxa, rxb + 1):
                for v in range(rya, ryb + 1):
                    if (u - x)**2 + (v - y)**2 <= radius**2:
                        pixel = max_pixel(x, y, u, v)
                        j = index(u, v)
                        W[i, j] = np.exp(-pixel / rho)

    return W
    
