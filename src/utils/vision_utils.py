import numpy as np

def compute_center(mask):
    """
    Returns (cx, cy) pixel centroid of nonzero mask region.
    """
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return None
    return int(x.mean()), int(y.mean())