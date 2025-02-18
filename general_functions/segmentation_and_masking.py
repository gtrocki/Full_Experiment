import numpy as np

# -----------------------------------------------------------------------------
def is_in_trapezoid(points, trapezoid):
    """ Vectorized check if points are within a trapezoid using cross products. """
    def sign(p1, p2, p3):
        return (p1[:, 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[:, 1] - p3[1])

    # Calculate the cross product for each edge of the trapezoid
    d1 = sign(points, trapezoid[0], trapezoid[1])
    d2 = sign(points, trapezoid[1], trapezoid[2])
    d3 = sign(points, trapezoid[2], trapezoid[3])
    d4 = sign(points, trapezoid[3], trapezoid[0])

    # Check if the signs are consistent (all positive or all negative)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0) | (d4 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0) | (d4 > 0)

    return ~(has_neg & has_pos)
# -----------------------------------------------------------------------------
def mask_points_within_trapezoid(colors: np.ndarray, color_differences: np.ndarray, trapezoid: np.ndarray, extra_color: int=None, extra_color_range: tuple=0):

    if extra_color == None:
        # create the mask for the segmentation
        mask = np.array(is_in_trapezoid(color_differences, trapezoid))
    else:
        n = extra_color

        mask = np.array(is_in_trapezoid(color_differences, trapezoid) &
                          (extra_color_range[0] < colors[:, n]) &
                          (colors[:, n] < extra_color_range[1]))
    return mask
# -----------------------------------------------------------------------------
def get_color_differences(colors: np.ndarray):

    color_differences = np.hstack(
        ((colors[:, 1] - colors[:, 0]).reshape((len(colors), 1)),
         (colors[:, 2] - colors[:, 1]).reshape((len(colors), 1))))

    return color_differences
