"""

Author: Martin Benes
Affiliation: University of INnsbruck
"""

import numba
import numpy as np
from PIL import Image
import typing

from ._common import CLAMP, parse_xy_tuple

@numba.jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def kernel(t):
    return int(abs(t) < 1/2.)

@numba.jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=False)
def _scale_image_antialias(
    array_in,
    width_in,
    height_in,
    channels_in,
    #
    array_out,
    width_out,
    height_out,
    #
    grid_shift_x,
    grid_shift_y,
):

    # Compute area sizes
    h_x: float = width_in / width_out
    h_y: float = height_in / height_out

    for i in numba.prange(height_out):
        for j in numba.prange(width_out):
            # Relative coordinates of the pixel in output space
            x_out: float = j / width_out
            y_out: float = i / height_out

            # Corresponding absolute coordinates of the pixel in input space
            x_in: float = (x_out * width_in) + grid_shift_x
            y_in: float = (y_out * height_in) + grid_shift_x

            # Get boundaries of the rectangle
            x_min: float = x_in - h_x
            y_min: float = y_in - h_y
            x_max: float = x_in + h_x + 1
            y_max: float = y_in + h_y + 1

            # # Get boundaries of the rectangle
            # x_min = (j) * kernel_width  # (x_in // kernel_width) * kernel_width
            # y_min = (i) * kernel_height  # (y_in // kernel_height) * kernel_height
            # x_max = (j + 1) * kernel_width  # (x_in // kernel_width + 1) * kernel_width
            # y_max = (i + 1) * kernel_height  # (y_in // kernel_height + 1) * kernel_height

            # Furthest within-rectangle coordinates in input space
            x_low: int = int(np.ceil(x_min))
            y_low: int = int(np.ceil(y_min))
            x_high: int = int(np.ceil(x_max))
            y_high: int = int(np.ceil(y_max))

            # Sanitize bounds
            x_low: int = CLAMP(x_low, 0, width_in-1)
            y_low: int = CLAMP(y_low, 0, height_in-1)
            x_high: int = CLAMP(x_high, 0, width_in)
            y_high: int = CLAMP(y_high, 0, height_in)

            # Number of pixels in the area
            N: int = (x_high - x_low) * (y_high - y_low)

            # Interpolate over 3 RGB layers
            for c in range(channels_in):

                # Mean of the area
                array_out[i][j][c] = 0
                for y in numba.prange(y_low, y_high):
                    for x in numba.prange(x_low, x_high):
                        array_out[i][j][c] += array_in[y][x][c]
                array_out[i][j][c] /= N

    return array_out


@numba.jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def _scale_image(
    array_in,
    width_in,
    height_in,
    channels_in,
    #
    array_out,
    width_out,
    height_out,
    #
    grid_shift_x,
    grid_shift_y,
):
    for i in numba.prange(height_out):
        for j in numba.prange(width_out):
            # Relative coordinates of the pixel in output space
            x_out = j / width_out
            y_out = i / height_out

            # Corresponding absolute coordinates of the pixel in input space
            x_in = (x_out * width_in) + grid_shift_x
            y_in = (y_out * height_in) + grid_shift_y

            # Nearest neighbor
            x_near = int(np.floor(x_in))
            y_near = int(np.floor(y_in))

            # Sanitize bounds - no need to check for < 0
            x_near = CLAMP(x_near, 0, width_in-1)
            y_near = CLAMP(y_near, 0, height_in-1)

            # Interpolate over 3 RGB layers
            for c in range(channels_in):
                array_out[i][j][c] = array_in[y_near][x_near][c]

    return array_out


def scale_image(
    x: np.ndarray,
    shape: typing.Tuple[int],
    grid_shift: typing.Tuple[float] = (-.5, -.5),
    use_antialiasing: bool = False,
) -> np.ndarray:
    """Scales image using nearest neighbor interpolation.

    Args:
        x (np.ndarray): Input image.
        shape (tuple): Output shape.
        grid_shift (tuple, optional): Grid shift, by default (-.5,-.5).
        use_antialising (bool, optional): Whether antialiasing filter should be used. False by default.
    Returns:
        (np.ndarray): Scaled image.
    """
    # parse input
    h_x, w_x, channels_x = x.shape
    w_y, h_y = tuple(shape)
    w_shift, h_shift = tuple(grid_shift)
    # allocate output
    y = np.zeros((h_y, w_y, channels_x))

    # plain scaling
    if not use_antialiasing:
        y = _scale_image(x, w_x, h_x, channels_x, y, w_y, h_y, w_shift, h_shift)
    # scaling with anti-aliasing filter
    else:
        assert w_x > w_y, "can't use anti-aliasing for supersampling"
        assert h_x > h_y, "can't use anti-aliasing for supersampling"
        y = _scale_image_antialias(x, w_x, h_x, channels_x, y, w_y, h_y, w_shift, h_shift)

    return y
