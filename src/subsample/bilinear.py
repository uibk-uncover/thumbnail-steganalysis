"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numba
import numpy as np
from PIL import Image
import typing

from ._defs import CLAMP

@numba.jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=False)
def kernel(t):
    return max(0, 1 - abs(t))


@numba.jit(nopython=True, fastmath=True, nogil=False, cache=True, parallel=False)
def _scale_image_antialias(
    array_in,
    width_in,
    height_in,
    channels_in,
    array_out,
    width_out,
    height_out,
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
            y_in: float = (y_out * height_in) + grid_shift_y

            # Get boundaries of the rectangle
            x_min: float = x_in - h_x
            y_min: float = y_in - h_y
            x_max: float = x_in + h_x + 1
            y_max: float = y_in + h_y + 1

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

            # Compute weights
            phi_x = np.zeros(int(np.ceil(2 * width_in / width_out) + 1), dtype=np.float64)
            phi_y = np.zeros(int(np.ceil(2 * height_in / height_out) + 1), dtype=np.float64)
            for x in numba.prange(x_low, x_high):
                phi_x[x - x_low] = kernel((x-x_in) / h_x) / h_x
            phi_x = phi_x / phi_x.sum()
            for y in numba.prange(y_low, y_high):
                phi_y[y - y_low] = kernel((y-y_in) / h_y) / h_y
            phi_y = phi_y / phi_y.sum()

            # Interpolate over 3 RGB layers
            for c in numba.prange(channels_in):
                array_out[i][j][c] = 0

                # Iterate kernel area
                for y in numba.prange(y_low, y_high):

                    weighted_sum_x = 0
                    for x in numba.prange(x_low, x_high):
                        weighted_sum_x += array_in[y][x][c] * phi_x[x - x_low]

                    array_out[i][j][c] += weighted_sum_x * phi_y[y - y_low]

    return array_out

@numba.jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def _scale_image(
    array_in,
    width_in,
    height_in,
    channels_in,
    array_out,
    width_out,
    height_out,
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

            # Nearest neighbours coordinates in input space
            x_prev = int(np.floor(x_in))
            x_next = x_prev + 1
            y_prev = int(np.floor(y_in))
            y_next = y_prev + 1

            # Sanitize bounds - no need to check for < 0
            x_prev = CLAMP(x_prev, 0, width_in - 1)
            x_next = CLAMP(x_next, 0, width_in - 1)
            y_prev = CLAMP(y_prev, 0, height_in - 1)
            y_next = CLAMP(y_next, 0, height_in - 1)

            # Distances between neighbour nodes in input space
            Dy_next = y_next - y_in
            Dy_prev = 1. - Dy_next  # because next - prev = 1
            Dx_next = x_next - x_in
            Dx_prev = 1. - Dx_next  # because next - prev = 1

            # Interpolate over channel layers
            for c in numba.prange(channels_in):
                array_out[i][j][c] = (
                    Dy_prev * (
                        array_in[y_next][x_prev][c] * Dx_next +
                        array_in[y_next][x_next][c] * Dx_prev) +
                    Dy_next * (
                        array_in[y_prev][x_prev][c] * Dx_next +
                        array_in[y_prev][x_next][c] * Dx_prev))

    return array_out


def scale_image(
    x: np.ndarray,
    shape: typing.Tuple[int],
    grid_shift: typing.Tuple[int] = (-.5,-.5),
    use_antialiasing: bool = False
) -> np.ndarray:
    """Scales image using bilinear interpolation.

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
        # phi_x = np.zeros(int(np.ceil(2 * w_x / w_y) + 1), dtype=np.float64)
        # phi_y = np.zeros(int(np.ceil(2 * h_x / h_y) + 1), dtype=np.float64)
        y = _scale_image_antialias(x, w_x, h_x, channels_x, y, w_y, h_y, w_shift, h_shift)

    return y
