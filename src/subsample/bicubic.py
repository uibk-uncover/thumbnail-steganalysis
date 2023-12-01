"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numba
import numpy as np
from PIL import Image
import typing

from ._defs import CLAMP, getPixelClamped


@numba.jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def cubic_hermite(A: float, B: float, C: float, D: float, t: float):
    a: float = -1/2.*A + 3/2.*B - 3/2.*C + 1/2.*D
    b: float = 1/1.*A - 5/2.*B + 2/1.*C - 1/2.*D
    c: float = -1/2.*A + 1/2.*C
    d: float = 1/1.*B
    return a*t**3 + b*t**2 + c*t + d


@numba.jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def kernel(t):
    a = -.5
    t = abs(t)  # center and normalize
    if t > 2:
        return 0.
    elif t > 1:
        return a*t**3-5*a*t**2+8*a*t-4*a
    else:
        return (a+2)*t**3-(a+3)*t**2+1


@numba.jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=False)
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
                phi_x[x-x_low] = kernel((x-x_in) / h_x) / h_x
            phi_x = phi_x / phi_x.sum()
            for y in numba.prange(y_low,y_high):
                phi_y[y-y_low] = kernel((y-y_in) / h_y) / h_y
            phi_y = phi_y / phi_y.sum()

            # Interpolate over channel layers
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
            x_int = int(np.floor(x_in))
            y_int = int(np.floor(y_in))
            x_frac = x_in - x_int
            y_frac = y_in - y_int

            # 1st row
            p00 = getPixelClamped(array_in, width_in, height_in, x_int-1, y_int-1)
            p10 = getPixelClamped(array_in, width_in, height_in, x_int+0, y_int-1)
            p20 = getPixelClamped(array_in, width_in, height_in, x_int+1, y_int-1)
            p30 = getPixelClamped(array_in, width_in, height_in, x_int+2, y_int-1)
            # 2nd row
            p01 = getPixelClamped(array_in, width_in, height_in, x_int-1, y_int+0)
            p11 = getPixelClamped(array_in, width_in, height_in, x_int+0, y_int+0)
            p21 = getPixelClamped(array_in, width_in, height_in, x_int+1, y_int+0)
            p31 = getPixelClamped(array_in, width_in, height_in, x_int+2, y_int+0)
            # 3rd row
            p02 = getPixelClamped(array_in, width_in, height_in, x_int-1, y_int+1)
            p12 = getPixelClamped(array_in, width_in, height_in, x_int+0, y_int+1)
            p22 = getPixelClamped(array_in, width_in, height_in, x_int+1, y_int+1)
            p32 = getPixelClamped(array_in, width_in, height_in, x_int+2, y_int+1)
            # 4th row
            p03 = getPixelClamped(array_in, width_in, height_in, x_int-1, y_int+2)
            p13 = getPixelClamped(array_in, width_in, height_in, x_int+0, y_int+2)
            p23 = getPixelClamped(array_in, width_in, height_in, x_int+1, y_int+2)
            p33 = getPixelClamped(array_in, width_in, height_in, x_int+2, y_int+2)

            # Interpolate over 3 RGB layers
            for c in numba.prange(channels_in):
                # along width
                col0 = cubic_hermite(p00[c], p10[c], p20[c], p30[c], x_frac)
                col1 = cubic_hermite(p01[c], p11[c], p21[c], p31[c], x_frac)
                col2 = cubic_hermite(p02[c], p12[c], p22[c], p32[c], x_frac)
                col3 = cubic_hermite(p03[c], p13[c], p23[c], p33[c], x_frac)
                # along height
                array_out[i][j][c] = CLAMP(cubic_hermite(col0, col1, col2, col3, y_frac), 0, 1)

    return array_out


def scale_image(
    x: np.ndarray,
    shape: typing.Tuple[int],
    grid_shift: typing.Tuple[float] = (-.5, -.5),
    use_antialiasing=False,
) -> np.ndarray:
    """Scales image using bicubic interpolation.

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
