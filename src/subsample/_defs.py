"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numba
import numpy as np
import typing


@numba.jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=False)
def CLAMP(
    v: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """"""
    return max(vmin, min(v, vmax))


@numba.jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=False)
def getPixelClamped(
    array_in: np.ndarray,
    width_in: int,
    height_in: int,
    x: int,
    y: int
) -> float:
    """"""
    x = CLAMP(x, 0, int(width_in - 1))
    y = CLAMP(y, 0, int(height_in - 1))
    return array_in[y][x]


def parse_xy_tuple(
    tpl: typing.Union[typing.Tuple[typing.Any], typing.Any],
) -> typing.Tuple[typing.Any]:
    """"""
    try:
        w_y, h_y = tuple(tpl)
    except TypeError:
        w_y, h_y = [int(tpl)]*2
    return w_y, h_y
