"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import numpy as np
import typing
import warnings

# common
from . import _defs

# subsamplers
from . import nearest
from . import bilinear
from . import bicubic
from . import magick


def scale_image(
    x: np.ndarray,
    shape: typing.Union[int, typing.Tuple[int]],
    kernel: str = None,
    implementation: str = 'textbook',
    grid_shift: typing.Union[float, typing.Tuple[float]] = -.5,
    use_antialiasing: bool = False,
    **kw,
) -> np.ndarray:
    """Scales image.

    Args:
        x (np.ndarray): Input image.
        shape (tuple): Output shape.
        kernel (str): Type of interpolation.
        implementation (str): Type of implementation.
        grid_shift (float, tuple, optional): Grid shift, by default (-.5,-.5).
        use_antialising (bool, optional): Whether antialiasing filter should be used. False by default.
    Returns:
        (np.ndarray): Scaled image.
    """
    # parse input
    x = np.array(x)
    h_x, w_x = x.shape[:2]
    w_y, h_y = _defs.parse_xy_tuple(shape)
    if abs(h_x/w_x - h_y/w_y) > .01:
        warnings.warn('significant aspect ratio mismatch!', stacklevel=2)
    if use_antialiasing:
        assert w_x > w_y, "can't use anti-aliasing for supersampling"
        assert h_x > h_y, "can't use anti-aliasing for supersampling"
    w_shift, h_shift = _defs.parse_xy_tuple(grid_shift)

    if implementation == 'textbook':
        x = x / 255.  # uint8 -> float64
        if kernel in {'nearest'}:
            y = nearest.scale_image(x, (w_y, h_y), (w_shift, h_shift), use_antialiasing)
        elif kernel in {'linear', 'bilinear'}:
            y = bilinear.scale_image(x, (w_y, h_y), (w_shift, h_shift), use_antialiasing)
        elif kernel in {'cubic', 'bicubic'}:
            y = bicubic.scale_image(x, (w_y, h_y), (w_shift, h_shift), use_antialiasing)
        else:
            raise NotImplementedError(f'kernel {kernel} not implemented')
        y = (y * 255.).astype(np.uint8)
    elif implementation == 'magick':
        y = magick.scale_image(x, (w_y, h_y), kernel=kernel, use_antialiasing=use_antialiasing)
    else:
        raise NotImplementedError(f'kernel {kernel} not implemented')

    return y
