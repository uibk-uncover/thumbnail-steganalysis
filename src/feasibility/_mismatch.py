"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import jpeglib
import numpy as np
from PIL import Image
import tempfile
import typing

from . import _defs


def dct_size(dct: np.ndarray) -> int:
    return dct.size


def dct_absolute_mismatch(dct1: np.ndarray, dct2: np.ndarray) -> int:
    """Number of unequal DCT coefficients. Only in single component."""
    return (dct1 != dct2).sum()


def compute_mismatch(
    cover_name: str,
    stego_name: str,
    embedding: str,
    embedding_rate: float,
    # sampling_rate: float = .25,
    sampling_method: str = 'nearest',
    use_antialiasing: bool = True,
    post_compress: bool = True
) -> typing.Dict[str, typing.Any]:

    # open temporary file
    tmp = tempfile.NamedTemporaryFile(suffix='.jpeg')

    # load cover
    if post_compress:
        x0 = jpeglib.read_spatial(cover_name, jpeglib.JCS_GRAYSCALE).spatial
        # x0 = jpeglib.read_spatial(cover_name).spatial
    else:
        raise NotImplementedError('not implemented for colored JPEG')
        x0 = np.array(Image.open(cover_name))
        if len(x0.shape) == 2:
            x0 = np.expand_dim(x0, 2)

    # load stego
    xm = jpeglib.read_spatial(stego_name, jpeglib.JCS_GRAYSCALE).spatial

    # create thumbnail
    sampling_rate = 128 / x0.shape[0]
    x0_t, y0_t = _defs.scale_compress_image(
        x0,
        sampling_method=sampling_method,
        use_antialiasing=use_antialiasing,
        thumbnail_shape=(128, 128),
        tmp=tmp,
    )
    xm_t, ym_t = _defs.scale_compress_image(
        xm,
        sampling_method=sampling_method,
        use_antialiasing=use_antialiasing,
        thumbnail_shape=(128, 128),
        tmp=tmp,
    )

    # result of comparison
    res = {
        _defs.EMBEDDING_LABEL: embedding,
        _defs.EMBEDDING_RATE_LABEL: embedding_rate,
        _defs.SAMPLING_METHOD_LABEL: sampling_method,
        _defs.SAMPLING_RATE_LABEL: sampling_rate,
        _defs.STAGE_LABEL: 'post' if post_compress else 'pre',
        _defs.ANTIALIASING_LABEL: 'aa' if use_antialiasing else 'noaa'
    }

    # compare spatial
    if xm_t is not None and x0_t is not None:
        res[_defs.BEFORE_COMPRESSION] = (x0_t != xm_t).mean()
    else:
        res[_defs.BEFORE_COMPRESSION] = None

    # compare compressed
    cover_size = dct_size(y0_t.Y)
    dct_mismatch = dct_absolute_mismatch(y0_t.Y, ym_t.Y)
    # in case of color
    if y0_t.has_chrominance:
        cover_size = (
            cover_size +
            dct_size(y0_t.Cb) +
            dct_size(y0_t.Cr)
        )
        dct_mismatch = (
            dct_mismatch +
            dct_absolute_mismatch(y0_t.Cb, ym_t.Cb) +
            dct_absolute_mismatch(y0_t.Cr, ym_t.Cr)
        )
    #
    res[_defs.AFTER_COMPRESSION] = dct_mismatch / cover_size

    # decompress
    with tempfile.NamedTemporaryFile(suffix='.jpeg') as tmp:
        y0_t.write_dct(tmp.name)
        x0_t_d = jpeglib.read_spatial(tmp.name).spatial
        ym_t.write_dct(tmp.name)
        xm_t_d = jpeglib.read_spatial(tmp.name).spatial

    res[_defs.AFTER_DECOMPRESSION] = (x0_t_d != xm_t_d).mean()

    return res
