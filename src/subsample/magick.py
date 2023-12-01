"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import jpeglib
import numpy as np
from PIL import Image
import subprocess
import tempfile
import typing


def scale_image(
    x: np.ndarray,
    shape: typing.Tuple[int] = (160, 120),
    kernel: str = 'thumbnail',
    use_antialiasing: bool = False
) -> np.ndarray:
    """Scales RGB image into the shape (maintains aspect ratio) using ImageMagick.

    Args:
        x (np.ndarray): Source RGB.
        shape (tuple): Shape to sample to.
        kernel (str): Method to use.
    Returns:
        (np.ndarray): Resampled RGB.
    """
    # prepare source file
    assert kernel is not None, "kernel cannot be None"

    with tempfile.NamedTemporaryFile(suffix='.png') as src:
        if len(x.shape) == 3 and x.shape[2] == 1:
            x = x[:, :, 0]
        Image.fromarray(x).save(src.name)
        src.flush()
        # low-pass filter for anti-aliasing
        anti_alias = ()
        if use_antialiasing:
            # anti_alias = ('-unsharp', '0x.5')
            anti_alias = ('-unsharp', '0x.5')
        # reshape to destination file
        with tempfile.NamedTemporaryFile(suffix='.png') as dst:
            res = subprocess.run([
                'convert',
                src.name,
                # '-auto-orient',
                f'-{kernel}', '%dx%d' % (shape),
                '-alpha', 'deactivate',
                *anti_alias,
                dst.name
            ])
            if res.returncode != 0:
                raise Exception(res.stderr.decode('ascii'))

            # read destination file
            y = np.array(Image.open(dst.name))
    if len(y.shape) == 2:
        y = np.repeat(y[..., None], 3, axis=2)
    return y


def scale_compress_image(
    x: np.ndarray,
    shape: typing.Tuple[int] = (160, 120),
    use_antialiasing: bool = False,
) -> jpeglib.DCTJPEG:
    """Scales RGB image into the shape (maintains aspect ratio) and compress using ImageMagick.

    Args:
        x (np.ndarray): Source RGB.
        shape (tuple): Shape to sample to.
    Returns:
        (np.ndarray): Resampled RGB.
    """
    # prepare source file
    with tempfile.NamedTemporaryFile(suffix='.png') as src:
        if len(x.shape) == 3 and x.shape[2] == 1:
            x = x[:, :, 0]
        Image.fromarray(x).save(src.name)
        src.flush()
        # low-pass filter for anti-aliasing
        anti_alias = ()
        if use_antialiasing:
            # anti_alias = ('-unsharp', '0x.5')
            anti_alias = ('-unsharp', '0x.5')
        # reshape to destination file
        with tempfile.NamedTemporaryFile(suffix='.jpeg') as dst:
            res = subprocess.run([
                'convert',
                src.name,
                # '-auto-orient',
                '-thumbnail', '%dx%d' % (shape),
                '-alpha', 'deactivate',
                *anti_alias,
                dst.name
            ])
            if res.returncode != 0:
                raise Exception(res.stderr.decode('ascii'))

            # read destination file
            jpeg = jpeglib.read_dct(dst.name)
            jpeg.Y  # to load data

    return jpeg
