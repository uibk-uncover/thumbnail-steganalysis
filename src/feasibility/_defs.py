"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import imageops
import joblib
import jpeglib
import numpy as np
import tempfile
import tqdm
import typing

from .. import subsample

DIFF_LABEL: str = 'changerate'
STAGE_LABEL: str = 'stage'
BEFORE_COMPRESSION: str = 'pre'
AFTER_COMPRESSION: str = 'post'
AFTER_DECOMPRESSION: str = 'dec'
EMBEDDING_LABEL: str = 'embedding'
EMBEDDING_RATE_LABEL: str = 'embrate'
SAMPLING_METHOD_LABEL: str = 'sampmethod'
SAMPLING_RATE_LABEL: str = 'samprate'
ANTIALIASING_LABEL: str = 'antialiasing'


class ProgressParallel(joblib.Parallel):
    def __call__(self, *args, total: int = None, **kwargs):
        with tqdm.tqdm(total=total) as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        # self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def scale_compress_image(
    x: np.ndarray,
    sampling_method: str,
    thumbnail_shape: typing.Tuple[int],
    use_antialiasing: bool,
    tmp=None
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """"""
    # subsample
    if sampling_method != 'magick':
        if sampling_method == 'qlmanage':
            x_t = subsample.qlmanage.scale_image(x, 128)
        else:
            x_t = subsample.scale_image(
                x, thumbnail_shape,
                sampling_method,
                use_antialiasing=use_antialiasing
            )
        if tmp is None:
            tmp = tempfile.NamedTemporaryFile(suffix='.jpeg')
        jpeglib.from_spatial(x_t).write_spatial(tmp.name)
        tmp.flush()
        ym_t = jpeglib.read_dct(tmp.name)
        ym_t.load()
        return x_t, ym_t
    # subsample & compress
    else:
        y_t = subsample.magick.scale_compress_image(
            x, thumbnail_shape,
            use_antialiasing=use_antialiasing
        )
        return None, y_t
