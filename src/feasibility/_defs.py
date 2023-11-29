
import imageops
import joblib
import jpeglib
import numpy as np
import tempfile
from tqdm import tqdm
import typing

DIFF_LABEL = 'changerate'
STAGE_LABEL = 'stage'
BEFORE_COMPRESSION = 'pre'
AFTER_COMPRESSION = 'post'
AFTER_DECOMPRESSION = 'dec'
EMBEDDING_LABEL = 'embedding'
EMBEDDING_RATE_LABEL = 'embrate'
SAMPLING_METHOD_LABEL = 'sampmethod'
SAMPLING_RATE_LABEL = 'samprate'
ANTIALIASING_LABEL = 'antialiasing'

class ProgressParallel(joblib.Parallel):
    def __call__(self, *args, total:int=None, **kwargs):
        with tqdm(total=total) as self._pbar:
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
            x_t = imageops.subsample.qlmanage.scale_image(x, 128)
        else:
            x_t = imageops.scale_image(
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
        y_t = imageops.subsample.magick.scale_compress_image(
            x, thumbnail_shape,
            use_antialiasing=use_antialiasing
        )
        return None, y_t
