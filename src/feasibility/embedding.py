"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import joblib
import numpy as np
import pandas as pd
import pathlib
import scipy.stats
import typing

from . import _defs
from . import _mismatch


def run_embedding(
    dataset_path: str,
    # iterable parameters
    embeddings: typing.List[str],
    alphas: typing.List[float],
    # marginalized parameters
    sampling_rate: float = .25,
    sampling_method: str = 'nearest',
    use_antialiasing: bool = True,
    **kw
) -> pd.DataFrame:
    # data paths
    dataset_path = pathlib.Path(dataset_path)

    post_compress = True
    cover_dir = 'jpegs_q75_512'

    # construct image pairs
    images = pd.DataFrame()
    # cover_dirs = [f'jpegs_q{quality:02d}', 'images']
    # for post_compress, cover_dir in zip([True, False], cover_dirs):

    for embedding in embeddings:
        for alpha in alphas:

            # get images
            cover_path = dataset_path / cover_dir
            stego_path = dataset_path / f'stego_{embedding}_alpha_{alpha}_{cover_dir}'
            cover_images = pd.read_csv(cover_path / 'files.csv')
            stego_images = pd.read_csv(stego_path / 'files.csv')

            # merge image pairs
            cover_images['fname'] = cover_images['name'].apply(lambda s: pathlib.Path(s).stem).astype('str')
            stego_images['fname'] = stego_images['name'].apply(lambda s: pathlib.Path(s).stem).astype('str')
            cover_images = cover_images.set_index('fname')
            stego_images = stego_images.set_index('fname')

            #
            cover_stego_images = cover_images.join(
                stego_images[['name', 'stego_method', 'alpha']],
                how='inner', lsuffix='_cover', rsuffix='_stego',
            ).reset_index(drop=True)
            cover_stego_images['post_compress'] = post_compress
            images = pd.concat([images, cover_stego_images])

    # iterate image pairs (embarrasingly parallel)
    gen = (
        joblib.delayed(_mismatch.compute_mismatch)(
            cover_name=dataset_path / row.name_cover,
            stego_name=dataset_path / row.name_stego,
            embedding=row.stego_method,
            embedding_rate=row.alpha,
            # sampling_rate=sampling_rate,
            sampling_method=sampling_method,
            use_antialiasing=use_antialiasing,
            post_compress=row.post_compress,
        )
        for index, row in images.iterrows()
    )
    res = _defs.ProgressParallel(**kw)(gen, total=len(images))

    # to long format (for processing)
    res = pd.DataFrame(res)
    # print(res)
    # pre-/post- w.r.t. thumbnail compression
    res = pd.melt(
        res[[
            _defs.EMBEDDING_LABEL, _defs.EMBEDDING_RATE_LABEL,
            'pre', 'post', 'dec',
        ]],
        id_vars=(_defs.EMBEDDING_LABEL, _defs.EMBEDDING_RATE_LABEL),
        var_name=_defs.STAGE_LABEL,
        value_name=_defs.DIFF_LABEL,
    )
    # # pre-/post- w.r.t. main image compression
    # res = res.rename({'post': _defs.DIFF_LABEL}, axis=1)
    # res = res.drop(['pre'], axis=1)

    # compute 95%CI
    res = (
        res.groupby([_defs.EMBEDDING_LABEL, _defs.EMBEDDING_RATE_LABEL, _defs.STAGE_LABEL])
        .agg({_defs.DIFF_LABEL: ['mean', 'std', 'count']})
    ).reset_index(drop=False)
    res[(_defs.DIFF_LABEL,'ci95')] = (
        scipy.stats.norm.ppf(.975) *
        res[_defs.DIFF_LABEL]['std'] /
        res[_defs.DIFF_LABEL]['count'].apply(np.sqrt)
    )
    res.columns = ['_'.join([c for c in col if c]) for col in res.columns]

    # to wide format (for plotting)
    res = res.pivot(
        index=_defs.EMBEDDING_RATE_LABEL,
        columns=[_defs.STAGE_LABEL, _defs.EMBEDDING_LABEL],
        values=[_defs.DIFF_LABEL+'_mean', _defs.DIFF_LABEL+'_ci95']
    ).reset_index(drop=False)
    res.columns = ['_'.join([c for c in reversed(col) if c]) for col in res.columns]

    # print(res)
    return res
