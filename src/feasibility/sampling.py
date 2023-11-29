
import joblib
import numpy as np
import pandas as pd
import pathlib
import scipy.stats
import typing

from . import _defs
from . import _mismatch


def run_sampling(
    dataset_path_prefix: str,
    # iterable parameters
    sampling_methods: typing.List[str],
    sampling_rates: typing.List[float],
    use_antialiasing: typing.List[bool],
    # marginalized parameters
    embedding: str,
    alpha: float,
    beta: float,
    quality: int = 75,
    thumbnail_quality: int = 75,
    **kw
):
    # data paths
    cover_dir = f'jpegs_q{quality:02d}'
    stego_dir = f'stego_{embedding}_alpha_{alpha}_beta_{beta}_{cover_dir}'

    # construct image pairs
    images = pd.DataFrame()
    for rate in sampling_rates:
        for method in sampling_methods:
            for aa in use_antialiasing:

                # data path
                shape = int(np.round(128/rate))
                dataset_path = pathlib.Path(f'{dataset_path_prefix}_{shape}')
                # thumb_dir = f'thumbnail_{method}_q{thumbnail_quality}_{cover_dir}'
                # get images
                cover_path = dataset_path / cover_dir
                stego_path = dataset_path / stego_dir
                cover_images = pd.read_csv(cover_path / 'files.csv')
                stego_images = pd.read_csv(stego_path / 'files.csv')

                # image absolute path
                cover_images['name'] = cover_images['name'].apply(lambda s: dataset_path / s).astype('str')
                stego_images['name'] = stego_images['name'].apply(lambda s: dataset_path / s).astype('str')

                # merge image pairs
                cover_images['fname'] = cover_images['name'].apply(lambda s: pathlib.Path(s).name).astype('str')
                stego_images['fname'] = stego_images['name'].apply(lambda s: pathlib.Path(s).name).astype('str')
                cover_images = cover_images.set_index('fname')
                stego_images = stego_images.set_index('fname')

                #
                cover_stego_images = cover_images.join(
                    stego_images[['name']],
                    how='inner', lsuffix='_cover', rsuffix='_stego',
                )
                # cover_stego_images = stego_images[['name']].join(
                #     thumb_images[['name','quality','thumbnail_type','thumbnail_quality']],
                #     how='inner', lsuffix='_stego', rsuffix='_thumb',
                # )
                cover_stego_images['sampling_method'] = method
                cover_stego_images['sampling_rate'] = rate
                cover_stego_images['use_antialiasing'] = aa
                images = pd.concat([images, cover_stego_images])

    # iterate image pairs (embarrasingly parallel)
    gen = (
        joblib.delayed(_mismatch.compute_mismatch)(
            cover_name=row.name_cover,
            stego_name=row.name_stego,
            embedding=embedding,
            embedding_rate=alpha,
            # sampling_rate=row.sampling_rate,
            sampling_method=row.sampling_method,
            use_antialiasing=row.use_antialiasing,
        )
        for index, row in images.iterrows()
    )
    res = _defs.ProgressParallel(**kw)(gen, total=len(images))

    # to long format (for processing)
    res = pd.DataFrame(res)
    res = res[[
        _defs.SAMPLING_METHOD_LABEL, _defs.SAMPLING_RATE_LABEL,
        _defs.ANTIALIASING_LABEL,
        'post',
    ]]
    res = res.rename({'post': _defs.DIFF_LABEL}, axis=1)
    print(res)
    # res = pd.melt(
    #     res[[
    #         _defs.SAMPLING_METHOD_LABEL, _defs.SAMPLING_RATE_LABEL,
    #         _defs.ANTIALIASING_LABEL,
    #         'post',
    #     ]],
    #     id_vars=(_defs.SAMPLING_METHOD_LABEL, _defs.SAMPLING_RATE_LABEL),
    #     var_name=_defs.ANTIALIASING_LABEL,
    #     value_name=_defs.DIFF_LABEL,
    # )
    # print(res)

    # compute 95%CI
    res = (
        res.groupby([_defs.SAMPLING_METHOD_LABEL, _defs.SAMPLING_RATE_LABEL, _defs.ANTIALIASING_LABEL])
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
        index=_defs.SAMPLING_RATE_LABEL,
        columns=[_defs.ANTIALIASING_LABEL, _defs.SAMPLING_METHOD_LABEL],
        values=[_defs.DIFF_LABEL+'_mean', _defs.DIFF_LABEL+'_ci95']
    ).reset_index(drop=False)
    res.columns = ['_'.join([c for c in reversed(col) if c]) for col in res.columns]

    print(res)
    return res
