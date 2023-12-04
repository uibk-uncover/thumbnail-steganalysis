"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import joblib
import jpeglib
import pandas as pd
import pathlib
import typing

from feasibility import _defs
from training.metrics import PEMeter


def compute_match(
    cover_name: pathlib.Path,
    stego_name: pathlib.Path,
    stego_method: str,
    alpha: float,
    thumbnail_shape: typing.Tuple[int],
    sampling_method: str,
    use_antialiasing: bool,
    **kw
) -> typing.Dict[str, typing.Any]:
    """Compute mismatch between cover and stego thumbnail.

    Args:
        cover_name (pathlib.Path): Cover name.
        stego_name (pathlib.Path): Stego name.
        stego_method (str): Steganographic method.
        alpha (float): Embedding rate.
        thumbnail_shape (tuple): Thumbnail shape.
        sampling_method (str): Subsampling kernel.
        use_antialiasing (bool): Whether to use antialiasing.

    """

    # load images
    x0 = jpeglib.read_spatial(cover_name, jpeglib.JCS_GRAYSCALE).spatial
    xm = jpeglib.read_spatial(stego_name, jpeglib.JCS_GRAYSCALE).spatial

    # generate thumbnails
    _, y0_t = _defs.scale_compress_image(
        x=x0,
        thumbnail_shape=thumbnail_shape,
        sampling_method=sampling_method,
        use_antialiasing=use_antialiasing,
    )
    _, ym_t = _defs.scale_compress_image(
        x=xm,
        thumbnail_shape=thumbnail_shape,
        sampling_method=sampling_method,
        use_antialiasing=use_antialiasing,
    )

    # compare DCT
    match = [(y0_t.Y != ym_t.Y).sum()]
    if y0_t.has_chrominance:
        match.append((y0_t.Cb != ym_t.Cb).sum())
        match.append((y0_t.Cr != ym_t.Cr).sum())
    res = {
        'name': cover_name.name,
        'stego_method': stego_method,
        'alpha': alpha,
        # total match (0 / 1)
        'match': int(all([m == 0 for m in match])),
        # number of mismatched coefficients
        'mismatch': int(sum(match)),
        **kw
    }
    return res


def run_reproduction(
    dataset_path: str,
    # iterable parameters
    embeddings: typing.List[str],
    alphas: typing.List[float],
    # marginalized parameter
    sampling_method: str = 'nearest',
    thumbnail_shape: typing.Tuple[int] = (128, 128),
    use_antialiasing: bool = True,
    **kw,
) -> pd.DataFrame:
    """Run the reproduction"""
    # data paths
    dataset_path = pathlib.Path(dataset_path)

    # construct image pairs
    images = pd.DataFrame()
    for embedding in embeddings:
        for alpha in alphas:

            # get images
            cover_dir = 'jpegs_q75'
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
                stego_images[['name','stego_method', 'alpha']],
                how='inner', lsuffix='_cover', rsuffix='_stego',
            ).reset_index(drop=True)
            images = pd.concat([images, cover_stego_images])

    # iterate image pairs (embarrasingly parallel)
    gen = (
        joblib.delayed(compute_match)(
            cover_name=dataset_path / row.name_cover,
            stego_name=dataset_path / (row.name_stego if label else row.name_cover),
            stego_method=row.stego_method if label == 1 else None,
            alpha=row.alpha if label == 1. else 0.,
            sampling_method=sampling_method,
            thumbnail_shape=thumbnail_shape,
            use_antialiasing=use_antialiasing,
            label=label,
        )
        for label in [0., 1.]
        for index, row in images.iterrows()
    )
    res = _defs.ProgressParallel(**kw)(gen, total=len(images)*2)
    res = pd.DataFrame(res)

    # def get_PE(df: pd.DataFrame) -> float:
    #     """Compute PE"""
    #     pemeter = PEMeter('P_E')
    #     pemeter.update(
    #         df['label'],
    #         df['mismatch'] / df['mismatch'].max(),
    #     )
    #     return pemeter.avg

    # # get covers
    # res_cover = res[res['stego_method'].isna()]
    # res_cover = res_cover.drop_duplicates(['name'])
    # print(f'{len(res_cover)=}')

    scores = []
    for (stego_method, alpha), res_stego in res.groupby(['stego_method', 'alpha']):
        if not stego_method:
            continue

        scores.append({
            'stego_method': stego_method,
            'alpha': alpha,
            # 'P_E': get_PE(pd.concat([res_cover, res_stego])),
            'P_FP': (res_stego['match'] == 1).mean(),
            'mismatch': res_stego['match'].sum(),
            'count': len(res_stego),
        })
    return pd.DataFrame(scores)


if __name__ == '__main__':
    # marginal parameters
    sampling_method = 'nearest'
    thumbnail_shape = (128, 128)
    use_antialiasing = False

    # run embedding
    res = run_reproduction(
        # '/home/martin/fabrika/alaska1000_512',
        'data/dataset',
        # iterable parameters
        embeddings=['nsF5'],#,'J-UNIWARD', 'UERD'],
        alphas=[.4],#, .2, .1],
        # marginalized parameter
        sampling_method=sampling_method,
        thumbnail_shape=thumbnail_shape,
        use_antialiasing=use_antialiasing,
        # joblib
        n_jobs=20, backend='loky'
        # prefer='threads'
    )

    # export
    res.to_csv(
        'results/reproduction/reproduction'
        f'_sampling_{sampling_method}{"_aa" if use_antialiasing else ""}'
        '.csv',
        index=False,
    )

    # save latex table
    res.to_latex(
        'results/reproduction/reproduction_table.tex',
        index=False,
        float_format='%.4f',
        escape=False,
        multicolumn_format='c',
        header=False,
    )




