"""Script to prepare dataset for main experiment.

$ python3 data/generate_dataset --dataset /home/martin/ALASKA_v2_TIFF_512_COLOR

The program expects you already downloaded ALASKA2 dataset.

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import argparse
import glob
import jpeglib
import logging
import numpy as np
import os
import pandas as pd
import pathlib
from PIL import Image
import sys
import tqdm
import typing
sys.path.append('src')

import generate_feasibility as feasibility
import subsample


def prepare_jpegs(
    input_dir: pathlib.Path,
    rotation: int = 0,
):
    """
    Compress given images using the given JPEG quality and libjpeg-turbo 2.0.1

    :param input_dir: directory containing the images to be compressed
    :param rotation: rotate angle in counter-clockwise direction
    :return: data frame
    """
    # Find uncompressed cover images
    input_filepaths = sorted(glob.glob(str(input_dir / '*.tif')))
    input_filepaths = input_filepaths[:100]  # DEBUG

    # Create destination directory
    output_dirname = 'jpegs_q75'
    if rotation != 0:
        output_dirname += f'_rotation_{rotation}'

    # Create output directory
    dst = pathlib.Path('data/dataset')
    output_dir = dst / output_dirname
    output_dir.mkdir(parents=False, exist_ok=False)

    # Store records
    output_records = []

    # Iterate over images
    logging.info(f'_cover.compress on {input_dir}')
    pbar = tqdm.tqdm(enumerate(input_filepaths), total=len(input_filepaths))
    for i, input_filepath in pbar:
        input_basename = pathlib.Path(input_filepath).stem
        pbar.set_description(f'{input_basename}.jpeg')

        # Read image and convert from PIL to numpy
        x = np.array(Image.open(input_filepath).convert('L'))

        if rotation != 0:
            x = np.rot90(x, rotation // 90)
            x = np.ascontiguousarray(x)

        # Expand shape
        if len(x.shape) == 2:
            x = x[..., None]

        # write
        compressed_filepath = output_dir / f'{input_basename}.jpeg'
        jpeglib_img = jpeglib.from_spatial(x)
        try:
            jpeglib_img.write_spatial(compressed_filepath, qt=75)
        except OSError as e:
            logging.error(f'Failed to compress cover image to file \"{compressed_filepath}\". Error: \"{str(e)}\" - Skipping')
            continue

        # collect to dataframe
        output_records.append({
            'name': os.path.relpath(compressed_filepath, dst),
            'height': x.shape[0],
            'width': x.shape[1],
            'quality': 'q75',
            'rotation': rotation,
        })

    # Concatenate records
    output_df = pd.DataFrame(output_records)
    output_df = output_df.sort_values('name')
    logging.info(f'compressed {len(output_df)} images')
    output_files_csv = output_dir / 'files.csv'
    output_df.to_csv(output_files_csv, index=False)
    return output_dirname


def postprocess_dataset(
    dataset_path: typing.Union[pathlib.Path, str],
    splitpoints: typing.Tuple[float],
    seed: int = None
) -> typing.Tuple[pd.DataFrame]:
    """Shuffles and splits fabrika dataset into disjoint sets."""

    # process parameters
    dataset_path = pathlib.Path(dataset_path)
    K = len(splitpoints)

    # get number of images
    dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    if 'jpeg' in set([d.name[:4] for d in dirs]):
        covers = [d for d in dirs if 'jpeg' == d.name[:4]]
        dir_config = pd.read_csv(covers[0] / 'files.csv')
    else:
        raise Exception('cover directory not found!')

    N = len(dir_config)

    # split function
    assert np.sum(splitpoints) == 1, 'splits do not sum up to 1'
    splitpoints = np.hstack([[0], np.cumsum(np.array(splitpoints)*N)]).astype('int')

    # permute
    rng = np.random.default_rng(seed)
    names = rng.permutation([pathlib.Path(d).stem for d in dir_config.name])

    # for each dataset
    splits = [pd.DataFrame() for _ in range(K)]
    for dir in dataset_path.iterdir():
        if dir.is_dir():

            # load dataset files
            try:
                dir_config = pd.read_csv(dir / 'files.csv')
            except FileNotFoundError:
                logging.warning(f'ignoring {dir}, files.csv not present')
                continue
            # parse quality
            if 'quality' in dir_config:
                dir_config = dir_config[dir_config.quality.str.startswith('q')]
                dir_config['quality'] = dir_config.quality.apply(lambda s: int(s[1:]))
            # reorder by permutation
            dir_config['basename'] = dir_config.name.apply(lambda d: pathlib.Path(d).stem)
            dir_config = dir_config.set_index('basename')
            dir_config = dir_config.reindex(index=names)
            #
            for i in range(K):
                # get indices
                idx = names[splitpoints[i]:splitpoints[i+1]]
                dir_config_segment = dir_config.loc[idx]
                # drop missing stego files (e.g., when embedding fails)
                dir_config_segment = dir_config_segment.dropna(subset=['name'])
                # get images from this segment
                splits[i] = pd.concat([splits[i], dir_config_segment], ignore_index=True)

    for split in splits:
        if 'height' in split:
            split.height = split.height.astype(dtype=pd.Int64Dtype())
        if 'width' in split:
            split.width = split.width.astype(dtype=pd.Int64Dtype())
        if 'rotation' in split:
            split.rotation = split.rotation.astype(dtype=pd.Int64Dtype())

    return splits


def construct_output_dirname(
    dir_image: str,
    implementation: str,
    kernel: str,
    quality: int,
    antialiasing: bool,
    rotation: int,
) -> pathlib.Path:
    output_dirname = f'thumbnail_{implementation}_{kernel}'
    if antialiasing:
        output_dirname += '_aa'
    output_dirname += f'_q{quality:02d}'
    output_dirname += f'_{dir_image}'
    if rotation != 0:
        output_dirname += f'_rotation_{rotation}'
    output_dirname = pathlib.Path(output_dirname)
    return output_dirname


def is_jpeg(
    path: typing.Union[str, pathlib.Path],
) -> bool:
    """Analyses image to be JPEG.

    :param path: path to the file
    :returns: whether the file is valid JPEG
    """
    try:
        i = Image.open(path)
        format = i.format
    except IOError:
        format = None
    else:
        i.close()
    finally:
        return format == 'JPEG'


def process_thumbnail(
    image_dir: pathlib.Path,
    path: pathlib.Path,
    kernel: str = 'nearest',
    implementation: str = 'textbook',
    antialiasing: bool = False,
):
    """Produces thumbnail into dst/ from images from src/."""
    # Get uncompressed cover images
    path = pathlib.Path(path)
    src = path / image_dir
    df = pd.read_csv(src / 'files.csv')
    df['name'] = df['name'].apply(lambda fname: src / pathlib.Path(fname).name)
    # src_files = [src / pathlib.Path(fname).name for fname in df['name']]

    # Constuct names
    dst_names = df['name'].apply(lambda fname: f'{pathlib.Path(fname).stem}.jpeg').to_list()
    # dst_names = [f'{pathlib.Path(fname).stem}.jpeg' for fname in src_files]

    # Parse quality and thumbnail type
    thumbnail_type = f'{implementation}_{kernel}{":aa" if antialiasing else ""}'

    # Create destination directory
    output_dirname = f'thumbnail_{thumbnail_type}_q75_{pathlib.Path(src).name}'
    output_dirname = pathlib.Path(output_dirname)
    dst_thumbnails = path / output_dirname
    dst_thumbnails.mkdir(parents=False, exist_ok=False)

    # Store records
    output_records = []

    # Iterate over images
    logging.info(f'_thumbnail.process to {output_dirname}')
    pbar = tqdm.tqdm(df.iterrows(), total=len(df), file=sys.stdout)
    for i, row in pbar:
        pbar.set_description(pathlib.Path(row['name']).name)

        # check type
        if is_jpeg(row['name']):
            x = jpeglib.read_spatial(row['name']).spatial
        else:
            x = np.array(Image.open(row['name']))
            if len(x.shape) == 2:
                x = x[:, :, None]

        # create thumbnail
        xt = subsample.scale_image(
            x,  # pre-cover
            (128, 128),
            kernel=kernel,
            implementation=implementation,
            use_antialiasing=antialiasing,
        )

        # write
        jpeglib_img = jpeglib.from_spatial(xt)
        jpeglib_img.samp_factor = '4:4:4'
        jpeglib_img.write_spatial(
            dst_thumbnails / dst_names[i],
            qt=75,
        )

        # collect to dataframe
        output_records.append({
            **row.to_dict(),
            'name': output_dirname / dst_names[i],
            'height': 128,
            'width': 128,
            'thumbnail_type': thumbnail_type,
            'thumbnail_quality': 'q75',
            'thumbnail_postcompress': False,
        })
    #
    output_df = pd.DataFrame(output_records)
    output_df = output_df.sort_values('name')

    output_files_csv = dst_thumbnails / 'files.csv'
    output_df.to_csv(output_files_csv, index=False)


if __name__ == '__main__':
    # logging configuration
    logging.basicConfig(level=logging.INFO)
    jpeglib.version.set('turbo210')

    # arguments
    parser = argparse.ArgumentParser(
        description='Processing the dataset for R/I experiemnts'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        type=str,
        help='path or URL to the dataset',
    )
    args = parser.parse_args()

    # generate covers
    args.dataset = pathlib.Path(args.dataset)
    for rotation in [0, 90, 180, 270]:
        jpeg_dir = prepare_jpegs(args.dataset, rotation=rotation)

        for thumbnail_kernel, thumbnail_antialiasing in zip(
            ['nearest', 'bicubic', 'nearest'],
            [False, False, True],
        ):
            process_thumbnail(
                jpeg_dir,
                path='data/dataset',
                kernel=thumbnail_kernel,
                implementation='textbook',
                antialiasing=thumbnail_antialiasing,
            )


        # generate stegos
        for alpha in [.4, .2, .1]:
            for method in ['nsF5', 'UERD', 'J-UNIWARD']:
                stego_dir = feasibility.prepare_stegos(
                    jpeg_dir,
                    path='data/dataset',
                    method=method,
                    alpha=alpha,
                )
                process_thumbnail(
                    stego_dir,
                    path='data/dataset',
                    kernel='nearest',
                    implementation='textbook',
                    antialiasing=False,
                )


        # generate thumbnails
        # TODO: cover
        # TODO: stego (only one specific)

    # postprocess dataset
    splits = postprocess_dataset(
        'data/dataset',
        (.6, .2, .2),
        12345,
    )
    for split, name in zip(splits, ['tr', 'va', 'te']):
        split.to_csv(
            f'data/dataset/split_{name}.csv',
            index=False,
        )
