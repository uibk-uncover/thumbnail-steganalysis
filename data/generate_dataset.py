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
import tqdm

import generate_feasibility as feasibility


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
    input_filepaths = input_filepaths[:10]  # DEBUG

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
        im = Image.open(input_filepath)

        if im.mode == '1':
            im = im.convert('L')

        x = np.array(im)

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

        # generate stegos
        for alpha in [.4, .2, .1]:
            for method in ['nsF5', 'UERD', 'J-UNIWARD']:
                stego_dir = feasibility.prepare_stegos(
                    jpeg_dir,
                    path='data/dataset',
                    method=method,
                    alpha=alpha,
                )

        # generate thumbnails
        # TODO: cover
        # TODO: stego (only one specific)
