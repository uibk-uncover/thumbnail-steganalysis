"""Generates dataset for feasibility study.

$ python src/generate_feasibility.py

The file will download raw images

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import argparse
import jpeglib
import logging
import numpy as np
import os
import pandas as pd
import pathlib
from PIL import Image
import requests
import subprocess
import tempfile
import tqdm
import typing
import warnings

import _seeds
import stegolab2 as sl2

# feasibility = pd.read_csv('feasibility.csv')
# feasibility['stem'] = feasibility['name'].apply(lambda n: pathlib.Path(n).stem)
# raws = pd.read_csv('raws.csv', header=None, names=['name'])
# raws['stem'] = raws['name'].apply(lambda n: pathlib.Path(n).stem)
# fnames = raws.merge(feasibility, on='stem', suffixes=('_raw', None))
# fnames[['name_raw', 'name']].to_csv('files1000.csv', index=False)


def get_imread(path: str) -> typing.Callable:
    download = not pathlib.Path(path).is_dir()

    def imread_download(fname: str) -> bytes:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                category=requests.packages.urllib3.exceptions.InsecureRequestWarning
            )
            res = requests.get(f'{path}/{fname}', verify=False)
        assert res.ok
        return res.content

    def imread_load(fname: str) -> bytes:
        with open(f'{path}/{fname[:2]}/{fname}', 'rb') as fp:
            return fp.read()

    return imread_download if download else imread_load


def generate_dataset(
    path: str,
) -> pd.DataFrame:
    """"""
    # prepare output
    outdir = pathlib.Path('data') / 'feasibility' / 'images'
    outdir.mkdir(parents=False, exist_ok=False)

    # imread
    imread = get_imread(path=path)

    # iterate files
    files = pd.read_csv('data/files_feasibility.csv')
    # files = files[:3]  # DEBUG
    pbar = tqdm.tqdm(files['name_raw'])
    images = []
    for f in pbar:
        pbar.set_description(pathlib.Path(f).name)

        # download raw
        try:
            content = imread(f)
        except Exception as e:
            logging.error(f'error fetching {f}: {str(e)}')
            raise
            continue

        # process
        with tempfile.NamedTemporaryFile() as tmp:
            with open(tmp.name, 'wb') as fp:
                fp.write(content)
            res = subprocess.run(
                [
                    'dcraw',
                    '-w',  # no white balance
                    '-d',  # grayscale
                    '-c',  # write to standard output
                    '-v',  # verbose
                    tmp.name,
                ],
                capture_output=True,
            )

        # parse PPM (output of dcraw)
        with tempfile.NamedTemporaryFile(suffix='.ppm') as tmp:
            tmp.write(res.stdout)
            x = np.array(Image.open(tmp.name))

        output_f = f'{pathlib.Path(f).stem}.png'
        res = {
            'name': str(pathlib.Path('images') / output_f),
            'height_original': x.shape[0],
            'width_original': x.shape[1],
        }

        # throw small images (<2560 in on dimension, 34 images)
        if x.shape[0] >= 2560 and x.shape[1] >= 2560:

            # top-left crop to 2560x2560
            y = np.ascontiguousarray(x[:2560, :2560])

            # write output
            Image.fromarray(y).save(outdir / output_f)

            # add to dataframe
            res['height'] = y.shape[0]
            res['width'] = y.shape[1]

        else:
            logging.warning(f'ignoring {f}: too small, shape {x.shape}')

        images.append(res)

    images = pd.DataFrame(images)
    logging.info(f'exported {len(images)} images (top-left crop 2560x2560)')
    images.to_csv(outdir / 'files.csv', index=False)
    return images


def prepare_covers(crop_size: int):
    """Compress images with libjpeg-turbo at QF75."""
    # input path
    path = pathlib.Path('data/feasibility')
    input_files = pd.read_csv(f'{path}/images/files.csv')
    input_filepaths = input_files['name'].apply(lambda f: str(path / f))

    # Create destination directory
    output_dirname = f'images_{crop_size}'
    output_dir = path / output_dirname
    output_dir.mkdir(parents=False, exist_ok=False)

    # Store records
    output_records = []

    # Iterate over images
    logging.info(f'generate_feasibility.prepare_covers on {output_dir}')
    pbar = tqdm.tqdm(enumerate(input_filepaths), total=len(input_filepaths))
    for i, input_filepath in pbar:
        input_basename = pathlib.Path(input_filepath).stem
        pbar.set_description(input_basename)

        # Ensure that cover image exists
        assert os.path.exists(input_filepath)

        # Read image and convert from PIL to numpy
        x = np.array(Image.open(input_filepath))

        # Crop image
        x = x[:crop_size, :crop_size]

        # save spatial
        compressed_filepath = output_dir / f'{input_basename}.png'
        try:
            Image.fromarray(x).save(compressed_filepath)
        except OSError as e:
            logging.error(
                f'Failed saving \"{compressed_filepath}\". '
                f'Error: \"{str(e)}\" - Skipping'
            )
            continue

        # collect to dataframe
        output_records.append({
            'name': os.path.relpath(compressed_filepath, path),
            'height': x.shape[0],
            'width': x.shape[1],
        })

    # Concatenate records
    output_df = pd.DataFrame(output_records)
    output_df = output_df.sort_values('name')
    logging.info(f'compressed {len(output_df)} images')

    output_files_csv = output_dir / 'files.csv'
    output_df.to_csv(output_files_csv, index=False)
    return output_dirname


def prepare_jpegs(crop_size: int):
    """Compress images with libjpeg-turbo at QF75."""
    # input path
    path = pathlib.Path(f'data/feasibility')
    input_files = pd.read_csv(f'{path}/images_{crop_size}/files.csv')
    input_filepaths = input_files['name'].apply(lambda f: str(path / f))

    # Create destination directory

    output_dirname = f'jpegs_q75_{crop_size}'
    output_dir = path / output_dirname
    output_dir.mkdir(parents=False, exist_ok=False)

    # Store records
    output_records = []

    # Iterate over images
    logging.info(f'generate_feasibility.prepare_jpegs on {output_dir}')
    pbar = tqdm.tqdm(enumerate(input_filepaths), total=len(input_filepaths))
    for i, input_filepath in pbar:
        input_basename = pathlib.Path(input_filepath).stem
        pbar.set_description(input_basename)

        # Ensure that cover image exists
        assert os.path.exists(input_filepath)

        # Read image and convert from PIL to numpy
        x = np.array(Image.open(input_filepath))[..., None]

        # save spatial
        compressed_filepath = output_dir / f'{input_basename}.jpeg'
        jpeglib_img = jpeglib.from_spatial(x)
        try:
            jpeglib_img.write_spatial(compressed_filepath, qt=75)
        except OSError as e:
            logging.error(
                f'Failed compressing \"{compressed_filepath}\". '
                f'Error: \"{str(e)}\" - Skipping'
            )
            continue

        # collect to dataframe
        output_records.append({
            'name': os.path.relpath(compressed_filepath, path),
            'height': x.shape[0],
            'width': x.shape[1],
        })

    # Concatenate records
    output_df = pd.DataFrame(output_records)
    output_df = output_df.sort_values('name')
    logging.info(f'compressed {len(output_df)} images')

    output_files_csv = output_dir / 'files.csv'
    output_df.to_csv(output_files_csv, index=False)
    return output_dirname


EMBEDDINGS = {
    'nsF5': sl2.nsF5,
    'UERD': sl2.uerd,
    'J-UNIWARD': sl2.juniward,
}


def prepare_stegos(
    cover_dir: str,
    method: str,
    alpha: float,
):
    """
    :param method: steganography method
    :param alpha: embedding rate
    :param skip_num_images: skip the given number of images from the beginning
    :param take_num_images: take only a given number of images
    :param skip_existing: if True, skip existing images. Otherwise, overwrite existing images.
    """
    # Find JPEG covers
    path = pathlib.Path('data/feasibility')
    cover_files = pd.read_csv(path / cover_dir / 'files.csv')
    cover_filepaths = cover_files['name'].apply(lambda f: str(path / f))

    # Create stego directory
    output_dirname = f'stego_{method}_alpha_{alpha}_{cover_dir}'
    output_dir = path / output_dirname
    output_dir.mkdir(parents=False, exist_ok=False, mode=0o775)

    # Store records for each processed image
    output_records = []

    # Iterate over cover images
    logging.info(f'stego.embed_{method} on {output_dir}')
    logging.info('using adaptive seed')

    # Find steganographic implementation
    simulate_single_channel = EMBEDDINGS[method].simulate_single_channel  # update

    pbar = tqdm.tqdm(cover_filepaths, total=len(cover_filepaths))
    for cover_filepath in pbar:
        cover_basename = pathlib.Path(cover_filepath).stem
        pbar.set_description(cover_basename)

        # Ensure that cover image exists
        assert os.path.exists(cover_filepath)

        # Load cover
        cover = jpeglib.read_dct(cover_filepath)

        # Embedding-specific parameters
        kw = {}

        # provide decompressed image
        if method in {'J-UNIWARD'}:
            kw['cover_spatial'] = jpeglib.read_spatial(
                cover_filepath,
            ).spatial[..., 0]

        # provide implementation
        if method in {'J-UNIWARD'}:
            kw['implementation'] = sl2.juniward.IMPLEMENTATION_ORIGINAL

        # provide QT
        if method in {'J-UNIWARD', 'UERD'}:
            kw['quantization_table'] = cover.qt[cover.quant_tbl_no[0]]

        # provide payload mode (for comparability)
        if method in {'UERD'}:
            kw['payload_mode'] = 'bpnzAC'

        # Generate per-component seeds
        component_seeds = _seeds.filename_to_component_seeds(cover_basename)

        # Embed
        stego = cover.copy()
        try:
            stego.Y = simulate_single_channel(
                cover_dct_coeffs=cover.Y,
                embedding_rate=alpha,
                seed=component_seeds[0],
                **kw,
            )

        except Exception as e:
            logging.error(
                f'Failed to embed {method} (alpha={alpha}) '
                f'for cover \"{cover_filepath}\". '
                f'Error: \"{str(e)}\" - Skipping'
            )
            continue

        # Save stego
        stego_filepath = output_dir / f'{cover_basename}.jpeg'
        stego.write_dct(str(stego_filepath))

        # Keep records
        output_records.append({
            'name': os.path.relpath(stego_filepath, path),
            'height': cover.height,
            'width': cover.width,
            'stego_method': method,
            'alpha': alpha,
        })

    # Concatenate records
    output_df = pd.DataFrame(output_records)
    output_df = output_df.sort_values('name')
    output_df.to_csv(output_dir / 'files.csv', index=False)
    return output_dirname


if __name__ == '__main__':
    # logging configuration
    logging.basicConfig(level=logging.INFO)
    jpeglib.version.set('turbo210')

    # arguments
    parser = argparse.ArgumentParser(
        description='Processing the steganographic dataset for training'
    )
    parser.add_argument(
        '--dataset',
        default='http://alaska.utt.fr/DATASETS/ALASKA_v2_RAWs/',
        required=False,
        type=str,
        help='path or URL to the dataset',
    )
    args = parser.parse_args()

    #
    args.dataset = pathlib.Path(args.dataset)
    generate_dataset(args.dataset)

    # generate feasibility dataset
    crop_size = 512
    prepare_covers(crop_size)
    jpeg_dir = prepare_jpegs(crop_size)
    for method in ['nsF5', 'UERD', 'J-UNIWARD']:
        for alpha in [.05, .1, .15, .2, .25, .3, .35, .4]:
            stego_dir = prepare_stegos(jpeg_dir, method=method, alpha=alpha)

    #
    for crop_size in [256, 640, 800, 960, 1024, 1280, 1600, 2048, 2560]:
        prepare_covers(crop_size)
        jpeg_dir = prepare_jpegs(crop_size)
        prepare_stegos(jpeg_dir, method='nsF5', alpha=.2)
