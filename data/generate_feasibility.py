"""Generates dataset for feasibility study.

$ python src/generate_feasibility.py

Author: Martin Benes
Affiliation: University of Innsbruck
"""


import csv
import logging
import numpy as np
import pandas as pd
import pathlib
from PIL import Image
import requests
import subprocess
import tempfile
import tqdm
import warnings


# feasibility = pd.read_csv('feasibility.csv')
# feasibility['stem'] = feasibility['name'].apply(lambda n: pathlib.Path(n).stem)
# raws = pd.read_csv('raws.csv', header=None, names=['name'])
# raws['stem'] = raws['name'].apply(lambda n: pathlib.Path(n).stem)
# fnames = raws.merge(feasibility, on='stem', suffixes=('_raw', None))
# fnames[['name_raw', 'name']].to_csv('files1000.csv', index=False)


URL = 'http://alaska.utt.fr/DATASETS/ALASKA_v2_RAWs/'


def generate():
    """"""
    # prepare output
    outdir = pathlib.Path('feasibility')
    outdir.mkdir(parents=False, exist_ok=False)

    # iterate files
    files = pd.read_csv('files_feasibility.csv')
    files = files[:10]
    pbar = tqdm.tqdm(files['name_raw'])
    images = []
    for f in pbar:
        pbar.set_description(pathlib.Path(f).name)

        # download raw
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    category=requests.packages.urllib3.exceptions.InsecureRequestWarning
                )
                res = requests.get(f'{URL}/{f}', verify=False)
            assert res.ok
        except Exception as e:
            logging.error(f'error fetching {f}: {str(e)}')
            continue

        # process
        with tempfile.NamedTemporaryFile() as tmp:
            with open(tmp.name, 'wb') as fp:
                fp.write(res.content)
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
            'name': output_f,
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


if __name__ == '__main__':
    generate()
