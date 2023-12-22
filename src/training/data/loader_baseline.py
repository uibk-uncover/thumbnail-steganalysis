
import jpeglib
import logging
import numpy as np
import pandas as pd
import pathlib
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import typing

pd.options.mode.chained_assignment = None  # suppress Pandas warning


def imread(f):
    return jpeglib.read_spatial(f).spatial


class StegoDataset(Dataset):
    """"""
    def __init__(
        self,
        # get dataset
        dataset_path: typing.Union[pathlib.Path, str],
        # get split
        config_path: typing.Union[pathlib.Path, str],
        # filter dataset
        shape: typing.Tuple[int],
        quality: int,
        stego_method: str,
        alpha: float,
        # rotation
        rotation: int = None,
        # pair constraint
        pair_constraint: bool = False,
        seed: int = None,
        # additional
        imread: typing.Optional[typing.Callable] = imread,
        transform: typing.Optional[typing.Callable] = None,
        target_transform: typing.Optional[typing.Callable] = None,
        drop_unpaired: bool = True,
        hshift: int = 0,
        #
        debug: bool = False,  # returning filenames instead of images
    ):
        # parameters
        self.dataset_path = pathlib.Path(dataset_path)
        self.shape = shape
        self.quality = quality
        self.stego_method = stego_method
        self.alpha = alpha
        self.rotation = rotation
        self.pair_constraint = pair_constraint
        self.imread = imread
        self.debug = debug
        self.transform = transform
        self.target_transform = target_transform
        self.drop_unpaired = drop_unpaired
        self.hshift = hshift

        # get selected dataset
        self.config = pd.read_csv(config_path, low_memory=False)

        # filter based on parameters
        config_cover, config_stego = self.filter_cover_stego(self.config)

        # set cover and stego
        self.config_cover = config_cover.reset_index(drop=True)
        self.config_stego = config_stego.reset_index(drop=True)

        # drop unpaired samples
        if self.drop_unpaired:
            self._sync_stego()

            no_cover = self.config_stego.index.difference(
                self.config_cover.index
            )
            no_stego = self.config_cover.index.difference(
                self.config_stego.index
            )
            for f in no_cover:
                logging.warning(f'dropping stego {f}, corresponding cover not found')
            for f in no_stego:
                logging.warning(f'dropping cover {f}, corresponding stego not found')
            self.config_cover = self.config_cover.drop(no_stego)
            self.config_stego = self.config_stego.drop(no_cover)

        # for dataset reshuffling
        self._rng = np.random.default_rng(seed)

        # check dataset
        assert len(self.config_cover) > 0, 'no such covers found, did you forget running preprocess_dataset.py?'
        assert len(self.config_stego) > 0, 'no such stegos found, did you forget running preprocess_dataset.py?'
        assert len(self.config_cover) == len(self.config_stego), 'imbalanced cover-stego dataset'

    @staticmethod
    def basename(r):
        return f'{pathlib.Path(r["name"]).stem}_{int(r["rotation"])}'

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.config_cover) * 2

    def reshuffle(self):
        """Call after each epoch to reshuffle."""
        # shuffle cover
        seed = self._rng.integers(2**32-1)
        self.config_cover = self.config_cover.sample(
            frac=1,
            random_state=seed,
        )
        # shuffle stego
        if not self.pair_constraint:
            seed = self._rng.integers(2**32-1)
            self.config_stego = self.config_stego.sample(
                frac=1,
                random_state=seed,
            )
        # sync stego
        else:
            self._sync_stego()

        # reset indices
        self._reset_indices()

    def filter_cover_stego(
        self,
        config: pd.DataFrame,
    ) -> typing.Tuple[pd.DataFrame]:
        """"""
        # filter dataset
        config = config[~config.quality.isnull()]
        config = config[config.quality == self.quality]
        config = config[
            (config.height == self.shape[0]) &
            (config.width == self.shape[1])
        ]
        # rotation
        config[config.rotation.isnull()] = 0
        config['rotation'] = config['rotation'].astype(int)
        if self.rotation is None:
            config = config[config.rotation == 0]
        # get cover
        config_cover = config[config.stego_method.isnull()]
        # get stego
        config_stego = config[config.stego_method == self.stego_method]
        config_stego = config_stego[config_stego.alpha == self.alpha]
        #
        return config_cover, config_stego

    def cover_index(self):
        df = self.config_cover.apply(self.basename, axis=1)
        return df

    def stego_index(self):
        return self.config_stego.apply(self.basename, axis=1)

    def __contains__(self, name: str) -> bool:
        return name in self.config_cover.index

    def _reset_indices(self):
        """Reset indices of cover/stego datasets."""
        self.config_cover = self.config_cover.reset_index(drop=True)
        self.config_stego = self.config_stego.reset_index(drop=True)

    def _sync_stego(self):
        """Order stego based on cover."""
        # convert to common key (basename)
        self.config_cover['fname'] = self.cover_index()
        self.config_cover = self.config_cover.set_index('fname')
        self.config_stego['fname'] = self.stego_index()
        self.config_stego = self.config_stego.set_index('fname')
        # sort jointly
        self.config_stego = self.config_stego.reindex(
            index=self.config_cover.index,
        )
        # drop missing
        self.config_stego = self.config_stego.dropna(subset=['name'])

    def __getitem__(self, idx: int):
        """Index dataset."""
        # cover or stego
        is_stego = idx % 2 != 0  # zigzag cover-stego
        config = self.config_stego if is_stego else self.config_cover
        target = int(is_stego)

        # load image
        image_name = config.iloc[idx // 2, :]['name']
        if self.debug:
            return image_name
        image = self.imread(self.dataset_path / image_name)

        # desync JPEG phase
        if self.hshift > 0:
            image[:, :-self.hshift] = image[:, self.hshift:]

        # transform
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # retreive image+target pairs
        return image, target

    def __iter__(self):
        """Iterate over cover-stego pairs."""
        for i in range(len(self) // 2):
            yield self[2*i]  # cover
            yield self[2*i+1]  # stego

    @classmethod
    def get_side_information(cls, args):
        """Retreive side information."""
        return {}


# for split in ['tr', 'va', 'te']:

#     print(f'{split=}')
#     dataset = StegoDataset(
#         # dataset root
#         'data/dataset/',
#         # config file
#         f'data/dataset/split_{split}.csv',
#         # training parameters
#         shape=(512, 512),
#         quality=75,
#         stego_method='J-UNIWARD',
#         alpha=.4,
#         rotation=90,
#         # pair constraint
#         pair_constraint=False,
#         seed=12345,  # for shuffling, if PC=False
#         #
#         debug=True,
#     )

#     dataset.reshuffle()
#     for i, x in enumerate(dataset):
#         if i > 2:
#             break
#         print(x)
