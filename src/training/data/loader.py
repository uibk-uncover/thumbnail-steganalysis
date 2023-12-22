
import jpeglib
import json
import logging
import numpy as np
import pandas as pd
import pathlib
import timm
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import typing

try:
    from .loader_baseline import StegoDataset, imread
    from .loader_cover_thumbnail import CoverThumbnailDataset
    from .loader_thumbnail import ThumbnailDataset
except ImportError:
    from loader_baseline import StegoDataset
    from loader_cover_thumbnail import CoverThumbnailDataset
    from loader_thumbnail import ThumbnailDataset

jpeglib.version.set('turbo210')


def get_timm_transform(
    in_chans: int,
    shape: typing.Tuple[int],
    mean: float,
    std: float,
    post_flip: bool = False,
):
    transform = [
        transforms.ToTensor(),
        transforms.CenterCrop(512),
        transforms.Normalize(mean, std),
    ]
    if post_flip:
        transform += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    return transforms.Compose(transform)


class RandomRotationDataset(Dataset):
    def __init__(
        self,
        datasets: typing.Sequence[Dataset],
        augment_seed: int = None,
    ):
        # super().__init__(datasets)
        self.datasets = datasets
        self.D = len(self.datasets)

        # get indices
        indices = [d.cover_index() for d in self.datasets]

        # merge non-duplicated indices, sort
        self.index = (
            pd.concat(indices)
            .drop_duplicates()
            .sort_values()
        )

        # remove missing
        missing = []
        for c in self.index:
            if any([c not in d for d in self.datasets]):
                missing.append(c)
        for f in missing:
            logging.warning(f'dropping {f}, some rotations missing')
        self.index = self.index.drop(missing)

        # for rotation selection
        self._rng = np.random.default_rng(augment_seed)
        self._perm = list(range(len(self)))
        self.rotations = np.zeros(len(self), dtype='uint8')

    def reshuffle(self):
        """"""
        # shuffle
        self._rng.shuffle(self._perm)
        self.rotations = self._rng.choice(
            range(self.D),
            size=len(self._perm),
        )

        # shuffle datasets
        for d in self.datasets:
            d.reshuffle()

    def __getitem__(self, idx: int):

        # get permuted dataset index
        perm_idx = self._perm[idx]

        # index rotation
        d_idx = self.rotations[perm_idx]

        # get dataset
        return self.datasets[d_idx][perm_idx]

    def __len__(self) -> int:
        return 2 * len(self.index)


def get_data_loader(
    config_path: pathlib.Path,
    args: typing.Dict[str, typing.Any],
    in_chans: int,
    augment: bool = False,
    debug: bool = False,
):
    """"""
    # Normalization using green
    mean = list(timm.data.constants.IMAGENET_DEFAULT_MEAN)[1:2]
    std = list(timm.data.constants.IMAGENET_DEFAULT_STD)[1:2]
    if args['thumbnail']:
        mean = list([*mean, *mean])
        std = list([*std, *std])

    # Dataset
    if args['thumbnail']:
        if args.get('thumbnail_stego', False):
            constructDataset = ThumbnailDataset
        else:
            constructDataset = CoverThumbnailDataset
    else:
        constructDataset = StegoDataset
    kw_side_information = constructDataset.get_side_information(args)
    print('Side-information')
    print(json.dumps(kw_side_information, indent=4))

    # Dataset transform
    transform = get_timm_transform(
        in_chans,
        args['shape'],
        mean=mean,
        std=std,
        post_flip=args.get('post_flip', False),
    )
    print('Data transform')
    print(transform)

    dataset = constructDataset(
        # dataset
        args['dataset'],
        config_path,
        imread=imread,
        # training parameters
        shape=args['shape'],
        quality=args['quality'],
        stego_method=args['stego_method'],
        alpha=args['alpha'],
        rotation=args['pre_rotate'],  #
        # pair constraint
        seed=args['seed'],  # for shuffling, if PC=False
        # other
        hshift=args.get('hshift', 0),
        transform=transform,
        debug=debug,
        **kw_side_information,
    )

    # Create data loaders
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=True,
    )

    #
    return loader, dataset


# for split in ['tr', 'va', 'te']:
#     split = 'te'
#     print(f'{split=}')
#     datasets = []
#     for rotate in range(4):
#         dataset = StegoDataset(
#             # dataset root
#             'data/dataset/',
#             # config file
#             f'data/dataset/split_{split}.csv',
#             # training parameters
#             shape=(512, 512),
#             quality=75,
#             stego_method='J-UNIWARD',
#             alpha=.4,
#             rotation=rotate * 90,
#             # pair constraint
#             pair_constraint=False,
#             seed=12345,  # for shuffling, if PC=False
#             #
#             debug=True,
#         )

#         datasets.append(dataset)

#     # create rotation cover-stego loader
#     rot_dataset = RandomRotationDataset(datasets)
#     loader = torch.utils.data.DataLoader(
#         rot_dataset,
#         batch_size=8,
#         shuffle=False,
#         num_workers=0,
#         pin_memory=True,
#     )

#     rot_dataset.reshuffle()
#     for i, f in enumerate(loader):
#         print(f)
#         break

