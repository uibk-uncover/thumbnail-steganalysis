
# import imageops
import logging
import numpy as np
import pandas as pd
import skimage
import typing

try:
    from .loader_baseline import StegoDataset
except:
    from loader_baseline import StegoDataset

import subsample

pd.options.mode.chained_assignment = None  # suppress Pandas warning


class CoverThumbnailDataset(StegoDataset):
    """"""
    def __init__(
        self,
        #
        *args,
        # thumbnail parameters
        thumbnail_shape: typing.Tuple[int],
        thumbnail_implementation: str,
        thumbnail_kernel: str,
        thumbnail_antialiasing: bool,
        thumbnail_quality: int,
        thumbnail_precompress: bool,
        thumbnail_hshift: int = None,
        #
        **kw,
    ):
        # parameters
        self.thumbnail_shape = thumbnail_shape
        self.thumbnail_type = f'{thumbnail_implementation}_{thumbnail_kernel}{":aa" if thumbnail_antialiasing else ""}'
        self.thumbnail_quality = f'q{thumbnail_quality}'
        self.thumbnail_precompress = thumbnail_precompress
        self.thumbnail_hshift = thumbnail_hshift

        # call parent constructor
        super().__init__(*args, **kw)

        try:
            if self.skip_thumbnail_processing:
                return
        except AttributeError:
            pass

        # filter based on parameters
        config_thumb = self.filter_thumbnail(self.config)

        # set thumbnail
        self.config_thumb = config_thumb.reset_index(drop=True)

        # drop unpaired samples
        if self.drop_unpaired:
            self._sync_thumb()

            # drop unpaired samples
            no_image = self.config_thumb.index.difference(
                self.config_cover.index
            )
            no_thumbnail = self.config_cover.index.difference(
                self.config_thumb.index
            )
            for f in no_image:
                logging.warning(
                    f'dropping thumbnail {f}, corresponding images not found'
                )
            for f in no_thumbnail:
                logging.warning(
                    f'dropping image {f}, corresponding thumbnail not found'
                )
            self.config_cover = self.config_cover.drop(no_thumbnail)
            self.config_stego = self.config_stego.drop(no_thumbnail)
            self.config_thumb = self.config_thumb.drop(no_image)

            # check dataset
            Nc, Nt = len(self.config_cover), len(self.config_thumb)
            assert Nt > 0, 'no such thumbnails found'
            assert Nc == Nt, 'imbalanced thumbnail dataset'

    def filter_thumbnail(self, config: pd.DataFrame, stego: bool = False) -> pd.DataFrame:
        if not self.thumbnail_precompress:
            config = config[config.quality == self.quality]
        if self.thumbnail_shape is not None:
            config = config[config.height == self.thumbnail_shape[0]]
            config = config[config.width == self.thumbnail_shape[1]]
        if not self.rotation:
            config = config[config.rotation == 0]
        config = config[config.thumbnail_type == self.thumbnail_type]
        config = config[config.thumbnail_quality == self.thumbnail_quality]
        config = config[config.thumbnail_postcompress != self.thumbnail_precompress]
        if not stego:
            config = config[config.stego_method.isna()]
        else:
            config = config[config.stego_method == self.stego_method]
            config = config[config.alpha == self.alpha]
        return config

    def _to_common_key(self, config):
        config['fname'] = config.apply(self.basename, axis=1)
        return config.set_index('fname')

    def _sync_thumb(self):
        """Order thumb based on cover."""

        # convert to common key (basename)
        self.config_cover = self._to_common_key(self.config_cover)
        self.config_stego = self._to_common_key(self.config_stego)
        self.config_thumb = self._to_common_key(self.config_thumb)

        # sort jointly
        self.config_thumb = self.config_thumb.reindex(
            index=self.config_cover.index
        )
        # drop missing
        self.config_thumb = self.config_thumb.dropna(subset=['name'])
        self.config_thumb['name'] = self.config_thumb.name.astype(str)

    def __getitem__(self, idx):
        """Index dataset"""
        # cover or stego
        is_stego = idx % 2 != 0  # zigzag cover-stego
        config = self.config_stego if is_stego else self.config_cover
        target = int(is_stego)

        # load image
        image_row = config.iloc[idx // 2, :]
        thumb_name = self.config_thumb.loc[self.basename(image_row), 'name']
        if self.debug:
            return image_row['name'], thumb_name
        image = self.imread(self.dataset_path / image_row['name'])
        # generate empty thumbnail
        # thumb = np.zeros(image.shape, dtype=np.float32)
        thumb = self.imread(self.dataset_path / thumb_name)
        # upsample thumbnail
        # thumb = skimage.transform.rescale(
        #     thumb,
        #     image.shape[0] / thumb.shape[0],  # sampling rate
        #     order=0,  # nearest
        #     channel_axis=2,
        # )

        thumb = subsample.scale_image(
            thumb, image.shape[1::-1],
            kernel='nearest'
        )

        # desync JPEG phase
        if self.hshift > 0:
            image[:, :-self.hshift] = image[:, self.hshift:]
        if self.thumbnail_hshift > 0:
            thumb[:, :-self.thumbnail_hshift] = thumb[:, self.thumbnail_hshift:]

        # concatenate along channels
        image = np.concatenate([image, thumb], axis=2)

        # transform
        if self.transform is not None:
            image = self.transform(image)
            # image = self.transform(image=image)
            # image = image['image']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    @classmethod
    def get_side_information(cls, args):
        kw_side_information = {
            k: args[k]
            for k in [
                'thumbnail_shape',
                'thumbnail_implementation',
                'thumbnail_kernel',
                'thumbnail_antialiasing',
                'thumbnail_quality',
                'thumbnail_precompress',
                'thumbnail_stego',
                'thumbnail_hshift',
            ]
        }
        # kw_side_information['thumbnail_stego'] = False
        return kw_side_information


# for split in ['tr', 'va', 'te']:

#     print(f'{split=}')
#     dataset = CoverThumbnailDataset(
#         # dataset root
#         'data/dataset/',
#         # config file
#         f'data/dataset/split_{split}.csv',
#         # training parameters
#         shape=(512, 512),
#         quality=75,
#         stego_method='J-UNIWARD',
#         alpha=.4,
#         # thumbnail
#         thumbnail_shape=(128, 128),
#         thumbnail_kernel='nearest',
#         thumbnail_implementation='textbook',
#         thumbnail_antialiasing=False,
#         thumbnail_quality=75,
#         thumbnail_precompress=True,
#         # pair constraint
#         pair_constraint=True,
#         seed=12345,  # for shuffling, if PC=False
#         #
#         debug=True,
#     )

#     dataset.reshuffle()
#     for i, x in enumerate(dataset):
#         if i > 3:
#             break
#         print(x)


