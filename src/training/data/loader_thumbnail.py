
import logging
import numpy as np
import pandas as pd
import skimage
import typing

try:
    from .loader_cover_thumbnail import CoverThumbnailDataset
except:
    from loader_cover_thumbnail import CoverThumbnailDataset

import subsample

pd.options.mode.chained_assignment = None  # suppress Pandas warning


class ThumbnailDataset(CoverThumbnailDataset):
    """"""
    def __init__(
        self,
        *args,
        **kw,
    ):
        # call parent constructor
        self.skip_thumbnail_processing = True
        super().__init__(*args, **kw)
        print('using ThumbnailDataset')

        # filter based on parameters
        config_cover_thumb = self.filter_thumbnail(self.config, stego=False)
        config_stego_thumb = self.filter_thumbnail(self.config, stego=True)

        # set thumbnails
        self.config_cover_thumb = config_cover_thumb.reset_index(drop=True)
        self.config_stego_thumb = config_stego_thumb.reset_index(drop=True)

        # drop unpaired samples
        if self.drop_unpaired:
            self._sync_thumb()

            # drop unpaired with cover thumbnail
            no_image = self.config_cover_thumb.index.difference(
                self.config_cover.index
            )
            no_cover_thumbnail = self.config_cover.index.difference(
                self.config_cover_thumb.index
            )
            for f in no_image:
                logging.warning(
                    f'dropping thumbnail {f}, corresponding images not found'
                )
            for f in no_cover_thumbnail:
                logging.warning(
                    f'dropping image {f}, corresponding cover thumbnail not found'
                )
            self.config_cover = self.config_cover.drop(no_cover_thumbnail)
            self.config_stego = self.config_stego.drop(no_cover_thumbnail)
            self.config_stego_thumb = self.config_stego_thumb.drop(no_cover_thumbnail)
            self.config_cover_thumb = self.config_cover_thumb.drop(no_image)

            # drop unpaired with stego thumbnail
            no_image = self.config_stego_thumb.index.difference(
                self.config_cover.index
            )
            no_stego_thumbnail = self.config_cover.index.difference(
                self.config_stego_thumb.index
            )
            for f in no_image:
                logging.warning(
                    f'dropping cover thumbnail {f}, corresponding images not found'
                )
            for f in no_stego_thumbnail:
                logging.warning(
                    f'dropping image {f}, corresponding stego thumbnail not found'
                )
            self.config_cover = self.config_cover.drop(no_stego_thumbnail)
            self.config_stego = self.config_stego.drop(no_stego_thumbnail)
            self.config_cover_thumb = self.config_cover_thumb.drop(no_stego_thumbnail)
            self.config_stego_thumb = self.config_stego_thumb.drop(no_image)

        # check dataset
        Nc, Ntc = len(self.config_cover), len(self.config_cover_thumb)
        assert Ntc > 0, 'no such cover thumbnails found'
        assert Nc == Ntc, 'imbalanced cover thumbnail dataset'
        Ns, Nts = len(self.config_stego), len(self.config_stego_thumb)
        assert Nts > 0, 'no such stego thumbnails found'
        assert Ns == Nts, 'imbalanced stego thumbnail dataset'

    def _sync_thumb(self):
        """Order thumb based on cover and stego"""

        # convert to common key (basename)
        self.config_cover = self._to_common_key(self.config_cover)
        self.config_stego = self._to_common_key(self.config_stego)
        self.config_cover_thumb = self._to_common_key(self.config_cover_thumb)
        self.config_stego_thumb = self._to_common_key(self.config_stego_thumb)

        # sort jointly
        self.config_cover_thumb = self.config_cover_thumb.reindex(
            index=self.config_cover.index
        )
        self.config_stego_thumb = self.config_stego_thumb.reindex(
            index=self.config_stego.index
        )
        # drop missing
        self.config_cover_thumb = self.config_cover_thumb.dropna(subset=['name'])
        self.config_cover_thumb['name'] = self.config_cover_thumb.name.astype(str)
        self.config_stego_thumb = self.config_stego_thumb.dropna(subset=['name'])
        self.config_stego_thumb['name'] = self.config_stego_thumb.name.astype(str)

    def __getitem__(self, idx):
        """Index dataset"""
        # cover or stego
        is_stego = idx % 2 != 0  # zigzag cover-stego
        config_image = self.config_stego if is_stego else self.config_cover
        config_thumb = self.config_stego_thumb if is_stego else self.config_cover_thumb
        target = int(is_stego)

        # load image
        image_row = config_image.iloc[idx // 2, :]
        thumb_name = config_thumb.loc[self.basename(image_row), 'name']
        if self.debug:
            return image_row['name'], thumb_name
        image = self.imread(self.dataset_path / image_row['name'])
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

        # concatenate along channels
        image = np.concatenate([image, thumb], axis=2)

        # transform
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    @classmethod
    def get_side_information(cls, args):
        kw_side_information = super(cls, cls).get_side_information(args)
        kw_side_information['thumbnail_stego'] = True
        return kw_side_information


# for split in ['tr', 'va', 'te']:

#     print(f'{split=}')
#     dataset = StegoThumbnailDataset(
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
#         if i > 6:
#             break
#         print(x)

