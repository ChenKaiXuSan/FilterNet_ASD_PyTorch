#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/project/project/dataloader/data_loader.py
Project: /workspace/project/project/dataloader
Created Date: Thursday January 9th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday January 9th 2025 12:29:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from torchvision.transforms import (
    Compose,
    Resize,
)

from typing import Any, Callable, Dict, Optional
from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import functional as F, Transform

from pytorchvideo.data import make_clip_sampler
from pytorchvideo.data.labeled_video_dataset import labeled_video_dataset

from filter.dataloader.filter_gait_video_dataset import (
    labeled_gait_video_dataset,
)


class UniformTemporalSubsample(Transform):
    """Uniformly subsample ``num_samples`` indices from the temporal dimension of the video.

    Videos are expected to be of shape ``[..., T, C, H, W]`` where ``T`` denotes the temporal dimension.

    When ``num_samples`` is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.

    Args:
        num_samples (int): The number of equispaced samples to be selected
    """

    _transformed_types = (torch.Tensor,)

    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = num_samples

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = inpt.permute(1, 0, 2, 3)  # [C, T, H, W] -> [T, C, H, W]
        return self._call_kernel(F.uniform_temporal_subsample, inpt, self.num_samples)


disease_to_num_mapping_Dict: Dict = {
    2: {"ASD": 0, "non-ASD": 1},
    3: {"ASD": 0, "DHS": 1, "LCS_HipOA": 2},
    4: {"ASD": 0, "DHS": 1, "LCS_HipOA": 2, "normal": 3},
}


class ApplyTransformToKey:
    """
    Applies transform to key of dictionary input.

    Args:
        key (str): the dictionary key the transform is applied to
        transform (callable): the transform that is applied

    Example:
        >>>   transforms.ApplyTransformToKey(
        >>>       key='video',
        >>>       transform=UniformTemporalSubsample(num_video_samples),
        >>>   )
    """

    def __init__(self, key: str, transform: Callable):
        self._key = key
        self._transform = transform

    def __call__(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x[self._key] = self._transform(x[self._key])
        return x


class Div255(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.div_255``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale clip frames from [0, 255] to [0, 1].
        Args:
            x (Tensor): A tensor of the clip's RGB frames with shape:
                (C, T, H, W).
        Returns:
            x (Tensor): Scaled tensor by dividing 255.
        """
        return x / 255.0


class WalkDataModule(LightningDataModule):
    def __init__(self, opt, dataset_idx: Dict = None):
        super().__init__()

        self._gait_cycle_batch_size = opt.data.gait_cycle_batch_size

        self._NUM_WORKERS = opt.data.num_workers
        self._IMG_SIZE = opt.data.img_size

        # frame rate
        self._CLIP_DURATION = opt.train.clip_duration
        self.uniform_temporal_subsample_num = opt.train.uniform_temporal_subsample_num

        # * this is the dataset idx, which include the train/val dataset idx.
        self._dataset_idx = dataset_idx
        self._class_num = opt.model.model_class_num

        self._experiment = opt.train.experiment
        self._backbone = opt.train.backbone

        if "2dcnn" in self._backbone or "vit" in self._backbone:
            self.mapping_transform = Compose(
                [Div255(), Resize(size=[self._IMG_SIZE, self._IMG_SIZE])]
            )
        else:
            self.mapping_transform = Compose(
                [
                    UniformTemporalSubsample(self.uniform_temporal_subsample_num),
                    Div255(),
                    Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                ]
            )

        self.train_video_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Div255(),
                            Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                            UniformTemporalSubsample(
                                self.uniform_temporal_subsample_num
                            ),
                        ]
                    ),
                ),
            ]
        )

        self.val_video_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Div255(),
                            Resize(size=[self._IMG_SIZE, self._IMG_SIZE]),
                            UniformTemporalSubsample(
                                self.uniform_temporal_subsample_num
                            ),
                        ]
                    ),
                ),
            ]
        )

    def prepare_data(self) -> None:
        """here prepare the temp val data path,
        because the val dataset not use the gait cycle index,
        so we directly use the pytorchvideo API to load the video.
        AKA, use whole video to validate the model.
        """
        ...

    def setup(self, stage: Optional[str] = None) -> None:
        """
        assign tran, val, predict datasets for use in dataloaders

        Args:
            stage (Optional[str], optional): trainer.stage, in ('fit', 'validate', 'test', 'predict'). Defaults to None.
        """

        if "whole" in self._experiment:
            # * Here we use 1s30 frames to get a static image

            # train dataset
            self.train_gait_dataset = labeled_video_dataset(
                data_path=self._dataset_idx[2],
                clip_sampler=make_clip_sampler("uniform", 1),
                transform=self.train_video_transform,
            )

            # val dataset
            self.val_gait_dataset = labeled_video_dataset(
                data_path=self._dataset_idx[3],
                clip_sampler=make_clip_sampler("uniform", 1),
                transform=self.val_video_transform,
            )

            # test dataset
            self.test_gait_dataset = labeled_video_dataset(
                data_path=self._dataset_idx[3],
                clip_sampler=make_clip_sampler("uniform", 1),
                transform=self.val_video_transform,
            )

        elif "stance" or "swing" in self._experiment:

            # train dataset
            self.train_gait_dataset = labeled_gait_video_dataset(
                experiment=self._experiment,
                dataset_idx=self._dataset_idx[
                    0
                ],  # train mapped path, include gait cycle index.
                transform=self.mapping_transform,
            )

            # val dataset
            self.val_gait_dataset = labeled_gait_video_dataset(
                experiment=self._experiment,
                dataset_idx=self._dataset_idx[
                    1
                ],  # val mapped path, include gait cycle index.
                transform=self.mapping_transform,
            )

            # test dataset
            self.test_gait_dataset = labeled_gait_video_dataset(
                experiment=self._experiment,
                dataset_idx=self._dataset_idx[
                    1
                ],  # val mapped path, include gait cycle index.
                transform=self.mapping_transform,
            )

        else:
            raise ValueError("the experiment backbone is not supported.")

    def collate_fn(self, batch):
        """this function process the batch data, and return the batch data.

        Args:
            batch (list): the batch from the dataset.
            The batch include the one patient info from the json file.
            Here we only cat the one patient video tensor, and label tensor.

        Returns:
            dict: {video: torch.tensor, label: torch.tensor, info: list}
        """

        if "whole" in self._experiment:
            label = []
            video = torch.cat([i["video"] for i in batch], dim=0)

            for i in batch:
                _label = torch.tensor(i["label"])
                label.append(torch.repeat_interleave(_label, i["video"].shape[0]))

            return {
                "video": video.permute(1, 0, 2, 3),
                "label": torch.cat(label, dim=0),
                "info": batch,
            }

        # ! why here need to unpack the batch data, maybe can deprecated.
        batch_label = []
        batch_video = []

        # * mapping label
        for i in batch:
            # logging.info(i['video'].shape)
            c, t, h, w = i["video"].shape
            disease = i["disease"]

            batch_video.append(i["video"])
            for _ in range(t):

                if disease in disease_to_num_mapping_Dict[self._class_num].keys():

                    batch_label.append(
                        disease_to_num_mapping_Dict[self._class_num][disease]
                    )
                else:
                    # * if the disease not in the mapping dict, then set the label to non-ASD.
                    batch_label.append(
                        disease_to_num_mapping_Dict[self._class_num]["non-ASD"]
                    )

        assert t == len(batch_label)

        return {
            "video": torch.cat(batch_video, dim=1),
            "label": torch.tensor(batch_label),
            "info": batch,
        }

    def train_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        if "whole" in self._experiment:
            train_data_loader = DataLoader(
                self.train_gait_dataset,
                batch_size=self._gait_cycle_batch_size,
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                shuffle=False,  # whole do not need shuffle
                drop_last=True,
                collate_fn=self.collate_fn,
            )
        else:
            train_data_loader = DataLoader(
                self.train_gait_dataset,
                batch_size=self._gait_cycle_batch_size,
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                shuffle=True,
                drop_last=True,
                collate_fn=self.collate_fn,
            )

        return train_data_loader

    def val_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        val_data_loader = DataLoader(
            self.val_gait_dataset,
            batch_size=self._gait_cycle_batch_size,
            num_workers=self._NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

        return val_data_loader

    def test_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        test_data_loader = DataLoader(
            self.test_gait_dataset,
            batch_size=self._gait_cycle_batch_size,
            num_workers=self._NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

        return test_data_loader
