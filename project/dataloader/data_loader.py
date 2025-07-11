#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/project/project/dataloader/data_loader.py
Project: /workspace/project/project/dataloader
Created Date: Friday January 10th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday July 11th 2025 11:07:08 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

from torchvision.transforms import (
    Compose,
    Resize,
)

from typing import Any, Callable, Dict, Optional
from pytorch_lightning import LightningDataModule

from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import functional as F, Transform

from pytorchvideo.data import make_clip_sampler
from pytorchvideo.data.labeled_video_dataset import labeled_video_dataset

from project.dataloader.gait_video_dataset import labeled_gait_video_dataset

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
        inpt = inpt.permute(1, 0, 2, 3) # [C, T, H, W] -> [T, C, H, W]
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

        # self._seg_path = opt.data.seg_data_path
        # self._gait_seg_path = opt.data.gait_seg_data_path

        # ? 感觉batch size对最后的结果有影响，所以分开使用不同的batch size
        self._gait_cycle_batch_size = opt.data.gait_cycle_batch_size
        self._default_batch_size = opt.data.default_batch_size

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
        self._temporal_mix = opt.train.temporal_mix

        self.opt = opt

        if self._temporal_mix == True:
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

        # train dataset
        self.train_gait_dataset = labeled_gait_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx[
                0
            ],  # train mapped path, include gait cycle index.
            transform=self.mapping_transform,
            hparams=self.opt,
        )

        # val dataset
        self.val_gait_dataset = labeled_gait_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx[
                1
            ],  # val mapped path, include gait cycle index.
            transform=self.mapping_transform,
            hparams=self.opt,
        )

        # test dataset
        self.test_gait_dataset = labeled_gait_video_dataset(
            experiment=self._experiment,
            dataset_idx=self._dataset_idx[
                1
            ],  # val mapped path, include gait cycle index.
            transform=self.mapping_transform,
            hparams=self.opt,
        )

    def collate_fn(self, batch):
        """this function process the batch data, and return the batch data.

        Args:
            batch (list): the batch from the dataset.
            The batch include the one patient info from the json file.
            Here we only cat the one patient video tensor, and label tensor.

        Returns:
            dict: {video: torch.tensor, label: torch.tensor, info: list}
        """

        batch_label = []
        batch_video = []

        # * mapping label
        for i in batch:
            # logging.info(i['video'].shape)
            gait_num, *_ = i["video"].shape
            disease = i["disease"]

            batch_video.append(i["video"])
            for _ in range(gait_num):

                if disease in disease_to_num_mapping_Dict[self._class_num].keys():

                    batch_label.append(
                        disease_to_num_mapping_Dict[self._class_num][disease]
                    )
                else:
                    # * if the disease not in the mapping dict, then set the label to non-ASD.
                    batch_label.append(
                        disease_to_num_mapping_Dict[self._class_num]["non-ASD"]
                    )

        # video, b, c, t, h, w, which include the video frame from sample info
        # label, b, which include the video frame from sample info
        # sample info, the raw sample info from dataset
        return {
            "video": torch.cat(batch_video, dim=0),
            "label": torch.tensor(batch_label),
            "info": batch,
        }

    def train_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        if self._temporal_mix:
            train_data_loader = DataLoader(
                self.train_gait_dataset,
                batch_size=self._gait_cycle_batch_size,
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                shuffle=True,
                drop_last=True,
                collate_fn=self.collate_fn,
                # worker_init_fn=lambda x: torch.initial_seed(),
            )
        else:
            train_data_loader = DataLoader(
                self.train_gait_dataset,
                batch_size=self._default_batch_size,
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                shuffle=False,
                drop_last=True,
            )

        return train_data_loader

    def val_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        if self._temporal_mix:
            val_data_loader = DataLoader(
                self.val_gait_dataset,
                # batch_size=self._gait_cycle_batch_size,
                batch_size = 16, 
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                shuffle=False,
                drop_last=True,
                collate_fn=self.collate_fn,
            )
        else:
            val_data_loader = DataLoader(
                self.val_gait_dataset,
                batch_size=self._default_batch_size,
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                shuffle=False,
                drop_last=True,
            )

        return val_data_loader

    def test_dataloader(self) -> DataLoader:
        """
        create the Walk train partition from the list of video labels
        in directory and subdirectory. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """

        if self._temporal_mix:
            test_data_loader = DataLoader(
                self.test_gait_dataset,
                # batch_size=self._gait_cycle_batch_size,
                batch_size = 16,
                num_workers=self._NUM_WORKERS,
                shuffle=False,
                drop_last=True,
                collate_fn=self.collate_fn,
            )
        else:
            test_data_loader = DataLoader(
                self.test_gait_dataset,
                batch_size=self._default_batch_size,
                num_workers=self._NUM_WORKERS,
                shuffle=False,
                drop_last=True,
            )

        return test_data_loader
