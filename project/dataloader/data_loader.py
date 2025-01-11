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
Last Modified: Friday January 10th 2025 11:01:28 am
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
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    Div255,
)

from typing import Any, Callable, Dict, Optional, Type
from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.data.labeled_video_dataset import labeled_video_dataset

from project.dataloader.gait_video_dataset import labeled_gait_video_dataset

disease_to_num_mapping_Dict: Dict = {
    2: {"ASD": 0, "non-ASD": 1},
    3: {"ASD": 0, "DHS": 1, "LCS_HipOA": 2},
    4: {"ASD": 0, "DHS": 1, "LCS_HipOA": 2, "normal": 3},
}


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

        if self._temporal_mix:

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

        else: # ? in this experiment, do not use the whole dataset.

            if "single" in self._backbone:

                # train dataset
                if "random" in self._experiment:
                    self.train_gait_dataset = labeled_video_dataset(
                        data_path=self._dataset_idx[2],
                        clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                        transform=self.train_video_transform,
                    )

                else:
                    self.train_gait_dataset = labeled_gait_video_dataset(
                        experiment=self._experiment,
                        dataset_idx=self._dataset_idx[
                            0
                        ],  # train mapped path, include gait cycle index.
                        transform=self.mapping_transform,
                    )

                # val dataset
                self.val_gait_dataset = labeled_video_dataset(
                    data_path=self._dataset_idx[3],
                    clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                    transform=self.val_video_transform,
                )

                # test dataset
                self.test_gait_dataset = labeled_video_dataset(
                    data_path=self._dataset_idx[3],
                    clip_sampler=make_clip_sampler("uniform", self._CLIP_DURATION),
                    transform=self.val_video_transform,
                )

            elif "late_fusion" in self._experiment:

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

            elif (
                "two_stream" in self._backbone
                or "cnn_lstm" in self._backbone
                or "2dcnn" in self._backbone
            ):
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
                shuffle=True,
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
                pin_memory=True,
                shuffle=True,
                drop_last=True,
                collate_fn=self.collate_fn,
            )
        else:
            test_data_loader = DataLoader(
                self.test_gait_dataset,
                batch_size=self._default_batch_size,
                num_workers=self._NUM_WORKERS,
                pin_memory=True,
                shuffle=False,
                drop_last=True,
            )

        return test_data_loader
