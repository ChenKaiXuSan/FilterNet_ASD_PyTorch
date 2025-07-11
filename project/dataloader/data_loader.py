#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
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
"""

from torchvision.transforms import (
    Compose,
    Resize,
)

from typing import Any, Callable, Dict, Optional
from pytorch_lightning import LightningDataModule


import torch
from torch.utils.data import DataLoader

from project.dataloader.gait_video_dataset import labeled_gait_video_dataset
from project.dataloader.utils import UniformTemporalSubsample, Div255, ApplyTransformToKey


disease_to_num_mapping_Dict: Dict = {
    2: {"ASD": 0, "non-ASD": 1},
    3: {"ASD": 0, "DHS": 1, "LCS_HipOA": 2},
    4: {"ASD": 0, "DHS": 1, "LCS_HipOA": 2, "normal": 3},
}


class WalkDataModule(LightningDataModule):
    def __init__(self, opt, dataset_idx: Dict = None):
        super().__init__()

        self._batch_size = opt.data.batch_size

        self._NUM_WORKERS = opt.data.num_workers
        self._IMG_SIZE = opt.data.img_size

        # * this is the dataset idx, which include the train/val dataset idx.
        self._dataset_idx = dataset_idx
        self._class_num = opt.model.model_class_num

        self._experiment = opt.train.experiment

        self.opt = opt

        self.mapping_transform = Compose(
            [Div255(), Resize(size=[self._IMG_SIZE, self._IMG_SIZE])]
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

        train_data_loader = DataLoader(
            self.train_gait_dataset,
            batch_size=self._batch_size,
            num_workers=self._NUM_WORKERS,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
            # worker_init_fn=lambda x: torch.initial_seed(),
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
            batch_size=self._batch_size,
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
            batch_size=self._batch_size,
            num_workers=self._NUM_WORKERS,
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

        return test_data_loader
