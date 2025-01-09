#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /Users/chenkaixu/deep-learning-project-template/project/dataloader/data_loader.py
Project: /Users/chenkaixu/deep-learning-project-template/project/dataloader
Created Date: Saturday November 30th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday November 30th 2024 12:34:29 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
from typing import Any, Callable, Dict, Optional, Type

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from pytorchvideo.data import make_clip_sampler
from pytorchvideo.data.labeled_video_dataset import labeled_video_dataset


class DataModule(LightningDataModule):
    def __init__(self, opt):
        super().__init__()

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
        ...

    def collate_fn(self, batch):
        """this function process the batch data, and return the batch data.

        Args:
            batch (list): the batch from the dataset.
            The batch include the one patient info from the json file.
            Here we only cat the one patient video tensor, and label tensor.

        Returns:
            dict: {video: torch.tensor, label: torch.tensor, info: list}
        """
        ...

    def train_dataloader(self) -> DataLoader:
        """ create the train data loader

        Returns:
            DataLoader: _description_
        """        
        train_data_loader = DataLoader(
            self.train_gait_dataset,
            batch_size=self._default_batch_size,
            num_workers=self._NUM_WORKERS,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        return train_data_loader

    def val_dataloader(self) -> DataLoader:
        """ create the val data loader

        Returns:
            DataLoader: _description_
        """        
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
        """ create the test data loader

        Returns:
            DataLoader: 
        """
        test_data_loader = DataLoader(
            self.test_gait_dataset,
            batch_size=self._default_batch_size,
            num_workers=self._NUM_WORKERS,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )

        return test_data_loader
