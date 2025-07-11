#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/project/project/dataloader/gait_video_dataset.py
Project: /workspace/project/project/dataloader
Created Date: Friday January 10th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday July 9th 2025 10:04:27 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from __future__ import annotations

import logging
import json

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import torch

from torchvision.io import read_video, write_png

from project.dataloader.phase_mix import PhaseMix
from project.dataloader.filter import Filter

logger = logging.getLogger(__name__)


class LabeledGaitVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        experiment: str,
        labeled_video_paths: list[Tuple[str, Optional[dict]]],
        transform: Optional[Callable[[dict], Any]] = None,
        hparams: Dict = None,
    ) -> None:
        super().__init__()

        self._transform = transform
        self._labeled_videos = labeled_video_paths
        self._experiment = experiment

        self.current_fold = hparams.train.current_fold

        self.filter = hparams.train.filter
        self.temporal_mix = hparams.train.temporal_mix

        if self.filter:
            self._filter = Filter(hparams)
        else:
            self._filter = False

        if self.temporal_mix:
            self._temporal_mix = PhaseMix(hparams)
        else:
            self._temporal_mix = False

    def move_transform(self, vframes: list[torch.Tensor]) -> None:

        if self._transform is not None:
            video_t_list = []
            for video_t in vframes:
                transformed_img = self._transform(video_t)
                video_t_list.append(transformed_img)

            return torch.stack(video_t_list, dim=0)  # c, t, h, w
        else:
            print("no transform")
            return torch.stack(vframes, dim=0)

    def __len__(self):
        return len(self._labeled_videos)

    def __getitem__(self, index) -> Any:

        # load the video tensor from json file
        with open(self._labeled_videos[index]) as f:
            file_info_dict = json.load(f)

        # load video info from json file
        video_name = file_info_dict["video_name"]
        video_path = file_info_dict["video_path"]
        vframes, _, _ = read_video(video_path, output_format="TCHW", pts_unit="sec")
        label = file_info_dict["label"]
        disease = file_info_dict["disease"]
        gait_cycle_index = file_info_dict["gait_cycle_index"]
        bbox_none_index = file_info_dict["none_index"]
        bbox = file_info_dict["bbox"]

        filter_info = file_info_dict["filter_info"]

        # FIXME: 下面的两部分功能重叠了，但是不影响使用
        if self.filter:
            defined_vframes = self._filter(
                vframes, gait_cycle_index, bbox, label, filter_info
            )
            defined_vframes = self.move_transform(defined_vframes)

        if self.temporal_mix:

            defined_vframes = self._temporal_mix(
                vframes, gait_cycle_index, bbox, label, filter_info
            )
            defined_vframes = self.move_transform(defined_vframes)

        if not self.filter and not self.temporal_mix:
            defined_vframes = self.move_transform(vframes)

        sample_info_dict = {
            "video": defined_vframes,
            "label": label,
            "disease": disease,
            "video_name": video_name,
            "video_index": index,
            "gait_cycle_index": gait_cycle_index,
            "bbox_none_index": bbox_none_index,
        }

        return sample_info_dict


def labeled_gait_video_dataset(
    experiment: str,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    dataset_idx: Dict = None,
    hparams: Dict = None,
) -> LabeledGaitVideoDataset:

    dataset = LabeledGaitVideoDataset(
        experiment=experiment,
        labeled_video_paths=dataset_idx,
        transform=transform,
        hparams=hparams,
    )

    return dataset
