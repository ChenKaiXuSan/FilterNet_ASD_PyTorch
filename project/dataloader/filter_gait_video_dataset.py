#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/project/project/dataloader/filter_gait_video_dataset.py
Project: /workspace/project/project/dataloader
Created Date: Thursday January 9th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday January 9th 2025 2:55:43 pm
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

# from pytorchvideo.transforms.functional import uniform_temporal_subsample
from torchvision.transforms.v2.functional import (
    uniform_temporal_subsample_video,
    uniform_temporal_subsample,
)


logger = logging.getLogger(__name__)


class LabeledGaitVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        experiment: str,
        labeled_video_paths: list[Tuple[str, Optional[dict]]],
        transform: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        super().__init__()

        self._transform = transform
        self._labeled_videos = labeled_video_paths
        self._experiment = experiment

        self.backbone, self.phase = experiment.split("_")

    def move_transform(self, vframes: list[torch.Tensor]) -> None:

        video_t_list = []
        for video_t in vframes:
            if self._transform is not None:
                transformed_img = self._transform(video_t.permute(1, 0, 2, 3))
                video_t_list.append(transformed_img)
            else:
                video_t_list.append(video_t)

        if self.backbone == "3dcnn":
            return torch.stack(video_t_list, dim=1)
        elif self.backbone == "2dcnn" or self.backbone == 'vit':
            return torch.cat(video_t_list, dim=1)
        else:
            raise ValueError("backbone should be 2dcnn or 3dcnn")

    @staticmethod
    def split_gait_cycle(
        video_tensor: torch.Tensor, gait_cycle_index: list, gait_cycle: int
    ):

        use_idx = []
        ans_list = []
        if gait_cycle == 0 or len(gait_cycle_index) == 2:
            for i in range(0, len(gait_cycle_index) - 1, 2):
                ans_list.append(
                    video_tensor[gait_cycle_index[i] : gait_cycle_index[i + 1], ...]
                )
                use_idx.append(gait_cycle_index[i])

        elif gait_cycle == 1:

            # FIXME: maybe here do not -1 for upper limit.
            for i in range(1, len(gait_cycle_index) - 1, 2):
                ans_list.append(
                    video_tensor[gait_cycle_index[i] : gait_cycle_index[i + 1], ...]
                )
                use_idx.append(gait_cycle_index[i])

        # print(f"used split gait cycle index: {use_idx}")

        return ans_list, use_idx  # needed gait cycle video tensor

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

        # * step1: first find the phase frames and phase index.
        if self.phase == "stance":
            phase = 0
        elif self.phase == "swing":
            phase = 1
        else:
            raise ValueError("phase should be stance or swing")

        phase_list, phase_idx = self.split_gait_cycle(vframes, gait_cycle_index, phase)

        # * step2: move the frames through the transform function.
        defined_vframes = self.move_transform(phase_list)  # c, t, h, w

        # * step3: return the sample info dict.
        sample_info_dict = {
            "video": defined_vframes,
            "label": label,
            "disease": disease,
            "video_name": video_name,
            "video_index": index,
            "gait_cycle_index": gait_cycle_index,
            "bbox_none_index": bbox_none_index,
            "phase_idx": phase_idx,
        }

        return sample_info_dict


def labeled_gait_video_dataset(
    experiment: str,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    dataset_idx: Dict = None,
) -> LabeledGaitVideoDataset:

    dataset = LabeledGaitVideoDataset(
        experiment=experiment,
        labeled_video_paths=dataset_idx,
        transform=transform,
    )

    return dataset
