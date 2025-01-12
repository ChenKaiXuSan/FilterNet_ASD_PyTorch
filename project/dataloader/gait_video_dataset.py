#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/project/project/dataloader/gait_video_dataset.py
Project: /workspace/project/project/dataloader
Created Date: Friday January 10th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday January 10th 2025 9:08:20 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

from __future__ import annotations

import logging
import json

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import torch

from torchvision.io import read_video, write_png
# from pytorchvideo.transforms.functional import uniform_temporal_subsample
from torchvision.transforms.v2.functional import uniform_temporal_subsample_video, uniform_temporal_subsample

from project.dataloader.phase_mix import PhaseMix

logger = logging.getLogger(__name__)

def split_gait_cycle(video_tensor: torch.Tensor, gait_cycle_index: list, gait_cycle: int):

    use_idx = []
    ans_list = []
    if gait_cycle == 0 or len(gait_cycle_index) == 2 :
        for i in range(0, len(gait_cycle_index)-1, 2):
            ans_list.append(video_tensor[gait_cycle_index[i]:gait_cycle_index[i+1], ...])
            use_idx.append(gait_cycle_index[i])

    elif gait_cycle == 1:
    
        # FIXME: maybe here do not -1 for upper limit.
        for i in range(1, len(gait_cycle_index)-1, 2):
            ans_list.append(video_tensor[gait_cycle_index[i]:gait_cycle_index[i+1], ...])
            use_idx.append(gait_cycle_index[i])

    # print(f"used split gait cycle index: {use_idx}")
    
    return ans_list, use_idx # needed gait cycle video tensor

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

        self.backbone, self._temporal_mix, *self.phase = experiment.split("_")

        if self._temporal_mix:
            self._temporal_mix = PhaseMix(hparams)
        else:
            self._temporal_mix = False


    def move_transform(self, vframes: list[torch.Tensor]) -> None:

        if self._transform is not None:
            video_t_list = []
            for video_t in vframes:
                transformed_img = self._transform(video_t.permute(1, 0, 2, 3))
                video_t_list.append(transformed_img)

            return torch.stack(video_t_list, dim=0) # c, t, h, w
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

        # TODO: here should judge the frame with pre-trained model.
        if "True" in self._experiment:
            # should return the new frame, named temporal mix.
            defined_vframes = self._temporal_mix(vframes, gait_cycle_index, bbox, label)
            defined_vframes = self.move_transform(defined_vframes)

        elif "single" in self._experiment:
            if "stance" in self._experiment:    
                defined_vframes, used_gait_idx = split_gait_cycle(vframes, gait_cycle_index, 0)
            elif "swing" in self._experiment:
                defined_vframes, used_gait_idx = split_gait_cycle(vframes, gait_cycle_index, 1)
            
            defined_vframes = self.move_transform(defined_vframes)
                
        else:
            raise ValueError("experiment name is not correct")

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
