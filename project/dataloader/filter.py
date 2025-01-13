#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/project/project/dataloader/filter.py
Project: /workspace/project/project/dataloader
Created Date: Friday January 10th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday January 10th 2025 11:04:22 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

12-01-2025	Kaixu Chen	filter code implementation.
'''

import os 
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import torch
import torch.nn as nn

from torchvision.transforms.functional import resize

from project.models.make_model import MakeVideoModule, MakeImageModule, MakeViTModule

class Filter(nn.Module):

    def __init__(self, hparams) -> None:
        
        super(Filter, self).__init__()

        self.gpu_num = hparams.train.gpu_num
        self.phase = hparams.filter.phase
        self._IMG_SIZE = hparams.data.img_size

        self.init_filter_model(hparams)
    
    def load_model(self, hparams) -> nn.Module:

        filter_model = hparams.filter.backbone

        if filter_model == "3dcnn":
            model = MakeVideoModule(hparams).make_resnet()
        elif filter_model == "2dcnn":
            model = MakeImageModule(hparams).make_resnet()
        elif filter_model == "vit":
            model = MakeViTModule(hparams).make_vit()
        else:
            raise ValueError(f"the {filter_model} is not supported.")
        
        return model.to(self.gpu_num)

    def init_filter_model(self, hparams) -> nn.Module:

        logging.info(f"filter with {hparams.filter.phase} phase.")

        if hparams.filter.phase == "mix":
            # * load the stance and swing model

            stance_ckpt_path = os.path.join(hparams.filter.path, "stance", f"{hparams.train.current_fold}_best_model.ckpt")
            self.stance_model = self.load_model(hparams)
            _ckpt = self.convert_to_torch_model(stance_ckpt_path)
            self.stance_model.load_state_dict((_ckpt['state_dict']))

            logging.info(f"load stance model from {stance_ckpt_path}")

            swing_ckpt_path = os.path.join(hparams.filter.path, "swing", f"{hparams.train.current_fold}_best_model.ckpt")
            self.swing_model = self.load_model(hparams)
            _ckpt = self.convert_to_torch_model(swing_ckpt_path)
            self.swing_model.load_state_dict((_ckpt['state_dict']))

            logging.info(f"load swing model from {swing_ckpt_path}")

        else:

            # * load the model
            ckpt_path = os.path.join(hparams.filter.path, hparams.filter.phase, f"{hparams.train.current_fold}_best_model.ckpt")
            self._model = self.load_model(hparams)
            _ckpt = self.convert_to_torch_model(ckpt_path)
            self._model.load_state_dict(_ckpt['state_dict'])

            logging.info(f"load model from {ckpt_path}")

    def convert_to_torch_model(self, ckpt_path: str) -> Dict[str, Any]:
        """convert pytorch lightning model to torch model

        Args:
            ckpt_path (str): ckpt path

        Returns:
            Dict[str, Any]: loaded model info
        """

        _ckpt = torch.load(ckpt_path)
        
        for k in list(_ckpt['state_dict'].keys()):
            if k.startswith('model.'):
                _ckpt['state_dict'][k[6:]] = _ckpt['state_dict'].pop(k)

        return _ckpt

    def preprocess(self, vframes: list[torch.Tensor]) -> torch.Tensor:
        """preprocess the video frames

        Args:
            vframes (list[torch.Tensor]): the video frames

        Returns:
            torch.Tensor: the preprocessed video frames
        """        
        
        video_t_list = []
        for video_t in vframes:
            transformed_img = resize(video_t, self._IMG_SIZE)
            transformed_img = transformed_img / 255.0
            video_t_list.append(transformed_img)

        return torch.stack(video_t_list, dim=0) # c, t, h, w

    def inference(self, phase: List[torch.Tensor], label, model: nn.Module) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        """inference the phase

        Args:
            phase (List[torch.Tensor]): phase with video frames
            label (_type_): phase label
            model (nn.Module): filter model

        Returns:
            Tuple[list[torch.Tensor], list[torch.Tensor]]: filtered scores and sorted indices
        """        

        ans_filtered_scores = []
        ans_sorted_indices = []

        for one_phase in phase:
            
            preprocessed_phase = self.preprocess(one_phase)
            # FIXME: now only can use 1 process to inference.
            preprocessed_phase = preprocessed_phase.to(self.gpu_num)

            with torch.no_grad():
                one_phase_preds = model(preprocessed_phase)

            # compare the phase prediction with the phase_idx
            # extract the score of each sample on its target category
            filtered_scores = one_phase_preds[:, label]

            # sort the scores, return the sorted indices
            sorted_indices = torch.argsort(filtered_scores, descending=True)

            ans_filtered_scores.append(filtered_scores.tolist())
            ans_sorted_indices.append(sorted_indices.tolist())

        return ans_filtered_scores, ans_sorted_indices
        
    def forward(self, phase: dict):
        
        first_phase = phase["first_phase"]
        second_phase = phase["second_phase"]
        label = phase["label"]

        if self.phase == "mix":
            stance_ans_filtered_scores, stance_ans_sorted_indices =  self.inference(first_phase, label, self.stance_model)

            swing_ans_filtered_scores, swing_ans_sorted_indices =  self.inference(second_phase, label, self.swing_model)

        else:
            stance_ans_filtered_scores, stance_ans_sorted_indices =  self.inference(first_phase, label, self._model)
            swing_ans_filtered_scores, swing_ans_sorted_indices =  self.inference(second_phase, label, self._model)

        return {
            "first_phase": [stance_ans_filtered_scores, stance_ans_sorted_indices],
            "second_phase": [swing_ans_filtered_scores, swing_ans_sorted_indices],
        }