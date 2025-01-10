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
'''

import torch
import torch.nn as nn

from project.models.make_model import MakeVideoModule, MakeImageModule, MakeViTModule

class Filter(nn.Module):

    def __init__(self, hparams, ckpt_path: str, filter_model: str) -> None:

        super(Filter, self).__init__()

        self.filter_model = self.init_filter_model(hparams, ckpt_path, filter_model)

    def init_filter_model(self, hparams, ckpt_path: str, filter_model: str) -> nn.Module:

        if filter_model == "3dcnn":
            model = MakeVideoModule(hparams).make_resnet()
            model.load_state_dict(torch.load(ckpt_path))
        elif filter_model == "2dcnn":
            model = MakeImageModule(hparams).make_resnet()
            model.load_state_dict(torch.load(ckpt_path))
        elif filter_model == "vit":
            model = MakeViTModule(hparams).make_vit()
            model.load_state_dict(torch.load(ckpt_path))
        else:
            raise ValueError(f"the {filter_model} is not supported.")

        return model


    def forward(self, x):

        preds = self.filter_model(x)

        return x