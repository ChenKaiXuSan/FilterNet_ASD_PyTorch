#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/skeleton/project/models/make_model.py
Project: /workspace/skeleton/project/models
Created Date: Thursday October 19th 2023
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday January 9th 2025 12:29:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2023 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

26-11-2024	Kaixu Chen	remove x3d network.
'''

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchvideo.models import resnet
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import vit_l_32, ViT_L_32_Weights

class MakeVideoModule(nn.Module):
    '''
    make 3D CNN model from the PytorchVideo lib.

    '''

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_name = hparams.model.model
        self.model_class_num = hparams.model.model_class_num
        self.model_depth = hparams.model.model_depth
        self.transfer_learning = hparams.train.transfer_learning

    def initialize_walk_resnet(self, input_channel:int = 3) -> nn.Module:

        if self.transfer_learning:
            slow = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
            
            # for the folw model and rgb model 
            slow.blocks[0].conv = nn.Conv3d(input_channel, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
            # change the knetics-400 output 400 to model class num
            slow.blocks[-1].proj = nn.Linear(2048, self.model_class_num)

        else:
            slow = resnet.create_resnet(
                input_channel=input_channel,
                model_depth=self.model_depth,
                model_num_class=self.model_class_num,
                norm=nn.BatchNorm3d,
                activation=nn.ReLU,
            )

        return slow

class MakeImageModule(nn.Module):
    '''
    the module zoo from the torchvision lib, to make the different 2D model.

    '''

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_name = hparams.model.model
        self.model_class_num = hparams.model.model_class_num

    def make_resnet(self, input_channel:int = 3) -> nn.Module:

        model = resnet101(weights=ResNet101_Weights.DEFAULT)

        model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(2048, self.model_class_num)
    
        return model

class MakeViTModule(nn.Module):

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_name = hparams.model.model
        self.model_class_num = hparams.model.model_class_num

    def make_vit(self, input_channel:int = 3) -> nn.Module:
        
        # model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        model = vit_l_32(weights=ViT_L_32_Weights.DEFAULT)

        # model.patch_embed = nn.Conv2d(input_channel, 768, kernel_size=16, stride=16, padding=0)
        model.heads.head = nn.Linear(1024, self.model_class_num)

        return model