#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/models/make_model.py
Project: /workspace/skeleton/project/models
Created Date: Thursday October 19th 2023
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday July 9th 2025 10:04:27 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2023 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

26-11-2024	Kaixu Chen	remove x3d network.
"""

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MakeVideoModule(nn.Module):
    """
    make 3D CNN model from the PytorchVideo lib.

    """

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_name = hparams.model.model
        self.model_class_num = hparams.model.model_class_num
        self.model_depth = hparams.model.model_depth

    def make_resnet(self, input_channel: int = 3) -> nn.Module:

        slow = torch.hub.load(
            "facebookresearch/pytorchvideo", "slow_r50", pretrained=True
        )

        # for the folw model and rgb model
        slow.blocks[0].conv = nn.Conv3d(
            input_channel,
            64,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False,
        )
        # change the knetics-400 output 400 to model class num
        slow.blocks[-1].proj = nn.Linear(2048, self.model_class_num)

        return slow

    def __call__(self, *args: Any, **kwds: Any) -> Any:

        if self.model_name == "3dcnn":
            return self.make_resnet()
        else:
            raise KeyError(f"the model name {self.model_name} is not in the model zoo")


class MakeImageModule(nn.Module):
    """
    the module zoo from the torchvision lib, to make the different 2D model.

    """

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_name = hparams.model.model
        self.model_class_num = hparams.model.model_class_num

    def make_resnet(self, input_channel: int = 3) -> nn.Module:

        model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
        model.conv1 = nn.Conv2d(
            input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.fc = nn.Linear(2048, self.model_class_num)

        return model

    def __call__(self, *args: Any, **kwds: Any) -> Any:

        if self.model_name == "resnet":
            return self.make_resnet()
        else:
            raise KeyError(f"the model name {self.model_name} is not in the model zoo")


class MakeOriginalTwoStream(nn.Module):
    """
    from torchvision make resnet 50 network.
    input is single figure.
    """

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_class_num = hparams.model.model_class_num

    def make_resnet(self, input_channel: int = 3):

        model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)

        # from pytorchvision, use resnet 50.
        # weights = ResNet50_Weights.DEFAULT
        # model = resnet50(weights=weights)

        # for the folw model and rgb model
        model.conv1 = nn.Conv2d(
            input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # change the output 400 to model class num
        model.fc = nn.Linear(2048, self.model_class_num)

        return model


class CNNLSTM(nn.Module):
    """
    the cnn lstm network, use the resnet 50 as the cnn part.
    """

    def __init__(self, hparams) -> None:

        super().__init__()

        self.model_class_num = hparams.model.model_class_num

        self.cnn = self.make_cnn()
        # LSTM
        self.lstm = nn.LSTM(
            input_size=300, hidden_size=512, num_layers=2, batch_first=True
        )
        self.fc = nn.Linear(512, self.model_class_num)

    def make_cnn(self, input_channel: int = 3):

        model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)

        # from pytorchvision, use resnet 50.
        # weights = ResNet50_Weights.DEFAULT
        # model = resnet50(weights=weights)

        # for the folw model and rgb model
        model.conv1 = nn.Conv2d(
            input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # change the output 400 to model class num
        model.fc = nn.Linear(2048, 300)

        return model

    def forward(self, x):

        b, c, t, h, w = x.size()

        res = []

        for i in range(b):
            hidden = None
            out = self.cnn(x[i].permute(1, 0, 2, 3))
            out, hidden = self.lstm(out, hidden)

            out = F.relu(out)
            out = self.fc(out)

            res.append(out)

        return torch.cat(res, dim=0)
