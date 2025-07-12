#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/models/optical_flow.py
Project: /workspace/skeleton/project/models
Created Date: Friday June 7th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday July 10th 2025 9:41:08 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import torch.nn as nn
import torch

from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


class Optical_flow(nn.Module):

    def __init__(self, of_ckpt: str = None):
        super().__init__()

        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.optical_flow_path = of_ckpt

        # define the network
        self.model = self.init_model(self.weights, self.optical_flow_path)

    @staticmethod
    def init_model(weights, of_path=None):
        """
        load the optical flow model.
        """

        if of_path is not None:
            model = raft_large(weights=None, progress=False).cuda()
            try:
                model.load_state_dict(torch.load(of_path, map_location="cpu"))
                print(f"Optical flow model loaded from {of_path}")
            except Exception as e:
                print(f"Error loading optical flow model: {e}")

        else:
            model = raft_large(weights=weights, progress=False).cuda()

        return model

    def get_Optical_flow(self, frame_batch):
        """
        catch one by one batch optical flow, use RAFT method.

        Args:
            frame_batch (tensor): one batch frame, (c, f, h, w)

        Returns:
            tensor: one batch pred optical flow
        """

        c, f, h, w = frame_batch.shape

        frame_batch = frame_batch.permute(1, 0, 2, 3)  # c, f, h, w to f, c, h, w

        current_frame = frame_batch[:-1, :, :, :]  # 0~-1 frame
        next_frame = frame_batch[1:, :, :, :]  # 1~last frame

        # transforms
        current_frame_batch, next_frame_batch = self.transforms(
            current_frame, next_frame
        )

        # start predict
        self.model.eval()

        with torch.no_grad():
            pred_flows = self.model(current_frame_batch, next_frame_batch)[-1]

        # empty cache
        torch.cuda.empty_cache()

        return pred_flows  # f, c, h, w

    def process_batch(self, batch):
        """
        predict one batch optical flow.

        Args:
            batch (nn.Tensor): batches of videos. (b, c, f, h, w)

        Returns:
            nn.Tensor: stacked predict optical flow, (b, 2, f, h, w)
        """

        b, c, f, h, w = batch.shape

        pred_optical_flow_list = []

        for batch_index in range(b):
            one_batch_pred_flow = self.get_Optical_flow(
                batch[batch_index]
            )  # f, c, h, w
            pred_optical_flow_list.append(one_batch_pred_flow)

        return torch.stack(pred_optical_flow_list).permute(
            0, 2, 1, 3, 4
        )  # b, c, f, h, w
