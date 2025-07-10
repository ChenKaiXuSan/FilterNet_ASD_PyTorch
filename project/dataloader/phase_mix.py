#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/project/project/dataloader/phase_mix.py
Project: /workspace/project/project/dataloader
Created Date: Saturday January 11th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday July 10th 2025 9:41:08 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

12-01-2025	Kaixu Chen	add the fileter implementation.
"""
from __future__ import annotations

import logging

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import torch

from torchvision.io import read_video, write_png


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


class PhaseMix(object):
    """
    This class is temporal mix, which is used to mix the first phase and second phase of gait cycle.
    """

    def __init__(self, hparams) -> None:

        # self.filter = Filter(hparams)
        self.current_fold = hparams.train.current_fold
        self.uniform_temporal_subsample = hparams.train.uniform_temporal_subsample_num

    @staticmethod
    def process_phase(
        phase_frame: List[torch.Tensor], phase_idx: List[int], bbox: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Crop the human area with bbox, and normalize them with max width.

        Args:
            phase_frame (List[torch.Tensor]): procedded frame by gait index
            phase_idx (List[int]): gait cycle index
            bbox (List[torch.Tensor]): _description_

        Returns:
            List[torch.Tensor]: _description_
        """
        # find the max width of phase and crop them.
        cropped_frame_list: List[torch.Tensor] = []

        for i in range(len(phase_idx)):

            one_pack_frames = phase_frame[i]  # frame pack
            one_pack_start_idx = phase_idx[i]
            b, c, h, w = one_pack_frames.shape  # b is one frame pack size

            stored_x_max = float("-inf")
            stored_xmax = 0
            stored_xmin = 0

            one_pack_frames_list: List[torch.Tensor] = []

            # * step1: find the max width and max height for one frame pack.
            for k in range(b):

                frame_bbox = bbox[one_pack_start_idx + k]
                x, y, w, h = frame_bbox
                xmin = int(x - w / 2)
                xmax = int(x + w / 2)

                if xmax - xmin > stored_x_max:
                    stored_x_max = xmax - xmin
                    stored_xmax = xmax
                    stored_xmin = xmin

            # * step2: crop human area with bbox, and normalized with max width
            for k in range(b):

                frame_bbox = bbox[i + k]
                x, y, w, h = frame_bbox

                frame = one_pack_frames[k]
                cropped_one_frame_human = frame[:, :, stored_xmin:stored_xmax]
                one_pack_frames_list.append(cropped_one_frame_human)

                # write_png(input=cropped_one_frame_human, filename=f'/workspace/skeleton/logs/img/test{k}.png')

            # * step3: stack the cropped frame, for next step to fuse them
            cropped_frame_list.append(
                torch.stack(one_pack_frames_list, dim=0)
            )  # b, c, h, w

        # shape check
        assert (
            len(cropped_frame_list) == len(phase_frame) == len(phase_idx)
        ), "frame pack length is not equal"
        for i in range(len(phase_frame)):
            assert (
                cropped_frame_list[i].size()[0] == phase_frame[i].size()[0]
            ), f"the {i} frame pack size is not equal"

        return cropped_frame_list

    def fuse_frames(
        self,
        processed_first_phase: List[torch.Tensor],
        processed_second_phase: List[torch.Tensor],
        first_phase_sorted_idx: List[torch.Tensor],
        second_phase_sorted_idx: List[torch.Tensor],
    ) -> torch.Tensor:

        res_fused_frames: List[torch.Tensor] = []

        for pack in range(len(processed_first_phase)):

            fuse_frame_num = self.uniform_temporal_subsample

            first_phase_frame_ans = []
            second_phase_frame_ans = []

            ##############
            # first phase
            ##############
            # * split > sort > alignment > select from the sorted idx
            # * resort the frame idx with fuse_frame_num, keep the time order.
            first_phase_sorted_idx[pack] = first_phase_sorted_idx[pack][:fuse_frame_num]
            first_phase_sorted_idx[pack] = sorted(first_phase_sorted_idx[pack])

            # * keep the frame num equal to fuse_frame_num
            if processed_first_phase[pack].size()[0] < fuse_frame_num:
                for _ in range(fuse_frame_num - processed_first_phase[pack].size()[0]):
                    processed_first_phase[pack] = torch.cat(
                        [
                            processed_first_phase[pack],
                            processed_first_phase[pack][-1].unsqueeze(0),
                        ],
                        dim=0,
                    )
                    first_phase_sorted_idx[pack].append(
                        first_phase_sorted_idx[pack][-1]
                    )

            for idx in range(fuse_frame_num):
                first_phase_frame_ans.append(
                    processed_first_phase[pack][first_phase_sorted_idx[pack][idx]]
                )

            uniform_first_phase = torch.stack(first_phase_frame_ans, dim=0)

            ##############
            # second phase
            ##############

            # * resort the frame idx with fuse_frame_num, keep the time order.
            second_phase_sorted_idx[pack] = second_phase_sorted_idx[pack][
                :fuse_frame_num
            ]
            second_phase_sorted_idx[pack] = sorted(second_phase_sorted_idx[pack])

            # * keep the frame num equal to fuse_frame_num
            if processed_second_phase[pack].size()[0] < fuse_frame_num:
                for _ in range(fuse_frame_num - processed_second_phase[pack].size()[0]):
                    processed_second_phase[pack] = torch.cat(
                        [
                            processed_second_phase[pack],
                            processed_second_phase[pack][-1].unsqueeze(0),
                        ],
                        dim=0,
                    )
                    second_phase_sorted_idx[pack].append(
                        second_phase_sorted_idx[pack][-1]
                    )

            for idx in range(fuse_frame_num):
                second_phase_frame_ans.append(
                    processed_second_phase[pack][second_phase_sorted_idx[pack][idx]]
                )

            uniform_second_phase = torch.stack(second_phase_frame_ans, dim=0)

            #################
            # fuse width dim
            #################
            fused_frames = torch.cat([uniform_first_phase, uniform_second_phase], dim=3)

            # write the fused frame to png
            # for i in range(fused_frames.size()[0]):
            #     write_png(input=fused_frames[i], filename=f'/workspace/project/logs/img/fused{i}.png')

            res_fused_frames.append(fused_frames)

        return res_fused_frames

    def __call__(
        self,
        video_tensor: torch.Tensor,
        gait_cycle_index: list,
        bbox: List[torch.Tensor],
        label: List[torch.Tensor],
        filter_info: Dict[str, dict],
    ) -> torch.Tensor:

        # * step1: first find the phase frames (pack) and phase index.
        first_phase, first_phase_idx = split_gait_cycle(
            video_tensor, gait_cycle_index, 0
        )
        second_phase, second_phase_idx = split_gait_cycle(
            video_tensor, gait_cycle_index, 1
        )

        first_phase_filtered_scores = filter_info["first_phase"][
            f"fold{self.current_fold}"
        ]["filtered_scores"]
        first_phase_sorted_idx = filter_info["first_phase"][f"fold{self.current_fold}"][
            "sorted_idx"
        ]

        second_phase_filtered_scores = filter_info["second_phase"][
            f"fold{self.current_fold}"
        ]["filtered_scores"]
        second_phase_sorted_idx = filter_info["second_phase"][
            f"fold{self.current_fold}"
        ]["sorted_idx"]

        # * keep the frame pack length equal.
        if len(first_phase) > len(second_phase):
            second_phase.append(second_phase[-1])
            second_phase_idx.append(second_phase_idx[-1])
            second_phase_sorted_idx.append(second_phase_sorted_idx[-1])
        elif len(first_phase) < len(second_phase):
            first_phase.append(first_phase[-1])
            first_phase_idx.append(first_phase_idx[-1])
            first_phase_sorted_idx.append(first_phase_sorted_idx[-1])

        # * step3: process on pack, crop the human area with bbox
        processed_first_phase = self.process_phase(first_phase, first_phase_idx, bbox)
        processed_second_phase = self.process_phase(
            second_phase, second_phase_idx, bbox
        )

        # * step3: fuse the first phase and second phase
        fused_vframes = self.fuse_frames(
            processed_first_phase,
            processed_second_phase,
            first_phase_sorted_idx,
            second_phase_sorted_idx,
        )

        return fused_vframes
