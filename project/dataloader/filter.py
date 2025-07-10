from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import torch


def split_gait_cycle(
    video_tensor: torch.Tensor, gait_cycle_index: list, gait_cycle: int
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Split the video tensor into gait cycles based on the provided indices.
    """
    use_idx, ans_list = [], []

    start = 0 if gait_cycle == 0 or len(gait_cycle_index) == 2 else 1
    for i in range(start, len(gait_cycle_index) - 1, 2):
        ans_list.append(
            video_tensor[gait_cycle_index[i] : gait_cycle_index[i + 1], ...]
        )
        use_idx.append(gait_cycle_index[i])

    return ans_list, use_idx


class Filter:
    """
    A class to filter video frames based on gait cycle phases and filtering scores.
    """

    def __init__(self, hparams) -> None:
        self.filter = hparams.train.filter
        self.backbone, self._temporal_mix, *self.phase = hparams.train.experiment.split(
            "_"
        )
        self.current_fold = hparams.train.current_fold
        self.uniform_temporal_subsample = hparams.train.uniform_temporal_subsample_num

    @staticmethod
    def filter_video_frames(
        video_tensor: torch.Tensor,
        phase_sorted_idx: List[List[int]],
        uniform_temporal_subsample: int = 8,
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        Filter video frames based on sorted phase indices and uniform temporal subsampling.

        Args:
            video_tensor (torch.Tensor): Tensor of shape (B, T, C, H, W)
            phase_sorted_idx (List[List[int]]): Frame indices per sample.
            uniform_temporal_subsample (int): Target number of frames.

        Returns:
            Tuple[torch.Tensor, List[List[int]]]: Filtered video tensor (B, S, C, H, W), and used indices.
        """
        res_batch_frames = []
        used_indices = []

        B = len(phase_sorted_idx)

        assert B == len(video_tensor), "Batch size mismatch"

        for i in range(B):
            frame_indices = phase_sorted_idx[i]
            num_frames = len(frame_indices)

            if num_frames >= uniform_temporal_subsample:
                # 按照已有顺序取前 uniform_temporal_subsample 个并升序排列
                selected_idx = sorted(frame_indices[:uniform_temporal_subsample])
            else:
                # 均匀补帧：用重复或插值策略（此处为重复）
                repeat_idx = torch.linspace(
                    0, num_frames - 1, steps=uniform_temporal_subsample
                ).long().tolist()
                selected_idx = [frame_indices[j] for j in repeat_idx]

            used_indices.append(selected_idx)

            selected_frames = [video_tensor[i][f] for f in selected_idx]
            res_batch_frames.append(torch.stack(selected_frames, dim=1))  # (C, T, H, W)

        return torch.stack(res_batch_frames, dim=0), used_indices  # (B, C, T, H, W)


    def __call__(
        self,
        video_tensor: torch.Tensor,
        gait_cycle_index: list,
        bbox: List[torch.Tensor],
        label: List[torch.Tensor],
        filter_info: Dict[str, dict],
    ) -> torch.Tensor:

        # Split video into gait phases
        first_phase, first_phase_idx = split_gait_cycle(
            video_tensor, gait_cycle_index, 0
        )
        second_phase, second_phase_idx = split_gait_cycle(
            video_tensor, gait_cycle_index, 1
        )

        first_scores = filter_info["first_phase"][f"fold{self.current_fold}"]
        second_scores = filter_info["second_phase"][f"fold{self.current_fold}"]

        first_phase_filtered_scores = first_scores["filtered_scores"]
        first_phase_sorted_idx = first_scores["sorted_idx"]

        second_phase_filtered_scores = second_scores["filtered_scores"]
        second_phase_sorted_idx = second_scores["sorted_idx"]

        # Pad the shorter phase to match length
        len_diff = len(first_phase) - len(second_phase)
        if len_diff > 0:
            second_phase.extend([second_phase[-1]] * len_diff)
            second_phase_idx.extend([second_phase_idx[-1]] * len_diff)
            second_phase_sorted_idx.extend([second_phase_sorted_idx[-1]] * len_diff)
        elif len_diff < 0:
            first_phase.extend([first_phase[-1]] * (-len_diff))
            first_phase_idx.extend([first_phase_idx[-1]] * (-len_diff))
            first_phase_sorted_idx.extend([first_phase_sorted_idx[-1]] * (-len_diff))

        # first phase selection
        first_phase, _ = self.filter_video_frames(
            first_phase, first_phase_sorted_idx, self.uniform_temporal_subsample
        )
        second_phase, _ = self.filter_video_frames(
            second_phase, second_phase_sorted_idx, self.uniform_temporal_subsample
        )

        return torch.cat(
            [first_phase, second_phase], dim=0
        )
