#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/project/project/filter_score/main.py
Project: /workspace/project/project/filter_score
Created Date: Monday January 13th 2025
Author: Kaixu Chen
-----
Comment:
from segmentation_dataset_512/json_mix to load the video info, and inference the video with the filter model.
Then save the filtered score to the json file.
Have a good code time :)
-----
Last Modified: Monday January 13th 2025 3:31:32 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

from pathlib import Path
from typing import Dict

import json
import logging
import hydra
import os
import multiprocessing

import torch
from torchvision.io import read_video
from project.filter_score.filter import Filter

class_num_mapping_Dict: Dict = {
    2: {
        0: "ASD",
        1: "non-ASD"
    },
    3: {
        0: "ASD",
        1: "DHS",
        2: "LCS_HipOA"
    },
    4: {
        0: "ASD",
        1: "DHS",
        2: "LCS_HipOA",
        3: "normal"
    }
}


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

def map_class_num(class_num: int, raw_video_path: Path) -> Dict:

    _class_num = class_num_mapping_Dict[class_num]

    res_dict = {v:[] for k,v in _class_num.items()}

    for disease in raw_video_path.iterdir():

        for one_json_file in disease.iterdir():

            if disease.name in res_dict.keys():
                res_dict[disease.name].append(one_json_file)
            elif disease.name == 'log':
                continue;
            else:
                res_dict["non-ASD"].append(one_json_file)

    return res_dict

def inference_one_path(one_path: Path, config) -> Dict:

    with open(one_path, 'r') as f:
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

    first_phase, first_phase_idx = split_gait_cycle(vframes, gait_cycle_index, 0)
    second_phase, second_phase_idx = split_gait_cycle(vframes, gait_cycle_index, 1)

    filter_info = {
        "first_phase": first_phase,
        "second_phase": second_phase,
        "label": label,
    }  

    first_phase_info = {}
    second_phase_info = {}

    for fold_idx in range(config.train.fold):

        config.train.current_fold = fold_idx
        filter_model = Filter(config)

        filtered_res: dict = filter_model(filter_info)

        first_phase_filtered_scores, first_phase_sorted_idx = filtered_res["first_phase"]
        second_phase_filtered_scores, second_phase_sorted_idx = filtered_res["second_phase"]

        first_phase_info[f"fold{fold_idx}"] = {
            "filtered_scores": first_phase_filtered_scores,
            "sorted_idx": first_phase_sorted_idx
        }
        second_phase_info[f"fold{fold_idx}"] = {
            "filtered_scores": second_phase_filtered_scores,
            "sorted_idx": second_phase_sorted_idx
        }

    file_info_dict["filter_info"] = {    
            "first_phase": first_phase_info,
            "second_phase": second_phase_info
        }

    # save to json file
    target_path = Path(config.data.gait_seg_data_path_with_score) / config.filter.phase / disease / one_path.name
    if not target_path.parent.exists():
        target_path.parent.mkdir(parents=True)

    with open(target_path, 'w') as f:
        json.dump(file_info_dict, f, indent=4, sort_keys=True)

    logging.info(f"update the json file: {one_path}")

def process(path_list: list, config) -> Dict:

    for one_path in path_list:
        logging.info(one_path)

        inference_one_path(one_path, config)

    logging.info("finish all inference")

@hydra.main(
    version_base=None,
    config_path="../../configs",  # * the config_path is relative to location of the python script
    config_name="filter_score.yaml",
)
def init_params(config):

    gait_seg_data_path = Path(config.data.gait_seg_data_path)

    mapped_class_Dict = map_class_num(3, Path(gait_seg_data_path))

    # define the process
    config.train.gpu_num = 0

    asd = multiprocessing.Process(target=process, args=(mapped_class_Dict['ASD'], config), name="process_ASD")
    asd.start()

    config.train.gpu_num = 0
    dhs = multiprocessing.Process(target=process, args=(mapped_class_Dict['DHS'], config), name="process_DHS")
    dhs.start()

    lcs_HipOA = multiprocessing.Process(target=process, args=(mapped_class_Dict['LCS_HipOA'], config), name="process_LCS_HipOA")
    lcs_HipOA.start()

    logging.info("finish all inference")

if __name__ == '__main__':
    
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()