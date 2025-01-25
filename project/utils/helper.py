#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/project/project/helpet.py
Project: /workspace/project/project
Created Date: Thursday January 9th 2025
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

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
    MulticlassAUROC,
)

from pytorch_grad_cam import (
    GradCAMPlusPlus
)

def save_inference(all_pred: list, all_label: list, fold: str, save_path: str):
    """save the inference results to .pt file.

    Args:
        all_pred (list): predict result.
        all_label (list): label result.
        fold (str): fold number.
        save_path (str): save path.
    """       

    pred = torch.tensor(all_pred)
    label = torch.tensor(all_label)

    # save the results
    save_path = Path(save_path) / "best_preds"

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    torch.save(
        pred,
        save_path / f"{fold}_pred.pt",
    )
    torch.save(
        label,
        save_path / f"{fold}_label.pt",
    )

    logging.info(
        f"save the pred and label into {save_path} / {fold}"
    )

def save_metrics(all_pred: list, all_label: list, fold: str, save_path: str, num_class: int):
    """save the metrics to .txt file.

    Args:
        all_pred (list): all the predict result.
        all_label (list): all the label result.
        fold (str): the fold number.
        save_path (str): the path to save the metrics.
        num_class (int): number of class.
    """    

    save_path = Path(save_path) / "metrics.txt"
    all_pred = torch.tensor(all_pred)
    all_label = torch.tensor(all_label)

    _accuracy = MulticlassAccuracy(num_class)
    _precision = MulticlassPrecision(num_class)
    _recall = MulticlassRecall(num_class)
    _f1_score = MulticlassF1Score(num_class)
    _auroc = MulticlassAUROC(num_class)
    _confusion_matrix = MulticlassConfusionMatrix(num_class, normalize="true")

    logging.info("*" * 100)
    logging.info("accuracy: %s" % _accuracy(all_pred, all_label))
    logging.info("precision: %s" % _precision(all_pred, all_label))
    logging.info("recall: %s" % _recall(all_pred, all_label))
    logging.info("f1_score: %s" % _f1_score(all_pred, all_label))
    logging.info("aurroc: %s" % _auroc(all_pred, all_label.long()))
    logging.info("confusion_matrix: %s" % _confusion_matrix(all_pred, all_label))
    logging.info("#" * 100)

    with open(save_path, "a") as f:
        f.writelines(f"Fold {fold}\n")
        f.writelines(f"accuracy: {_accuracy(all_pred, all_label)}\n")
        f.writelines(f"precision: {_precision(all_pred, all_label)}\n")
        f.writelines(f"recall: {_recall(all_pred, all_label)}\n")
        f.writelines(f"f1_score: {_f1_score(all_pred, all_label)}\n")
        f.writelines(f"aurroc: {_auroc(all_pred, all_label.long())}\n")
        f.writelines(f"confusion_matrix: {_confusion_matrix(all_pred, all_label)}\n")
        f.writelines("#" * 100)
        f.writelines("\n")


def save_CM(all_pred: list, all_label: list, save_path: str, num_class: int, fold: str):
    """save the confusion matrix to file.

    Args:
        all_pred (list): predict result.
        all_label (list): label result.
        save_path (Path): the path to save the confusion matrix.
        num_class (int): the number of class.
        fold (str): the fold number.
    """    

    save_path = Path(save_path) / "CM"
    all_pred = torch.Tensor(all_pred)
    all_label = torch.Tensor(all_label)

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    _confusion_matrix = MulticlassConfusionMatrix(num_class, normalize="true")

    logging.info("_confusion_matrix: %s" % _confusion_matrix(all_pred, all_label))

    # set the font and title
    plt.rcParams.update({"font.size": 30, "font.family": "sans-serif"})

    confusion_matrix_data = _confusion_matrix(all_pred, all_label).cpu().numpy() * 100

    axis_labels = ["ASD", "DHS", "LCS_HipOA"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix_data,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        xticklabels=axis_labels,
        yticklabels=axis_labels,
        vmin=0,
        vmax=100,
    )
    plt.title(f"Fold {fold} (%)", fontsize=30)
    plt.ylabel("Actual Label", fontsize=30)
    plt.xlabel("Predicted Label", fontsize=30)

    plt.savefig(
        save_path / f"fold{fold}_confusion_matrix.png", dpi=300, bbox_inches="tight"
    )

    logging.info(
        f"save the confusion matrix into {save_path}/fold{fold}_confusion_matrix.png"
    )

# @warnings.deprecated('not used')
# def save_CAM(
#     config,
#     model: torch.nn.Module,
#     input_tensor: torch.Tensor,
#     inp_label,
#     fold,
#     flag,
#     i,
#     random_index,
# ):
#     """_summary_

#     Args:
#         config (_type_): _description_
#         model (torch.nn.Module): _description_
#         input_tensor (torch.Tensor): _description_
#         inp_label (_type_): _description_
#         fold (_type_): _description_
#         flag (_type_): _description_
#         i (_type_): _description_
#         random_index (_type_): _description_
#     """    
#     # FIXME: 由于backbone的不同，需要修改target_layer的位置。
#     # guided grad cam method
#     target_layer = [model.blocks[-2].res_blocks[-1]]
#     # target_layer = [model.model.blocks[-2]]

#     cam = GradCAMPlusPlus(model, target_layer)

#     # save the CAM
#     save_path = Path(config.train.log_path) / "CAM" / f"fold{fold}" / flag

#     if save_path.exists() is False:
#         save_path.mkdir(parents=True)

#     for idx, num in enumerate(random_index):

#         grayscale_cam = cam(
#             input_tensor[num : num + 1], aug_smooth=True, eigen_smooth=True
#         )
#         output = cam.outputs

#         # prepare save figure
#         inp_tensor = (
#             input_tensor[num].permute(1, 2, 3, 0)[-1].cpu().detach().numpy()
#         )  # display original image
#         cam_map = grayscale_cam.squeeze().mean(axis=2, keepdims=True) # b, c, t, h, w

#         # figure, axis = viz.visualize_image_attr(
#         #     cam_map,
#         #     inp_tensor,
#         #     method="blended_heat_map",
#         #     sign="positive",
#         #     show_colorbar=True,
#         #     cmap="jet",
#         #     title=f"label: {int(inp_label[num])}, pred: {output.argmax(dim=1)[0]}",
#         # )

#         # figure.savefig(
#         #     save_path / f"fold{fold}_{flag}_step{i}_num{idx}_{num}.png",
#         # )

#     # save with pytorch_grad_cam
#     # visualization = show_cam_on_image(inp_tensor, cam_map, use_rgb=True)

#     print(f"save the CAM into {save_path}")
