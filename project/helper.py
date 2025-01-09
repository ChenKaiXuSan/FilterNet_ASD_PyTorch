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
Last Modified: Thursday January 9th 2025 1:23:51 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

import logging
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import random
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
from captum.attr import visualization as viz

@warnings.deprecated('not used')
def save_helper(config, model, dataloader, fold):

    if "late_fusion" in config.train.experiment:
        total_pred, total_label = save_inference_late_fusion(
            config, model, dataloader, fold
        )
    elif "two_stream" in config.train.experiment:
        total_pred, total_label = save_inference_two_stream(
            config, model, dataloader, fold
        )
    elif "3dcnn_atn" in config.train.experiment:
        total_pred, total_label = save_inference_atn(config, model, dataloader, fold)
    else:
        total_pred, total_label = save_inference(config, model, dataloader, fold)

    save_metrics(total_pred, total_label, fold, config)
    save_CM(total_pred, total_label, fold, config)

@warnings.deprecated('not used')
def save_inference_two_stream(config, model, dataloader, fold):

    total_pred_list = []
    total_label_list = []

    test_dataloader = dataloader.test_dataloader()
    optical_flow_model = model.optical_flow_model
    model_rgb = model.model_rgb
    model_flow = model.model_flow

    for i, batch in enumerate(test_dataloader):

        # input and label
        video = (
            batch["video"].detach().to(f"cuda:{config.train.gpu_num}")
        )  # b, c, t, h, w
        label = (
            batch["label"].detach().to(f"cuda:{config.train.gpu_num}")
        )  # b, class_num
        label = label.repeat_interleave(video.size()[2]-1)

        model.eval().to(f"cuda:{config.train.gpu_num}")

        video_flow = optical_flow_model.process_batch(video)  # b, c, t, h, w

        b, c, t, h, w = video.shape

        single_img = video[:, :, :-1, :].reshape(-1, 3, h, w)
        single_flow = video_flow.contiguous().view(-1, 2, h, w)

        # pred the video frames
        with torch.no_grad():
            pred_video_rgb = model_rgb(single_img)
            pred_video_flow = model_flow(single_flow)

        pred_total = (pred_video_rgb + pred_video_flow) / 2
        pred_video_softmax = torch.softmax(pred_total, dim=1)

        # random_index = random.sample(range(0, video.size()[0]), 2)
        # save_CAM(
        #     config, model.model_rgb, video[..., 0], label, fold, "rgb", i, random_index
        # )
        # save_CAM(
        #     config,
        #     model.model_flow,
        #     video[..., 1],
        #     label,
        #     fold,
        #     "flow",
        #     i,
        #     random_index,
        # )

        for i in pred_video_softmax.tolist():
            total_pred_list.append(i)
        for i in label.tolist():
            total_label_list.append(i)

    pred = torch.tensor(total_pred_list)
    label = torch.tensor(total_label_list)

    # save the results
    save_path = Path(config.train.log_path) / "best_preds"

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    torch.save(
        pred,
        save_path / f"{config.model.model}_{config.data.sampling}_{fold}_pred.pt",
    )
    torch.save(
        label,
        save_path / f"{config.model.model}_{config.data.sampling}_{fold}_label.pt",
    )

    logging.info(
        f"save the pred and label into {save_path} / {config.model.model}_{config.data.sampling}_{fold}"
    )

    return pred, label

@warnings.deprecated('not used')
def save_inference_late_fusion(config, model, dataloader, fold):

    total_pred_list = []
    total_label_list = []

    device = f"cuda:{config.train.gpu_num}"

    test_dataloader = dataloader.test_dataloader()
    stance_cnn = model.stance_cnn
    swing_cnn = model.swing_cnn

    for i, batch in enumerate(test_dataloader):

        # if i == 5: break; # ! debug only

        stance_video = (
            batch["video"][..., 0].detach().to(f"cuda:{config.train.gpu_num}")
        )  # b, c, t, h, w
        swing_video = (
            batch["video"][..., 1].detach().to(f"cuda:{config.train.gpu_num}")
        )  # b, c, t, h, w
        # sample_info = batch["info"] # b is the video instance number

        label = batch["label"]

        stance_cnn.eval().to(device)
        swing_cnn.eval().to(device)

        with torch.no_grad():

            # * slove OOM problem, cut the large batch, when >= 30
            # if stance_video.size()[0] + swing_video.size()[0] >= 30:
            #     stance_preds = stance_cnn(stance_video[:14])
            #     swing_preds = swing_cnn(swing_video[:14])

            #     label = label[:14]

            #     stance_video = stance_video[:14]
            #     swing_video = swing_video[:14]

            # else:
            stance_preds = stance_cnn(stance_video)
            swing_preds = swing_cnn(swing_video)

        predict = (stance_preds + swing_preds) / 2
        preds_softmax = torch.softmax(predict, dim=1)

        # * Since saving the video tensor is too GPU memory intensive, the content is extracted in a batch to be saved
        random_index = random.sample(range(0, stance_video.size()[0]), 2)
        save_CAM(
            config, stance_cnn, stance_video, label, fold, "stance", i, random_index
        )
        save_CAM(config, swing_cnn, swing_video, label, fold, "swing", i, random_index)

        for i in preds_softmax.tolist():
            total_pred_list.append(i)
        for i in label.tolist():
            total_label_list.append(i)

    pred = torch.tensor(total_pred_list)
    label = torch.tensor(total_label_list)

    # save the results
    save_path = Path(config.train.log_path) / "best_preds"

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    torch.save(
        pred,
        save_path / f"{config.model.model}_{config.data.sampling}_{fold}_pred.pt",
    )
    torch.save(
        label,
        save_path / f"{config.model.model}_{config.data.sampling}_{fold}_label.pt",
    )

    logging.info(
        f"save the pred and label into {save_path}/{config.model.model}_{config.data.sampling}_{fold}"
    )

    return pred, label

# @warnings.deprecated('not used')
# def save_inference(config, model, dataloader, fold):

#     total_pred_list = []
#     total_label_list = []

#     test_dataloader = dataloader.test_dataloader()

#     for i, batch in enumerate(test_dataloader):

#         # input and label
#         video = (
#             batch["video"].detach().to(f"cuda:{config.train.gpu_num}")
#         )  # b, c, t, h, w
#         label = (
#             batch["label"].detach().to(f"cuda:{config.train.gpu_num}")
#         )  # b, class_num

#         if "cnn_lstm" in config.train.experiment:
#             label = label.repeat_interleave(video.size()[2])
        
#         if "2dcnn" in config.train.experiment:
#             b, c, t, h, w = video.size()
#             label = label.repeat_interleave(video.size()[2])
#             video = video.reshape(b*t, c, h, w)
            

#         model.eval().to(f"cuda:{config.train.gpu_num}")

#         # pred the video frames
#         with torch.no_grad():
#             preds = model(video)

#         # when torch.size([1]), not squeeze.
#         if preds.size()[0] != 1 or len(preds.size()) != 1:
#             preds = preds.squeeze(dim=-1)
#             preds_softmax = torch.softmax(preds, dim=1)
#         else:
#             preds_softmax = torch.softmax(preds, dim=1)

#         # random_index = random.sample(range(0, video.size()[0]), 2)
#         # save_CAM(
#         #     config,
#         #     model.video_cnn,
#         #     video,
#         #     label,
#         #     fold,
#         #     config.train.experiment,
#         #     i,
#         #     random_index,
#         # )

#         for i in preds_softmax.tolist():
#             total_pred_list.append(i)
#         for i in label.tolist():
#             total_label_list.append(i)

#     pred = torch.tensor(total_pred_list)
#     label = torch.tensor(total_label_list)

#     # save the results
#     save_path = Path(config.train.log_path) / "best_preds"

#     if save_path.exists() is False:
#         save_path.mkdir(parents=True)

#     torch.save(
#         pred,
#         save_path / f"{config.model.model}_{config.data.sampling}_{fold}_pred.pt",
#     )
#     torch.save(
#         label,
#         save_path / f"{config.model.model}_{config.data.sampling}_{fold}_label.pt",
#     )

#     logging.info(
#         f"save the pred and label into {save_path} / {config.model.model}_{config.data.sampling}_{fold}"
#     )

#     return pred, label

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

@warnings.deprecated('not used')
def save_CAM(
    config,
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    inp_label,
    fold,
    flag,
    i,
    random_index,
):
    """_summary_

    Args:
        config (_type_): _description_
        model (torch.nn.Module): _description_
        input_tensor (torch.Tensor): _description_
        inp_label (_type_): _description_
        fold (_type_): _description_
        flag (_type_): _description_
        i (_type_): _description_
        random_index (_type_): _description_
    """    
    # FIXME: 由于backbone的不同，需要修改target_layer的位置。
    # guided grad cam method
    target_layer = [model.blocks[-2].res_blocks[-1]]
    # target_layer = [model.model.blocks[-2]]

    cam = GradCAMPlusPlus(model, target_layer)

    # save the CAM
    save_path = Path(config.train.log_path) / "CAM" / f"fold{fold}" / flag

    if save_path.exists() is False:
        save_path.mkdir(parents=True)

    for idx, num in enumerate(random_index):

        grayscale_cam = cam(
            input_tensor[num : num + 1], aug_smooth=True, eigen_smooth=True
        )
        output = cam.outputs

        # prepare save figure
        inp_tensor = (
            input_tensor[num].permute(1, 2, 3, 0)[-1].cpu().detach().numpy()
        )  # display original image
        cam_map = grayscale_cam.squeeze().mean(axis=2, keepdims=True) # b, c, t, h, w

        figure, axis = viz.visualize_image_attr(
            cam_map,
            inp_tensor,
            method="blended_heat_map",
            sign="positive",
            show_colorbar=True,
            cmap="jet",
            title=f"label: {int(inp_label[num])}, pred: {output.argmax(dim=1)[0]}",
        )

        figure.savefig(
            save_path / f"fold{fold}_{flag}_step{i}_num{idx}_{num}.png",
        )

    # save with pytorch_grad_cam
    # visualization = show_cam_on_image(inp_tensor, cam_map, use_rgb=True)

    print(f"save the CAM into {save_path}")
