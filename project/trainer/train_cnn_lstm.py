#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/trainer/train_two_stream.py
Project: /workspace/skeleton/project/trainer
Created Date: Friday June 7th 2024
Author: Kaixu Chen
-----
Comment:
This file implements the training process for cnn lstm method.
Here, saving the results and calculating the metrics are done in separate functions.

Have a good code time :)
-----
Last Modified: Friday June 7th 2024 7:50:12 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from torchvision.io import write_video
from torchvision.utils import save_image, flow_to_image

from project.models.make_model import CNNLSTM

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)

class CNNLstmModule(LightningModule):

    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.model_type = hparams.model.model
        self.lr = hparams.optimizer.lr
        self.num_classes = hparams.model.model_class_num

        # model define

        self.model = CNNLSTM(hparams)
        
        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        train steop when trainer.fit called

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss
        """

        video = batch["video"].detach() # b, c, t, h, w
        label = batch["label"].detach()  # b, c, t, h, w
        label = label.repeat_interleave(video.size()[2])

        loss = self.single_logic(label, video)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        val step when trainer.fit called.

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_

        Returns:
            loss: the calc loss
            accuract: selected accuracy result.
        """

        # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach()  # b

        label = label.repeat_interleave(video.size()[2])
        loss = self.single_logic(label, video)

    def test_step(self, batch, batch_idx):
        """
        test step when trainer.test called

        Args:
            batch (3D tensor): b, c, t, h, w
            batch_idx (_type_): _description_
        """
         # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach()  # b

        # not use the last frame
        label = label.repeat_interleave(video.size()[2])
        loss = self.single_logic(label, video)

    def configure_optimizers(self):
        """
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        """

        optimzier = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimzier,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimzier),
                "monitor": "val/loss",
            },
        }
        # return torch.optim.SGD(self.parameters(), lr=self.lr)

    def _get_name(self):
        return self.model_type

    def single_logic(self, label: torch.Tensor, video: torch.Tensor):

        b, c, t, h, w = video.shape

        # eval model, feed data here
        if self.training:
            preds = self.model(video)
            
        else:
            with torch.no_grad():
                preds = self.model(video)

        # squeeze(dim=-1) to keep the torch.Size([1]), not null.

        loss = F.cross_entropy(
            preds.squeeze(dim=-1), label.long()
        )

        self.save_log(preds, label, loss)

        return loss

    def save_log(self, pred: torch.Tensor, label: torch.Tensor, loss):

        if self.training:

            preds = pred

            # when torch.size([1]), not squeeze.
            if preds.size()[0] != 1 or len(preds.size()) != 1:
                preds = preds.squeeze(dim=-1)
                pred_softmax = torch.softmax(preds, dim=-1)
            else:
                pred_softmax = torch.softmax(preds)

            # video rgb metrics
            accuracy = self._accuracy(pred_softmax, label)
            precision = self._precision(pred_softmax, label)
            recall = self._recall(pred_softmax, label)
            f1_score = self._f1_score(pred_softmax, label)
            confusion_matrix = self._confusion_matrix(pred_softmax, label)

            # log to tensorboard
            self.log_dict(
                {
                    "train/loss": loss,
                    "train/video_acc": accuracy,
                    "train/video_precision": precision,
                    "train/video_recall": recall,
                    "train/video_f1_score": f1_score,
                },
                on_epoch=True,
                on_step=True,
                batch_size=label.size()[0],
            )

        else:

            preds = pred

            # when torch.size([1]), not squeeze.
            if preds.size()[0] != 1 or len(preds.size()) != 1:
                preds = preds.squeeze(dim=-1)
                pred_softmax = torch.sigmoid(preds)
            else:
                pred_softmax = torch.sigmoid(preds)

            # video rgb metrics
            accuracy = self._accuracy(pred_softmax, label)
            precision = self._precision(pred_softmax, label)
            recall = self._recall(pred_softmax, label)
            f1_score = self._f1_score(pred_softmax, label)
            confusion_matrix = self._confusion_matrix(pred_softmax, label)

            # log to tensorboard
            self.log_dict(
                {
                    "val/loss": loss,
                    "val/video_acc": accuracy,
                    "val/video_precision": precision,
                    "val/video_recall": recall,
                    "val/video_f1_score": f1_score,
                },
                on_epoch=True,
                on_step=True,
                batch_size=label.size()[0],
            )
