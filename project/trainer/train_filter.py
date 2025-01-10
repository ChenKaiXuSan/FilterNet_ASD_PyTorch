#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/skeleton/project/trainer/train_two_stream copy.py
Project: /workspace/skeleton/project/trainer
Created Date: Sunday June 9th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday January 9th 2025 12:29:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import torch
import logging
import torch.nn.functional as F
from pytorch_lightning import LightningModule

# from torchvision.io import write_video
# from torchvision.utils import save_image, flow_to_image

from project.models.make_model import MakeImageModule, MakeViTModule
from project.utils.helper import save_inference, save_metrics, save_CM

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)


class CNNModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # return model type name
        self.lr = hparams.optimizer.lr
        self.num_classes = hparams.model.model_class_num

        # model define
        # model = MakeImageModule(hparams)
        # self.model = model.make_resnet(self.num_classes)

        self.model = self.init_model(hparams, hparams.model.model, self.num_classes)

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

    def init_model(self, hparams, model: str, num_classes: int):

        if model == "2dcnn":
            model = MakeImageModule(hparams)
            model = model.make_resnet(num_classes)

        elif model == "vit":
            model = MakeViTModule(hparams)
            model = model.make_vit(num_classes)

        else:
            raise ValueError("the model is not supported.")

        return model

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

        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach()  # b, c, t, h, w
        # label = label.repeat_interleave(video.size()[2])

        video = video.permute(1, 0, 2, 3)
        t, c, h, w = video.size()

        if t > 128:
            for i in range(0, c, 128):  # 128 is the batch size
                preds = self.model(video[i : i + 128, ...])

                loss = F.cross_entropy(
                    preds.squeeze(dim=-1), label[i : i + 128].long()
                )
                self.save_log(preds, label[i : i + 128], loss)
        else:
            preds = self.model(video)

            loss = F.cross_entropy(preds.squeeze(dim=-1), label.long())

            self.save_log(preds, label, loss)

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

        video = video.permute(1, 0, 2, 3)

        with torch.no_grad():
            preds = self.model(video)

        loss = F.cross_entropy(preds.squeeze(dim=-1), label.long())

        self.save_log(preds, label, loss)

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

    def _get_name(self):
        return self.model_type

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

    ##############
    # test step
    ##############
    # the order of the hook function is:
    # on_test_start -> test_step -> on_test_batch_end -> on_test_epoch_end -> on_test_end

    def on_test_start(self) -> None:
        """hook function for test start"""
        self.test_outputs = []
        self.test_pred_list = []
        self.test_label_list = []

        logging.info("test start")

    def on_test_end(self) -> None:
        """hook function for test end"""
        logging.info("test end")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        # input and model define
        video = batch["video"].detach()  # c, t, h, w
        label = batch["label"].detach().float().squeeze()  # b

        # label = label.repeat_interleave(video.size()[1])
        video = video.permute(1, 0, 2, 3)

        # eval model, feed data here
        with torch.no_grad():
            preds = self.model(video)

        loss = F.cross_entropy(preds.squeeze(dim=-1), label.long())

        self.log("test/loss", loss, on_epoch=True, on_step=True, batch_size=video.size()[0])

        # log metrics
        video_acc = self._accuracy(preds, label)
        video_precision = self._precision(preds, label)
        video_recall = self._recall(preds, label)
        video_f1_score = self._f1_score(preds, label)
        video_confusion_matrix = self._confusion_matrix(preds, label)

        metric_dict = {
            "test/video_acc": video_acc,
            "test/video_precision": video_precision,
            "test/video_recall": video_recall,
            "test/video_f1_score": video_f1_score,
        }
        self.log_dict(metric_dict, on_epoch=True, on_step=True, batch_size=video.size()[0])

        return preds

    def on_test_batch_end(
        self,
        outputs: list[torch.Tensor],
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """hook function for test batch end

        Args:
            outputs (torch.Tensor | logging.Mapping[str, Any] | None): current output from batch.
            batch (Any): the data of current batch.
            batch_idx (int): the index of current batch.
            dataloader_idx (int, optional): the index of all dataloader. Defaults to 0.
        """

        perds = outputs
        label = batch["label"].detach().float().squeeze()

        self.test_outputs.append(outputs)
        # tensor to list
        for i in perds.tolist():
            self.test_pred_list.append(i)
        for i in label.tolist():
            self.test_label_list.append(i)

    def on_test_epoch_end(self) -> None:
        """hook function for test epoch end"""

        # save inference
        save_inference(
            self.test_pred_list,
            self.test_label_list,
            fold=self.logger.name,
            save_path=self.hparams.hparams.train.log_path,
        )
        # save metrics
        save_metrics(
            self.test_pred_list,
            self.test_label_list,
            fold=self.logger.name,
            save_path=self.hparams.hparams.train.log_path,
            num_class=self.num_classes,
        )
        # save confusion matrix
        save_CM(
            self.test_pred_list,
            self.test_label_list,
            save_path=self.hparams.hparams.train.log_path,
            num_class=self.num_classes,
            fold=self.logger.name,
        )

        # save CAM
        # save_CAM(self.test_pred_list, self.test_label_list, self.num_classes)

        logging.info("test epoch end")
