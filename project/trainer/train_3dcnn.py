"""
File: train.py
Project: project
Created Date: 2023-10-19 02:29:47
Author: chenkaixu
-----
Comment:
 This file is the train/val/test process for the project.


Have a good code time!
-----
Last Modified: Thursday July 10th 2025 9:41:08 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

16-05-2024	Kaixu Chen	do not output metrics in test_step to log file.

22-03-2024	Kaixu Chen	add different class number mapping, now the class number is a hyperparameter.

14-12-2023	Kaixu Chen refactor the code, now it a simple code to train video frame from dataloader.

"""

from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from pytorch_lightning import LightningModule

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)

from project.models.make_model import MakeVideoModule

from project.helper import save_helper

logger = logging.getLogger(__name__)


class Res3DCNNModule(LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.img_size = hparams.data.img_size
        self.lr = hparams.optimizer.lr

        self.num_classes = hparams.model.model_class_num

        # define model
        self.video_cnn = MakeVideoModule(hparams)()

        # save the hyperparameters to the file and ckpt
        self.save_hyperparameters()

        self._accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self._precision = MulticlassPrecision(num_classes=self.num_classes)
        self._recall = MulticlassRecall(num_classes=self.num_classes)
        self._f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self._confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)

    def forward(self, x):
        return self.video_cnn(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):

        # prepare the input and label
        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach().float().squeeze()  # b
        # sample_info = batch["info"] # b is the video instance number

        b, c, t, h, w = video.shape

        video_preds = self.video_cnn(video)
        video_preds_softmax = torch.softmax(video_preds, dim=1)

        # check shape
        if b == 1:
            label = label.unsqueeze(0)

        assert label.shape[0] == video_preds.shape[0]

        loss = F.cross_entropy(video_preds, label.long())

        self.log("train/loss", loss, on_epoch=True, on_step=True, batch_size=b)

        # log metrics
        video_acc = self._accuracy(video_preds_softmax, label)
        video_precision = self._precision(video_preds_softmax, label)
        video_recall = self._recall(video_preds_softmax, label)
        video_f1_score = self._f1_score(video_preds_softmax, label)
        video_confusion_matrix = self._confusion_matrix(video_preds_softmax, label)

        self.log_dict(
            {
                "train/video_acc": video_acc,
                "train/video_precision": video_precision,
                "train/video_recall": video_recall,
                "train/video_f1_score": video_f1_score,
            },
            on_epoch=True,
            on_step=True,
            batch_size=b,
        )

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):

        # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach().float().squeeze()  # b

        b, c, t, h, w = video.shape

        video_preds = self.video_cnn(video)
        video_preds_softmax = torch.softmax(video_preds, dim=1)

        if b == 1:
            label = label.unsqueeze(0)

        # check shape
        assert label.shape[0] == b

        loss = F.cross_entropy(video_preds, label.long())

        self.log("val/loss", loss, on_epoch=True, on_step=True, batch_size=b)

        # log metrics
        video_acc = self._accuracy(video_preds_softmax, label)
        video_precision = self._precision(video_preds_softmax, label)
        video_recall = self._recall(video_preds_softmax, label)
        video_f1_score = self._f1_score(video_preds_softmax, label)
        video_confusion_matrix = self._confusion_matrix(video_preds_softmax, label)

        self.log_dict(
            {
                "val/video_acc": video_acc,
                "val/video_precision": video_precision,
                "val/video_recall": video_recall,
                "val/video_f1_score": video_f1_score,
            },
            on_epoch=True,
            on_step=True,
            batch_size=b,
        )

    ##############
    # test step
    ##############
    # the order of the hook function is:
    # on_test_start -> test_step -> on_test_batch_end -> on_test_epoch_end -> on_test_end

    def on_test_start(self) -> None:
        """hook function for test start"""
        self.test_outputs: list[torch.Tensor] = []
        self.test_pred_list: list[torch.Tensor] = []
        self.test_label_list: list[torch.Tensor] = []

        logger.info("test start")

    def on_test_end(self) -> None:
        """hook function for test end"""
        logger.info("test end")

    def test_step(self, batch: torch.Tensor, batch_idx: int):

        # input and model define
        video = batch["video"].detach()  # b, c, t, h, w
        label = batch["label"].detach().float().squeeze()  # b

        b, c, t, h, w = video.shape

        video_preds = self.video_cnn(video)
        video_preds_softmax = torch.softmax(video_preds, dim=1)

        if b == 1:
            label = label.unsqueeze(0)

        # check shape
        assert label.shape[0] == b

        loss = F.cross_entropy(video_preds, label.long())

        self.log("test/loss", loss, on_epoch=True, on_step=True, batch_size=b)

        # log metrics
        video_acc = self._accuracy(video_preds_softmax, label)
        video_precision = self._precision(video_preds_softmax, label)
        video_recall = self._recall(video_preds_softmax, label)
        video_f1_score = self._f1_score(video_preds_softmax, label)
        video_confusion_matrix = self._confusion_matrix(video_preds_softmax, label)

        self.log_dict(
            {
                "test/video_acc": video_acc,
                "test/video_precision": video_precision,
                "test/video_recall": video_recall,
                "test/video_f1_score": video_f1_score,
            },
            on_epoch=True,
            on_step=True,
            batch_size=b,
        )

        return video_preds_softmax, video_preds

    def on_test_batch_end(
        self,
        outputs: list[torch.Tensor],
        batch: Any,
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

        pred_softmax, pred = outputs
        label = batch["label"].detach().float()

        self.test_outputs.append(outputs)
        self.test_pred_list.append(pred_softmax)
        self.test_label_list.append(label)

    def on_test_epoch_end(self) -> None:
        """hook function for test epoch end"""

        # save the metrics to file
        save_helper(
            all_pred=self.test_pred_list,
            all_label=self.test_label_list,
            fold=self.logger.root_dir.split("/")[-1],
            save_path=self.logger.save_dir,
            num_class=self.num_classes,
        )

        logger.info("test epoch end")

    def configure_optimizers(self):
        """
        configure the optimizer and lr scheduler

        Returns:
            optimizer: the used optimizer.
            lr_scheduler: the selected lr scheduler.
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.trainer.estimated_stepping_batches,
                ),
                "monitor": "train/loss",
            },
        }
