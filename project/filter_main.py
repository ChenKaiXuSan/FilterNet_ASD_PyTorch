#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/project/project/main.py
Project: /workspace/project/project
Created Date: Thursday January 9th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday January 9th 2025 2:08:28 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import os
import logging
import hydra
import shutil
from omegaconf import DictConfig

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    TQDMProgressBar,
    RichModelSummary,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from project.dataloader.filter_data_loader import WalkDataModule
from project.trainer.train_filter import CNNModule
from project.filter_cross_validation import DefineCrossValidation


def train(hparams: DictConfig, dataset_idx, fold: int):
    """the train process for the one fold.

    Args:
        hparams (hydra): the hyperparameters.
        dataset_idx (int): the dataset index for the one fold.
        fold (int): the fold index.

    Returns:
        list: best trained model, data loader
    """

    seed_everything(42, workers=True)

    if hparams.train.backbone == "3dcnn":
        pass
    elif hparams.train.backbone == "2dcnn":
        classification_module = CNNModule(hparams)
    elif hparams.train.backbone == "vit":
        classification_module = CNNModule(hparams)
    else:
        raise ValueError("the experiment backbone is not supported.")

    data_module = WalkDataModule(hparams, dataset_idx)

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hparams.train.log_path),
        name=str(fold),  # here should be str type.
        default_hp_metric=False,
    )

    # some callbacks
    progress_bar = TQDMProgressBar(refresh_rate=100)
    rich_model_summary = RichModelSummary(max_depth=2)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        filename="{epoch}-{filter_val/loss:.2f}-{filter_val/acc:.4f}",
        auto_insert_metric_name=False,
        monitor="filter_val/acc",
        mode="max",
        save_last=False,
        save_top_k=2,
    )

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor="filter_val/acc",
        patience=3,
        mode="max",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        devices=[
            int(hparams.train.gpu_num),
        ],
        accelerator="gpu",
        max_epochs=hparams.train.max_epochs,
        # limit_train_batches=2,
        # limit_val_batches=2,
        logger=tb_logger,  # wandb_logger,
        check_val_every_n_epoch=1,
        callbacks=[
            progress_bar,
            rich_model_summary,
            model_check_point,
            early_stopping,
            lr_monitor,
        ],
        fast_dev_run=hparams.train.fast_dev_run,  # if use fast dev run for debug.
    )

    trainer.fit(classification_module, data_module)
 
    # use test step to save log.
    trainer.test(
        classification_module,
        data_module,
        ckpt_path="best",
    )

    # save the best model to file.
    best_model_path = os.path.join(model_check_point.best_model_path)
    logging.info(f"best model path: {best_model_path}")
    log_best_model_path = os.path.join(hparams.train.log_path, f"filter_ckpt/{str(fold)}_best_model.ckpt")
    if os.path.exists(os.path.join(hparams.train.log_path, "filter_ckpt")) is False:
        os.makedirs(os.path.join(hparams.train.log_path, "filter_ckpt"))
    shutil.copyfile(best_model_path, log_best_model_path)

@hydra.main(
    version_base=None,
    config_path="../configs",  # * the config_path is relative to location of the python script
    config_name="filter_config.yaml",
)
def init_params(config):

    #######################
    # prepare dataset index
    #######################

    fold_dataset_idx = DefineCrossValidation(config)()

    logging.info("#" * 50)
    logging.info("Start train all fold")
    logging.info("#" * 50)

    #########
    # K fold
    #########
    # * for one fold, we first train/val model, then save the best ckpt preds/label into .pt file.

    for fold, dataset_value in fold_dataset_idx.items():
        logging.info("#" * 50)
        logging.info("Start train fold: {}".format(fold))
        logging.info("#" * 50)

        train(config, dataset_value, fold)

        logging.info("#" * 50)
        logging.info("finish train fold: {}".format(fold))
        logging.info("#" * 50)

    logging.info("#" * 50)
    logging.info("finish train all fold")
    logging.info("#" * 50)


if __name__ == "__main__":

    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
