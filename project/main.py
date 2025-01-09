#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/deep-learning-project-template/project/main.py
Project: /workspace/deep-learning-project-template/project
Created Date: Friday November 29th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday November 29th 2024 12:51:51 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
"""
File: main.py
Project: project
Created Date: 2023-10-19 02:29:35
Author: chenkaixu
-----
Comment:
 
Have a good code time!
-----
Last Modified: Thursday October 19th 2023 2:29:35 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

26-11-2024	Kaixu Chen	refactor the code, now run script in python -m project.main

26-11-2024	Kaixu Chen	add attention branch network (ATN) for compare experiment.

23-09-2024	Kaixu Chen	add compare experiment, phasemix with different backbone, like 3dcnn, 2dcnn, cnn_lstm.

25-06-2024	Kaixu Chen	Splitting the backbone and temporal mix was used for more detailed comparison tests

07-06-2024	Kaixu Chen	add two stream compare experiment.

14-05-2024	Kaixu Chen	1. move the train process inside the new folder "trainer" and select based on "experiment" keyword.
                        2. add the save helper to save the inference results. deplucate the save_inference code in the main.py.
04-04-2024	Kaixu Chen	add save inference method. now it can save the pred/label to the disk, for the further analysis.
2023-10-29	KX.C	add the lr monitor, and fast dev run to trainer.

"""

import os
import logging
import hydra
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

from project.dataloader.data_loader import WalkDataModule

#####################################
# select different experiment trainer 
#####################################

# 3D CNN model
from project.trainer.train_single import SingleModule
from project.trainer.train_late_fusion import LateFusionModule
from project.trainer.train_temporal_mix import TemporalMixModule
# compare experiment
from project.trainer.train_two_stream import TwoStreamModule
from project.trainer.train_cnn_lstm import CNNLstmModule
from project.trainer.train_cnn import CNNModule
# Attention Branch Network
from project.trainer.train_backbone_atn import BackboneATNModule


from project.cross_validation import DefineCrossValidation
from project.helper import save_helper


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

    # * select experiment
    if hparams.train.backbone == "3dcnn":
        # * ablation study 2: different training strategy
        if "late_fusion" in hparams.train.experiment:
            classification_module = LateFusionModule(hparams)
        elif "single" in hparams.train.experiment:
            classification_module = SingleModule(hparams)
        elif hparams.train.temporal_mix:
            classification_module = TemporalMixModule(hparams)
        else:
            raise ValueError(f"the {hparams.train.experiment} is not supported.")
    elif hparams.train.backbone == "3dcnn_atn":
        classification_module = BackboneATNModule(hparams)
    # * compare experiment
    elif hparams.train.backbone == "two_stream":
        classification_module = TwoStreamModule(hparams)
    # * compare experiment
    elif hparams.train.backbone == "cnn_lstm":
        classification_module = CNNLstmModule(hparams)
    # * compare experiment
    elif hparams.train.backbone == "2dcnn":
        classification_module = CNNModule(hparams)

    else:
        raise ValueError("the experiment backbone is not supported.")

    data_module = WalkDataModule(hparams, dataset_idx)

    # for the tensorboard
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hparams.train.log_path),
        name=str(fold),  # here should be str type.
    )

    # some callbacks
    progress_bar = TQDMProgressBar(refresh_rate=100)
    rich_model_summary = RichModelSummary(max_depth=2)

    # define the checkpoint becavier.
    model_check_point = ModelCheckpoint(
        filename="{epoch}-{val/loss:.2f}-{val/video_acc:.4f}",
        auto_insert_metric_name=False,
        monitor="val/video_acc",
        mode="max",
        save_last=False,
        save_top_k=2,
    )

    # define the early stop.
    early_stopping = EarlyStopping(
        monitor="val/video_acc",
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

    # trainer.fit(classification_module, data_module)

    # the validate method will wirte in the same log twice, so use the test method.
    trainer.test(
        classification_module,
        data_module,
        # ckpt_path="best",
    )

    # TODO: the save helper for 3dnn_atn not implemented yet.
    if hparams.train.backbone == "3dcnn_atn":
        pass
    else:
        # save_helper(hparams, classification_module, data_module, fold) #! debug only
        save_helper(
            hparams,
            classification_module.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path
            ),
            data_module,
            fold,
        )


@hydra.main(
    version_base=None,
    config_path="../configs", # * the config_path is relative to location of the python script
    config_name="config.yaml",
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
