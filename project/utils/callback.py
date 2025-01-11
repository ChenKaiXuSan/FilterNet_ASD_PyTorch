#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/project/project/utils/callback.py
Project: /workspace/project/project/utils
Created Date: Saturday January 11th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday January 11th 2025 5:03:24 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

from pytorch_lightning import Callback
import shutil
import os

class MoveBestModelCallback(Callback):
    def __init__(self, target_dir):
        self.target_dir = target_dir
        os.makedirs(target_dir, exist_ok=True)

    def on_train_end(self, trainer, pl_module):
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            shutil.move(best_model_path, os.path.join(self.target_dir, os.path.basename(best_model_path)))
            print(f"Best model moved to {self.target_dir}")

# # 使用自定义回调
# target_dir = "final_models/"
# move_callback = MoveBestModelCallback(target_dir)

# trainer = Trainer(
#     max_epochs=10,
#     callbacks=[checkpoint_callback, move_callback]  # 添加回调
# )
