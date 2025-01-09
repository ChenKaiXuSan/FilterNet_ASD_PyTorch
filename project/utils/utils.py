'''
File: utils.py
Project: utils
Created Date: 2023-09-03 13:02:25
Author: chenkaixu
-----
Comment:
 
Have a good code time!
-----
Last Modified: 2023-09-03 13:03:05
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

'''

import os, shutil
import torch
from torchvision.transforms.functional import crop, pad, resize

def clip_pad_with_bbox(
    imgs: torch.tensor, boxes: list, img_size: int = 256, bias: int = 10
):
    """
    based torchvision function to crop, pad, resize img.

    clip with the bbox, (x1-bias, y1) and padd with the (gap-bais) in left and right.

    Args:
        imgs (list): imgs with (h, w, c)
        boxes (list): (x1, y1, x2, y2)
        img_size (int, optional): croped img size. Defaults to 256.
        bias (int, optional): the bias of bbox, with the (x1-bias) and (x2+bias). Defaults to 5.

    Returns:
        tensor: (c, t, h, w)
    """
    object_list = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  # dtype must int for resize, crop function

        box_width = x2 - x1
        box_height = y2 - y1

        width_gap = int(((box_height - box_width) / 2))  # keep int type

        img = imgs  # (h, w, c) to (c, h, w), for pytorch function

        # give a bias for the left and right crop bbox.
        croped_img = crop(
            img,
            top=y1,
            left=(x1 - bias),
            height=box_height,
            width=(box_width + 2 * bias),
        )

        pad_img = pad(croped_img, padding=(width_gap - bias, 0), fill=0)

        resized_img = resize(pad_img, size=(img_size, img_size))

        object_list.append(resized_img)

    return object_list  # c, t, h, w

def del_folder(path, *args):
    """
    delete the folder which path/version

    Args:
        path (str): path
        version (str): version
    """
    if os.path.exists(os.path.join(path, *args)):
        shutil.rmtree(os.path.join(path, *args))


def make_folder(path, *args):
    """
    make folder which path/version

    Args:
        path (str): path
        version (str): version
    """
    if not os.path.exists(os.path.join(path, *args)):
        os.makedirs(os.path.join(path, *args))
        print("success make dir! where: %s " % os.path.join(path, *args))
    else:
        print("The target path already exists! where: %s " % os.path.join(path, *args))
