# Copyright 2020 - 2023 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import os
from typing import List, Optional
import numpy as np
import torch
from monai import data, transforms
from monai.transforms import (
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Rotate90d,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    RemoveSmallObjectsd,
    RandRotated,
    ScaleIntensityRanged,
    ScaleIntensityd,
    ClipIntensityPercentilesd,
    Spacingd,
    OneOf,
)
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from .config import DataConfig, TestConfig
import json



def get_dataset(
    data_cfg: DataConfig | TestConfig,
    train_files: Optional[List] = None,
    val_files: Optional[List] = None,
    test_files: Optional[List] = None,
    test=False,
):
    if not test:
        train_transform = transforms.Compose(
            [
                LoadImaged(keys=["image", "label"], image_only=True),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAI"),
                #Rotate90d(keys=["image", "label"], k=1),
                #Spacingd(
                #    keys=["image", "label"],
                #    pixdim=(1.5, 1.5, 1.5),
                #    mode=("bilinear", "nearest"),
                #),
                #ScaleIntensityRanged(
                #    keys=["image"],
                #    a_min=data_cfg.a_min,
                #    a_max=data_cfg.a_max,
                #    b_min=data_cfg.b_min,
                #    b_max=data_cfg.b_max,
                #    clip=True,
                #),
                ClipIntensityPercentilesd(keys=["image"], lower=0.5, upper=99.5),
                ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                RemoveSmallObjectsd(
                    keys=["label"],
                    min_size=100,
                    connectivity=8,
                ),
                #OneOf(
                #    transforms=[
                #        Rotate90d(keys=["image", "label"], k=0, spatial_axes=(0, 1)),
                #        Rotate90d(keys=["image", "label"], k=1, spatial_axes=(1, 2)),
                #        Rotate90d(keys=["image", "label"], k=1, spatial_axes=(0, 2)),
                #    ],
                #),
                RandRotated(
                    keys=["image", "label"],
                    prob=0.25,
                    range_x=0.26,
                    range_y=0.26,
                    mode=("bilinear", "nearest"),
                ),
                #        RandRotate90d(
                #            keys=["image", "label"],
                #            prob=0.25,
                #            max_k=3,
                #        ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.25,
                    prob=0.25,
                ),
                RandGaussianNoised(keys=["image"], prob=0.15, std=0.01),
                RandGaussianSmoothd(keys=["image"], prob=0.15, sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15),sigma_z=(0.5, 1.15)),
            ]
        )

        val_transform = transforms.Compose(
            [
                LoadImaged(keys=["image", "label"], image_only=True),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAI"),
                #Rotate90d(keys=["image", "label"], k=1),
                #Spacingd(
                #    keys=["image", "label"],
                #    pixdim=(1.5, 1.5, 1.5),
                #    mode=("bilinear", "nearest"),
                #),
                # ScaleIntensityRanged(
                #    keys=["image"],
                #    a_min=data_cfg.a_min,
                #    a_max=data_cfg.a_max,
                #    b_min=data_cfg.b_min,
                #    b_max=data_cfg.b_max,
                #    clip=True,
                # ),
                ClipIntensityPercentilesd(keys=["image"], lower=0.5, upper=99.5),
                ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                RemoveSmallObjectsd(
                    keys=["label"],
                    min_size=100,
                    connectivity=8,
                ),
                #OneOf(
                #    transforms=[
                #        Rotate90d(keys=["image", "label"], k=0, spatial_axes=(0, 1)),
                #        Rotate90d(keys=["image", "label"], k=1, spatial_axes=(1, 2)),
                #        Rotate90d(keys=["image", "label"], k=1, spatial_axes=(0, 2)),
                #    ],
                #),
            ]
        )


        #if use_normal_dataset:
        #    train_ds = data.Dataset(data=train_files, transform=train_transform)
        #else:
        #train_ds = data.CacheDataset(
        #            data=train_files,
        #            transform=train_transform,
        #            # cache_rate=1,
        #            cache_num=150,
        #            num_workers=0,
        #        )
        train_ds = data.PersistentDataset(data=train_files, transform=train_transform, cache_dir='/home/iscarinci/total_liifa_cache')
            
        #val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_ds = data.PersistentDataset(
            data=val_files,
            transform=val_transform,
            cache_dir="/home/iscarinci/total_liifa_cache",
        )
        return train_ds, val_ds
    else:
        test_transform = transforms.Compose(
            [
                LoadImaged(keys=["image", "label"], image_only=True),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 1.5),
                    mode=("bilinear", "nearest"),
                ),
                ClipIntensityPercentilesd(keys=["image"], lower=0.5, upper=99.5),
                ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                RemoveSmallObjectsd(
                    keys=["label"],
                    min_size=100,
                    connectivity=8,
                ),
            ]
        )
        
        test_ds = data.Dataset(data=test_files, transform=test_transform)
        
        return test_ds


def split_data(data_cfg: DataConfig):
    data_dir = data_cfg.data_dir
    

    with open(data_cfg.json_list, "r") as f:
        json_data = json.load(f)

    if "validation" in json_data.keys():
        list_train = json_data["training"]
        list_valid = json_data["validation"]
        list_test = json_data["testing"]
    else:
        list_train = json_data["training"]
        list_test = json_data["testing"]
        
        list_train = sorted(list_train, key=lambda x: x["image"])
        import random
        random.seed(12345)
        random.shuffle(list_train)
        l_val = int(len(list_train) * data_cfg.splitval)
        l_val = (l_val // 2) * 2  # aseguro que l_val sea multiplo de 2
        list_valid = list_train[-l_val:]
        list_train = list_train[:-l_val]

    #    if hasattr(args, "rank") and args.rank == 0:
    #        print("train files", len(list_train), [os.path.basename(_["image"]).split(".")[0] for _ in list_train])
    #        print("val files", len(list_valid), [os.path.basename(_["image"]).split(".")[0] for _ in list_valid])
    #        print("test files", len(list_test), [os.path.basename(_["image"]).split(".")[0] for _ in list_test])

    # training data
    files = []
    for _i in range(len(list_train)):
        str_img = os.path.join(data_dir, list_train[_i]["image"])
        str_seg = os.path.join(data_dir, list_train[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})

    train_files = copy.deepcopy(files)

    files = []
    for _i in range(len(list_valid)):
        str_img = os.path.join(data_dir, list_valid[_i]["image"])
        str_seg = os.path.join(data_dir, list_valid[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})
    val_files = copy.deepcopy(files)

    return train_files, val_files



def test_files(data_cfg: TestConfig):
    data_dir = data_cfg.data_dir

    with open(data_cfg.json_list, "r") as f:
        json_data = json.load(f)
        list_test = json_data["testing"]
        labels = json_data["labels"]

    files = []
    for _i in range(len(list_test)):
        str_img = os.path.join(data_dir, list_test[_i]["image"])
        str_seg = os.path.join(data_dir, list_test[_i]["label"])

        if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
            continue

        files.append({"image": str_img, "label": str_seg})
    test_files = copy.deepcopy(files)
    return test_files, labels
