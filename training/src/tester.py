from dataclasses import asdict
import json
import os
import random
import time
from typing import Sequence

import fsspec
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

from monai.metrics import DiceMetric, compute_dice
from monai.transforms import Activations, AsDiscrete, Compose, RemoveSmallObjects
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch, ThreadDataLoader

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import  Snapshot, TrainerConfig
from .utils import (
    AverageMeter,
    prepare_sam_val_input_cp_only,
    prepare_sam_val_input_pp_only,
    prepare_sam_val_input_bb_only,
)

import matplotlib.pyplot as plt

class Tester:
    def __init__(
        self,
        device,
        tester_cfg: TrainerConfig,
        model: nn.Module,
        sam_image_size,
        test_dataset: Dataset,
    ):
        self.device = device
        self.config = tester_cfg
        self.model = model
        self.sam_image_size = sam_image_size
        self.test_dataset = test_dataset
        
        self.test_loader = self._prepare_dataloader(self.test_dataset)

        self._load_snapshot()
        self.model = model.to(device)
        #
        if self.config.use_amp:
            self.scaler = GradScaler()
            
        
        
        self.post_label = AsDiscrete(to_onehot=115)
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.dice_acc = DiceMetric(
            include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True
        )

        self.writer = None
        if self.config.logdir is not None:
            if self.config.label_prompt:
                self.path_test = os.path.join(self.config.logdir, 
                                        self.config.experiment_name, 
                                        "test",
                                        "label")
            elif self.config.bbox_prompt:
                self.path_test = os.path.join(self.config.logdir, 
                                        self.config.experiment_name, 
                                        "test",
                                        "bbox")
            elif self.config.point_prompt:
                self.path_test = os.path.join(self.config.logdir, 
                                        self.config.experiment_name, 
                                        "test",
                                        str(self.config.point_pos)+"_point")
            os.makedirs(self.path_test, exist_ok=True)
            self.writer = SummaryWriter(self.path_test, comment="Testing")
            print("Writing Tensorboard logs to ", self.path_test)
            
                
            

    def _prepare_dataloader(self, dataset: Dataset):
        return ThreadDataLoader(
            dataset=dataset,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.data_loader_workers,
        )

    def _load_snapshot(self):
        try:
            snapshot = fsspec.open(self.config.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location="cpu")  # type: ignore
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch")
            return

        snapshot = Snapshot(**snapshot_data)
        self.model.load_state_dict(snapshot.model_state)
        print("Testing from snapshot")
        

    def test_epoch(self, dataloder):
        self.model.eval()
        self.dice_acc.reset()
        with torch.no_grad():
            for batch in dataloder:
                inputs_l = batch["image"]
                labels_l = batch['label']
                self.files = inputs_l.meta["filename_or_obj"][0]
                print(f'File: {inputs_l.meta["filename_or_obj"]}')
                labels_l.shape[-1]
                inputs_l = inputs_l.squeeze()
                labels_l = labels_l.squeeze()
                
                
                n_slices = self.config.roi_z_iter
                # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
                pd = (n_slices // 2, n_slices // 2)
                
                inputs_l = F.pad(inputs_l, pd, "constant", 0)
                labels_l = F.pad(labels_l, pd, "constant", 0)
                n_z_after_pad = labels_l.shape[-1]
                
                
                segmentation = torch.zeros_like(labels_l)
                for idx in range(n_slices // 2, n_z_after_pad - n_slices // 2):
                    
                    inputs = inputs_l[..., idx - n_slices // 2 : idx + n_slices // 2 + 1].permute(2, 0, 1)
                    labels = labels_l[..., idx - n_slices // 2 : idx + n_slices // 2 + 1][..., n_slices // 2]
                    
                    if self.config.label_prompt:
                        data, target, _ = prepare_sam_val_input_cp_only(inputs.to(self.device), labels.to(self.device))
                    elif self.config.point_prompt:
                        data, target, _ = prepare_sam_val_input_pp_only(inputs.to(self.device), labels.to(self.device), self.config, self.sam_image_size, self.config.point_pos, self.config.point_neg)
                    elif self.config.bbox_prompt:
                        data, target, _ = prepare_sam_val_input_bb_only(inputs.to(self.device), labels.to(self.device))

                    with torch.no_grad():
                        with autocast(enabled=self.config.use_amp):
                            outputs = self.model(data)
                            logit = outputs[0]['high_res_logits']
                    #segmentation[...,idx] = self.post_pred(logit)
                    segmentation[..., idx] = torch.argmax(self.post_pred(logit), dim=0)

                print(f'Segmentation shape {segmentation.shape}')
                self.save_predictions(segmentation)
                
                #f_name = batch_data["image"].meta["filename_or_obj"]
                #segmentation.numpy()
    
    def save_predictions(self, segmentation):
        
        name = self.files.split("/")[-1].split(".")[0]
        segmentation = segmentation.detach().cpu().numpy()
        # save plot of segmentation to disk each 10 slices
        for idx in range(0, segmentation.shape[-1], 10):
            plt.imsave(os.path.join(self.path_test, f"segmentation_{name}_{idx}.png"), segmentation[...,idx])
    
    def dice_score(self):
        pass
    
    def test(self):
        print("ENTRO AL TEST")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        print("~~~~~~~~~Start testing~~~~~~~~\n")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        self.test_epoch(self.test_loader)
        
        
        
                