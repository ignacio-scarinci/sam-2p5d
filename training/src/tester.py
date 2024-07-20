from dataclasses import asdict
import json
import os
import random
import time
from typing import Dict, Sequence

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

from monai.metrics import DiceMetric, compute_dice, compute_hausdorff_distance
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch, ThreadDataLoader

from tqdm import tqdm

from .config import  TestConfig
from .utils import (
    prepare_sam_val_input_pp_only,
    prepare_sam_val_input_bb_only,
)

import matplotlib.pyplot as plt

class Tester:
    def __init__(
        self,
        device,
        tester_cfg: TestConfig,
        model: nn.Module,
        sam_image_size,
        test_dataset: Dataset,
        labels: Sequence
    ):
        self.device = device
        self.config = tester_cfg
        self.model = model
        self.sam_image_size = sam_image_size
        self.test_dataset = test_dataset
        
        self.test_loader = self._prepare_dataloader(self.test_dataset)

        self._weights()
        self.model = model.to(device)
        #
        if self.config.use_amp:
            self.scaler = GradScaler()
            
        
        self.post_label = AsDiscrete(to_onehot=115)
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.dice_acc = DiceMetric(
            include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True
        )

        self.log_acc = open(self.config.logdir + "/resultados.json", 'w')
        
        self.labels = labels
            

    def _prepare_dataloader(self, dataset: Dataset):
        return ThreadDataLoader(
            dataset=dataset,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.data_loader_workers,
        )

    def _weights(self):

        model_state = torch.load(self.config.weight, map_location="cpu")
        self.model.load_state_dict(model_state["state_dict"])
        
    def test(self):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        print("~~~~~~~~  START TEST  ~~~~~~~~\n")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        
        self.model.eval()
        self.dice_acc.reset()
        with torch.no_grad():
            for batch in self.test_loader:
                inputs_l = batch["image"]
                labels_l = batch['label']
                files = inputs_l.meta["filename_or_obj"][0]
                print(f'File: {inputs_l.meta["filename_or_obj"]}')
                labels_l.shape[-1]
                inputs_l = inputs_l.squeeze()
                labels_l = labels_l.squeeze()
                
                # TODO: revisar que sea correcta la permutacion
                if self.config.axis == "sagital":
                    inputs_l = inputs_l.permute(2, 0, 1) 
                    labels_l = labels_l.permute(2, 0, 1)
                elif self.config.axis == "coronal":
                    inputs_l = inputs_l.permute(2, 1, 0)
                    labels_l = labels_l.permute(2, 1, 0)
                
                # Creo un tensor de zeros para la segmentacion
                segmentation = torch.zeros_like(labels_l)
                
                n_slices = self.config.roi_z_iter
                # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
                pd = (n_slices // 2, n_slices // 2)
                
                inputs_l = F.pad(inputs_l, pd, "constant", 0)
                labels_l = F.pad(labels_l, pd, "constant", 0)
                n_z_after_pad = labels_l.shape[-1]
                
                for idx in tqdm(range(n_slices // 2, n_z_after_pad - n_slices // 2), desc='Slice'):
                    
                    inputs = inputs_l[..., idx - n_slices // 2 : idx + n_slices // 2 + 1].permute(2, 0, 1)
                    labels = labels_l[..., idx - n_slices // 2 : idx + n_slices // 2 + 1][..., n_slices // 2]
                    
                    if self.config.point_prompt:
                        data, target, _ = prepare_sam_val_input_pp_only(inputs.to(self.device), labels.to(self.device), self.config, self.sam_image_size, self.config.point_pos, self.config.point_neg)
                    elif self.config.bbox_prompt:
                        data, target, _ = prepare_sam_val_input_bb_only(inputs.to(self.device), labels.to(self.device))

                    with torch.no_grad():
                        with autocast(enabled=self.config.use_amp):
                            outputs = self.model(data)
                            logit = outputs[0]['high_res_logits']
        
                    
                    segmentation[..., idx] = torch.argmax(self.post_pred(logit))

                print(f'Segmentation shape {segmentation.shape}')
                
                acc_per_label = compute_dice(logit, labels, include_background=False)
                acc_sum, not_nans = (
                        torch.nansum(acc_per_label).item(),
                        114 - torch.sum(torch.isnan(acc_per_label).float()).item(),
                    )
                
                acc_mean = acc_sum / not_nans
                hausdorsff = compute_hausdorff_distance(logit, labels, include_background=False, spacing=1.5)
                hausdorsff = hausdorsff.tolist()
                res = {"Image":files,
                       "Dice_mean":acc_mean,
                       "dice_per_label":dict(zip(self.labels, acc_per_label)),
                       "Hausdorff_distance":dict(zip(self.labels, hausdorsff))}
                
                self.log_acc.seek(0, 2)
                json.dump(res, self.log_acc)
        self.log_acc.close()
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        print("~~~~~~~    END TEST    ~~~~~~~\n")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    #    def save_predictions(self, segmentation):
#        
#        name = self.files.split("/")[-1].split(".")[0]
#        segmentation = segmentation.detach().cpu().numpy()
#        # save plot of segmentation to disk each 10 slices
#        for idx in range(0, segmentation.shape[-1], 10):
#            plt.imsave(os.path.join(self.path_test, f"segmentation_{name}_{idx}.png"), segmentation[...,idx])
    
    def dice_score(self):
        pass
    



        
        
        
                