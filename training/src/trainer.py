from dataclasses import asdict
import json
import os
import random
import time
from typing import Sequence

import numpy as np
import torch
import fsspec

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

from .focal_loss import FocalDiceloss_IoULoss
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, compute_dice
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch, ThreadDataLoader
from tensorboardX import SummaryWriter

from .config import TrainerConfig, Snapshot
from .utils import (
    AverageMeter,
    distributed_all_gather,
    prepare_sam_training_input,
    generate_point_prompt,
    prepare_sam_val_input_pp_only,
    prepare_sam_val_input_bb_only,
    prepare_sam_val_input_np_only
)


class Trainer:
    def __init__(self, trainer_cfg: TrainerConfig, model, mod, sam_image_size, optimizer, scheduler, train_dataset, val_dataset):

        self.config = trainer_cfg
        self.experiment_name = self.config.experiment_name
        self.mod = mod
        self.sam_image_size = sam_image_size
        

        self.point_prob = 0.5
        self.bbox_prob = 0.5

        # set torchrun variables
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])  
        
        #data stuff
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.val_loader = self._prepare_dataloader(val_dataset)
        
        #initialize train states
        self.epochs_run = 0
        self.model = model.to(self.local_rank)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = self.config.save_every
        
        # 
        if self.config.use_amp:
            self.scaler = GradScaler()
        
        # load snapshot if available. only necessary on the first node.
        if self.config.snapshot_path is None:
            self.config.snapshot_path = os.path.join(self.config.logdir, self.experiment_name,"snapshot.pt")
        self.best_acc = 0.0
        self._load_snapshot()
        
        # initialize DDP
        self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
        
        self.dice_loss = DiceCELoss(sigmoid=True)
        #self.dice_loss = FocalDiceloss_IoULoss()
        self.post_label = AsDiscrete(to_onehot=115)
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        self.dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
        
        self.writer = None
        if self.config.logdir is not None and self.global_rank == 0:
            self.writer = SummaryWriter(self.config.logdir)
            if self.global_rank == 0:
                print("Writing Tensorboard logs to ", self.config.logdir)

    def _prepare_dataloader(self, dataset: Dataset):
        return ThreadDataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.data_loader_workers,
            sampler=DistributedSampler(dataset),
        )
    
    def _load_snapshot(self):
        try:
            snapshot = fsspec.open(self.config.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location="cpu") # type: ignore
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch")
            return
        snapshot = Snapshot(**snapshot_data)
        self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        self.best_acc = snapshot.best_acc
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    
    def train_epoch(self, epoch: int, dataloader: DataLoader) -> float:
        """
        Trains the model for one epoch.

        Args:
            epoch (int): The current epoch number.
            dataloader (DataLoader): The data loader for the training data.

        Returns:
            float: The average loss for the epoch.
        """
        self.model.train()
        epoch_loss = AverageMeter()
        assert self.config.roi_z_iter % 2 == 1
        dataloader.sampler.set_epoch(epoch)  # type: ignore

        for batch_idx, batch in enumerate(dataloader):
            start_time = time.time()

            # Process batch
            images = batch["image"].squeeze()
            labels = batch["label"].squeeze()
            z_dim = labels.shape[-1]

            slice_count = self.config.roi_z_iter
            padding = (slice_count // 2, slice_count // 2)
            images = F.pad(images, padding, "constant", 0)
            labels = F.pad(labels, padding, "constant", 0)

            batch_loss = torch.tensor(0.0, device=self.local_rank)

            for _ in range(self.config.num_patch):
                start_idx = np.random.randint(low=slice_count // 2, high=(slice_count // 2 + z_dim))

                input_slices = images[
                    ..., start_idx - slice_count // 2: start_idx + slice_count // 2 + 1
                ].permute(2, 0, 1)
                label_slices = labels[
                    ..., start_idx - slice_count // 2: start_idx + slice_count // 2 + 1
                ][..., slice_count // 2]

                data, target, _, skip = prepare_sam_training_input(
                    inputs=input_slices.to(self.local_rank),
                    labels=label_slices.to(self.local_rank),
                    config=self.config,
                    model=self.model,
                    sam_image_size=self.sam_image_size,
                    point_prob=self.point_prob,
                    bbox_prob=self.bbox_prob,
                )

                # Zero gradients
                self.optimizer.zero_grad()

                with autocast(enabled=self.config.use_amp):
                    outputs = self.model(data, is_train=True)

                loss = self.dice_loss(outputs[0]["low_res_logits"], target)

                if skip:
                    loss *= 0.0

                if self.config.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.config.clip > -1.0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.config.clip > -1.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                    self.optimizer.step()

                batch_loss += loss.detach()

            batch_loss /= min(self.config.num_patch, z_dim)

            if self.config.distributed:
                loss_list = distributed_all_gather([batch_loss], out_numpy=True)
                epoch_loss.update(np.mean(np.stack(loss_list, axis=0)), n=self.config.batch_size * dist.get_world_size())
            else:
                epoch_loss.update(batch_loss.item(), n=self.config.num_patch)

            if self.global_rank == 0:
                print(f"Epoch {epoch}/{self.config.max_epochs} {batch_idx}/{len(dataloader)}",
                      f"loss: {epoch_loss.avg:.4f}",
                      f"time: {time.time() - start_time:.2f}s")

        return epoch_loss.avg
    
    def val_epoch(self, epoch: int, dataloader: DataLoader, iterative: bool = False):
        self.model.eval()
        accuracy_meter = AverageMeter()
        val_start_time = time.time()
        
        with torch.no_grad():
            dataloader.sampler.set_epoch(epoch)  # type: ignore
            for batch_index, batch_data in enumerate(dataloader):
                prompt_type = random.choice(['point', 'bbox'])
                print(f"Rank: {self.global_rank}, Prompt: {prompt_type}")
                
                images = batch_data["image"].squeeze()
                labels = batch_data["label"].squeeze()
                
                slice_count = self.config.roi_z_iter
                pad_size = (slice_count // 2, slice_count // 2)

                images = F.pad(images, pad_size, "constant", 0)
                labels = F.pad(labels, pad_size, "constant", 0)
                num_slices_after_pad = labels.shape[-1]

                total_accuracy_sum = 0.0
                total_not_nans = 0.0

                for start_index in range(
                    num_slices_after_pad // 2 - self.config.num_patch_val // 2, 
                    num_slices_after_pad // 2 + self.config.num_patch_val // 2
                ):
                    input_slices = images[
                        ..., start_index - slice_count // 2 : start_index + slice_count // 2 + 1
                    ].permute(2, 0, 1)

                    label_slice = labels[
                        ..., start_index - slice_count // 2 : start_index + slice_count // 2 + 1
                    ][..., slice_count // 2]

                    if prompt_type == 'point':
                        data, target, _ = prepare_sam_val_input_pp_only(
                            input_slices.to(self.local_rank),
                            label_slice.to(self.local_rank),
                            self.config,
                            self.sam_image_size
                        )
                    elif prompt_type == 'bbox':
                        data, target, _ = prepare_sam_val_input_bb_only(
                            inputs=input_slices.to(self.local_rank),
                            sam_image_size=self.sam_image_size,
                            labels=label_slice.to(self.local_rank)
                        )

                    with autocast(enabled=self.config.use_amp):
                        outputs = self.model(data)
                        logits = outputs[0]["high_res_logits"]

                    predictions = torch.stack(self.post_pred(decollate_batch(logits)), 0)

                    batch_accuracy = compute_dice(y_pred=predictions, y=target)
                    accuracy_sum, not_nans = (
                        torch.nansum(batch_accuracy).item(),
                        114 - torch.sum(torch.isnan(batch_accuracy).float()).item(),
                    )
                    total_accuracy_sum += accuracy_sum
                    total_not_nans += not_nans

                avg_accuracy = total_accuracy_sum / total_not_nans
                file_name = batch_data["image"].meta["filename_or_obj"]
                print(f"Rank: {self.global_rank}, Case: {file_name}, Accuracy: {avg_accuracy:.4f}, Prompts: {int(total_not_nans)}")

                avg_accuracy_tensor = torch.tensor(avg_accuracy).cuda(self.local_rank)
                not_nans_tensor = torch.tensor(total_not_nans).cuda(self.local_rank)

                if self.config.distributed:
                    accuracy_list, not_nans_list = distributed_all_gather([avg_accuracy_tensor, not_nans_tensor], out_numpy=True)
                    for acc_value, not_nan_value in zip(accuracy_list, not_nans_list):
                        accuracy_meter.update(acc_value, n=not_nan_value)
                else:
                    accuracy_meter.update(avg_accuracy_tensor.cpu().numpy(), n=not_nans_tensor.cpu().numpy())

                if self.global_rank == 0:
                    avg_accuracy = np.mean(accuracy_meter.avg)
                    print(
                        f"Validation {epoch}/{self.config.max_epochs} {batch_index + 1}/{len(dataloader)}",
                        "Accuracy:",
                        avg_accuracy,
                        f"Time: {time.time() - val_start_time:.2f}s"
                    )
                val_start_time = time.time()

        return accuracy_meter.avg

    def _save_snapshot(self, epoch, best_acc):
        # capture snapshot
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(), # type: ignore
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch,
            scheduler_state=self.scheduler.state_dict() if self.scheduler is not None else None,
            best_acc=best_acc
        )
        # save snapshot
        snapshot = asdict(snapshot)
        torch.save(snapshot, self.config.snapshot_path)

        print(f"Snapshot saved at epoch {epoch}")

    def save_checkpoint(self,
        model, epoch, args, filename="model.pt", best_acc:float=0.0, loss:float=0.0, optimizer=None, scheduler=None
    ):
        state_dict = (
            model.state_dict() if not self.config.distributed else self.model.module.state_dict()
        )
        save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
        if optimizer is not None:
            save_dict["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            save_dict["scheduler"] = scheduler.state_dict()
        filename = os.path.join(args.logdir, filename)
        torch.save(save_dict, filename)
        print("Saving checkpoint", filename)

    def train(self):
        """
        Trains the model for a specified number of epochs.

        This function iterates over a specified number of epochs and performs the following steps for each epoch:
        1. Synchronizes the GPU if distributed training is enabled.
        2. Prints the current epoch and local rank if global rank is 0.
        3. Prints the current learning rate if a scheduler is provided.
        4. Updates the label, point, and bounding box probabilities based on the epoch and configuration.
        5. Trains the model using the train_epoch method.
        6. Prints the training loss and time taken for the epoch if global rank is 0.
        7. Adds the training loss to the tensorboard writer if available.
        8. Validates the model every 'val_every' epochs and calculates the average accuracy.
        9. Prints the validation accuracy, moving average accuracy, best validation accuracy, and time taken for the validation.
        10. Updates the best validation accuracy, best epoch, and saves the model if the validation accuracy is higher.
        11. Saves the model every 'save_every' epochs or at the end of training if the training loss is lower than the previous best loss.
        12. Saves the model if the training loss is lower than the previous best loss.
        13. Steps the scheduler if available.
        14. Closes the tensorboard writer if global rank is 0.
        15. Prints the final training results, including the best validation accuracy and best epoch.

        Parameters:
        None    

        Returns:
        None
        """
        val_acc_max = self.best_acc
        best_epoch = -1
        val_MA = None
        best_log = {}
        best_loss = 1000.0
        for epoch in range(self.epochs_run, self.config.max_epochs):
            torch.cuda.synchronize()
            if self.config.distributed:
                dist.barrier()
            
            print(self.local_rank, time.ctime(), "Epoch:", epoch, "\n")
            epoch_time = time.time()
            if self.global_rank == 0:
                if self.scheduler is not None:
                    print(f"Current LR: {self.scheduler.get_last_lr()}")
                    self.writer.add_scalar("lr", self.scheduler.get_last_lr(), epoch)
                else:
                    print(f"Current LR: {self.optimizer.param_groups[0]['lr']}")
                    
            train_loss = self.train_epoch(epoch, self.train_loader)
            if self.global_rank == 0:
                print(
                    "Final training  {}/{}".format(epoch, self.config.max_epochs - 1),
                    "loss: {:.4f}".format(train_loss),
                    "time {:.2f}s".format(time.time() - epoch_time))

            if self.global_rank == 0:
                self.writer.add_scalar("train_loss", train_loss, epoch) # type: ignore
            
            if (epoch+1) % self.config.val_every == 0:
                if self.config.distributed:
                    dist.barrier()
                if self.global_rank == 0:
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
                    print("~~~~~~~Start validation~~~~~~~\n")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
                    print(f"point_prompt: {self.config.point_prompt}")
                epoch_time = time.time()
                val_avg_acc = self.val_epoch(iterative=False, 
                                             dataloader=self.val_loader,
                                             epoch=epoch)

                val_avg_acc = np.mean(val_avg_acc)
                if val_MA is None:
                    val_MA = val_avg_acc
                else:
                    val_MA = 0.9 * val_MA + 0.1 * val_avg_acc
                
                if self.global_rank == 0:
                    print(
                        "Final validation  {}/{},".format(epoch, self.config.max_epochs - 1),
                        f"Acc {val_avg_acc:.4f},",
                        f"mv Acc {val_MA:.4f},",
                        "Previous Best validation at epoch {} is {:.4f},".format(
                            best_epoch, val_acc_max
                        ),
                        "time {:.2f}s".format(time.time() - epoch_time),
                    )
                    if self.writer is not None:
                        self.writer.add_scalar("val_acc", val_avg_acc, epoch)
                    if val_avg_acc > val_acc_max:
                        print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                        val_acc_max = val_avg_acc
                        best_log[epoch] = float(val_acc_max)
                        best_epoch = epoch
                        if self.global_rank == 0 and self.config.logdir and self.config.save_checkpoint:
                            self.save_checkpoint(
                                self.model,
                                epoch,
                                self.config,
                                filename="model_best.pt",
                                best_acc=val_acc_max,
                                loss=train_loss,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                            )
                    with open(os.path.join(self.config.logdir, "best_log.json"), "w") as f:
                        json.dump(best_log, f)
                        
            if self.global_rank == 0 and (epoch % self.save_every == 0 or epoch == self.config.max_epochs - 1):
                    self._save_snapshot(epoch, val_acc_max)
            
            if self.global_rank == 0 and (train_loss < best_loss):
                self.save_checkpoint(
                    self.model,
                    epoch,
                    self.config,
                    filename="model_best_loss.pt",
                    best_acc=val_acc_max,
                    loss=train_loss,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                best_loss = train_loss

            if self.scheduler is not None:
                self.scheduler.step()
        if self.global_rank == 0 and self.writer is not None:
            self.writer.close()
        
        if self.global_rank == 0:
            print("Training Finished !, Best Accuracy: ", val_acc_max, "at epoch", best_epoch
        )
        
        
