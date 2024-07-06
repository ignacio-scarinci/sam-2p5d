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
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    
    def train_epoch(self, epoch:int, dataloader: DataLoader):
        """
        Trains the model for one epoch.

        Args:
            epoch (int): The current epoch number.
            dataloader (DataLoader): The data loader for the training data.

        Returns:
            float: The average loss for the epoch.

        """
        self.model.train()
        run_loss = AverageMeter()
        assert self.config.roi_z_iter % 2 == 1
        dataloader.sampler.set_epoch(epoch)  # type: ignore
        for idx, batch_data in enumerate(dataloader):
            start_time = time.time()

            # only take 1 batch
            inputs_l = batch_data["image"]
            labels_l = batch_data["label"]
            # TODO: we only support batch_size = 1 for data loader.
            inputs_l = inputs_l.squeeze()
            labels_l = labels_l.squeeze()
            n_z_before_pad = labels_l.shape[-1]
            
            n_slice = self.config.roi_z_iter
            # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
            pd = (n_slice // 2, n_slice // 2)
            inputs_l = F.pad(inputs_l, pd, "constant", 0)
            labels_l = F.pad(labels_l, pd, "constant", 0)
            _loss = torch.tensor(0.0).to(self.local_rank)
            #for param in self.model.parameters():
            #    param.grad = None
                
            for _k in range(self.config.num_patch):
                # Return random integers from `low` (inclusive) to `high` (exclusive).
                start_idx = int(
                    np.random.randint(
                        low=n_slice // 2, high=(n_slice // 2 + n_z_before_pad)
                    )
                )

                inputs = inputs_l[
                    ..., start_idx - n_slice // 2 : start_idx + n_slice // 2 + 1
                ].permute(2, 0, 1)
                # we only need the label for the center slice
                labels = labels_l[
                    ..., start_idx - n_slice // 2 : start_idx + n_slice // 2 + 1
                ][..., n_slice // 2]

                data, target, _, skip = prepare_sam_training_input(
                    inputs=inputs.to(self.local_rank),
                    labels=labels.to(self.local_rank),
                    config=self.config,
                    model=self.model,
                    sam_image_size=self.sam_image_size,
                    point_prob=self.point_prob,
                    bbox_prob=self.bbox_prob
                )
                
                for param in self.model.parameters():
                    param.grad = None
                
                with autocast( enabled=self.config.use_amp):
                    outputs = self.model(data, is_train=True)
                
                loss = self.dice_loss(outputs[0]["low_res_logits"], target)

                if skip:
                    loss = loss * 0.0
                
                if self.config.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.config.clip > -1.0 :   #is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.config.clip > -1.0: 
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                    self.optimizer.step()

                _loss += loss.detach()
            
            _loss /= min(self.config.num_patch, n_z_before_pad)
            if self.config.distributed:
                loss_list = distributed_all_gather([_loss],out_numpy=True)
                run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),n=self.config.batch_size * dist.get_world_size())
            else:
                run_loss.update(_loss.item(), n=self.config.num_patch)
            if self.global_rank == 0:
                print(
                    "Epoch {}/{} {}/{}".format(epoch, self.config.max_epochs, idx, len(dataloader)),
                    "loss: {:.4f}".format(run_loss.avg),
                    "time {:.2f}s".format(time.time() - start_time),
                )

        for param in self.model.parameters():
            param.grad = None
        return run_loss.avg

    def train_iterative_epoch(self, epoch, dataloader):
        self.model.train()
        run_loss = AverageMeter()
        # we need to make sure the number of 2.5D input is an odd number.
        assert self.config.roi_z_iter % 2 == 1
        dataloader.sampler.set_epoch(epoch)
        for idx, batch_data in enumerate(dataloader):
            start_time = time.time()

            # only take 1 batch
            inputs_l = batch_data["image"]
            labels_l = batch_data["label"]
            # TODO: we only support batch_size = 1 for data loader.
            inputs_l = inputs_l.squeeze()
            labels_l = labels_l.squeeze()
            n_z_before_pad = labels_l.shape[-1]

            n_slice = self.config.roi_z_iter
            # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
            pd = (n_slice // 2, n_slice // 2)
            inputs_l = F.pad(inputs_l, pd, "constant", 0)
            labels_l = F.pad(labels_l, pd, "constant", 0)
            _loss = torch.tensor(0.0).to(self.local_rank)

            for _k in range(min(self.config.num_patch, n_z_before_pad)):
                # Return random integers from `low` (inclusive) to `high` (exclusive).
                start_idx = int(np.random.randint(low=n_slice // 2, high=(n_slice // 2 + n_z_before_pad)))

                inputs = inputs_l[..., start_idx - n_slice // 2 : start_idx + n_slice // 2 + 1].permute(2, 0, 1)

                # we only need the label for the center slice
                labels = labels_l[..., start_idx - n_slice // 2 : start_idx + n_slice // 2 + 1][..., n_slice // 2]

                data, target, target_original, skip = prepare_sam_training_input(
                    inputs=inputs.to(self.local_rank),
                    labels=labels.to(self.local_rank),
                    config=self.config,
                    model=self.model,
                    sam_image_size=self.sam_image_size,
                    point_prob=self.point_prob,
                    bbox_prob=self.bbox_prob
                )

                for param in self.model.parameters():
                    param.grad = None

                with autocast(enabled=self.config.use_amp):
                    if self.config.distributed:
                        image_embeddings = self.model.module.get_image_embeddings(data)
                    else:
                        image_embeddings = self.model.get_image_embeddings(data)

                if skip:
                    with autocast(enabled=self.config.use_amp):
                        if self.config.distributed:
                            outputs = self.model.module.get_mask_prediction(data, image_embeddings)
                        else:
                            outputs = self.model.get_mask_prediction(data, image_embeddings)
                    loss = loss = (
                        self.dice_loss(
                            outputs[0]["low_res_logits"],
                            target
                        )
                        * 0.0
                    )
                else:
                    # iterative training
                    loss = 0
                    drop_iter = random.randint(0, self.config.num_iterative_step - 2)
                    for i in range(self.config.num_iterative_step):
                        with autocast(enabled=self.config.use_amp):
                            if self.config.distributed:
                                outputs = self.model.module.get_mask_prediction(data, image_embeddings)
                            else:
                                outputs = self.model.get_mask_prediction(data, image_embeddings)
                        loss +=  self.dice_loss(outputs[0]["low_res_logits"], target)
                        if i == self.config.num_iterative_step - 1:
                            # no need to perform the following operations after the last step
                            continue
                        # we also supply the mask prediction from the previous iteration
                        # as an additional prompt to our model (follow original SAM).
                        data[0]["mask_inputs"] = outputs[0]["low_res_logits"].detach()
                        if i == drop_iter:
                            # for drop iter, no additional points are sampled (follow original SAM).
                            continue

                        previous_point_coords = data[0].get("point_coords", None)
                        previous_point_labels = data[0].get("point_labels", None)

                        if previous_point_coords is None and self.config.no_more_points_for_cp_only:
                            # if no point prompt at the first prompt generation,
                            # we will not add more additional pointa during iterative training.
                            continue

                        # sample one pos and on neg point based on previous prediction
                        previous_pred = (F.sigmoid(outputs[0]["high_res_logits"].detach()) > 0.5).float()
                        point_coords, point_labels = generate_point_prompt(
                            target_original, 
                            config=self.config, 
                            sam_image_size=self.sam_image_size,
                            points_pos=1, 
                            points_neg=1, 
                            previous_pred=previous_pred
                        )

                        if previous_point_coords is not None:
                            data[0]["point_coords"] = torch.cat([previous_point_coords, point_coords], dim=1)
                            data[0]["point_labels"] = torch.cat([previous_point_labels, point_labels], dim=1)
                        else:
                            data[0]["point_coords"] = point_coords
                            data[0]["point_labels"] = point_labels

                if self.config.use_amp:
                    self.scaler.scale(loss).backward() # type: ignore
                    if self.config.clip > -1 : #is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward() # type: ignore
                    if self.config.clip > -1: #is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                    self.optimizer.step()

                _loss += loss.detach() / self.config.num_iterative_step
            _loss /= min(self.config.num_patch, n_z_before_pad)
            if self.config.distributed:
                loss_list = distributed_all_gather(
                    [_loss],
                    out_numpy=True,
                )
                run_loss.update(
                    np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=self.config.batch_size * dist.get_world_size()
                )
            else:
                run_loss.update(_loss.item(), n=self.config.num_patch)
            if self.global_rank == 0:
                print(
                    "Epoch {}/{} {}/{}".format(epoch, self.config.max_epochs, idx, len(dataloader)),
                    "loss: {:.4f}".format(run_loss.avg),
                    "time {:.2f}s".format(time.time() - start_time),
                )
        for param in self.model.parameters():
            param.grad = None
        return run_loss.avg

    
    def val_epoch(self, epoch: int, dataloader: DataLoader ,iterative: bool = False):
        self.model.eval()
        run_acc = AverageMeter()
        start_time = time.time()
        with torch.no_grad():
            dataloader.sampler.set_epoch(epoch) # type: ignore
            for idx, batch_data in enumerate(dataloader):
                prompt = random.choices(['point', 'bbox'], weights=(1.0, 1.0), k=1)[0]
                print(f"Rank: {self.global_rank}, prompt: {prompt}")
                inputs_l = batch_data["image"]
                labels_l = batch_data["label"]
                labels_l.shape[-1]
                
                inputs_l = inputs_l.squeeze()
                labels_l = labels_l.squeeze()
                
                n_slice = self.config.roi_z_iter
                # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
                pd = (n_slice // 2, n_slice // 2)

                inputs_l = F.pad(inputs_l, pd, "constant", 0)
                labels_l = F.pad(labels_l, pd, "constant", 0)
                n_z_after_pad = labels_l.shape[-1]

                acc_sum_total = 0.0
                not_nans_total = 0.0

            # We only loop the center args.num_patch_val slices to save val time
                for start_idx in range(
                    n_z_after_pad // 2 - self.config.num_patch_val // 2, n_z_after_pad // 2 + self.config.num_patch_val // 2
                ):
                    inputs = inputs_l[..., start_idx - n_slice // 2 : start_idx + n_slice // 2 + 1].permute(2, 0, 1)

                    # we only need the label for the center slice
                    labels = labels_l[..., start_idx - n_slice // 2 : start_idx + n_slice // 2 + 1][..., n_slice // 2]
                    if  prompt == 'point':
                        data, target, _ = prepare_sam_val_input_pp_only(inputs.to(self.local_rank), labels.to(self.local_rank), self.config, self.sam_image_size)
                    elif prompt == 'bbox':
                        data, target, _ = prepare_sam_val_input_bb_only(inputs.to(self.local_rank), labels.to(self.local_rank))
                    
                    with autocast(enabled=self.config.use_amp):
                        outputs = self.model(data)
                        logit = outputs[0]["high_res_logits"]
                    
                    #y_pred = self.post_pred(logit)
                    y_pred = torch.stack(self.post_pred(decollate_batch(logit)), 0) 

                    acc_batch = compute_dice(y_pred=y_pred, y=target)
                    acc_sum, not_nans = (
                        torch.nansum(acc_batch).item(),
                        114 - torch.sum(torch.isnan(acc_batch).float()).item(),
                    )
                    acc_sum_total += acc_sum
                    not_nans_total += not_nans
                    
                acc, not_nans = acc_sum_total / not_nans_total, not_nans_total
                f_name = batch_data["image"].meta["filename_or_obj"]
                print(f"Rank: {self.global_rank}, Case: {f_name}, Acc: {acc:.4f}, N_prompts: {int(not_nans)} ")

                acc = torch.tensor(acc).cuda(self.local_rank)
                not_nans = torch.tensor(not_nans).cuda(self.local_rank)
                if self.config.distributed:
                    acc_list, not_nans_list = distributed_all_gather([acc, not_nans], out_numpy=True)
                    for al, nl in zip(acc_list, not_nans_list):
                        run_acc.update(al, n=nl)

                else:
                    run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
                    
                if self.global_rank == 0:
                    avg_acc = np.mean(run_acc.avg)
                    print(
                        "Val {}/{} {}/{}".format(epoch, self.config.max_epochs, idx + 1, len(dataloader)),
                        "acc",
                        avg_acc,
                        "time {:.2f}s".format(time.time() - start_time),
                    )
                start_time = time.time()
        return run_acc.avg

    def _save_snapshot(self, epoch):
        # capture snapshot
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(), # type: ignore
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch,
            scheduler_state=self.scheduler.state_dict() if self.scheduler is not None else None
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
        val_acc_max = 0.0
        best_epoch = -1
        val_MA = None
        best_log = {}
        best_loss = 2.0
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
                    

            
            if epoch > self.config.iterative_training_warm_up_epoch:
                if self.global_rank == 0:
                    print("Iterative training\n")
                train_loss = self.train_iterative_epoch(epoch, self.train_loader)
            else:
                print(f"GPU {self.global_rank}  Single-Step Training\n")
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
                    print(f"label_prompt: {self.config.label_prompt}, point_prompt: {self.config.point_prompt}")
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
                        
            if self.local_rank == 0 and (epoch % self.save_every == 0 or epoch == self.config.max_epochs - 1 or train_loss < best_loss):
                    self._save_snapshot(epoch)
                    best_loss = train_loss
            
            if self.local_rank == 0 and (train_loss < best_loss):
                self.save_checkpoint(
                    self.model,
                    epoch,
                    self.config,
                    filename="model_best_loss.pt",
                    best_acc=val_avg_acc,
                    loss=train_loss,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )

            if self.scheduler is not None:
                self.scheduler.step()
        if self.global_rank == 0 and self.writer is not None:
            self.writer.close()
        
        if self.global_rank == 0:
            print("Training Finished !, Best Accuracy: ", val_acc_max, "at epoch", best_epoch
        )
        
        
