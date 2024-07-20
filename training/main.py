import os
from typing import Optional

import hydra
import torch
from torch.distributed import init_process_group, destroy_process_group
from omegaconf import DictConfig
from dataclasses import dataclass

from src.trainer import Trainer
from src.data import split_data, get_dataset
from src.model import sam_model_registry
from src.config import TrainerConfig, OptimizerConfig, SamConfig, DataConfig
import numpy as np

from monai.utils import set_determinism


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def optimizer_setup(model, opt_config: OptimizerConfig, mod):

    if mod == 'sam_adpt':
        for n, value in model.image_encoder.named_parameters(): 
            if ("Adapter" in n) or ("patch_embed" in n) or ("pos_embed" in n):
                value.requires_grad = True
            else:
                value.requires_grad = False
    elif mod == "sam":
        for n, value in model.image_encoder.named_parameters(): 
            value.requires_grad = True

    params = filter(lambda p: p.requires_grad, model.parameters())
    if (opt_config.optim_name).lower() == "adam":
        optimizer =  torch.optim.Adam(params, lr=opt_config.optim_lr, weight_decay=opt_config.reg_weight)
    elif (opt_config.optim_name).lower() == "adamw":
        optimizer = torch.optim.AdamW(
            params, lr=opt_config.optim_lr, weight_decay=opt_config.reg_weight
        )
    elif (opt_config.optim_name).lower() == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=opt_config.optim_lr,
            momentum=opt_config.momentum,
            nesterov=True,
            weight_decay=opt_config.reg_weight,
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(opt_config.optim_name))

    if opt_config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, opt_config.t0, T_mult=opt_config.tmult, eta_min=opt_config.e_min,
        )
    else:
        scheduler = None
    
    return optimizer, scheduler
    

def get_train_objs(model_cfg: SamConfig, opt_cfg: OptimizerConfig, data_cfg: DataConfig):
    train_files, val_files = split_data(data_cfg)
    train_ds, val_ds = get_dataset(train_files=train_files, val_files=val_files, data_cfg=data_cfg)

    model = sam_model_registry[model_cfg.sam_base_model](
        checkpoint=model_cfg.checkpoint,
        image_size=model_cfg.sam_image_size,
        encoder_in_chans=model_cfg.encoder_in_channels,
        mod=model_cfg.mod,
    )
    
    optimizer, scheduler = optimizer_setup(model, opt_cfg, model_cfg.mod)

    if int(os.environ["RANK"]) == 0:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Total parameters count", pytorch_total_params * 1.0e-6, "M\n")
        pytorch_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print("Total trainable parameters count", pytorch_total_params * 1.0e-6, "M\n")
        pytorch_total_params = sum(p.numel() for p in model.image_encoder.parameters())
        print(
            "Total image encoder parameters count", pytorch_total_params * 1.0e-6, "M\n"
        )
        pytorch_total_params = sum(
            p.numel() for p in model.image_encoder.parameters() if p.requires_grad
        )
        print(
            "Total trainable image encoder parameters count",
            pytorch_total_params * 1.0e-6,
            "M\n",
        )
        pytorch_total_params = sum(p.numel() for p in model.mask_decoder.parameters())
        print(
            "Total mask decoder parameters count", pytorch_total_params * 1.0e-6, "M\n"
        )
        pytorch_total_params = sum(
            p.numel() for p in model.mask_decoder.parameters() if p.requires_grad
        )
        print(
            "Total trainable mask decoder parameters count",
            pytorch_total_params * 1.0e-6,
            "M\n",
        )
        pytorch_total_params = sum(p.numel() for p in model.prompt_encoder.parameters())
        print(
            "Total prompt encoder parameters count",
            pytorch_total_params * 1.0e-6,
            "M\n",
        )
        pytorch_total_params = sum(
            p.numel() for p in model.prompt_encoder.parameters() if p.requires_grad
        )
        print(
            "Total trainable prompt encoder parameters count",
            pytorch_total_params * 1.0e-6,
            "M\n",
        )

    return model, optimizer, scheduler, train_ds, val_ds


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig):
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)    
    
    trainer_cfg = TrainerConfig(**cfg["trainer"])
    if trainer_cfg.seed is not None:
        set_determinism(seed=trainer_cfg.seed)
    
    torch.backends.cudnn.benchmark = True
    if trainer_cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_cfg = SamConfig(**cfg['model'])
    opt_cfg = OptimizerConfig(**cfg['optimizer'])
    data_cfg = DataConfig(**cfg['data'])
    
    
    model, optimizer, scheduler, train_data, val_data = get_train_objs(model_cfg, opt_cfg, data_cfg)
    ddp_setup()
    trainer = Trainer(trainer_cfg=trainer_cfg, 
                      model=model, 
                      mod=model_cfg.mod ,
                      sam_image_size=model_cfg.sam_image_size,
                      optimizer=optimizer, 
                      train_dataset=train_data, 
                      val_dataset=val_data,
                      scheduler=scheduler)
    trainer.train()
    
    destroy_process_group()
    
if __name__ == "__main__":
    main()