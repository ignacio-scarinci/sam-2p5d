from dataclasses import dataclass
from typing import Optional, Any, Dict
from collections import OrderedDict
import torch

@dataclass
class SamConfig:
    checkpoint: str
    sam_base_model: str
    mod: str 
    mid_dim: Optional[int]
    sam_image_size: int
    encoder_in_channels: int


@dataclass
class OptimizerConfig:
    optim_name: str
    optim_lr: float
    momentum: float
    reg_weight: float
    scheduler: str | None
    t0: int
    tmult: int
    e_min: float
    
    


@dataclass
class TrainerConfig:
    experiment_name: str
    seed: Optional[int]
    max_epochs: int 
    batch_size: int 
    logdir: str 
    snapshot_path: Optional[str]
    use_amp: bool
    tf32: bool 
    save_every: int
    data_loader_workers: int
    label_prompt: bool
    point_prompt: bool
    bbox_prompt: bool
    point_pos: int | None
    point_neg: int | None
    label_prompt_warm_up_epoch: int
    iterative_training_warm_up_epoch: int
    num_patch: int
    clip: float
    distributed: bool
    skip_bk: bool
    num_prompt: int
    max_points: int
    roi_z_iter: int
    num_iterative_step: int
    no_more_points_for_cp_only: bool
    num_patch_val: int
    val_every: int
    save_checkpoint: bool


@dataclass
class DataConfig:
    data_dir: str
    json_list: str
    splitval: float
    a_min: float
    a_max: float
    b_min: float
    b_max: float
    
@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    scheduler_state: Optional[Dict[str, Any]]
    finished_epoch: int
