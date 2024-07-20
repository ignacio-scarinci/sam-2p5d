import os
from typing import Optional

import hydra
import torch
from torch.distributed import init_process_group, destroy_process_group
from omegaconf import DictConfig
from dataclasses import dataclass

from src.tester import Tester
from src.data import test_files, get_dataset
from src.model import sam_model_registry
from src.config import TestConfig, SamConfig, DataConfig
import numpy as np

    
def get_test_objs(
    model_cfg: SamConfig, data_cfg: TestConfig
):
    test_fs, labels = test_files(data_cfg)
    test_ds = get_dataset(
        test_files=test_fs, data_cfg=data_cfg, test=True
    )

    model = sam_model_registry[model_cfg.sam_base_model](
        checkpoint=None,
        image_size=model_cfg.sam_image_size,
        encoder_in_chans=model_cfg.encoder_in_channels,
        mod=model_cfg.mod,
    )


    return model, test_ds, labels


@hydra.main(version_base=None, config_path="./config", config_name="config_test")
def main(cfg: DictConfig):
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    print("INICIO")
    test_cfg = TestConfig(**cfg["test"])

    torch.backends.cudnn.benchmark = True
    if test_cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_cfg = SamConfig(**cfg["model"])

    model, test_data, labels= get_test_objs(
        model_cfg, test_cfg
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tester = Tester(
        device=device,
        tester_cfg=test_cfg,
        model=model,
        sam_image_size=model_cfg.sam_image_size,
        test_dataset=test_data, # type: ignore
        labels=labels
    )
    tester.test()

if __name__ == "__main__":
    main()