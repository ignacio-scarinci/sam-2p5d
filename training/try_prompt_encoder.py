import os

import numpy as np

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.utils import prepare_sam_training_input
from src.model import sam_model_registry
from src.data import split_data, get_dataset
from src.config import DataConfig, TrainerConfig
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
    ):
        
        self.config = TrainerConfig(
          experiment_name= 'try_prompt',
          seed= 5344,
          max_epochs= 50,
          batch_size= 1,
          logdir= "../runs/",
          snapshot_path= "../runs",
          use_amp= True,
          tf32= True,
          save_every= 5,
          data_loader_workers= 8,
          label_prompt= True,
          point_prompt= True,
          bbox_prompt= True,
          label_prompt_warm_up_epoch= 15,
          iterative_training_warm_up_epoch= 30,
          num_patch= 24,
          num_prompt= 8,
          clip= -1,
          distributed= False,
          skip_bk= True,
          max_points= 8,
          roi_z_iter= 9,
          num_iterative_step= 5,
          no_more_points_for_cp_only= False,
          num_patch_val= 30,
          val_every= 2,
          save_checkpoint= True
        )
        
        data_config = DataConfig(data_dir = "/home/iscarinci/total_liifa",
            json_list = "/home/iscarinci/total_liifa/dataset.json",
            splitval = 0.1,
            a_min = -1024,
            a_max = 1024,
            b_min = 0,
            b_max = 1,)
        # data stuff
        train_files, val_files, _ = split_data(data_config) # type: ignore
        train_ds, _ = get_dataset(train_files=train_files, val_files=val_files, data_cfg=data_config) # type: ignore

        self.loader = self._prepare_dataloader(train_ds)

        self.model = sam_model_registry['tiny_vit'](
        checkpoint=None,
        image_size=1024,
        encoder_in_chans=27,
        mod='sam',
    )

    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            num_workers=8,
        )

    def train_epoch(self, epoch: int, dataloader: DataLoader):
        self.model.train()
        prompt_elegido = np.zeros((8,))
        tqdm_load = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}")
        for idx, batch_data in enumerate(tqdm_load):
            # only take 1 batch
            inputs_l = batch_data["image"]
            labels_l = batch_data["label"]
            # TODO: we only support batch_size = 1 for data loader.
            inputs_l = inputs_l.squeeze()
            labels_l = labels_l.squeeze()
            n_z_before_pad = labels_l.shape[-1]

            n_slice = 9
            # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
            pd = (n_slice // 2, n_slice // 2)
            inputs_l = F.pad(inputs_l, pd, "constant", 0)
            labels_l = F.pad(labels_l, pd, "constant", 0)

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
                    inputs=inputs.to('cuda'),
                    labels=labels.to('cuda'),
                    config=self.config,
                    model=self.model,
                    sam_image_size=1024,
                    label_prob=self.label_prob,
                    point_prob=self.point_prob,
                    bbox_prob=self.bbox_prob,
                )

                keys = data[0].keys()
                if 'labels' in keys and 'point_labels' in keys and 'boxes' in keys:
                    prompt_elegido[0] += 1
                elif 'points_coords' in keys and 'labels' in keys:
                    prompt_elegido[1] += 1
                elif 'point_coords' in keys and 'boxes' in keys:
                    prompt_elegido[2] += 1
                elif 'boxes' in keys and 'labels' in keys:
                    prompt_elegido[3] += 1
                elif 'labels' in keys:
                    prompt_elegido[4] += 1
                elif 'boxes' in keys:
                    prompt_elegido[5] += 1
                elif 'points_coords' in keys:
                    prompt_elegido[6] += 1
                else:
                    prompt_elegido[7] += 1
        return prompt_elegido
    
    def train(self):
        prompt_elegido = np.zeros((8, 20))
        for epoch in range(0, 20):
            print(f'Epoch {epoch}')
            if self.config.label_prompt and self.config.point_prompt:
                if epoch < 10:
                    self.label_prob = 0.8
                    self.point_prob = 0.5
                    self.bbox_prob = 0.5
                else:
                    self.label_prob = 0.5
                    self.point_prob = 0.5
                    self.bbox_prob = 0.5
            
            prompt_elegido[:, epoch] = self.train_epoch(epoch, self.loader)
        np.save('prompt_elegido.npy', prompt_elegido)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()