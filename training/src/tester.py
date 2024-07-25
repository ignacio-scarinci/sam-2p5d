from dataclasses import asdict
import json
import os
from typing import Dict, List, Sequence, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.cuda.amp import GradScaler, autocast

from monai.metrics import compute_dice, compute_hausdorff_distance
from monai.transforms import Activations, AsDiscrete, Compose
from monai.data import ThreadDataLoader

from tqdm import tqdm

from .config import TestConfig
from .utils import prepare_sam_val_input_pp_only, prepare_sam_val_input_bb_only


import matplotlib.pyplot as plt


class Tester:
    def __init__(
        self,
        device,
        tester_cfg: TestConfig,
        model: nn.Module,
        sam_image_size,
        test_dataset: Dataset,
        labels: Sequence,
    ):
        """
        A function that initializes the Tester class with provided parameters.

        Parameters:
            device: The device to use for testing.
            tester_cfg (TestConfig): The configuration for testing.
            model (nn.Module): The model to be tested.
            sam_image_size: The size of the image for the Spatial Attention Module.
            test_dataset (Dataset): The dataset for testing.
            labels (Sequence): The labels for testing, excluding the background label.
        """
        self.device = device
        self.config = tester_cfg
        self.model = model
        self.sam_image_size = sam_image_size
        self.test_dataset = test_dataset
        self.labels = labels[1:]  # quito background

        self.test_loader = self._prepare_dataloader(self.test_dataset)

        self._weights()
        self.model = model.to(device)
        #
        if self.config.use_amp:
            self.scaler = GradScaler()

        self.post_label = AsDiscrete(to_onehot=115)
        self.post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    def _prepare_dataloader(self, dataset):
        """
        A function that prepares a dataloader for the given dataset.

        Parameters:
            dataset (Dataset): The dataset to be loaded into the dataloader.

        Returns:
            ThreadDataLoader: A dataloader with the specified batch size, memory pinning, shuffling, and number of workers.
        """
        return ThreadDataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.data_loader_workers,
        )

    def _weights(self):
        """
        Load the weights of the model from a checkpoint file specified in the configuration.

        This function loads the weights of the model from the checkpoint file specified in the
        `weight` attribute of the `config` object. It uses the `torch.load()` function to load
        the checkpoint file and then uses the `load_state_dict()` method of the `model` object
        to load the weights into the model. The `map_location` parameter is set to "cpu" to
        ensure that the weights are loaded onto the CPU.

        """

        model_state = torch.load(self.config.weight, map_location="cpu")
        self.model.load_state_dict(model_state["state_dict"])

    def test(self) -> None:
        """
        Function to perform testing on the model using the provided test_loader.
        It iterates through the test_loader batches, processes the inputs, and generates segmentations.
        Computes Dice scores and Hausdorff distances for evaluation.
        Saves the results to a JSON file in the specified log directory.
        """
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        print("~~~~~~~~  START TEST  ~~~~~~~~\n")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        self.model.eval()
        num_labels = len(self.labels)
        results: List[Dict[str, Union[str, float, Dict]]] = []

        with torch.no_grad():
            for batch in self.test_loader:
                inputs_l = batch["image"].squeeze()
                labels_l = batch["label"].squeeze()
                self.files = labels_l.meta["filename_or_obj"][0]
                self.file_path = self.files.split("/")[:-1]
                self.files = self.files.split("/")[-1][:-7]
                print(f"File: {self.files}")

                # TODO: revisar que sea correcta la permutacion
                if self.config.axis == "sagital":
                    inputs_l = inputs_l.permute(2, 0, 1)
                    labels_l = labels_l.permute(2, 0, 1)
                elif self.config.axis == "coronal":
                    inputs_l = inputs_l.permute(2, 1, 0)
                    labels_l = labels_l.permute(2, 1, 0)

                # Creo un tensor de zeros para la segmentacion
                segmentation = torch.zeros((num_labels, 1, *labels_l.shape))
                targets_total = torch.zeros((num_labels, 1, *labels_l.shape))

                n_slices = self.config.roi_z_iter
                # pad the z direction, so we can easily extract 2.5D input and predict labels for the center slice
                pd = (n_slices // 2, n_slices // 2)

                inputs_l = F.pad(inputs_l, pd, "constant", 0)
                labels_pad = F.pad(labels_l, pd, "constant", 0)
                n_z_after_pad = labels_l.shape[-1]

                for idx in tqdm(
                    range(n_slices // 2, n_z_after_pad - n_slices // 2), desc="Slice"
                ):
                    inputs = inputs_l[
                        ..., idx - n_slices // 2 : idx + n_slices // 2 + 1
                    ].permute(2, 0, 1)
                    labels = labels_pad[
                        ..., idx - n_slices // 2 : idx + n_slices // 2 + 1
                    ][..., n_slices // 2]

                    if self.config.point_prompt:
                        data, target, _ = prepare_sam_val_input_pp_only(
                            inputs.to(self.device),
                            labels.to(self.device),
                            self.config,
                            self.sam_image_size,
                            self.config.point_pos,
                            self.config.point_neg,
                        )
                    elif self.config.bbox_prompt:
                        data, target, _ = prepare_sam_val_input_bb_only(
                            inputs.to(self.device), labels.to(self.device)
                        )

                    with autocast(enabled=self.config.use_amp):
                        outputs = self.model(data)

                    segmentation[..., idx - n_slices // 2] = self.post_pred(
                        outputs[0]["high_res_logits"]
                    )  # type: ignore
                    targets_total[..., idx - n_slices // 2] = target

                acc_per_label = compute_dice(segmentation, targets_total)
                acc_sum = torch.nansum(acc_per_label).item()
                not_nans = (
                    num_labels - torch.sum(torch.isnan(acc_per_label).float()).item()
                )
                hausdorsff = compute_hausdorff_distance(
                    segmentation, targets_total, spacing=1.5, percentile=95
                )
                hausdorsff = hausdorsff.tolist()
                acc_mean = acc_sum / not_nans
                self.save_predictions(segmentation)
                print(f"Mean Dice: {acc_mean}")
                results.append(
                    {
                        "Image": self.files,
                        "Dice_mean": acc_mean,
                        "dice_per_label": dict(
                            zip(self.labels, acc_per_label.tolist())
                        ),
                        "Hausdorff_distance": dict(zip(self.labels, hausdorsff)),
                    }
                )

        with open(
            os.path.join(self.config.logdir, self.config.experiment_name + ".json"), "w"
        ) as f:
            json.dump(results, f, indent=4)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        print("~~~~~~~    END TEST    ~~~~~~~\n")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    def save_predictions(self, segmentation):
        try:
            os.mkdir(os.path.join("/", *self.file_path, self.config.experiment_name))
        except FileExistsError:
            pass
        segmentation = segmentation.squeeze()
        segmentation = torch.argmax(segmentation, dim=0)
        _, _, z = segmentation.shape
        segmentation = segmentation.detach().cpu().numpy()
        # save plot of segmentation to disk each 10 slices
        np.save(
            os.path.join(
                "/", *self.file_path, self.config.experiment_name, f"{self.files}.npy"
            ),
            segmentation,
        )
        plt.imsave(
            os.path.join(
                "/",
                *self.file_path,
                self.config.experiment_name,
                f"segmentation_{self.files}.png",
            ),
            segmentation[..., z // 2],
        )
