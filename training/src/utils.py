# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.ndimage as ndimage
import torch
import random
from copy import deepcopy
import torch.distributed as dist
from skimage.measure import label, regionprops
from monai.transforms.utils import generate_spatial_bounding_box

def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = dist.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        dist.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            dist.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


def apply_coords_torch(coords, original_size, sam_image_size):
    """
    Expects a numpy array of length 2 in the final dimension. Requires the
    original image size in (H, W) format.
    """
    old = original_size
    new = sam_image_size
    coords = deepcopy(coords).float()
    # Here, we can apply a same scale factor to h and w, because we first pad the input to a square image along the
    # longest side then resize it to sam_image_size. In other words, the scale factor is determined by the longest side.
    coords[..., 0] = coords[..., 0] * (new / old)
    coords[..., 1] = coords[..., 1] * (new / old)
    return coords


def apply_coords_bbox(coords, original_size, sam_image_size):
    old = original_size
    new = sam_image_size
    coords = deepcopy(coords).float()
    # Here, we can apply a same scale factor to h and w, because we first pad the input to a square image along the
    return coords * (new / old)


def sample_points(labelpoints, n_points):
    idx = torch.randperm(len(labelpoints), dtype=torch.long, device=labelpoints.device)[
        :n_points
    ]
    return [labelpoints[idx]]


def generate_point_prompt(
    batch_labels_, sam_image_size, config, points_pos=None, points_neg=None, previous_pred=None
):
    """
    Generate point prompts for SAM.

    Args:
        batch_labels_ (torch.Tensor): Batch of labels.
        sam_image_size (tuple): Size of the SAM image.
        config (Config): SAM configuration.
        points_pos (int, optional): Number of positive points. Defaults to None.
        points_neg (int, optional): Number of negative points. Defaults to None.
        previous_pred (torch.Tensor, optional): Previous predictions. Defaults to None.

    Returns:
        tuple: Point coordinates and point labels.
    """

    # Number of positive and negative points
    Np = (
        points_pos
        if points_pos is not None
        else min(
            config.max_points,
            int(np.abs(random.gauss(mu=0, sigma=config.max_points // 2))) + 1,
        )
    )
    Nn = (
        points_neg
        if points_neg is not None
        else min(
            config.max_points,
            int(np.abs(random.gauss(mu=0, sigma=config.max_points // 2))),
        )
    )

    _point = []  # Point coordinates
    _point_label = []  # Point labels
    b, h, w = batch_labels_.shape
    device = batch_labels_.device

    for i in range(b):
        plabels = batch_labels_[i, ...]
        nlabels = (plabels == 0.0).float()

        if previous_pred is not None:
            ppred = previous_pred[i, 0, ...]
            npred = (previous_pred[i, 0, ...] == 0.0).float()

            # False positive mask (pixels that are predicted as positive but are actually negative)
            fp_mask = torch.logical_and(nlabels, ppred)
            # False negative mask (pixels that are predicted as negative but are actually positive)
            fn_mask = torch.logical_and(plabels, npred)
            # Sample positive points from false negative pred.
            # Sample negative points from false positive pred.
            plabelpoints = torch.nonzero(fn_mask)
            nlabelpoints = torch.nonzero(fp_mask)

        else:
            plabelpoints = torch.nonzero(plabels)
            nlabelpoints = torch.nonzero(nlabels)

        # Calculate number of placeholder points
        n_placeholder = (
            Np + Nn - min(len(plabelpoints), Np) - min(len(nlabelpoints), Nn)
        )

        # Generate point coordinates with placeholder points
        _point.append(
            torch.cat(
                sample_points(plabelpoints, min(len(plabelpoints), Np))
                + sample_points(nlabelpoints, min(len(nlabelpoints), Nn))
                + [torch.zeros((1, 2), device=device)] * n_placeholder,
                dim=0,
            )
        )
        _point_label.append(
            torch.tensor(
                [1] * min(len(plabelpoints), Np)
                + [0] * min(len(nlabelpoints), Nn)
                + [-1] * n_placeholder
            ).to(device)
        )

    point = torch.stack(_point)
    point_label = torch.stack(_point_label)
    point_coords = apply_coords_torch(point, max(h, w), sam_image_size)

    return point_coords, point_label


def generate_bbox_prompt(batch_labels_, std=0.1, max_pixel=20):
    device = batch_labels_.device
    bbox_per_image = []
    b, h, w = batch_labels_.shape
    bbox_per_image = torch.empty(b, 4, device=device, dtype=torch.float)
    for i in range(b):
        image = batch_labels_[i, ...]

        active_pixels = torch.nonzero(image.as_tensor(), as_tuple=False)
        box = torch.empty(4, device=device)
        if active_pixels.size(0) == 0:
            box[0], box[1] = torch.randint(
                low=10, high=15, size=[2], device=device
            )
            box[2], box[3] = torch.randint(
                low=20,
                high=25,
                size=[2],
                device=device,
            )
        else:
            (y0, x0), (y1, x1) = generate_spatial_bounding_box(
                image.unsqueeze(0), allow_smaller=False
            )
#            min_coords, _ = torch.min(active_pixels, dim=0)
#            max_coords, _ = torch.max(active_pixels, dim=0)
#
            h_b = abs(x1 - x0)
            w_b = abs(y1 - y0)
#                
            noise_std = min(h_b, w_b) * std
            max_noise = max(min(max_pixel, int(noise_std * 5)), 1)
#            
#            if max_noise == 0:
#                max_noise = 1
#            
            noise_x = random.randint(
                a=-max_noise, b=max_noise
            )
            noise_y = random.randint(
                a=-max_noise, b=max_noise
            )
            box[0], box[1] = x0 + noise_x, y0 + noise_y
            box[2], box[3] = x1 + noise_x, y1 + noise_y
        bbox_per_image[i, ...] = box
    return apply_coords_bbox(bbox_per_image, original_size=max(h,w), sam_image_size=1024)
    #return bbox_per_image


def prepare_sam_training_input(inputs, labels, config, model, sam_image_size, point_prob, bbox_prob):
    unique_labels = torch.unique(labels).as_tensor().long()
    
    device = labels.device
    if config.skip_bk:
        unique_labels = unique_labels[1:]

    if len(unique_labels) == 0:
        prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]
        batch_labels = torch.zeros(
            1, 1, sam_image_size // 4, sam_image_size // 4
        ).to(device)
        skip = True
        return prepared_input, batch_labels, None, skip

    # random sample args.num_prompt prompts, this will help to manage the GPU memory upper bound.
    if len(unique_labels) > config.num_prompt:
        idxs = random.sample(range(len(unique_labels)), config.num_prompt)
        idxs = torch.tensor(idxs)
        unique_labels = unique_labels[idxs]

    if len(unique_labels) < config.num_prompt:
        while len(unique_labels) < config.num_prompt:
            unique_labels = torch.cat([unique_labels, unique_labels], 0)
        unique_labels = unique_labels[: config.num_prompt]

    # add 4 background labels to every batch
    background_labels = list(
        set([i for i in range(1, 115)]) - set(unique_labels.cpu().numpy())
    )
    random.shuffle(background_labels)
    unique_labels = torch.cat(
        [unique_labels, torch.tensor(background_labels[:4]).to(device)]
    )

    # preprocess make the size of label same as low_res_logit
    batch_labels_ = torch.stack(
        [labels == unique_labels[i] for i in range(len(unique_labels))], dim=0
    ).float()

    if config.distributed:
        batch_labels = model.module.preprocess(batch_labels_, is_input=False)
    else:
        batch_labels = model.preprocess(batch_labels_, is_input=False)

    prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]

    prompt = random.choices(['point', 'bbox'], weights=(point_prob, bbox_prob), k=1)[0]
    

    if prompt == 'bbox':
        bbox_prompt = generate_bbox_prompt(batch_labels)
        prepared_input[0].update({"boxes": bbox_prompt})
    elif prompt == 'point':
        point_coords, point_labels = generate_point_prompt(batch_labels_, config=config, sam_image_size=sam_image_size)
        prepared_input[0].update(
            {"point_coords": point_coords, "point_labels": point_labels}
        )
    return prepared_input, batch_labels.unsqueeze(1).to(device), batch_labels_, False


def prepare_sam_val_input_pp_only(inputs, labels, config, sam_image_size, point_pos=None, point_neg=None):
    # Don't exclude background in val but will ignore it in metric calculation
    device = labels.device
    unique_labels = torch.tensor([i for i in range(1, 115)]).to(device)
    #unique_labels = torch.unique(labels).as_tensor().long()
    # preprocess make the size of lable same as high_res_logit
    batch_labels = torch.stack(
        [labels == unique_labels[i] for i in range(len(unique_labels))], dim=0
    ).float()

    prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]
    
    point_coords, point_labels = generate_point_prompt(batch_labels, sam_image_size=sam_image_size, config=config, points_neg=point_neg, points_pos=point_pos)
    prepared_input[0].update(
            {"point_coords": point_coords, "point_labels": point_labels}
        )

    return prepared_input, batch_labels.unsqueeze(1).to(device), unique_labels

def prepare_sam_val_input_bb_only(inputs, labels):
    # Don't exclude background in val but will ignore it in metric calculation
    device = labels.device
    unique_labels = torch.tensor([i for i in range(1, 115)]).to(device)
    #unique_labels = torch.unique(labels).as_tensor().long()
    #skip = False
    #if len(unique_labels) == 0:
    #    prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]
    #    batch_labels = torch.zeros(
    #        1, 1, 1024 // 4, 1024 // 4
    #    ).to(device)
    #    skip = True
    #    return prepared_input, batch_labels, None, skip
    
    # preprocess make the size of lable same as high_res_logit
    batch_labels = torch.stack(
        [labels == unique_labels[i] for i in range(len(unique_labels))], dim=0
    ).float()

    prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]

    bbox_prompt = generate_bbox_prompt(batch_labels, std=0.1, max_pixel=5)
    prepared_input[0].update({"boxes": bbox_prompt})

    return prepared_input, batch_labels.unsqueeze(1).to(device), unique_labels

def prepare_sam_val_input_np_only(inputs, labels):
    # Don't exclude background in val but will ignore it in metric calculation
    device = labels.device
    unique_labels = torch.tensor([i for i in range(1, 115)]).to(device)

    # preprocess make the size of lable same as high_res_logit
    batch_labels = torch.stack(
        [labels == unique_labels[i] for i in range(len(unique_labels))], dim=0
    ).float()

    prepared_input = [{"image": inputs, "original_size": tuple(labels.shape)}]

    return prepared_input, batch_labels.unsqueeze(1).to(device), unique_labels