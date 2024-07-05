# Take from https://github.com/wasserth/TotalSegmentator/tree/master

import nibabel as nib
import numpy as np


def combine_masks(mask_dir, class_type):
    """
    Combine classes to masks

    mask_dir: directory of totalsegmetator masks
    class_type: ribs | vertebrae | vertebrae_ribs | lung | heart

    returns: nibabel image
    """
    
    class_map_5_parts = {
        # 24 classes
        "class_map_part_organs": {
            1: "spleen",
            2: "kidney_right",
            3: "kidney_left",
            4: "gallbladder",
            5: "liver",
            6: "stomach",
            7: "pancreas",
            8: "adrenal_gland_right",
            9: "adrenal_gland_left",
            10: "lung_upper_lobe_left",
            11: "lung_lower_lobe_left",
            12: "lung_upper_lobe_right",
            13: "lung_middle_lobe_right",
            14: "lung_lower_lobe_right",
            15: "esophagus",
            16: "trachea",
            17: "thyroid_gland",
            18: "small_bowel",
            19: "duodenum",
            20: "colon",
            21: "urinary_bladder",
            22: "prostate",
            23: "kidney_cyst_left",
            24: "kidney_cyst_right",
        },
        # 26 classes
        "class_map_part_vertebrae": {
            1: "sacrum",
            2: "vertebrae_S1",
            3: "vertebrae_L5",
            4: "vertebrae_L4",
            5: "vertebrae_L3",
            6: "vertebrae_L2",
            7: "vertebrae_L1",
            8: "vertebrae_T12",
            9: "vertebrae_T11",
            10: "vertebrae_T10",
            11: "vertebrae_T9",
            12: "vertebrae_T8",
            13: "vertebrae_T7",
            14: "vertebrae_T6",
            15: "vertebrae_T5",
            16: "vertebrae_T4",
            17: "vertebrae_T3",
            18: "vertebrae_T2",
            19: "vertebrae_T1",
            20: "vertebrae_C7",
            21: "vertebrae_C6",
            22: "vertebrae_C5",
            23: "vertebrae_C4",
            24: "vertebrae_C3",
            25: "vertebrae_C2",
            26: "vertebrae_C1",
        },
        # 18
        "class_map_part_cardiac": {
            1: "heart",
            2: "aorta",
            3: "pulmonary_vein",
            4: "brachiocephalic_trunk",
            5: "subclavian_artery_right",
            6: "subclavian_artery_left",
            7: "common_carotid_artery_right",
            8: "common_carotid_artery_left",
            9: "brachiocephalic_vein_left",
            10: "brachiocephalic_vein_right",
            11: "atrial_appendage_left",
            12: "superior_vena_cava",
            13: "inferior_vena_cava",
            14: "portal_vein_and_splenic_vein",
            15: "iliac_artery_left",
            16: "iliac_artery_right",
            17: "iliac_vena_left",
            18: "iliac_vena_right",
        },
        # 23
        "class_map_part_muscles": {
            1: "humerus_left",
            2: "humerus_right",
            3: "scapula_left",
            4: "scapula_right",
            5: "clavicula_left",
            6: "clavicula_right",
            7: "femur_left",
            8: "femur_right",
            9: "hip_left",
            10: "hip_right",
            11: "spinal_cord",
            12: "gluteus_maximus_left",
            13: "gluteus_maximus_right",
            14: "gluteus_medius_left",
            15: "gluteus_medius_right",
            16: "gluteus_minimus_left",
            17: "gluteus_minimus_right",
            18: "autochthon_left",
            19: "autochthon_right",
            20: "iliopsoas_left",
            21: "iliopsoas_right",
            22: "brain",
            23: "skull",
        },
        # 26 classes
        # 12. ribs start from vertebrae T12
        # Small subset of population (roughly 8%) have 13. rib below 12. rib
        #  (would start from L1 then)
        #  -> this has label rib_12
        # Even smaller subset (roughly 1%) has extra rib above 1. rib   ("Halsrippe")
        #  (the extra rib would start from C7)
        #  -> this has label rib_1
        #
        # Quite often only 11 ribs (12. ribs probably so small that not found). Those
        # cases often wrongly segmented.
        "class_map_part_ribs": {
            1: "rib_left_1",
            2: "rib_left_2",
            3: "rib_left_3",
            4: "rib_left_4",
            5: "rib_left_5",
            6: "rib_left_6",
            7: "rib_left_7",
            8: "rib_left_8",
            9: "rib_left_9",
            10: "rib_left_10",
            11: "rib_left_11",
            12: "rib_left_12",
            13: "rib_right_1",
            14: "rib_right_2",
            15: "rib_right_3",
            16: "rib_right_4",
            17: "rib_right_5",
            18: "rib_right_6",
            19: "rib_right_7",
            20: "rib_right_8",
            21: "rib_right_9",
            22: "rib_right_10",
            23: "rib_right_11",
            24: "rib_right_12",
            25: "sternum",
            26: "costal_cartilages",
        }
    }

    if class_type == "ribs":
        masks = list(class_map_5_parts["class_map_part_ribs"].values())
    elif class_type == "vertebrae":
        masks = list(class_map_5_parts["class_map_part_vertebrae"].values())
    elif class_type == "vertebrae_ribs":
        masks = list(class_map_5_parts["class_map_part_vertebrae"].values()) + list(
            class_map_5_parts["class_map_part_ribs"].values()
        )
    elif class_type == "lung":
        masks = [
            "lung_upper_lobe_left",
            "lung_lower_lobe_left",
            "lung_upper_lobe_right",
            "lung_middle_lobe_right",
            "lung_lower_lobe_right",
        ]
    elif class_type == "lung_left":
        masks = ["lung_upper_lobe_left",
                 "lung_lower_lobe_left"]
    elif class_type == "lung_right":
        masks = [
            "lung_upper_lobe_right",
            "lung_middle_lobe_right",
            "lung_lower_lobe_right",
        ]
    elif class_type == "pelvis":
        masks = ["femur_left", "femur_right", "hip_left", "hip_right"]
    elif class_type == "body":
        masks = ["body_trunc", "body_extremities"]

    ref_img = None
    for mask in masks:
        if (mask_dir / f"{mask}.nii.gz").exists():
            ref_img = nib.load(mask_dir / f"{masks[0]}.nii.gz")
        else:
            raise ValueError(
                f"Could not find {mask_dir / mask}.nii.gz. Did you run TotalSegmentator successfully?"
            )

    combined = np.zeros(ref_img.shape, dtype=np.uint8)
    for idx, mask in enumerate(masks):
        if (mask_dir / f"{mask}.nii.gz").exists():
            img = nib.load(mask_dir / f"{mask}.nii.gz").get_fdata()
            combined[img > 0.5] = 1

    return nib.Nifti1Image(combined.astype(np.uint8), ref_img.affine)