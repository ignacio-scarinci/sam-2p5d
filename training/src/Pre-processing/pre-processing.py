# Take from https://github.com/wasserth/TotalSegmentator/tree/master

import shutil
import sys
import nibabel as nib
import numpy as np
import os
from pathlib import Path
import pandas as pd
import json

from tqdm import tqdm
from combine_mask import combine_masks


def generate_json_from_dir_v2(foldername, subjects_train, subjects_val, subjects_test, labels):
    print("Creating dataset.json...")
    out_base = foldername

    json_dict = {}
    json_dict["name"] = "Liifa"
    json_dict["description"] = "Liifa dataset base on TotalSegmentator classes"
    json_dict["reference"] = "https://zenodo.org/record/6802614"
    json_dict["licence"] = "Apache 2.0"
    json_dict["release"] = "1.0"
    json_dict["channel_names"] = {"0": "CT"}
    json_dict["labels"] = {
        val: idx
        for idx, val in enumerate(
            [
                "background",
            ]
            + list(labels)
        )
    }
    json_dict["numTotal"] = len(subjects_train + subjects_val+subjects_test)
    json_dict["numTraining"] = len(subjects_train)
    json_dict["numTest"] = len(subjects_test)
    json_dict["file_ending"] = ".nii.gz"
    json_dict["overwrite_image_reader_writer"] = "NibabelIOWithReorient"

    train = []
    val = []
    test = []
    for subject in tqdm(subjects_train+subjects_val):
        image = f"imagesTr/{subject}.nii.gz"
        label = f"labelsTr/{subject}.nii.gz"
        train.append({"image": image, "label": label})
    
    for subject in tqdm(subjects_test):
        image = f"imagesTs/{subject}.nii.gz"
        label = f"labelsTs/{subject}.nii.gz"
        test.append({"image": image, "label": label})

    json_dict["training"] = train
    json_dict["testing"] = test

    json.dump(
        json_dict, open(f"{out_base}/dataset.json", "w"), sort_keys=False, indent=4
    )


def combine_labels(ref_img, file_out, masks):
    ref_img = nib.load(ref_img)
    combined = np.zeros(ref_img.shape).astype(np.uint8)
    for idx, arg in enumerate(masks):
        file_in = Path(arg)
        if file_in.exists():
            img = nib.load(file_in)
            combined[img.get_fdata() > 0] = idx + 1
        else:
            print(f"Missing: {file_in}")
    nib.save(nib.Nifti1Image(combined.astype(np.uint8), ref_img.affine), file_out)


if __name__ == "__main__":
    """
    Convert the downloaded TotalSegmentator dataset (after unzipping it) and
    generate dataset.json and splits_final.json

    example usage:
    python convert_dataset_to_nnunet.py /my_downloads/TotalSegmentator_dataset /nnunet/raw/Dataset100_TotalSegmentator_part1 class_map_part_organs

    You must set nnUNet_raw and nnUNet_preprocessed environment variables before running this (see nnUNet documentation).
    """

    dataset_path = Path(sys.argv[1])  # directory containing all the subjects
    new_path = Path(sys.argv[2])  # directory of the new dataset
    # TotalSegmentator is made up of 5 models. Choose which one you want to produce. Choose from:
    #   class_map_part_organs
    #   class_map_part_vertebrae
    #   class_map_part_cardiac
    #   class_map_part_muscles
    #   class_map_part_ribs
    class_map_name = sys.argv[3]
    
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
        },
        "liifa": {
            1: "spleen",
            2: "kidney_right",
            3: "kidney_left",
            4: "gallbladder",
            5: "liver",
            6: "stomach",
            7: "pancreas",
            8: "adrenal_gland_right",
            9: "adrenal_gland_left",
            10: "lung_left",
            11: "lung_right",
            12: "esophagus",
            13: "trachea",
            14: "thyroid_gland",
            15: "small_bowel",
            16: "duodenum",
            17: "colon",
            18: "urinary_bladder",
            19: "prostate",
            20: "kidney_cyst_left",
            21: "kidney_cyst_right",
            22: "sacrum",
            23: "vertebrae",
            24: "heart",
            25: "humerus_left",
            26: "humerus_right",
            27: "scapula_left",
            28: "scapula_right",
            29: "clavicula_left",
            30: "clavicula_right",
            31: "femur_left",
            32: "femur_right",
            33: "hip_left",
            34: "hip_right",
            35: "spinal_cord",
            36: "brain",
            37: "skull",
            38: "ribs",
        },
            "total_liifa": {
                1: "spleen",
                2: "kidney_right",
                3: "kidney_left",
                4: "gallbladder",
                5: "liver",
                6: "stomach",
                7: "pancreas",
                8: "adrenal_gland_right",
                9: "adrenal_gland_left",
                10: "lung_left",
                11: "lung_right",
                12: "esophagus",
                13: "trachea",
                14: "thyroid_gland",
                15: "small_bowel",
                16: "duodenum",
                17: "colon",
                18: "urinary_bladder",
                19: "prostate",
                20: "kidney_cyst_left",
                21: "kidney_cyst_right",
                22: "sacrum",
                23: "vertebrae_S1",
                24: "vertebrae_L5",
                25: "vertebrae_L4",
                26: "vertebrae_L3",
                27: "vertebrae_L2",
                28: "vertebrae_L1",
                29: "vertebrae_T12",
                30: "vertebrae_T11",
                31: "vertebrae_T10",
                32: "vertebrae_T9",
                33: "vertebrae_T8",
                34: "vertebrae_T7",
                35: "vertebrae_T6",
                36: "vertebrae_T5",
                37: "vertebrae_T4",
                38: "vertebrae_T3",
                39: "vertebrae_T2",
                40: "vertebrae_T1",
                41: "vertebrae_C7",
                42: "vertebrae_C6",
                43: "vertebrae_C5",
                44: "vertebrae_C4",
                45: "vertebrae_C3",
                46: "vertebrae_C2",
                47: "vertebrae_C1",
                48: "heart",
                49: "aorta",
                50: "pulmonary_vein",
                51: "brachiocephalic_trunk",
                52: "subclavian_artery_right",
                53: "subclavian_artery_left",
                54: "common_carotid_artery_right",
                55: "common_carotid_artery_left",
                56: "brachiocephalic_vein_left",
                57: "brachiocephalic_vein_right",
                58: "atrial_appendage_left",
                59: "superior_vena_cava",
                60: "inferior_vena_cava",
                61: "portal_vein_and_splenic_vein",
                62: "iliac_artery_left",
                63: "iliac_artery_right",
                64: "iliac_vena_left",
                65: "iliac_vena_right",
                66: "humerus_left",
                67: "humerus_right",
                68: "scapula_left",
                69: "scapula_right",
                70: "clavicula_left",
                71: "clavicula_right",
                72: "femur_left",
                73: "femur_right",
                74: "hip_left",
                75: "hip_right",
                76: "spinal_cord",
                77: "gluteus_maximus_left",
                78: "gluteus_maximus_right",
                79: "gluteus_medius_left",
                80: "gluteus_medius_right",
                81: "gluteus_minimus_left",
                82: "gluteus_minimus_right",
                83: "autochthon_left",
                84: "autochthon_right",
                85: "iliopsoas_left",
                86: "iliopsoas_right",
                87: "brain",
                88: "skull",
                89: "rib_right_4",
                90: "rib_right_3",
                91: "rib_left_1",
                92: "rib_left_2",
                93: "rib_left_3",
                94: "rib_left_4",
                95: "rib_left_5",
                96: "rib_left_6",
                97: "rib_left_7",
                98: "rib_left_8",
                99: "rib_left_9",
                100: "rib_left_10",
                101: "rib_left_11",
                102: "rib_left_12",
                103: "rib_right_1",
                104: "rib_right_2",
                105: "rib_right_5",
                106: "rib_right_6",
                107: "rib_right_7",
                108: "rib_right_8",
                109: "rib_right_9",
                110: "rib_right_10",
                111: "rib_right_11",
                112: "rib_right_12",
                113: "sternum",
                114: "costal_cartilages",
            },
    }

    class_map = class_map_5_parts[class_map_name]

    (new_path / "imagesTr").mkdir(parents=True, exist_ok=True)
    (new_path / "labelsTr").mkdir(parents=True, exist_ok=True)
    (new_path / "imagesTs").mkdir(parents=True, exist_ok=True)
    (new_path / "labelsTs").mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(dataset_path / "meta.csv", sep=";")
    subjects_train = list(meta[meta["split"] == "train"]["image_id"].values)
    subjects_val = list(meta[meta["split"] == "val"]["image_id"].values)
    subjects_test = list(meta[meta["split"] == "test"]["image_id"].values)
    
#    print("Combining masks...")
#    for subject in tqdm(subjects_train + subjects_val + subjects_test):
#        mask_path = dataset_path / subject / "segmentations"
#        
#        combine = combine_masks(mask_path, 'lung_left')
#        file_out = mask_path / "lung_left.nii.gz"
#        nib.save(combine, file_out)
#        
#        combine = combine_masks(mask_path, "lung_right")
#        file_out = mask_path / "lung_right.nii.gz"
#        nib.save(combine, file_out)
#        
#        combine = combine_masks(mask_path, "ribs")
#        file_out = mask_path / "ribs.nii.gz"
#        nib.save(combine, file_out)
#        
#        combine = combine_masks(mask_path, "vertebrae")
#        file_out = mask_path / "vertebrae.nii.gz"
#        nib.save(combine, file_out)
        
    print("Copying train data...")
    for subject in tqdm(subjects_train + subjects_val):
        subject_path = dataset_path / subject
        shutil.copy(
            subject_path / "ct.nii.gz",
            new_path / "imagesTr" / f"{subject}.nii.gz",
        )
        combine_labels(
            subject_path / "ct.nii.gz",
            new_path / "labelsTr" / f"{subject}.nii.gz",
            [
                subject_path / "segmentations" / f"{roi}.nii.gz"
                for roi in class_map.values()
            ],
        )

    print("Copying test data...")
    for subject in tqdm(subjects_test):
        subject_path = dataset_path / subject
        shutil.copy(
            subject_path / "ct.nii.gz",
            new_path / "imagesTs" / f"{subject}.nii.gz",
        )
        combine_labels(
            subject_path / "ct.nii.gz",
            new_path / "labelsTs" / f"{subject}.nii.gz",
            [
                subject_path / "segmentations" / f"{roi}.nii.gz"
                for roi in class_map.values()
            ],
        )

    generate_json_from_dir_v2(
        new_path, subjects_train, subjects_val, subjects_test, class_map.values()
    )
    
# class_map = {
#     # classes of old TotalSegmentator v1
#     "total_v1": {
#         1: "spleen",
#         2: "kidney_right",
#         3: "kidney_left",
#         4: "gallbladder",
#         5: "liver",
#         6: "stomach",
#         7: "aorta",
#         8: "inferior_vena_cava",
#         9: "portal_vein_and_splenic_vein",
#         10: "pancreas",
#         11: "adrenal_gland_right",
#         12: "adrenal_gland_left",
#         13: "lung_upper_lobe_left",
#         14: "lung_lower_lobe_left",
#         15: "lung_upper_lobe_right",
#         16: "lung_middle_lobe_right",
#         17: "lung_lower_lobe_right",
#         18: "vertebrae_L5",
#         19: "vertebrae_L4",
#         20: "vertebrae_L3",
#         21: "vertebrae_L2",
#         22: "vertebrae_L1",
#         23: "vertebrae_T12",
#         24: "vertebrae_T11",
#         25: "vertebrae_T10",
#         26: "vertebrae_T9",
#         27: "vertebrae_T8",
#         28: "vertebrae_T7",
#         29: "vertebrae_T6",
#         30: "vertebrae_T5",
#         31: "vertebrae_T4",
#         32: "vertebrae_T3",
#         33: "vertebrae_T2",
#         34: "vertebrae_T1",
#         35: "vertebrae_C7",
#         36: "vertebrae_C6",
#         37: "vertebrae_C5",
#         38: "vertebrae_C4",
#         39: "vertebrae_C3",
#         40: "vertebrae_C2",
#         41: "vertebrae_C1",
#         42: "esophagus",
#         43: "trachea",
#         44: "heart_myocardium",
#         45: "heart_atrium_left",
#         46: "heart_ventricle_left",
#         47: "heart_atrium_right",
#         48: "heart_ventricle_right",
#         49: "pulmonary_artery",
#         50: "brain",
#         51: "iliac_artery_left",
#         52: "iliac_artery_right",
#         53: "iliac_vena_left",
#         54: "iliac_vena_right",
#         55: "small_bowel",
#         56: "duodenum",
#         57: "colon",
#         58: "rib_left_1",
#         59: "rib_left_2",
#         60: "rib_left_3",
#         61: "rib_left_4",
#         62: "rib_left_5",
#         63: "rib_left_6",
#         64: "rib_left_7",
#         65: "rib_left_8",
#         66: "rib_left_9",
#         67: "rib_left_10",
#         68: "rib_left_11",
#         69: "rib_left_12",
#         70: "rib_right_1",
#         71: "rib_right_2",
#         72: "rib_right_3",
#         73: "rib_right_4",
#         74: "rib_right_5",
#         75: "rib_right_6",
#         76: "rib_right_7",
#         77: "rib_right_8",
#         78: "rib_right_9",
#         79: "rib_right_10",
#         80: "rib_right_11",
#         81: "rib_right_12",
#         82: "humerus_left",
#         83: "humerus_right",
#         84: "scapula_left",
#         85: "scapula_right",
#         86: "clavicula_left",
#         87: "clavicula_right",
#         88: "femur_left",
#         89: "femur_right",
#         90: "hip_left",
#         91: "hip_right",
#         92: "sacrum",
#         93: "face",
#         94: "gluteus_maximus_left",
#         95: "gluteus_maximus_right",
#         96: "gluteus_medius_left",
#         97: "gluteus_medius_right",
#         98: "gluteus_minimus_left",
#         99: "gluteus_minimus_right",
#         100: "autochthon_left",
#         101: "autochthon_right",
#         102: "iliopsoas_left",
#         103: "iliopsoas_right",
#         104: "urinary_bladder",
#     },
#     # classes of new TotalSegmentator v2
#     "total": {
#         1: "spleen",
#         2: "kidney_right",
#         3: "kidney_left",
#         4: "gallbladder",
#         5: "liver",
#         6: "stomach",
#         7: "pancreas",
#         8: "adrenal_gland_right",
#         9: "adrenal_gland_left",
#         10: "lung_upper_lobe_left",
#         11: "lung_lower_lobe_left",
#         12: "lung_upper_lobe_right",
#         13: "lung_middle_lobe_right",
#         14: "lung_lower_lobe_right",
#         15: "esophagus",
#         16: "trachea",
#         17: "thyroid_gland",
#         18: "small_bowel",
#         19: "duodenum",
#         20: "colon",
#         21: "urinary_bladder",
#         22: "prostate",
#         23: "kidney_cyst_left",
#         24: "kidney_cyst_right",
#         25: "sacrum",
#         26: "vertebrae_S1",
#         27: "vertebrae_L5",
#         28: "vertebrae_L4",
#         29: "vertebrae_L3",
#         30: "vertebrae_L2",
#         31: "vertebrae_L1",
#         32: "vertebrae_T12",
#         33: "vertebrae_T11",
#         34: "vertebrae_T10",
#         35: "vertebrae_T9",
#         36: "vertebrae_T8",
#         37: "vertebrae_T7",
#         38: "vertebrae_T6",
#         39: "vertebrae_T5",
#         40: "vertebrae_T4",
#         41: "vertebrae_T3",
#         42: "vertebrae_T2",
#         43: "vertebrae_T1",
#         44: "vertebrae_C7",
#         45: "vertebrae_C6",
#         46: "vertebrae_C5",
#         47: "vertebrae_C4",
#         48: "vertebrae_C3",
#         49: "vertebrae_C2",
#         50: "vertebrae_C1",
#         51: "heart",
#         52: "aorta",
#         53: "pulmonary_vein",
#         54: "brachiocephalic_trunk",
#         55: "subclavian_artery_right",
#         56: "subclavian_artery_left",
#         57: "common_carotid_artery_right",
#         58: "common_carotid_artery_left",
#         59: "brachiocephalic_vein_left",
#         60: "brachiocephalic_vein_right",
#         61: "atrial_appendage_left",
#         62: "superior_vena_cava",
#         63: "inferior_vena_cava",
#         64: "portal_vein_and_splenic_vein",
#         65: "iliac_artery_left",
#         66: "iliac_artery_right",
#         67: "iliac_vena_left",
#         68: "iliac_vena_right",
#         69: "humerus_left",
#         70: "humerus_right",
#         71: "scapula_left",
#         72: "scapula_right",
#         73: "clavicula_left",
#         74: "clavicula_right",
#         75: "femur_left",
#         76: "femur_right",
#         77: "hip_left",
#         78: "hip_right",
#         79: "spinal_cord",
#         80: "gluteus_maximus_left",
#         81: "gluteus_maximus_right",
#         82: "gluteus_medius_left",
#         83: "gluteus_medius_right",
#         84: "gluteus_minimus_left",
#         85: "gluteus_minimus_right",
#         86: "autochthon_left",
#         87: "autochthon_right",
#         88: "iliopsoas_left",
#         89: "iliopsoas_right",
#         90: "brain",
#         91: "skull",
#         92: "rib_right_4",
#         93: "rib_right_3",
#         94: "rib_left_1",
#         95: "rib_left_2",
#         96: "rib_left_3",
#         97: "rib_left_4",
#         98: "rib_left_5",
#         99: "rib_left_6",
#         100: "rib_left_7",
#         101: "rib_left_8",
#         102: "rib_left_9",
#         103: "rib_left_10",
#         104: "rib_left_11",
#         105: "rib_left_12",
#         106: "rib_right_1",
#         107: "rib_right_2",
#         108: "rib_right_5",
#         109: "rib_right_6",
#         110: "rib_right_7",
#         111: "rib_right_8",
#         112: "rib_right_9",
#         113: "rib_right_10",
#         114: "rib_right_11",
#         115: "rib_right_12",
#         116: "sternum",
#         117: "costal_cartilages",
#     },
#     # total_fast not extra class map, because easier to use just "total" for fast model
#     "lung_vessels": {1: "lung_vessels", 2: "lung_trachea_bronchia"},
#     "covid": {
#         1: "lung_covid_infiltrate",
#     },
#     "cerebral_bleed": {
#         1: "intracerebral_hemorrhage",
#     },
#     "hip_implant": {
#         1: "hip_implant",
#     },
#     "coronary_arteries": {
#         1: "coronary_arteries",
#     },
#     "body": {
#         1: "body_trunc",
#         2: "body_extremities",
#     },
#     "pleural_pericard_effusion": {
#         # 1: "lung_pleural",
#         2: "pleural_effusion",
#         3: "pericardial_effusion",
#     },
#     "liver_vessels": {1: "liver_vessels", 2: "liver_tumor"},
#     "vertebrae_body": {1: "vertebrae_body"},
#     "heartchambers_highres": {
#         1: "heart_myocardium",
#         2: "heart_atrium_left",
#         3: "heart_ventricle_left",
#         4: "heart_atrium_right",
#         5: "heart_ventricle_right",
#         6: "aorta",
#         7: "pulmonary_artery",
#     },
#     "appendicular_bones": {
#         1: "patella",
#         2: "tibia",
#         3: "fibula",
#         4: "tarsal",
#         5: "metatarsal",
#         6: "phalanges_feet",
#         7: "ulna",
#         8: "radius",
#         9: "carpal",
#         10: "metacarpal",
#         11: "phalanges_hand",
#     },
#     # those classes need to be removed
#     "appendicular_bones_auxiliary": {
#         12: "humerus",
#         13: "femur",
#         14: "liver",
#         15: "spleen",
#     },
#     "tissue_types": {1: "subcutaneous_fat", 2: "torso_fat", 3: "skeletal_muscle"},
#     "face": {1: "face"},
#     "test": {1: "carpal"},
# }


# commercial_models = {
#     "heartchambers_highres": 301,
#     "appendicular_bones": 304,
#     "tissue_types": 481,
#     "vertebrae_body": 302,
#     "face": 303,
# }
# future
# - brain subparts
# - head subparts
# - everything highres?


