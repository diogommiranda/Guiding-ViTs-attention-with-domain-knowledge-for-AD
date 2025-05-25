# This file contains functions to load NIfTI files and create pytorch dataloaders for training and testing.

import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from pathlib import Path
import os
import random
import time

def get_paths_and_labels(data_dir, split, class_map=None):
    """
    Finds NIfTI file paths and corresponding integer labels for a given split.
    Uses an existing class_map if provided, otherwise infers from train dir.

    Args:
        data_dir (str or Path): Path to the root dataset directory.
        split (str): 'train' or 'test'.
        class_map (dict, optional): Predefined mapping from class names to indices.
                                    If None, inferred from 'train' directory.

    Returns:
        tuple: (list of file paths, list of integer labels, dict class_map)
               Returns ([], [], {}) if the split directory doesn't exist or is empty.
    """
    data_path = Path(data_dir)
    split_path = data_path / split

    # Infer class_map from train directory only if not provided or if split is 'train'
    if class_map is None:
        train_path = data_path / "train"
        class_names = sorted([p.name for p in train_path.iterdir() if p.is_dir()], reverse=True)
        class_map = {name: i for i, name in enumerate(class_names)}
    print(f"Class map: {class_map}")

    paths = []
    labels = []

    print(f"Scanning {split_path}...")
    for class_name, label_index in class_map.items():
        class_dir = split_path / class_name
        if class_dir.is_dir():
            nifti_files = list(class_dir.glob('*.nii'))
            if nifti_files:
                print(f"    Found {len(nifti_files)} files for class '{class_name}'")
                paths.extend([str(p) for p in nifti_files])
                labels.extend([label_index] * len(nifti_files))

    return paths, labels, class_map

def calculate_min_max(paths):
    """
    Calculates the minimum and maximum voxel values across a list of NIfTI files.
    These values are then used for min-max normalization of the data.

    Args:
        paths (list): List of paths to NIfTI files.

    Returns:
        tuple: (float, float) minimum and maximum values found.
    """
    if len(paths)==0:
        raise(ValueError("No paths provided for minmax calculation."))

    global_min = np.inf
    global_max = -np.inf

    for path_str in paths:
        try:
            img = nib.load(path_str)
            # We load as float32 instead of the original float64
            volume = img.get_fdata(dtype=np.float32)
            current_min = np.min(volume)
            current_max = np.max(volume)
            if current_min < global_min:
                global_min = current_min
            if current_max > global_max:
                global_max = current_max
        except Exception as e:
            print(f"Error processing file {path_str} for min/max: {e}")
            raise e

    print(f"Calculated Min: {global_min}, Max: {global_max}")
    return float(global_min), float(global_max)


class AdniDataset(Dataset):
    """ Dataset class for loading NIfTI files and their labels.

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, paths, labels, volume_shape, is_training, min_val, max_val, mask_path=None):
        """
        Args:
            paths (list): List of file paths (strings).
            labels (list): List of corresponding integer labels.
            volume_shape (tuple): The shape of the NIfTI volumes (W, H, D).
            min_val (float): The minimum value used for normalization.
            max_val (float): The maximum value used for normalization.
            mask_path (str): Path to the mask file (shares the same volume shape as the input data).
                             If None, no mask is applied.
        """
        self.paths = paths
        self.labels = labels
        self.volume_shape = volume_shape
        self.min_val = min_val
        self.max_val = max_val
        self.mask_path = mask_path
        self.mask = None
        self.is_training = is_training
        
        if mask_path is not None:
            try:
                mask_img = nib.load(mask_path)
                mask_volume = mask_img.get_fdata(dtype=np.float32)
                if mask_volume.shape != (volume_shape[0], volume_shape[1], volume_shape[2]):
                    raise ValueError(f"Mask shape {mask_volume.shape} does not match expected shape {(volume_shape[0], volume_shape[1], volume_shape[2])}.")
                self.mask = mask_volume
            except Exception as e:
                print(f"Error loading mask file {mask_path}: {e}")
                raise e
            
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        path = self.paths[idx]
        label = self.labels[idx]
        img_tensor = None
        label_tensor = None
        
        try:
            img = nib.load(path)
            volume = img.get_fdata(dtype=np.float32)
            # Perform intensity normalization
            volume = (volume - self.min_val) / (self.max_val - self.min_val)
            volume = np.clip(volume, 0.0, 1.0)

            if self.mask is not None:
                # Apply mask to volume 
                volume = np.multiply(volume, self.mask)
                
            if self.is_training and random.random() > 0.5:
                # Random coronal view flipping
                volume = np.flip(volume, axis=0)  
            
            # Add channel dimension
            volume = np.expand_dims(volume, axis=0)
            volume = np.transpose(volume, (0, 3, 2, 1)) # Change to (C, D, H, W)
            
            img_tensor = torch.tensor(volume.copy(), dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading file {path}: {e}")
            raise e
        
        return img_tensor, label_tensor
    
def create_dataloader(paths, labels, batch_size, volume_shape, is_training, seed, min_val, max_val, mask_path=None):
    """
    Creates a DataLoader from the list of paths and labels.

    Args:
        paths (list): List of file paths (strings).
        labels (list): List of corresponding integer labels.
        batch_size (int): The batch size for the DataLoader.
        volume_shape (tuple): The shape of the NIfTI volumes (W, H, D).
        is_training (bool): If True, shuffle the dataset.
        seed (int): Random seed for shuffling.
        min_val (float): The minimum value used for normalization.
        max_val (float): The maximum value used for normalization.
        mask_path (str): Path to the mask file (shares the same volume shape as the input data).
                         If None, no mask is applied.
    Returns:
        DataLoader: The DataLoader ready for training/evaluation.
                     Returns None if paths list is empty.
    """
    
    if len(paths) == 0:
        print(f"No data paths provided. Cannot create DataLoader.")
        return None

    generator = None
    if is_training:
        generator = torch.Generator()
        generator.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    dataset = AdniDataset(paths=paths, labels=labels, volume_shape=volume_shape, is_training=is_training, min_val=min_val, max_val=max_val, mask_path=mask_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_training, num_workers=4)
    
    return dataloader

def extract_subject_id(filepath):
    """
    Extracts subject ID from the filename.
    Assumes the filename starts with a string 'XXX_X_XXXX' that is the subject ID.
    """
    try:
        filename = Path(filepath).name
        parts = filename.split('_')
        subject_id = '_'.join(parts[:3])
        return subject_id
    except Exception as e:
        print(f"Warning: Could not extract subject ID from {filepath}. Error: {e}. Returning full filename.")
        return filename
    
def clean_zone_identifier_files(directory):
    """
    Remove all Zone.Identifier files in the given directory and its subdirectories.
    Zone.Identifier files are sometimes created when moving files across directories.
    """
    removed_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(':Zone.Identifier'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
    if removed_count > 0:
        print(f"Removed {removed_count} Zone.Identifier files")

if __name__ == '__main__':

    # Set path for dataset directory
    NORMALIZATION = "mni_reg_CV"
    DATASET = "smci_pmci"
    DATA_DIR = Path("datasets/") / NORMALIZATION / DATASET
    ROI_MASK_PATH = None
    
    VOLUME_SHAPE = (91, 109, 91)
    seed = 10
    
    BATCH_SIZE = 4

    print("\nCreating train set...\n")
    train_paths, train_labels, class_map = get_paths_and_labels(DATA_DIR, 'train')
    train_paths = np.array(train_paths)
    train_labels = np.array(train_labels)
    
    minmax_min, minmax_max = calculate_min_max(train_paths)
    print(f"Minmax values for normalization: {minmax_min}, {minmax_max}")

    train_data = create_dataloader(
        paths=train_paths,
        labels=train_labels,
        batch_size=BATCH_SIZE,
        volume_shape=VOLUME_SHAPE,
        is_training=True, 
        seed=seed,
        min_val=minmax_min,
        max_val=minmax_max,
        mask_path=ROI_MASK_PATH
        )
    
    print("\nCreating test set...")
    test_paths, test_labels, _ = get_paths_and_labels(DATA_DIR, 'test', class_map)
    test_paths = np.array(test_paths)
    test_labels = np.array(test_labels)

    test_data = create_dataloader(
        paths=test_paths,
        labels=test_labels,
        batch_size=BATCH_SIZE,
        volume_shape=VOLUME_SHAPE,
        is_training=False, 
        seed=seed,
        min_val=minmax_min,
        max_val=minmax_max,
        mask_path=ROI_MASK_PATH
    )
    
    if train_data:
        print("\nVerifying one train batch:\n")
        volume_batch, label_batch = next(iter(train_data))
        print("Train Volume batch shape:", volume_batch.shape)
        print("Train Label batch shape:", label_batch.shape)
        print("Train Volume batch dtype:", volume_batch.dtype)
        print("Train Label batch dtype:", label_batch.dtype)
        print("Sample label:", label_batch[0].numpy())
        print(f"Min value in batch: {volume_batch.min().item()}")
        print(f"Max value in batch: {volume_batch.max().item()}")
            
    if test_data:
        print("\nVerifying one test batch:\n")
        volume_batch, label_batch = next(iter(test_data))
        print("Test Volume batch shape:", volume_batch.shape)
        print("Test Label batch shape:", label_batch.shape)
        print("Test Volume batch dtype:", volume_batch.dtype)
        print("Test Label batch dtype:", label_batch.dtype)
        print("Sample label:", label_batch[0].numpy())
        print(f"Min value in batch: {volume_batch.min().item()}")
        print(f"Max value in batch: {volume_batch.max().item()}")
    
            