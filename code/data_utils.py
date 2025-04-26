# This file contains functions to load NIfTI files and create TensorFlow datasets for training and testing.

import tensorflow as tf
import nibabel as nib
import numpy as np
from pathlib import Path
import os

AUTOTUNE = tf.data.AUTOTUNE

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
        if not train_path.exists() or not train_path.is_dir():
             raise FileNotFoundError(f"Train directory '{train_path}' not found for class inference.")

        # Get class names from the train directory for consistency
        class_names = sorted([p.name for p in train_path.iterdir() if p.is_dir()], reverse=True)
        if not class_names:
            raise ValueError(f"No class subdirectories found in {train_path}")
        class_map = {name: i for i, name in enumerate(class_names)}
        print(f"Class map: {class_map}")
    else:
         print(f"Class map: {class_map}")

    paths = []
    labels = []

    if not split_path.exists() or not split_path.is_dir():
        print(f"Warning: Directory for split '{split}' not found at '{split_path}'. Returning empty lists.")
        return paths, labels, class_map 

    print(f"Scanning {split_path}...")
    for class_name, label_index in class_map.items():
        class_dir = split_path / class_name
        if class_dir.is_dir():
            # Get the nifti files 
            nifti_files = list(class_dir.glob('*.nii'))
            if nifti_files:
                print(f"Found {len(nifti_files)} files for class '{class_name}'")
                paths.extend([str(p) for p in nifti_files])
                labels.extend([label_index] * len(nifti_files))
            else:
                print(f"No *.nii files found for class '{class_name}' in split '{split}'")
        else:
             print(f"Directory for class '{class_name}' not found in '{split}' split.")

    if not paths:
         print(f"Warning: No *.nii files found in any class subdirectories within '{split_path}'.")

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

    print(f"Calculating minmax across {len(paths)} files...")
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


def create_dataset(paths, labels, batch_size, volume_shape, is_training, seed, min_val, max_val, mask_path=None):
    """
    Creates a tf.data.Dataset from the list of paths and labels.

    Args:
        paths (list): List of file paths (strings).
        labels (list): List of corresponding integer labels.
        batch_size (int): The batch size for the dataset.
        volume_shape (tuple): The shape of the NIfTI volumes (W, H, D).
        is_training (bool): If True, shuffle the dataset.
        seed (int): Random seed for shuffling.
        min_val (float): The minimum value used for normalization.
        max_val (float): The maximum value used for normalization.
        mask_path (str): Path to the mask file (shares the same volume shape as the input data).
                         If None, no mask is applied.
    Returns:
        tf.data.Dataset: The dataset ready for training/evaluation.
                         Returns None if paths list is empty.
    """
    
    if len(paths) == 0:
        print(f"No data paths provided. Cannot create dataset.")
        return None

    min_val_tf = tf.constant(min_val, dtype=tf.float32)
    max_val_tf = tf.constant(max_val, dtype=tf.float32)
    range_val_tf = max_val_tf - min_val_tf
    
    def load_nifti(path_str):
        """Loads NIfTI volume as float32."""
        try:
            img = nib.load(path_str)
            volume = img.get_fdata(dtype=np.float32) # Load as float32

            if volume.ndim == 3:
                volume = np.expand_dims(volume, axis=-1)
            elif volume.ndim != 4:
                raise ValueError(f"Unexpected shape: {volume.ndim} dimensions for file {path_str}. Expected 3.")

            return volume

        except Exception as e:
            print(f"Error loading NIfTI file {path_str}: {e}")
            raise e    
    
    # Create mask if provided
    roi_mask = None
    expected_mask_shape = (volume_shape[0], volume_shape[1], volume_shape[2], 1)
    if mask_path is not None:
        mask_volume = load_nifti(mask_path)
        if mask_volume.shape != expected_mask_shape:
            raise ValueError(f"Mask shape {mask_volume.shape} does not match expected shape {expected_mask_shape}.")
        roi_mask = tf.constant(mask_volume, dtype=tf.float32)

    def load_nifti_wrapper(path_bytes):
        """Loads NIfTI volume as float32."""
        path_str = path_bytes.numpy().decode('utf-8')
        volume = load_nifti(path_str)
        return volume

    def process_path(path, label):
        """Performs map loading, minmax normalization, shape setting and label casting."""
        volume = tf.py_function(
            func=load_nifti_wrapper,
            inp=[path],
            Tout=tf.float32
        )
        # Minmax normalization
        volume_normalized = (volume - min_val_tf) / range_val_tf
        
        # Apply the mask to the volume
        if roi_mask is not None:
            print(f"\nApplying mask from {mask_path}\n")
            volume_normalized = tf.multiply(volume_normalized, roi_mask)
        else:
            print(f"\nNo mask applied.\n")
            
        # Set shape
        expected_shape = (volume_shape[0], volume_shape[1], volume_shape[2], 1)
        volume_normalized.set_shape(expected_shape)

        # Cast label
        label = tf.cast(label, tf.int32)

        return volume_normalized, label

    # Create the initial dataset
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

    # Shuffle if training
    if is_training:
        BUFFER_SIZE = len(paths)
        print(f"Shuffling with buffer size: {BUFFER_SIZE}")
        ds = ds.shuffle(buffer_size=BUFFER_SIZE, seed=seed, reshuffle_each_iteration=True)

    # Map the loading and processing function
    ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)

    # Batch and Prefetch
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds

def clean_zone_identifier_files(directory):
    """Remove all Zone.Identifier files in the given directory and its subdirectories.
    Zone.Identifier files are created in Linux when moving files from Windows.
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
    
    print(f"Removed {removed_count} Zone.Identifier files")

if __name__ == '__main__':

    # Set path for dataset directory
    NORMALIZATION = "mni_reg"
    DATASET = "smci_pmci"
    DATA_DIR = Path("datasets/") / NORMALIZATION / DATASET
    ROI_MASK_PATH = str(Path("/home/diogommiranda/tese/masks/ROI_MASK.nii"))

    # Set batch size
    BATCH_SIZE = 4
    
    # Expected volume shape of PET file
    VOLUME_SHAPE = (91, 109, 91)
    
    # Random seed for shuffling
    seed = 42
    
    train_paths, train_labels, class_map = get_paths_and_labels(DATA_DIR, 'train')
    train_paths = np.array(train_paths)
    train_labels = np.array(train_labels)
    
    # Calculate minmax values for train set
    minmax_min, minmax_max = calculate_min_max(train_paths)
    print(f"Minmax values for normalization: {minmax_min}, {minmax_max}")

    # Create train dataset
    print("\n--- Creating Train Dataset ---")
    train_dataset = create_dataset(
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
    
    # Create validation dataset
    print("\n--- Creating Validation Dataset ---")
    validation_paths, validation_labels, _ = get_paths_and_labels(DATA_DIR, 'validation', class_map)
    validation_paths = np.array(validation_paths)
    validation_labels = np.array(validation_labels)
    
    validation_dataset = create_dataset(
        paths=validation_paths,
        labels=validation_labels,
        batch_size=BATCH_SIZE,
        volume_shape=VOLUME_SHAPE,
        is_training=False, 
        seed=seed,
        min_val=minmax_min, 
        max_val=minmax_max,
        mask_path=ROI_MASK_PATH
    )

    # Create test dataset
    print("\n--- Creating Test Dataset ---")
    test_paths, test_labels, _ = get_paths_and_labels(DATA_DIR, 'test', class_map)
    test_paths = np.array(test_paths)
    test_labels = np.array(test_labels)

    test_dataset = create_dataset(
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
    
    # Verify one batch if datasets were created
    if train_dataset:
        print("\n--- Verifying one train batch ---")
        for volume_batch, label_batch in train_dataset.take(1):
            print("Train Volume batch shape:", volume_batch.shape)
            print("Train Label batch shape:", label_batch.shape)
            print("Train Volume batch dtype:", volume_batch.dtype)
            print("Train Label batch dtype:", label_batch.dtype)
            print("Sample label:", label_batch[0].numpy()) 
            
    if validation_dataset:
        print("\n--- Verifying one validation batch ---")
        for volume_batch, label_batch in validation_dataset.take(1):
            print("Validation Volume batch shape:", volume_batch.shape)
            print("Validation Label batch shape:", label_batch.shape)
            print("Validation Volume batch dtype:", volume_batch.dtype)
            print("Validation Label batch dtype:", label_batch.dtype)
            print("Sample label:", label_batch[0].numpy()) 

    if test_dataset:
        print("\n--- Verifying one test batch ---")
        for volume_batch, label_batch in test_dataset.take(1):
            print("Test Volume batch shape:", volume_batch.shape)
            print("Test Label batch shape:", label_batch.shape)
            print("Test Volume batch dtype:", volume_batch.dtype)
            print("Test Label batch dtype:", label_batch.dtype)
            print("Sample label:", label_batch[0].numpy())
            
    print(len(train_dataset), len(validation_dataset), len(test_dataset))
