# This file contains functions for visualizing nifti images and plotting ML-related metrics.

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import random
import torch

def view_image_data(img_data):
    """Views a 3D image data

    Args:
        img_data (float): 3D Image data
    """
    # We assume the 3D image data is in the first 3 dimensions if it is a 4D tensor
    if len(img_data.shape) in (3, 4):
        if len(img_data.shape) == 4:
            img_data = img_data.squeeze(dim=0)
        # Transpose to X, Y, Z axis
        img_data = img_data.permute(2, 1, 0)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Get middle slices
        x_mid = (img_data.shape[0]) // 2
        y_mid = (img_data.shape[1]) // 2 
        z_mid = (img_data.shape[2]) // 2 
         
        # Display the slices
        axes[0].imshow(np.rot90(img_data[x_mid, :, :]), cmap='gray')
        axes[0].set_title(f'Sagittal Slice (X={x_mid})')
        axes[0].axis('off')
        
        axes[1].imshow(np.rot90(img_data[:, y_mid, :]), cmap='gray')
        axes[1].set_title(f'Coronal Slice (Y={y_mid})')
        axes[1].axis('off')
        
        axes[2].imshow(np.rot90(img_data[:, :, z_mid]), cmap='gray')
        axes[2].set_title(f'Axial Slice (Z={z_mid})')
        axes[2].axis('off')
        
        plt.tight_layout()
    else:
        print("Image is not 3D. Shape:", img_data.shape)

def view_image(dataset_dir, subject_id, date):
    """Views an image given its subject ID and date

    Args:
        dataset_dir (str): Directory of the dataset.
        subject_id (str): Subject ID.
        date (str): Date of the scan.

    Returns:
        tuple: (numpy array of the image data, nifti image) or (None, None) if not found/error.
    """
    filename_prefix = f"{subject_id}_{date}_"
    found_path = None
    found_filename = None
    target_class = None

    try:
        # Walk through the directory tree
        for root, dirs, files in os.walk(dataset_dir):
            for filename in files:
                if filename.startswith(filename_prefix):
                    found_path = os.path.join(root, filename)
                    found_filename = filename
                    target_class = os.path.basename(root)
            if found_path:
                 break
        else:
             print(f"Error: No image found in '{dataset_dir}' matching prefix '{filename_prefix}'")
             return None, None
    except FileNotFoundError:
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return None, None
    except Exception as e:
        print(f"Error during directory search in {dataset_dir}: {e}")
        return None, None

    if found_path:
        nifti_img = nib.load(found_path)
        img_data = nifti_img.get_fdata()
        
        # Print some information about the image
        print(f"Image path: {found_path}")
        print(f"Image name: {found_filename}")
        print(f"Image class: {target_class}")
        print(f"Image shape: {img_data.shape}")
        print(f"ImageData Type: {img_data.dtype}")
        print(f"Value Range: [{np.min(img_data)}, {np.max(img_data)}]")

        # Display a middle slice frome each dimension
        # We assume the 3D image data is in the first 3 dimensions if it is a 4D tensor
        if len(img_data.shape) in (3, 4):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
            # Get middle slices
            x_mid = img_data.shape[0] // 2
            y_mid = img_data.shape[1] // 2
            z_mid = img_data.shape[2] // 2

            
            
            # Display the slices
            axes[0].imshow(np.rot90(img_data[x_mid, :, :]), cmap='gray')
            axes[0].set_title(f'Sagittal Slice (X={x_mid})')
            axes[0].axis('off')
        
            axes[1].imshow(np.rot90(img_data[:, y_mid, :]), cmap='gray')
            axes[1].set_title(f'Coronal Slice (Y={y_mid})')
            axes[1].axis('off')
        
            axes[2].imshow(np.rot90(img_data[:, :, z_mid]), cmap='gray')
            axes[2].set_title(f'Axial Slice (Z={z_mid})')
            axes[2].axis('off')
        
            plt.suptitle(f"{target_class}: {found_filename}")
            plt.tight_layout()
        else:
            print("Image is not 3D. Shape:", img_data.shape)

        return img_data, nifti_img
    else:
        print(f"Error: No image found matching prefix '{filename_prefix}'") # Fallback message
        return None, None
    
def view_random_image(target_dir, random_arg):
    """Views a random image from a specified dataset.
    This function can either select a random image from a specific class or from a specific subject ID.

    Args:
        target_dir (str): Root directory of the dataset.
        random_arg (str): Class name ("smci", "pmci", "ad", "nc") or subject ID ("XXX_S_XXXX").

    Returns:
        tuple: (numpy array of the image data, nifti image) or (None, None) if no image found.
    """
    
    if random_arg in ["smci", "pmci", "ad", "nc"]:
        category = "class"
    elif "_S_" in random_arg:
        category = "subject"
    else:
        print(f"Error: Invalid random_arg '{random_arg}'. Must be a class or subject ID.")
        return None, None
        
    if category == "class":
        # Select a random image from the specified class
        target_folder = target_dir+random_arg
        img_class = random_arg
        random_image = random.choice(os.listdir(target_folder))
        img_path = os.path.join(target_folder, random_image)    
    else:
        # Search for subject_id
        subject_files = []
        for root, dirs, files in os.walk(target_dir):
            for filename in files:
                # Save files that match the subject ID
                if random_arg in filename:
                    subject_files.append(os.path.join(root, filename))
            if subject_files:
                break
        if not subject_files:
            print(f"Error: No images found for subject ID: {random_arg}.")
            return None, None

        # Choose a file randomly from the found subject files
        img_path = random.choice(subject_files)
        random_image = os.path.basename(img_path)
        img_class = os.path.basename(os.path.dirname(img_path))
        

    # Load the nifti image
    nifti_img = nib.load(img_path)
    # Convert the image to a numpy array
    img_data = nifti_img.get_fdata()
  
    # Print some information about the image
    print(f"Image path: {img_path}") 
    print(f"Image name: {random_image}")
    print(f'Image class: {img_class}')
    print(f"Image shape: {img_data.shape}")
    print(f"Data type: {img_data.dtype}")
    print(f"Value range: [{np.min(img_data)}, {np.max(img_data)}]\n")

    # Display a middle slice from each dimension
    # We assume the 3D image data is in the first 3 dimensions if it is a 4D tensor
    if len(img_data.shape) in (3, 4):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Get middle slices 
        x_mid = img_data.shape[0] // 2
        y_mid = img_data.shape[1] // 2
        z_mid = img_data.shape[2] // 2
         
        # Display the slices
        axes[0].imshow(np.rot90(img_data[x_mid, :, :]), cmap='gray')
        axes[0].set_title(f'Sagittal Slice (X={x_mid})')
        axes[0].axis('off')
        
        axes[1].imshow(np.rot90(img_data[:, y_mid, :]), cmap='gray')
        axes[1].set_title(f'Coronal Slice (Y={y_mid})')
        axes[1].axis('off')
        
        axes[2].imshow(np.rot90(img_data[:, :, z_mid]), cmap='gray')
        axes[2].set_title(f'Axial Slice (Z={z_mid})')
        axes[2].axis('off')
        
        plt.suptitle(f"{random_image}")
        plt.tight_layout()
    else:
        print("Image is not 3D. Shape:", img_data.shape)
    
    return img_data, nifti_img

def plot_loss_curves(history, save_dir=None):
    """
    Plots separate loss curves for training and validation metrics.
    Saves the plots in the directory specified by `save_dir` if provided.
    
    Args:
        history: History object from model training.
        save_dir: Directory to save the plots. If None, plots are not saved.
    """
    
    loss = history['train_loss']
    accuracy = history['train_accuracy']
    
    val_loss = None
    val_accuracy = None
    
    if 'val_loss' in history:
        val_loss = history['val_loss']
    if 'val_accuracy' in history:
        val_accuracy = history['val_accuracy']

    epochs = range(len(history['train_loss']))
    
    # Plot loss
    fig_losses = plt.figure(figsize=(9, 6))
    plt.plot(epochs, loss, label='training_loss')
    if val_loss:
        plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    if save_dir:
        save_path = os.path.join(save_dir, "loss_curves.png")
        try:
            plt.savefig(save_path)
            plt.close(fig_losses)
        except Exception as e:
            print(f"Error saving loss curves: {e}")

    # Plot accuracy
    fig_accuracies = plt.figure(figsize=(9, 6))
    plt.plot(epochs, accuracy, label='train_accuracy')
    if val_accuracy:
        plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
    if save_dir:
        save_path = os.path.join(save_dir, "accuracy_curves.png")
        try:
            plt.savefig(save_path)
            plt.close(fig_accuracies)
        except Exception as e:
            print(f"Error saving accuracy curves: {e}")
            
def plot_guidance_losses(history, save_dir):
    """
    Plots guidance-related losses.
    
    Args:
        history: History object.
        save_dir: Directory to save the plots.
    """
    
    epochs = range(len(history['train_guidance_loss']))
    
    # --- Plot guidance loss ---
    fig, ax = plt.subplots(figsize=(9, 6))
    legend_handles_plot = []

    line, = ax.plot(epochs, history['train_guidance_loss'], label=f'Train guidance loss')
    legend_handles_plot.append(line)
    line, = ax.plot(epochs, history['val_guidance_loss'], label=f'Val guidance loss')
    legend_handles_plot.append(line)

    ax.set_title(f'Guidance Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss Value')
    ax.legend(handles=legend_handles_plot)
    ax.grid(True)

    save_path = os.path.join(save_dir, f"guidance_loss_curve.png")
    try:
        fig.savefig(save_path)
    except Exception as e:
        print(f"Error saving guidance loss: {e}")
        
    plt.close(fig)
    
    # --- Plot penalization term and reward term scores ---
    fig, ax = plt.subplots(figsize=(9, 6))
    legend_handles_plot = []

    line, = ax.plot(epochs, history['train_penalization_term_loss'], label='Train penalization term', linestyle='-')
    legend_handles_plot.append(line)
    line, = ax.plot(epochs, history['val_penalization_term_loss'], label='Val penalization term',  linestyle='--')
    legend_handles_plot.append(line)
    line, = ax.plot(epochs, history['train_reward_term_loss'], label='Train reward term',  linestyle='-')
    legend_handles_plot.append(line)
    line, = ax.plot(epochs, history['val_reward_term_loss'], label='Val reward term',  linestyle='--')
    legend_handles_plot.append(line)
    
    ax.set_title(f'Penalization term and Reward term')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Average Attention Score')
    ax.legend(handles=legend_handles_plot)
    ax.grid(True)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"term_losses_curves.png")
    try:
        fig.savefig(save_path)
    except Exception as e:
        print(f"Error saving guidance term losses: {e}")
        
    plt.close(fig)
    
    
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, save_dir=None):
    """Makes a labelled confusion matrix comparing predictions and ground truth labels.

    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    Args:
        y_true: Array of truth labels (must be same shape as y_pred).
        y_pred: Array of predicted labels (must be same shape as y_true).
        classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
        figsize: Size of output figure (default=(10, 10)).
        text_size: Size of output figure text (default=15).

    Returns:
        A labelled confusion matrix plot comparing y_true and y_pred.
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(n_classes), # create enough axis slots for each class
            yticks=np.arange(n_classes),
            xticklabels=labels, # axes will labeled with class names (if they exist) or ints
            yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)
    
    # Save the figure if a save directory is provided
    if save_dir:
        save_path = os.path.join(save_dir, "cm.png")
        try:
            plt.savefig(save_path)
            plt.close(fig)
        except Exception as e:
            print(f"Error saving loss curves: {e}")