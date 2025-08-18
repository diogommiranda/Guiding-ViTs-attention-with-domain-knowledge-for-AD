# This file contains functions for visualizing nifti images and creating plots for analysis.

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import random
import torch

def view_image_data(img_data):
    """ Displays the FDG-PET image along the saggital, coronal, and axial planes.

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

def plot_loss_curves(history, save_dir=None):
    """
    Plots loss curves for training and validation.
    Saves the plots in the directory specified by 'save_dir' if provided.
    
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
    Plots guidance loss curves for training and validation.
    Plots penalization term and reward term scores as well.
    
    Args:
        history: History object.
        save_dir: Directory to save the plots.
    """
    
    epochs = range(len(history['train_guidance_loss']))
    
    # --- Plot guidance loss curves ---
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
    
    # --- Plot penalization and reward terms ---
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
    ax.set_ylabel('Loss Value')
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
    
def plot_average_attention_scores(history, save_dir):
    """
    Plots average attention scores for ROI and non-ROI tokens during training and validation.

    Args:
        history: History object.
        save_dir: Directory to save the plots.
    """
    
    epochs = range(len(history['train_avg_att_non_roi']))
    
    fig, ax = plt.subplots(figsize=(9, 6))
    legend_handles_plot = []

    line, = ax.plot(epochs, history['train_avg_att_non_roi'], label='Train Avg non-ROI attention', linestyle='-')
    legend_handles_plot.append(line)
    line, = ax.plot(epochs, history['val_avg_att_non_roi'], label='Val Avg non-ROI attention',  linestyle='--')
    legend_handles_plot.append(line)
    line, = ax.plot(epochs, history['train_avg_att_roi'], label='Train Avg ROI attention',  linestyle='-')
    legend_handles_plot.append(line)
    line, = ax.plot(epochs, history['val_avg_att_roi'], label='Val Avg ROI attention',  linestyle='--')
    legend_handles_plot.append(line)
    
    ax.set_title(f'Average attention scores')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Average Attention Score')
    ax.legend(handles=legend_handles_plot)
    ax.grid(True)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"avg_attention_scores.png")
    try:
        fig.savefig(save_path)
    except Exception as e:
        print(f"Error saving guidance term losses: {e}")
        
    plt.close(fig)
    
def plot_correlation(history, save_dir, corr):
    """
    Plots correlation between [CLS] attention vector and Token ROI Scores for training and validation.

    Args:
        history: History object.
        save_dir: Directory to save the plots.
        corr: Correlation type ('pearson' or 'spearmanr').
    """
    
    if corr == 'pearson':
        corr_name = 'Pearson Correlation'
        train_corr_values = history['train_pearson_corr']
        val_corr_values = history['val_pearson_corr']
    elif corr == 'spearmanr':
        corr_name = 'Spearmanr Correlation'
        train_corr_values = history['train_spearmanr_corr']
        val_corr_values = history['val_spearmanr_corr']
    else:
        raise ValueError(f"Unsupported correlation type: {corr}. Supported types are 'pearson' and 'spearmanr'.")
    
    epochs = range(len(train_corr_values))

    fig = plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_corr_values, label=f'Train {corr_name}')
    plt.plot(epochs, val_corr_values, label=f'Val {corr_name}')
    
    plt.title(f'{corr_name} of CLS Tokens and ROI scores')
    plt.xlabel('Epochs')
    plt.legend()
    
    if save_dir:
        save_path = os.path.join(save_dir, f"{corr}.png")
        try:
            plt.savefig(save_path)
            plt.close(fig)
        except Exception as e:
            print(f"Error saving loss curves: {e}")
    
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
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] 
    n_classes = cm.shape[0] 

    # Plot the figure
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(n_classes), 
            yticks=np.arange(n_classes),
            xticklabels=labels,
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