## Guiding Vision Transformers' Attention with Domain Knowledge for Alzheimer's Disease Diagnosis
----
# Purpose

Predicting conversion from Mild Cognitive Impairment (MCI) to Alzheimer’s Dementia is the most challenging task in Alzheimer’s Disease (AD) diagnosis. Correctly identifying which patients will progress to AD in the near future remains a challenge for even the most experienced doctors. While conventional Deep Learning (DL) models applied to neuroimaging data have shown great success in diagnosing AD, the more modern Vision Transformers (ViTs) have been proven to outperform them in natural imaging analysis tasks when trained on large amounts of data. Despite their promising results, their application
to the medical domain and AD diagnosis has been greatly constrained by the limited number of medical-labeled data to effectively train these models. Integrating domain knowledge into DL models is a technique that mitigates the limitations of small-sized medical datasets and improves the performance of DL
models. This work explores the integration of brain regions of interest (ROIs), known to have clinical significance for AD diagnosis, as domain knowledge into ViT-based networks for predicting conversion to AD. We introduce a Hybrid ViT and a pure ViT architecture, demonstrating that integrating domain knowledge compensates for the lack of inductive biases in ViTs trained on limited-data scenarios and increases their performance. The models are trained and evaluated on FDG-PET neuroimaging data from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset.
----
# Contents

This repository contains the following modules:
- models3D.py: Defines model architectures -> resnet extractor, ViT, hybrid ViT.
- resnets.ipynb: Trains resnets.
- vit_baselines.ipynb: Trains ViT and hybrid ViT without guidance.
- roi_guidance.ipynb: Trains ViT and hybrid ViT with guidance.
- data_utils_torch.py: Defines helper functions and classes for loading data, preprocessing, data augmentation and splitting subjects.
- plotting_utils_torch.py: Defines helper functions for visualizing data and plotting graphics.
- analyse_mask.ipynb: Notebook for performing EDA on the ROI Mask.
- visualize_attention.ipynb: Notebook to create and visualize normalized attention maps.
