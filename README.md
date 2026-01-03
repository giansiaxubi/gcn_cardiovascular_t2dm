# Interpretable GCN for Cardiovascular Disease Risk Prediction in T2DM

This repository provides the official PyTorch implementation of the core model and loss function for the paper: **"Interpretable Graph Convolutional Networks for cardiovascular disease risk prediction in patients with Type 2 Diabetes Mellitus"**.

**Paper:** [Link to be added upon publication]

## Overview

This research introduces a novel framework for CVD risk prediction that reframes the imbalanced classification task as a supervised graph anomaly detection problem. The key components, implemented in this repository, are:

1.  A **Graph Convolutional Network (GCN)** architecture adapted for learning patient embeddings from a population graph.
2.  A custom **hypersphere-based loss function** designed to learn a compact representation for the majority (non-CVD) class and identify high-risk patients as anomalies.

This repository provides these two components as a reference for researchers who wish to understand, reproduce, or apply our core methodology to other datasets.

## Repository Structure

*   `model.py`: Contains the `GCN_Anomaly` class, a two-layer Graph Convolutional Network that takes patient features and graph structure as input to generate embeddings.
*   `loss_functions.py`: Implements the `AnomalyDetectionLoss`, which combines a compactness term for normal nodes and a discriminative AUC approximation term to separate anomalous nodes.

**Note:** This repository contains the core algorithmic components of our work. It does not include data preprocessing, graph construction, or training/evaluation scripts, as these are highly specific to the private clinical dataset used in the study. The provided code is designed to be integrated into a user's own data pipeline.
