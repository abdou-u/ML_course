# Road Segmentation from Satellite Imagery

## Team Members

- [Ahmed Abdelmalek](https://people.epfl.ch/ahmed.abdelmalek)  
- [Nestor Lomba](https://people.epfl.ch/nestor.lombalomba)  
- [Khaled Miraoui](https://people.epfl.ch/mohamed.miraoui)  

---

## Datasets
The datasets used, as well as the augmented datasets and the models that are too big to fit in github are available here:
https://drive.google.com/drive/folders/1yLp-PiNDty-TXF4iqJktIC0VvGJXrRid?usp=sharing

## Instructions to generate the submission files

After having downloaded the models and the datasets, the final submission.csv can be generated running the run.ipynb notebook

## Project Overview

This project aims to extract road networks from satellite imagery using machine learning techniques, specifically **semantic segmentation**. In this task, each pixel in an input image is classified as either:

- **Road (1)**
- **Background (0)**

The dataset used for training and testing is provided by the [EPFL ML Road Segmentation Challenge](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation). It includes:
- **Satellite Images**: High-resolution aerial imagery.
- **Ground-Truth Masks**: Pixel-wise annotations indicating road and non-road regions.

The primary goal is to develop and compare robust deep learning architectures capable of achieving the **highest F1 score** on the test set.

---

## Repository Structure

This repository contains scripts, models, and utilities to support the entire workflow of data preparation, model training, evaluation, and submission generation:

### **Main Components**
- **Data Preprocessing**:
  - Scripts to crop, normalize, and augment images for training.
  - Utilities to convert predicted masks to submission format (`mask_to_submission.py`).

- **Model Implementations**:
  - **RFE-LinkNet**: A segmentation model that focuses on extracting more robust and fine-grained spatial details for segmentation.
  - **ResNet-based Model**: A ResNet backbone adapted for semantic segmentation.
  - **DeepLabV3**: An advanced segmentation model incorporating atrous spatial pyramid pooling.

- **Training Pipelines**:
  - Customized training loops with support for **Dice loss**, **cross-entropy loss**, and data augmentation.

- **Evaluation**:
  - Tools for evaluating model predictions using F1 score, IoU (Intersection over Union), and accuracy.

- **Submission Scripts**:
  - Generates predictions compatible with the EPFL AIcrowd competition format.

---

## Workflow Overview

### 1. **Data Preprocessing**
- Images and ground-truth masks are resized to a uniform resolution.
- The dataset is patched into smaller tiles (e.g., 512x512) for efficient training and inference.
- Augmentations such as rotations, flips, and normalization are applied during training.

### 2. **Model Development**
Our project implements and evaluates the three deep learning architectures mentioned earlier:

### 3. **Training and Validation**
- Models were trained on the augmented training dataset and validated using a dedicated validation set.
- Training pipelines include:
  - **Loss Functions**: Dice Loss and Cross-Entropy Loss.
  - **Optimizers**: Adam optimizer with learning rate scheduling.
  - **Metrics**: F1 Score, IoU, and Accuracy.

### 4. **Inference and Evaluation**
- The trained models are applied to the test dataset, which consists of unseen satellite images.
- Predictions are stitched back together from patches to reconstruct full-resolution masks.
- Results are evaluated locally using the F1 score before submission.

### 5. **Submission Generation**
- Predictions are saved as binary masks.
- The `mask_to_submission.py` script converts masks into a submission-ready `.csv` format, which can be uploaded to the AIcrowd platform.

---

## Results

The models were evaluated on the private test set of the EPFL challenge:
- **Primary Metric**: **F1 Score** (balancing precision and recall).
- Comparative performance of all three architectures is presented in the final report.

---

## Evaluation Metric

The **F1 Score** is the key evaluation metric used to assess the quality of predictions

---

## Acknowledgment

This project was developed as part of the **EPFL Machine Learning Course**. Special thanks to the teaching assistants and course instructors for their guidance.

---

## Disclaimer

This repository and its contents are intended for educational purposes only. All satellite images are sourced from the EPFL Road Segmentation Challenge.
