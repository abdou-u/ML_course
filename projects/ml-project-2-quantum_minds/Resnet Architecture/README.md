# ResNet Architecture for Road Segmentation

This project implements a U-Net-like segmentation model with a ResNet-50 encoder, tailored for road segmentation tasks from satellite imagery.

## Directory Structure

- **`data/predict_ai_crowd`**: Contains AI Crowd test images for prediction.
- **`epfl_daugmented`**: Folder containing the best-augmented dataset for training and validation.
- **`Resnet_notebook`**: A notebook for training the model and generating the submission file.
- **Python Files**:
  - **`dataset.py`**: Handles data loading and preprocessing.
  - **`model.py`**: Defines the ResNet-based U-Net segmentation model.
  - **`train.py`**: Contains the training loop and evaluation functions.
  - **`mask_to_submission.py`**: Converts predicted masks into the submission format for AI Crowd.

## Key Features

- **Model**: ResNet-50 encoder integrated into a U-Net architecture for multi-scale feature extraction and accurate segmentation.
- **Data Augmentation**: Includes techniques for improving model robustness and generalization.
- **Evaluation**: Calculates performance metrics such as F1-score and generates submission files for AI Crowd.
- **Framework**: Built with PyTorch for flexibility and scalability.

## Steps to Use

1. **Dataset Preparation**:
   - Place the training dataset in the `epfl_daugmented` folder.
   - Place test images for prediction in the `data/predict_ai_crowd` folder.

2. **Training**:
   - Use the `Resnet_notebook` to train the model and save the best-performing weights.

3. **Prediction**:
   - Use the `train.py` or `Resnet_notebook` to generate predictions for test images.

4. **Submission**:
   - Run `mask_to_submission.py` to create the submission file in the required format for AI Crowd.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- PIL