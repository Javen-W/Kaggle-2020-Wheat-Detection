# Kaggle 2020 Global Wheat Detection

This repository contains my solution for the Kaggle 2020 Global Wheat Detection competition, aimed at detecting wheat heads in agricultural images to estimate crop yield. The project implements a Faster R-CNN model with a ResNet-50 backbone, fine-tuned on the competition dataset using PyTorch. It includes a custom dataset pipeline with data augmentation, training and validation loops, and visualization of predictions with bounding boxes.

## Table of Contents
- [Kaggle 2020 Global Wheat Detection](#kaggle-2020-global-wheat-detection)
  - [Project Overview](#project-overview)
  - [Approach](#approach)
  - [Tools and Technologies](#tools-and-technologies)
  - [Results](#results)
  - [Skills Demonstrated](#skills-demonstrated)
  - [Setup and Usage](#setup-and-usage)
  - [References](#references)

## Project Overview
The Kaggle 2020 Global Wheat Detection competition tasks participants with detecting wheat heads in RGB images of wheat fields, providing bounding box annotations for training. This project develops an object detection pipeline using Faster R-CNN with a ResNet-50 backbone, pre-trained on COCO, and fine-tuned on the competition’s dataset (3373 training images, 10 test images). The pipeline includes data preprocessing, augmentation, model training, validation with mean Average Precision (mAP), and prediction visualization. The goal is to accurately detect wheat heads, applicable to automated agricultural monitoring and manufacturing defect detection.

## Approach
The project is structured as follows:
- **Dataset Pipeline (`dataset.py`)**: Implements a custom `WheatDataset` class to load images and bounding box annotations from `train.csv`. Uses pandas for data parsing, splits data into train (80%) and validation (20%) sets, and applies augmentations (e.g., RandomResizedCrop, HorizontalFlip, GaussNoise) using Albumentations for training robustness.
- **Model Training (`main.py`)**: Fine-tunes a Faster R-CNN model with a ResNet-50 backbone, replacing the classifier head for binary classification (wheat vs. background). Trains for 10 epochs with SGD optimizer (lr=0.005), using a batch size of 8 and a custom collate function for variable-sized inputs.
- **Evaluation**: Computes mAP@0.5:0.75 on the validation set using torchmetrics, tracking training loss and validation performance per epoch.
- **Inference and Visualization**: Generates predictions on the test set, formats outputs as `score x_min y_min width height` strings, and visualizes bounding boxes with confidence scores using Matplotlib.
- **Data Processing**: Normalizes images to [0,1], converts to tensors, and handles bounding box coordinates in Pascal VOC format.

The model processes 1024x1024 images, leveraging augmentations to handle variations in lighting, orientation, and noise, ensuring robust detection in diverse field conditions.

## Tools and Technologies
- **Python**: Core language for data processing, model training, and inference.
- **PyTorch**: Framework for Faster R-CNN implementation and training.
- **Albumentations**: Data augmentation for robust training (e.g., flips, noise, brightness).
- **pandas**: Parsing and processing `train.csv` for bounding box annotations.
- **OpenCV**: Image loading and preprocessing.
- **Matplotlib**: Visualizing predictions with bounding boxes and scores.
- **torchmetrics**: Computing mAP for validation performance.
- **CUDA**: GPU acceleration for training and inference.

## Results
- **Training Performance**: Achieved a training loss of 0.9026 and validation mAP@0.5:0.75 of 0.6224 after 10 epochs, demonstrating effective learning and generalization.
- **Test Predictions**: Generated submission file (`submission.csv`) with bounding box predictions for 10 test images, formatted as required by Kaggle.
- **Visualization**: Produced high-quality visualizations of test predictions with confidence-scored bounding boxes, aiding interpretability for agricultural analysis.

## Skills Demonstrated
- **Object Detection**: Fine-tuned Faster R-CNN for wheat head detection, applicable to manufacturing defect identification.
- **Data Preprocessing**: Built a custom dataset pipeline with pandas and Albumentations for efficient data loading and augmentation.
- **Model Training**: Implemented training and validation loops with PyTorch, optimizing hyperparameters for robust performance.
- **Evaluation**: Used mAP@0.5:0.75 to assess model accuracy, ensuring reliable detection metrics.
- **Visualization**: Developed Matplotlib-based tools for visualizing bounding box predictions, enhancing result interpretability.
- **GPU Programming**: Leveraged CUDA for accelerated training and inference.

## Setup and Usage
1. **Prerequisites**:
   - Clone the repository: `git clone <repository-url>`
   - Install dependencies: `pip install -r requirements.txt` (requires PyTorch, pandas, OpenCV, Albumentations, torchmetrics, matplotlib).
   - Download the dataset from [Kaggle Global Wheat Detection](https://www.kaggle.com/competitions/global-wheat-detection/data) and place it in `dataset/`.
2. **Running**:
- Train the model: `python main.py` (set `train_model=True` to train, or use pre-trained weights).
- Generate test predictions: `python main.py` (outputs `output/submission.csv` and visualizations in `output/visual/`).
3. **Notes**:
- Requires a CUDA-capable GPU for optimal performance; CPU fallback available.
- Pre-trained weights can be loaded from `models/fasterrcnn_resnet50_fpn_COCO-V1.pt`.

## References
- [Kaggle Global Wheat Detection Competition](https://www.kaggle.com/competitions/global-wheat-detection)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [Albumentations Documentation](https://albumentations.ai/docs/)
- [torchmetrics Documentation](https://torchmetrics.readthedocs.io/en/stable/)


