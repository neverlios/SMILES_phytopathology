# Tomato Plant Disease Detection Project

## Project Overview
This project focuses on **automated detection of phytopathologies** in tomato plants using computer vision. Leveraging a substantial real-life dataset of tomato plant images, we aim to develop robust deep learning models capable of identifying disease symptoms from camera captures. The solution will assist agricultural specialists in early disease diagnosis and crop management.

## Key Objectives
1. Analyze and preprocess the tomato plant image dataset
2. Implement state-of-the-art object detection models
3. Compare model performance across architectures

## Dataset Characteristics
- **Type**: Real-world tomato plant images
- **Content**: Healthy and diseased plant specimens
- **Annotations**: Bounding boxes for disease symptoms (leaf spots, blight, etc.)
- **Challenges**: Variable lighting, field conditions

## Technical Approach
### Phase 1: Data Preparation
- Perform exploratory data analysis (EDA)
- Implement data augmentation pipeline
- Split dataset (train/validation/test)
- Handle class imbalance

### Phase 2: Model Development
**Primary Model**  
`YOLOv8 Implementation`:
- Transfer learning with COCO pretrained weights
- Hyperparameter tuning using evolutionary algorithms

**Comparative Models** (State-of-the-art Alternatives):
- Faster R-CNN with ResNet-101 backbone
- DETR (Detection Transformer)
- etc.

### Phase 3: Evaluation
- Metrics: mAP@0.5, Precision-Recall curves, F1-score
- Error analysis: Confusion matrices, false positive inspection

## Expected Deliverables
1. Trained detection models (PyTorch)
2. Comprehensive performance comparison report
3. Inference pipeline with preprocessing module
