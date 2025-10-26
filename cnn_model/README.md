# ResNet-CBAM SAR Landcover Classification

This folder contains the complete CNN model implementation for SAR (Synthetic Aperture Radar) landcover classification using ResNet50 with Convolutional Block Attention Module (CBAM).

## Model Architecture

The model combines:
- **ResNet50**: Pre-trained backbone modified for 2-channel SAR input (VV, VH polarization)
- **CBAM**: Convolutional Block Attention Module for enhanced feature learning
- **Custom Classification Head**: 5-class landcover classification

## Landcover Classes
1. Bare_Soil
2. Forest  
3. Water
4. Snow_Ice
5. Grassland

## Files Structure

### Core Model Files
- `model.py` - ResNet-CBAM model architecture implementation
- `dataset.py` - SAR dataset loading and preprocessing utilities
- `config.py` - Configuration management for all components

### Training & Evaluation
- `train.py` - Complete training pipeline with logging and checkpointing
- `evaluate.py` - Model evaluation with metrics and visualizations
- `inference.py` - Inference script for new SAR images

### Utilities
- `test_model.py` - Model testing and validation utilities
- `example_usage.py` - Example usage demonstrations

## Quick Start

### 1. Training
```bash
python train.py --data_dir dataset --epochs 30 --batch_size 32
```

### 2. Evaluation
```bash
python evaluate.py --checkpoint_path checkpoints/resnet_cbam_sar.pth --data_dir dataset
```

### 3. Inference
```bash
python inference.py --model_path checkpoints/resnet_cbam_sar.pth --input path/to/sar/images
```

## Requirements

- PyTorch >= 1.8.0
- torchvision
- rasterio (for SAR image loading)
- scikit-learn
- matplotlib
- seaborn
- tqdm
- tensorboard

## Model Features

- **Attention Mechanism**: CBAM modules enhance feature representation
- **SAR-Optimized**: Designed specifically for dual-polarization SAR data
- **Robust Training**: Includes early stopping, learning rate scheduling, class weighting
- **Comprehensive Evaluation**: Confusion matrices, Grad-CAM visualizations, classification reports
- **Production Ready**: Inference pipeline for batch processing

## Configuration

The `config.py` file provides multiple pre-configured setups:
- `default`: Standard training configuration
- `quick_test`: Fast testing with reduced parameters
- `production`: Optimized for production training
- `research`: Extensive logging and evaluation

## Author
Kiro AI Assistant