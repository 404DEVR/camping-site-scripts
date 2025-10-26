"""
SAR Dataset and Data Loading Utilities

This module provides dataset classes and data loading utilities for SAR landcover
classification with proper augmentation and preprocessing.

Author: Kiro AI Assistant
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import rasterio
import numpy as np
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
from collections import Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SARDataset(Dataset):
    """Custom Dataset for SAR landcover classification"""
    
    def __init__(self, image_paths, labels, transform=None, class_names=None):
        """
        Args:
            image_paths (list): List of paths to SAR image files
            labels (list): List of corresponding class labels (integers)
            transform (callable, optional): Optional transform to be applied
            class_names (list): List of class names for reference
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_names = class_names or []
        
        # Validate inputs
        assert len(image_paths) == len(labels), "Number of images and labels must match"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load SAR image
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load using rasterio for TIFF files
            with rasterio.open(image_path) as src:
                # Read all bands (should be 2 for VV, VH)
                image = src.read()  # Shape: (channels, height, width)
                
                # Ensure we have exactly 2 channels
                if image.shape[0] != 2:
                    raise ValueError(f"Expected 2 channels, got {image.shape[0]} in {image_path}")
                
                # Convert to float32 and normalize
                image = image.astype(np.float32)
                
                # Handle potential NaN or infinite values
                image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
                
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return zeros as fallback
            image = np.zeros((2, 32, 32), dtype=np.float32)
        
        # Convert to PIL Image format for transforms (if needed)
        # For now, work directly with numpy arrays
        image = torch.from_numpy(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class SARTransforms:
    """SAR-specific data augmentation and preprocessing transforms"""
    
    @staticmethod
    def get_train_transforms():
        """Get training transforms with augmentation"""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, fill=0),
            SARNormalize(),
        ])
    
    @staticmethod
    def get_val_transforms():
        """Get validation/test transforms without augmentation"""
        return transforms.Compose([
            SARNormalize(),
        ])


class SARNormalize:
    """Custom normalization for SAR data"""
    
    def __init__(self, mean=None, std=None):
        # Default SAR normalization values (can be computed from dataset)
        self.mean = mean or [0.0, 0.0]  # Per channel mean
        self.std = std or [1.0, 1.0]    # Per channel std
    
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): SAR image tensor of shape (C, H, W)
        Returns:
            Tensor: Normalized tensor
        """
        if tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {tensor.dim()}D")
        
        # Normalize each channel
        for i in range(tensor.size(0)):
            tensor[i] = (tensor[i] - self.mean[i]) / self.std[i]
        
        return tensor


def load_dataset_from_folders(data_dir, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load dataset from folder structure: data_dir/{train,test}/{class_name}/*.tif
    
    Args:
        data_dir (str): Root directory containing train and test folders
        test_size (float): Proportion of data for testing (if no separate test folder)
        val_size (float): Proportion of training data for validation
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing train, val, test datasets and class info
    """
    
    # Define class names (adjust based on your dataset)
    class_names = ['Bare_Soil', 'Forest', 'Water', 'Snow_Ice', 'Grassland']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Check if separate train/test folders exist
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        # Load from separate train/test folders
        train_paths, train_labels = load_images_from_folder(train_dir, class_to_idx)
        test_paths, test_labels = load_images_from_folder(test_dir, class_to_idx)
        
        # Split training data into train/val
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=val_size, 
            random_state=random_state, stratify=train_labels
        )
        
    else:
        # Load all data and split
        all_paths, all_labels = load_images_from_folder(data_dir, class_to_idx)
        
        # First split: separate test set
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            all_paths, all_labels, test_size=test_size, 
            random_state=random_state, stratify=all_labels
        )
        
        # Second split: separate validation from training
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=train_val_labels
        )
    
    # Print dataset statistics
    logger.info(f"Dataset loaded successfully!")
    logger.info(f"Classes: {class_names}")
    logger.info(f"Training samples: {len(train_paths)}")
    logger.info(f"Validation samples: {len(val_paths)}")
    logger.info(f"Test samples: {len(test_paths)}")
    
    # Print class distribution
    train_dist = Counter(train_labels)
    logger.info("Training class distribution:")
    for class_name, count in zip(class_names, [train_dist[i] for i in range(len(class_names))]):
        logger.info(f"  {class_name}: {count}")
    
    return {
        'train_paths': train_paths,
        'train_labels': train_labels,
        'val_paths': val_paths,
        'val_labels': val_labels,
        'test_paths': test_paths,
        'test_labels': test_labels,
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'num_classes': len(class_names)
    }


def load_images_from_folder(folder_path, class_to_idx):
    """Load image paths and labels from folder structure"""
    image_paths = []
    labels = []
    
    for class_name, class_idx in class_to_idx.items():
        class_folder = os.path.join(folder_path, class_name)
        if not os.path.exists(class_folder):
            logger.warning(f"Class folder not found: {class_folder}")
            continue
        
        # Find all TIFF files in class folder
        pattern = os.path.join(class_folder, "*.tif")
        class_images = glob.glob(pattern)
        
        if not class_images:
            # Try alternative extensions
            for ext in ["*.tiff", "*.TIF", "*.TIFF"]:
                pattern = os.path.join(class_folder, ext)
                class_images.extend(glob.glob(pattern))
        
        logger.info(f"Found {len(class_images)} images for class {class_name}")
        
        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))
    
    return image_paths, labels


def create_data_loaders(dataset_info, batch_size=32, num_workers=4):
    """Create PyTorch DataLoaders for training, validation, and testing"""
    
    # Create transforms
    train_transform = SARTransforms.get_train_transforms()
    val_transform = SARTransforms.get_val_transforms()
    
    # Create datasets
    train_dataset = SARDataset(
        dataset_info['train_paths'], 
        dataset_info['train_labels'],
        transform=train_transform,
        class_names=dataset_info['class_names']
    )
    
    val_dataset = SARDataset(
        dataset_info['val_paths'], 
        dataset_info['val_labels'],
        transform=val_transform,
        class_names=dataset_info['class_names']
    )
    
    test_dataset = SARDataset(
        dataset_info['test_paths'], 
        dataset_info['test_labels'],
        transform=val_transform,
        class_names=dataset_info['class_names']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def compute_class_weights(labels, num_classes):
    """Compute class weights for handling class imbalance"""
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)  # Avoid division by zero
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)


if __name__ == "__main__":
    # Test dataset loading
    data_dir = "dataset"  # Adjust path as needed
    
    if os.path.exists(data_dir):
        dataset_info = load_dataset_from_folders(data_dir)
        train_loader, val_loader, test_loader = create_data_loaders(dataset_info, batch_size=4)
        
        # Test data loading
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"Batch {batch_idx}: Data shape: {data.shape}, Target shape: {target.shape}")
            if batch_idx >= 2:  # Just test first few batches
                break
    else:
        print(f"Dataset directory '{data_dir}' not found. Please adjust the path.")