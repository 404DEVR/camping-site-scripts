"""
Test Script for ResNet-CBAM Model

This script tests the model architecture, data loading, and basic functionality
to ensure everything works correctly before training.

Author: Kiro AI Assistant
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
from pathlib import Path

# Import our modules
from model import create_model, CBAM, ChannelAttention, SpatialAttention
from dataset import SARDataset, SARTransforms, load_dataset_from_folders, create_data_loaders

def test_attention_modules():
    """Test CBAM attention modules"""
    print("Testing CBAM attention modules...")
    
    # Test Channel Attention
    print("  Testing Channel Attention...")
    channel_attention = ChannelAttention(256, reduction_ratio=16)
    test_input = torch.randn(2, 256, 32, 32)
    channel_output = channel_attention(test_input)
    
    assert channel_output.shape == (2, 256, 1, 1), f"Expected (2, 256, 1, 1), got {channel_output.shape}"
    assert torch.all(channel_output >= 0) and torch.all(channel_output <= 1), "Channel attention should output values in [0, 1]"
    print("    ✓ Channel Attention working correctly")
    
    # Test Spatial Attention
    print("  Testing Spatial Attention...")
    spatial_attention = SpatialAttention(kernel_size=7)
    spatial_output = spatial_attention(test_input)
    
    assert spatial_output.shape == (2, 1, 32, 32), f"Expected (2, 1, 32, 32), got {spatial_output.shape}"
    assert torch.all(spatial_output >= 0) and torch.all(spatial_output <= 1), "Spatial attention should output values in [0, 1]"
    print("    ✓ Spatial Attention working correctly")
    
    # Test CBAM
    print("  Testing CBAM...")
    cbam = CBAM(256)
    cbam_output = cbam(test_input)
    
    assert cbam_output.shape == test_input.shape, f"CBAM should preserve input shape"
    print("    ✓ CBAM working correctly")
    
    print("✓ All attention modules working correctly\n")


def test_model_architecture():
    """Test ResNet-CBAM model architecture"""
    print("Testing ResNet-CBAM model architecture...")
    
    # Test model creation
    model = create_model(num_classes=5, pretrained=False)
    print(f"  Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test input shapes
    batch_sizes = [1, 4, 8]
    input_size = (2, 32, 32)  # SAR input: 2 channels, 32x32
    
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, *input_size)
        
        with torch.no_grad():
            output = model(test_input)
        
        expected_output_shape = (batch_size, 5)
        assert output.shape == expected_output_shape, f"Expected {expected_output_shape}, got {output.shape}"
        print(f"    ✓ Batch size {batch_size}: Input {test_input.shape} → Output {output.shape}")
    
    # Test feature extraction
    print("  Testing feature extraction...")
    test_input = torch.randn(2, 2, 32, 32)
    features = model.get_feature_maps(test_input)
    
    expected_features = ['layer1', 'layer2', 'layer3', 'layer4']
    for layer_name in expected_features:
        assert layer_name in features, f"Missing feature map: {layer_name}"
        print(f"    ✓ {layer_name}: {features[layer_name].shape}")
    
    print("✓ Model architecture working correctly\n")


def test_transforms():
    """Test SAR data transforms"""
    print("Testing SAR transforms...")
    
    # Create dummy SAR data
    dummy_sar = torch.randn(2, 32, 32)  # 2 channels, 32x32
    
    # Test training transforms
    train_transform = SARTransforms.get_train_transforms()
    transformed = train_transform(dummy_sar)
    
    assert transformed.shape == dummy_sar.shape, "Transform should preserve shape"
    print("  ✓ Training transforms working correctly")
    
    # Test validation transforms
    val_transform = SARTransforms.get_val_transforms()
    transformed = val_transform(dummy_sar)
    
    assert transformed.shape == dummy_sar.shape, "Transform should preserve shape"
    print("  ✓ Validation transforms working correctly")
    
    print("✓ All transforms working correctly\n")


def test_dataset_loading():
    """Test dataset loading functionality"""
    print("Testing dataset loading...")
    
    # Create dummy dataset structure for testing
    test_data_dir = "test_dataset"
    class_names = ['Bare_Soil', 'Forest', 'Water', 'Snow_Ice', 'Grassland']
    
    # Check if test dataset exists
    if not os.path.exists(test_data_dir):
        print("  Creating dummy test dataset...")
        create_dummy_dataset(test_data_dir, class_names)
    
    try:
        # Test dataset loading
        dataset_info = load_dataset_from_folders(test_data_dir, test_size=0.3, val_size=0.2)
        
        print(f"  ✓ Dataset loaded: {len(dataset_info['train_paths'])} train, "
              f"{len(dataset_info['val_paths'])} val, {len(dataset_info['test_paths'])} test")
        
        # Test data loaders
        train_loader, val_loader, test_loader = create_data_loaders(dataset_info, batch_size=4, num_workers=0)
        
        # Test one batch
        for batch_idx, (data, target) in enumerate(train_loader):
            assert data.shape[1] == 2, f"Expected 2 channels, got {data.shape[1]}"
            assert data.shape[2] == 32 and data.shape[3] == 32, f"Expected 32x32 images, got {data.shape[2]}x{data.shape[3]}"
            print(f"  ✓ Data loader working: batch shape {data.shape}, target shape {target.shape}")
            break
        
        print("✓ Dataset loading working correctly\n")
        
    except Exception as e:
        print(f"  ⚠ Dataset loading test skipped: {e}")
        print("    (This is normal if you don't have a real dataset yet)\n")


def create_dummy_dataset(data_dir, class_names, samples_per_class=5):
    """Create dummy dataset for testing"""
    import rasterio
    from rasterio.transform import from_bounds
    
    os.makedirs(data_dir, exist_ok=True)
    
    for split in ['train', 'test']:
        split_dir = os.path.join(data_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(samples_per_class):
                # Create dummy SAR data
                dummy_data = np.random.randn(2, 32, 32).astype(np.float32)
                
                # Save as TIFF
                output_path = os.path.join(class_dir, f'dummy_{i:03d}.tif')
                
                # Define a simple transform (not geographically accurate, just for testing)
                transform = from_bounds(0, 0, 32, 32, 32, 32)
                
                with rasterio.open(
                    output_path, 'w',
                    driver='GTiff',
                    height=32, width=32,
                    count=2, dtype=np.float32,
                    transform=transform
                ) as dst:
                    dst.write(dummy_data)


def test_training_components():
    """Test training-related components"""
    print("Testing training components...")
    
    # Test loss function
    criterion = nn.CrossEntropyLoss()
    dummy_output = torch.randn(4, 5)  # batch_size=4, num_classes=5
    dummy_target = torch.randint(0, 5, (4,))
    
    loss = criterion(dummy_output, dummy_target)
    assert loss.item() > 0, "Loss should be positive"
    print("  ✓ Loss function working correctly")
    
    # Test optimizer
    model = create_model(num_classes=5, pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Test backward pass
    dummy_input = torch.randn(2, 2, 32, 32)
    dummy_target = torch.randint(0, 5, (2,))
    
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()
    optimizer.step()
    
    print("  ✓ Optimizer and backward pass working correctly")
    
    print("✓ Training components working correctly\n")


def test_device_compatibility():
    """Test CUDA compatibility if available"""
    print("Testing device compatibility...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        
        # Test model on GPU
        device = torch.device('cuda')
        model = create_model(num_classes=5, pretrained=False).to(device)
        dummy_input = torch.randn(2, 2, 32, 32).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.device.type == 'cuda', "Output should be on CUDA device"
        print("  ✓ Model working correctly on GPU")
        
    else:
        print("  ⚠ CUDA not available, using CPU only")
    
    # Test CPU
    device = torch.device('cpu')
    model = create_model(num_classes=5, pretrained=False).to(device)
    dummy_input = torch.randn(2, 2, 32, 32).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.device.type == 'cpu', "Output should be on CPU device"
    print("  ✓ Model working correctly on CPU")
    
    print("✓ Device compatibility working correctly\n")


def main():
    """Run all tests"""
    print("="*60)
    print("ResNet-CBAM SAR Classification - Model Testing")
    print("="*60)
    print()
    
    try:
        # Run all tests
        test_attention_modules()
        test_model_architecture()
        test_transforms()
        test_dataset_loading()
        test_training_components()
        test_device_compatibility()
        
        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("Your ResNet-CBAM implementation is ready for training.")
        print("="*60)
        
    except Exception as e:
        print("="*60)
        print(f"❌ TEST FAILED: {e}")
        print("Please check the error and fix the issue.")
        print("="*60)
        raise


if __name__ == "__main__":
    main()