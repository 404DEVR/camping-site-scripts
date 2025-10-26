"""
ResNet-CBAM Model Implementation for SAR Landcover Classification

This module implements ResNet50 with Convolutional Block Attention Module (CBAM)
for multi-class landcover classification from SAR image patches.

Author: Kiro AI Assistant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class ChannelAttention(nn.Module):
    """Channel Attention Module of CBAM"""
    
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Average pooling branch
        avg_out = self.fc(self.avg_pool(x))
        # Max pooling branch
        max_out = self.fc(self.max_pool(x))
        # Combine and apply sigmoid
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module of CBAM"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along channel dimension
        x_cat = torch.cat([avg_out, max_out], dim=1)
        # Apply convolution and sigmoid
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # Apply channel attention
        x = x * self.channel_attention(x)
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        return x


class ResNetCBAM(nn.Module):
    """ResNet50 with CBAM for SAR Landcover Classification"""
    
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNetCBAM, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Modify first conv layer for 2-channel SAR input (VV, VH)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Initialize new conv1 weights
        if pretrained:
            # Average the RGB weights to initialize 2-channel weights
            with torch.no_grad():
                # Take mean of RGB channels for VV channel
                self.resnet.conv1.weight[:, 0, :, :] = original_conv1.weight.mean(dim=1)
                # Copy for VH channel
                self.resnet.conv1.weight[:, 1, :, :] = original_conv1.weight.mean(dim=1)
        
        # Get feature dimensions for CBAM modules
        # ResNet50 layer dimensions: [256, 512, 1024, 2048]
        self.cbam1 = CBAM(256)  # After layer1
        self.cbam2 = CBAM(512)  # After layer2
        self.cbam3 = CBAM(1024) # After layer3
        self.cbam4 = CBAM(2048) # After layer4
        
        # Modify final classifier
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
        # Store layer references for forward pass
        self.conv1 = self.resnet.conv1
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        self.avgpool = self.resnet.avgpool
        self.fc = self.resnet.fc
    
    def forward(self, x):
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers with CBAM
        x = self.layer1(x)
        x = self.cbam1(x)
        
        x = self.layer2(x)
        x = self.cbam2(x)
        
        x = self.layer3(x)
        x = self.cbam3(x)
        
        x = self.layer4(x)
        x = self.cbam4(x)
        
        # Global average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_feature_maps(self, x):
        """Extract feature maps for visualization (Grad-CAM)"""
        features = {}
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.cbam1(x)
        features['layer1'] = x
        
        x = self.layer2(x)
        x = self.cbam2(x)
        features['layer2'] = x
        
        x = self.layer3(x)
        x = self.cbam3(x)
        features['layer3'] = x
        
        x = self.layer4(x)
        x = self.cbam4(x)
        features['layer4'] = x
        
        return features


def create_model(num_classes=5, pretrained=True):
    """Factory function to create ResNet-CBAM model"""
    return ResNetCBAM(num_classes=num_classes, pretrained=pretrained)


if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_model(num_classes=5)
    
    # Test with dummy SAR input (batch_size=2, channels=2, height=32, width=32)
    dummy_input = torch.randn(2, 2, 32, 32)
    
    print("Model created successfully!")
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")