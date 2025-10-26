"""
Inference Script for ResNet-CBAM SAR Landcover Classification

This script provides functionality to predict landcover classes for new SAR patches
using a trained ResNet-CBAM model.

Author: Kiro AI Assistant
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import rasterio
from PIL import Image
import glob
import logging
from pathlib import Path

from model import create_model
from dataset import SARTransforms

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SARInference:
    """SAR landcover classification inference class"""
    
    def __init__(self, model_path, device='auto', class_names=None):
        """
        Initialize inference class
        
        Args:
            model_path (str): Path to trained model checkpoint
            device (str): Device to use for inference
            class_names (list): List of class names
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Default class names
        self.class_names = class_names or [
            'Bare_Soil', 'Forest', 'Water', 'Snow_Ice', 'Grassland'
        ]
        self.num_classes = len(self.class_names)
        
        # Load model
        self.model = self.load_model(model_path)
        
        # Set up transforms
        self.transform = SARTransforms.get_val_transforms()
        
        logger.info(f"Inference initialized on device: {self.device}")
        logger.info(f"Classes: {self.class_names}")
    
    def load_model(self, model_path):
        """Load trained model from checkpoint"""
        logger.info(f"Loading model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model
        model = create_model(num_classes=self.num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
    
    def load_sar_image(self, image_path):
        """Load and preprocess SAR image"""
        try:
            with rasterio.open(image_path) as src:
                # Read all bands (should be 2 for VV, VH)
                image = src.read()  # Shape: (channels, height, width)
                
                # Ensure we have exactly 2 channels
                if image.shape[0] != 2:
                    raise ValueError(f"Expected 2 channels, got {image.shape[0]} in {image_path}")
                
                # Convert to float32 and handle NaN/inf values
                image = image.astype(np.float32)
                image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
                
                return image
                
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def predict_single(self, image_path, return_probabilities=False):
        """
        Predict landcover class for a single SAR image
        
        Args:
            image_path (str): Path to SAR image file
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Prediction results
        """
        # Load image
        image = self.load_sar_image(image_path)
        if image is None:
            return None
        
        # Convert to tensor and apply transforms
        image_tensor = torch.from_numpy(image)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        # Prepare results
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        class_name = self.class_names[predicted_class]
        
        results = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'predicted_class_name': class_name,
            'confidence': confidence
        }
        
        if return_probabilities:
            class_probs = probabilities.cpu().numpy()[0]
            results['class_probabilities'] = {
                name: float(prob) for name, prob in zip(self.class_names, class_probs)
            }
        
        return results
    
    def predict_batch(self, image_paths, return_probabilities=False):
        """
        Predict landcover classes for multiple SAR images
        
        Args:
            image_paths (list): List of paths to SAR image files
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        logger.info(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(image_paths)} images")
            
            result = self.predict_single(image_path, return_probabilities)
            if result is not None:
                results.append(result)
        
        logger.info(f"Completed processing {len(results)} images")
        return results
    
    def predict_directory(self, directory_path, pattern="*.tif", return_probabilities=False):
        """
        Predict landcover classes for all SAR images in a directory
        
        Args:
            directory_path (str): Path to directory containing SAR images
            pattern (str): File pattern to match
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            list: List of prediction results
        """
        # Find all matching files
        search_pattern = os.path.join(directory_path, pattern)
        image_paths = glob.glob(search_pattern)
        
        if not image_paths:
            # Try alternative extensions
            for ext in ["*.tiff", "*.TIF", "*.TIFF"]:
                search_pattern = os.path.join(directory_path, ext)
                image_paths.extend(glob.glob(search_pattern))
        
        if not image_paths:
            logger.warning(f"No SAR images found in {directory_path} with pattern {pattern}")
            return []
        
        logger.info(f"Found {len(image_paths)} SAR images in {directory_path}")
        
        return self.predict_batch(image_paths, return_probabilities)
    
    def save_results(self, results, output_path, format='csv'):
        """
        Save prediction results to file
        
        Args:
            results (list): List of prediction results
            output_path (str): Path to output file
            format (str): Output format ('csv' or 'json')
        """
        if not results:
            logger.warning("No results to save")
            return
        
        if format.lower() == 'csv':
            import pandas as pd
            
            # Flatten results for CSV
            flattened_results = []
            for result in results:
                flat_result = {
                    'image_path': result['image_path'],
                    'predicted_class': result['predicted_class'],
                    'predicted_class_name': result['predicted_class_name'],
                    'confidence': result['confidence']
                }
                
                # Add class probabilities if available
                if 'class_probabilities' in result:
                    for class_name, prob in result['class_probabilities'].items():
                        flat_result[f'prob_{class_name}'] = prob
                
                flattened_results.append(flat_result)
            
            df = pd.DataFrame(flattened_results)
            df.to_csv(output_path, index=False)
            
        elif format.lower() == 'json':
            import json
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Results saved to {output_path}")
    
    def print_summary(self, results):
        """Print summary of prediction results"""
        if not results:
            logger.info("No results to summarize")
            return
        
        # Count predictions by class
        class_counts = {}
        total_confidence = 0
        
        for result in results:
            class_name = result['predicted_class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += result['confidence']
        
        # Print summary
        logger.info("="*50)
        logger.info("PREDICTION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total images processed: {len(results)}")
        logger.info(f"Average confidence: {total_confidence/len(results):.3f}")
        logger.info("")
        logger.info("Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            percentage = (count / len(results)) * 100
            logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SAR Landcover Classification Inference')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Path to output file')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'json'],
                       help='Output format')
    parser.add_argument('--pattern', type=str, default='*.tif',
                       help='File pattern for directory input')
    parser.add_argument('--probabilities', action='store_true',
                       help='Include class probabilities in output')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--class_names', type=str, nargs='+',
                       default=['Bare_Soil', 'Forest', 'Water', 'Snow_Ice', 'Grassland'],
                       help='List of class names')
    
    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_arguments()
    
    # Initialize inference
    inference = SARInference(
        model_path=args.model_path,
        device=args.device,
        class_names=args.class_names
    )
    
    # Check if input is file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file prediction
        logger.info(f"Predicting single file: {args.input}")
        result = inference.predict_single(args.input, args.probabilities)
        
        if result:
            results = [result]
            logger.info(f"Prediction: {result['predicted_class_name']} (confidence: {result['confidence']:.3f})")
        else:
            logger.error("Failed to process input file")
            return
            
    elif input_path.is_dir():
        # Directory prediction
        logger.info(f"Predicting directory: {args.input}")
        results = inference.predict_directory(args.input, args.pattern, args.probabilities)
        
    else:
        logger.error(f"Input path does not exist: {args.input}")
        return
    
    # Save results
    if results:
        inference.save_results(results, args.output, args.format)
        inference.print_summary(results)
    else:
        logger.warning("No predictions generated")


if __name__ == "__main__":
    main()