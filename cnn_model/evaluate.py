"""
Evaluation Script for ResNet-CBAM SAR Landcover Classification

This script handles model evaluation, confusion matrix generation, 
classification reports, and Grad-CAM visualization.

Author: Kiro AI Assistant
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import cv2
from tqdm import tqdm
import logging

from model import create_model
from dataset import load_dataset_from_folders, create_data_loaders

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping for feature visualization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, class_idx=None):
        """Generate Class Activation Map"""
        # Forward pass
        model_output = self.model(input_image)
        
        if class_idx is None:
            class_idx = np.argmax(model_output.cpu().data.numpy())
        
        # Backward pass
        self.model.zero_grad()
        class_loss = model_output[0, class_idx]
        class_loss.backward()
        
        # Generate CAM
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam) if np.max(cam) > 0 else cam
        
        return cam, class_idx


class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model, test_loader, device, class_names):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def evaluate(self):
        """Comprehensive model evaluation"""
        logger.info("Starting model evaluation...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = self.model(data)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        results = self.calculate_metrics(all_targets, all_predictions, all_probabilities)
        
        # Generate visualizations
        self.plot_confusion_matrix(all_targets, all_predictions)
        self.plot_class_distribution(all_targets, all_predictions)
        
        return results
    
    def calculate_metrics(self, targets, predictions, probabilities):
        """Calculate comprehensive evaluation metrics"""
        # Overall accuracy
        accuracy = accuracy_score(targets, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, labels=range(self.num_classes)
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            targets, predictions, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        
        # Classification report
        report = classification_report(
            targets, predictions, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        results = {
            'accuracy': accuracy,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support_per_class': support,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'classification_report': report
        }
        
        # Print results
        self.print_results(results)
        
        return results
    
    def print_results(self, results):
        """Print evaluation results"""
        logger.info("="*60)
        logger.info("EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Overall Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Macro F1-Score: {results['f1_macro']:.4f}")
        logger.info(f"Weighted F1-Score: {results['f1_weighted']:.4f}")
        logger.info("")
        
        logger.info("Per-Class Results:")
        logger.info("-" * 80)
        logger.info(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        logger.info("-" * 80)
        
        for i, class_name in enumerate(self.class_names):
            logger.info(f"{class_name:<15} {results['precision_per_class'][i]:<10.4f} "
                       f"{results['recall_per_class'][i]:<10.4f} "
                       f"{results['f1_per_class'][i]:<10.4f} "
                       f"{results['support_per_class'][i]:<10}")
        
        logger.info("-" * 80)
        logger.info(f"{'Macro Avg':<15} {results['precision_macro']:<10.4f} "
                   f"{results['recall_macro']:<10.4f} "
                   f"{results['f1_macro']:<10.4f} "
                   f"{sum(results['support_per_class']):<10}")
        logger.info(f"{'Weighted Avg':<15} {results['precision_weighted']:<10.4f} "
                   f"{results['recall_weighted']:<10.4f} "
                   f"{results['f1_weighted']:<10.4f} "
                   f"{sum(results['support_per_class']):<10}")
    
    def plot_confusion_matrix(self, targets, predictions, save_path='confusion_matrix.png'):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def plot_class_distribution(self, targets, predictions, save_path='class_distribution.png'):
        """Plot class distribution comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # True distribution
        unique_true, counts_true = np.unique(targets, return_counts=True)
        ax1.bar([self.class_names[i] for i in unique_true], counts_true, color='skyblue')
        ax1.set_title('True Class Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Predicted distribution
        unique_pred, counts_pred = np.unique(predictions, return_counts=True)
        ax2.bar([self.class_names[i] for i in unique_pred], counts_pred, color='lightcoral')
        ax2.set_title('Predicted Class Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Class distribution plot saved to {save_path}")
    
    def visualize_predictions(self, num_samples=16, save_path='predictions_visualization.png'):
        """Visualize model predictions on sample images"""
        self.model.eval()
        
        # Get a batch of test data
        data_iter = iter(self.test_loader)
        data, targets = next(data_iter)
        data, targets = data.to(self.device), targets.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(data)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)
        
        # Select samples to visualize
        num_samples = min(num_samples, data.size(0))
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()
        
        for i in range(num_samples):
            # Convert SAR image for visualization (use VV channel)
            img = data[i, 0].cpu().numpy()  # VV channel
            img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
            
            axes[i].imshow(img, cmap='gray')
            
            true_class = self.class_names[targets[i].item()]
            pred_class = self.class_names[predictions[i].item()]
            confidence = probabilities[i, predictions[i]].item()
            
            color = 'green' if targets[i] == predictions[i] else 'red'
            axes[i].set_title(f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.3f}', 
                            color=color, fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Predictions visualization saved to {save_path}")


def load_model_checkpoint(checkpoint_path, num_classes=5, device='cpu'):
    """Load model from checkpoint"""
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = create_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    logger.info(f"Model loaded successfully. Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    return model, checkpoint


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate ResNet-CBAM for SAR Landcover Classification')
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='dataset',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of landcover classes')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--visualize_predictions', action='store_true',
                       help='Generate prediction visualizations')
    
    return parser.parse_args()


def main():
    """Main evaluation function"""
    args = parse_arguments()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset_info = load_dataset_from_folders(args.data_dir)
    
    # Create test data loader
    _, _, test_loader = create_data_loaders(
        dataset_info, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    # Load model
    model, checkpoint = load_model_checkpoint(args.checkpoint_path, args.num_classes, device)
    
    # Create evaluator
    evaluator = ModelEvaluator(model, test_loader, device, dataset_info['class_names'])
    
    # Run evaluation
    results = evaluator.evaluate()
    
    # Generate visualizations if requested
    if args.visualize_predictions:
        evaluator.visualize_predictions(
            save_path=os.path.join(args.output_dir, 'predictions_visualization.png')
        )
    
    logger.info(f"Evaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()