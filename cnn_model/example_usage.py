"""
Example Usage Script for ResNet-CBAM SAR Landcover Classification

This script demonstrates how to use the ResNet-CBAM implementation for
SAR landcover classification with various configurations and workflows.

Author: Kiro AI Assistant
"""

import os
import torch
import argparse
from pathlib import Path

# Import our modules
from config import load_config, save_config
from model import create_model
from dataset import load_dataset_from_folders, create_data_loaders
from train import Trainer, set_seed
from evaluate import ModelEvaluator, load_model_checkpoint
from inference import SARInference

def example_training_workflow():
    """Example of complete training workflow"""
    print("="*60)
    print("EXAMPLE: Complete Training Workflow")
    print("="*60)
    
    # Load configuration
    config = load_config("quick_test")  # Use quick test for demonstration
    print(f"Using configuration: {config.experiment_name}")
    
    # Set random seed for reproducibility
    set_seed(config.training.seed)
    
    # Determine device
    if config.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # Check if dataset exists
    if not os.path.exists(config.data.data_dir):
        print(f"⚠ Dataset directory '{config.data.data_dir}' not found.")
        print("Please prepare your dataset first.")
        return
    
    try:
        # Load dataset
        print("\n1. Loading dataset...")
        dataset_info = load_dataset_from_folders(
            config.data.data_dir,
            test_size=config.data.test_size,
            val_size=config.data.val_size
        )
        
        # Create data loaders
        print("2. Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset_info,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers
        )
        
        # Create model
        print("3. Creating model...")
        model = create_model(
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained
        )
        model = model.to(device)
        
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create trainer
        print("4. Setting up trainer...")
        
        # Convert config to args-like object for compatibility
        class Args:
            def __init__(self, config):
                self.num_classes = config.model.num_classes
                self.use_class_weights = config.training.use_class_weights
                self.learning_rate = config.training.learning_rate
                self.weight_decay = config.training.weight_decay
                self.lr_factor = config.training.lr_factor
                self.lr_patience = config.training.lr_patience
                self.early_stopping_patience = config.training.early_stopping_patience
                self.early_stopping_delta = config.training.early_stopping_delta
                self.epochs = config.training.epochs
                self.checkpoint_dir = config.paths.checkpoint_dir
                self.experiment_name = config.experiment_name
        
        args = Args(config)
        trainer = Trainer(model, train_loader, val_loader, device, args)
        
        # Start training
        print("5. Starting training...")
        history = trainer.train()
        
        print("✅ Training completed successfully!")
        print(f"Best model saved to: {config.paths.best_model_path}")
        
        return history
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def example_evaluation_workflow():
    """Example of model evaluation workflow"""
    print("\n" + "="*60)
    print("EXAMPLE: Model Evaluation Workflow")
    print("="*60)
    
    # Configuration
    config = load_config("default")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if model checkpoint exists
    checkpoint_path = config.paths.best_model_path
    if not os.path.exists(checkpoint_path):
        print(f"⚠ Model checkpoint not found: {checkpoint_path}")
        print("Please train a model first.")
        return
    
    try:
        # Load dataset
        print("1. Loading test dataset...")
        dataset_info = load_dataset_from_folders(config.data.data_dir)
        
        # Create test data loader
        _, _, test_loader = create_data_loaders(
            dataset_info,
            batch_size=config.evaluation.batch_size,
            num_workers=config.evaluation.num_workers
        )
        
        # Load model
        print("2. Loading trained model...")
        model, checkpoint = load_model_checkpoint(
            checkpoint_path,
            num_classes=config.model.num_classes,
            device=device
        )
        
        # Create evaluator
        print("3. Running evaluation...")
        evaluator = ModelEvaluator(
            model, test_loader, device, dataset_info['class_names']
        )
        
        # Run comprehensive evaluation
        results = evaluator.evaluate()
        
        # Generate visualizations
        if config.evaluation.visualize_predictions:
            print("4. Generating prediction visualizations...")
            evaluator.visualize_predictions(
                num_samples=config.evaluation.num_visualization_samples
            )
        
        print("✅ Evaluation completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None

def example_inference_workflow():
    """Example of inference workflow"""
    print("\n" + "="*60)
    print("EXAMPLE: Inference Workflow")
    print("="*60)
    
    # Configuration
    config = load_config("default")
    checkpoint_path = config.paths.best_model_path
    
    # Check if model exists
    if not os.path.exists(checkpoint_path):
        print(f"⚠ Model checkpoint not found: {checkpoint_path}")
        print("Please train a model first.")
        return
    
    try:
        # Initialize inference
        print("1. Initializing inference...")
        inference = SARInference(
            model_path=checkpoint_path,
            device=config.device,
            class_names=config.model.class_names
        )
        
        # Example 1: Single image prediction
        print("\n2. Single image prediction example...")
        test_image_path = "path/to/test/image.tif"  # Replace with actual path
        
        if os.path.exists(test_image_path):
            result = inference.predict_single(
                test_image_path, 
                return_probabilities=True
            )
            
            if result:
                print(f"   Image: {result['image_path']}")
                print(f"   Predicted class: {result['predicted_class_name']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                
                if 'class_probabilities' in result:
                    print("   Class probabilities:")
                    for class_name, prob in result['class_probabilities'].items():
                        print(f"     {class_name}: {prob:.3f}")
        else:
            print(f"   ⚠ Test image not found: {test_image_path}")
        
        # Example 2: Directory prediction
        print("\n3. Directory prediction example...")
        test_dir = "dataset/test/Forest"  # Example directory
        
        if os.path.exists(test_dir):
            results = inference.predict_directory(
                test_dir,
                pattern="*.tif",
                return_probabilities=False
            )
            
            if results:
                print(f"   Processed {len(results)} images")
                
                # Save results
                output_path = "inference_results.csv"
                inference.save_results(results, output_path, format='csv')
                print(f"   Results saved to: {output_path}")
                
                # Print summary
                inference.print_summary(results)
        else:
            print(f"   ⚠ Test directory not found: {test_dir}")
        
        print("✅ Inference completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

def example_configuration_management():
    """Example of configuration management"""
    print("\n" + "="*60)
    print("EXAMPLE: Configuration Management")
    print("="*60)
    
    # Load different configurations
    print("1. Available configurations:")
    configs = ['default', 'quick_test', 'production', 'research']
    
    for config_name in configs:
        config = load_config(config_name)
        print(f"   {config_name}: {config.description}")
        print(f"     Epochs: {config.training.epochs}")
        print(f"     Batch size: {config.data.batch_size}")
        print(f"     Learning rate: {config.training.learning_rate}")
        print()
    
    # Create custom configuration
    print("2. Creating custom configuration...")
    custom_config = load_config("default")
    
    # Modify settings
    custom_config.experiment_name = "custom_experiment"
    custom_config.description = "Custom configuration example"
    custom_config.training.epochs = 25
    custom_config.training.learning_rate = 0.0005
    custom_config.data.batch_size = 16
    
    # Save custom configuration
    config_path = "custom_config.json"
    save_config(custom_config, config_path)
    print(f"   Custom configuration saved to: {config_path}")
    
    print("✅ Configuration management completed!")

def example_model_analysis():
    """Example of model architecture analysis"""
    print("\n" + "="*60)
    print("EXAMPLE: Model Architecture Analysis")
    print("="*60)
    
    try:
        # Create model
        print("1. Creating model...")
        model = create_model(num_classes=5, pretrained=False)
        
        # Model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
        
        # Test forward pass
        print("\n2. Testing forward pass...")
        dummy_input = torch.randn(1, 2, 32, 32)  # SAR input
        
        with torch.no_grad():
            output = model(dummy_input)
            features = model.get_feature_maps(dummy_input)
        
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print("   Feature map shapes:")
        for layer_name, feature_map in features.items():
            print(f"     {layer_name}: {feature_map.shape}")
        
        # Memory usage estimation
        print("\n3. Memory usage estimation...")
        input_memory = dummy_input.numel() * 4 / 1024 / 1024  # MB
        model_memory = total_params * 4 / 1024 / 1024  # MB
        
        print(f"   Input memory (1 sample): {input_memory:.2f} MB")
        print(f"   Model memory: {model_memory:.2f} MB")
        print(f"   Estimated batch memory (batch=32): {input_memory * 32:.2f} MB")
        
        print("✅ Model analysis completed!")
        return True
        
    except Exception as e:
        print(f"❌ Model analysis failed: {e}")
        return False

def main():
    """Main example function"""
    parser = argparse.ArgumentParser(description='ResNet-CBAM Usage Examples')
    parser.add_argument('--example', type=str, 
                       choices=['training', 'evaluation', 'inference', 'config', 'model', 'all'],
                       default='all',
                       help='Which example to run')
    
    args = parser.parse_args()
    
    print("ResNet-CBAM SAR Landcover Classification - Usage Examples")
    print("This script demonstrates various workflows and capabilities.")
    print()
    
    if args.example in ['training', 'all']:
        example_training_workflow()
    
    if args.example in ['evaluation', 'all']:
        example_evaluation_workflow()
    
    if args.example in ['inference', 'all']:
        example_inference_workflow()
    
    if args.example in ['config', 'all']:
        example_configuration_management()
    
    if args.example in ['model', 'all']:
        example_model_analysis()
    
    print("\n" + "="*60)
    print("Examples completed! Check the individual scripts for more details:")
    print("- train.py: For training models")
    print("- evaluate.py: For evaluating trained models")
    print("- inference.py: For making predictions")
    print("- config.py: For configuration management")
    print("="*60)

if __name__ == "__main__":
    main()