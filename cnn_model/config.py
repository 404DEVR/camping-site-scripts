"""
Configuration File for ResNet-CBAM SAR Landcover Classification

This file contains default configurations and hyperparameters for the model,
training, and evaluation processes.

Author: Kiro AI Assistant
"""

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    num_classes: int = 5
    pretrained: bool = True
    cbam_reduction_ratio: int = 16
    cbam_kernel_size: int = 7
    
    # Class names for landcover classification
    class_names: List[str] = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ['Bare_Soil', 'Forest', 'Water', 'Snow_Ice', 'Grassland']


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    data_dir: str = "dataset"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # Data splits
    test_size: float = 0.2
    val_size: float = 0.1
    
    # Image specifications
    image_size: tuple = (32, 32)
    num_channels: int = 2  # VV and VH polarization
    
    # Data augmentation
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    rotation_degrees: float = 15.0
    
    # Normalization (can be computed from dataset)
    sar_mean: List[float] = None
    sar_std: List[float] = None
    
    def __post_init__(self):
        if self.sar_mean is None:
            self.sar_mean = [0.0, 0.0]  # Per channel mean
        if self.sar_std is None:
            self.sar_std = [1.0, 1.0]   # Per channel std


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 30
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Loss function
    use_class_weights: bool = False
    label_smoothing: float = 0.0
    
    # Learning rate scheduler
    lr_scheduler: str = "ReduceLROnPlateau"  # Options: ReduceLROnPlateau, StepLR, CosineAnnealingLR
    lr_factor: float = 0.5
    lr_patience: int = 3
    lr_step_size: int = 10  # For StepLR
    lr_gamma: float = 0.1   # For StepLR
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_delta: float = 0.001
    
    # Gradient clipping
    gradient_clip_value: Optional[float] = None
    
    # Mixed precision training
    use_amp: bool = False
    
    # Reproducibility
    seed: int = 42


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    batch_size: int = 32
    num_workers: int = 4
    
    # Visualization
    visualize_predictions: bool = True
    visualize_gradcam: bool = True
    num_visualization_samples: int = 16
    num_gradcam_samples: int = 8
    
    # Output formats
    save_confusion_matrix: bool = True
    save_classification_report: bool = True
    save_predictions: bool = True


@dataclass
class InferenceConfig:
    """Inference configuration"""
    batch_size: int = 32
    num_workers: int = 4
    
    # Output options
    output_format: str = "csv"  # Options: csv, json
    include_probabilities: bool = False
    confidence_threshold: float = 0.5
    
    # File patterns
    image_patterns: List[str] = None
    
    def __post_init__(self):
        if self.image_patterns is None:
            self.image_patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]


@dataclass
class PathConfig:
    """Path configuration"""
    # Data paths
    data_dir: str = "dataset"
    train_dir: str = "dataset/train"
    test_dir: str = "dataset/test"
    
    # Output paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"
    tensorboard_dir: str = "runs"
    
    # Model paths
    best_model_path: str = "checkpoints/resnet_cbam_sar.pth"
    
    def __post_init__(self):
        # Create directories if they don't exist
        for dir_path in [self.checkpoint_dir, self.log_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    inference: InferenceConfig
    paths: PathConfig
    
    # Experiment metadata
    experiment_name: str = "resnet_cbam_sar"
    description: str = "ResNet-CBAM for SAR landcover classification"
    
    # Device configuration
    device: str = "auto"  # Options: auto, cpu, cuda, cuda:0, etc.
    
    def __post_init__(self):
        # Update paths with experiment name
        self.paths.checkpoint_dir = os.path.join("checkpoints", self.experiment_name)
        self.paths.results_dir = os.path.join("results", self.experiment_name)
        self.paths.tensorboard_dir = os.path.join("runs", self.experiment_name)
        
        # Create experiment directories
        for dir_path in [self.paths.checkpoint_dir, self.paths.results_dir]:
            os.makedirs(dir_path, exist_ok=True)


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration"""
    return ExperimentConfig(
        model=ModelConfig(),
        data=DataConfig(),
        training=TrainingConfig(),
        evaluation=EvaluationConfig(),
        inference=InferenceConfig(),
        paths=PathConfig()
    )


def get_config_for_quick_test() -> ExperimentConfig:
    """Get configuration for quick testing with reduced parameters"""
    config = get_default_config()
    
    # Reduce training time for testing
    config.training.epochs = 5
    config.training.early_stopping_patience = 2
    config.data.batch_size = 8
    config.data.num_workers = 2
    
    config.experiment_name = "resnet_cbam_quick_test"
    config.description = "Quick test configuration"
    
    return config


def get_config_for_production() -> ExperimentConfig:
    """Get configuration optimized for production training"""
    config = get_default_config()
    
    # Production settings
    config.training.epochs = 50
    config.training.use_class_weights = True
    config.training.use_amp = True  # Mixed precision for faster training
    config.training.gradient_clip_value = 1.0
    
    config.data.batch_size = 64  # Larger batch size if GPU memory allows
    config.data.num_workers = 8
    
    config.experiment_name = "resnet_cbam_production"
    config.description = "Production training configuration"
    
    return config


def get_config_for_research() -> ExperimentConfig:
    """Get configuration for research with extensive logging and evaluation"""
    config = get_default_config()
    
    # Research settings
    config.training.epochs = 100
    config.training.early_stopping_patience = 10
    config.evaluation.num_visualization_samples = 32
    config.evaluation.num_gradcam_samples = 16
    
    config.experiment_name = "resnet_cbam_research"
    config.description = "Research configuration with extensive evaluation"
    
    return config


# Predefined configurations
CONFIGS = {
    "default": get_default_config,
    "quick_test": get_config_for_quick_test,
    "production": get_config_for_production,
    "research": get_config_for_research
}


def load_config(config_name: str = "default") -> ExperimentConfig:
    """Load a predefined configuration"""
    if config_name not in CONFIGS:
        available_configs = list(CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available configs: {available_configs}")
    
    return CONFIGS[config_name]()


def save_config(config: ExperimentConfig, filepath: str):
    """Save configuration to file"""
    import json
    from dataclasses import asdict
    
    config_dict = asdict(config)
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Configuration saved to {filepath}")


def load_config_from_file(filepath: str) -> ExperimentConfig:
    """Load configuration from file"""
    import json
    
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    # Reconstruct the configuration object
    # Note: This is a simplified version. For complex nested structures,
    # you might want to use a more sophisticated serialization method.
    
    return ExperimentConfig(
        model=ModelConfig(**config_dict['model']),
        data=DataConfig(**config_dict['data']),
        training=TrainingConfig(**config_dict['training']),
        evaluation=EvaluationConfig(**config_dict['evaluation']),
        inference=InferenceConfig(**config_dict['inference']),
        paths=PathConfig(**config_dict['paths']),
        experiment_name=config_dict['experiment_name'],
        description=config_dict['description'],
        device=config_dict['device']
    )


if __name__ == "__main__":
    # Example usage
    print("Available configurations:")
    for name in CONFIGS.keys():
        config = load_config(name)
        print(f"  {name}: {config.description}")
    
    # Load and display default config
    print("\nDefault configuration:")
    default_config = load_config("default")
    print(f"  Experiment: {default_config.experiment_name}")
    print(f"  Epochs: {default_config.training.epochs}")
    print(f"  Batch size: {default_config.data.batch_size}")
    print(f"  Learning rate: {default_config.training.learning_rate}")
    
    # Save example config
    save_config(default_config, "example_config.json")