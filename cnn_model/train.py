"""
Training Script for ResNet-CBAM SAR Landcover Classification

This script handles the complete training pipeline including model initialization,
training loop, validation, and model checkpointing.

Author: Kiro AI Assistant
"""

import os
import argparse
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from model import create_model
from dataset import load_dataset_from_folders, create_data_loaders, compute_class_weights

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()


class Trainer:
    """Main trainer class for ResNet-CBAM"""
    
    def __init__(self, model, train_loader, val_loader, device, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.args = args
        
        # Loss function with class weights if specified
        if args.use_class_weights:
            # Compute class weights from training data
            train_labels = []
            for _, labels in train_loader:
                train_labels.extend(labels.tolist())
            class_weights = compute_class_weights(train_labels, args.num_classes)
            class_weights = class_weights.to(device)
            logger.info(f"Using class weights: {class_weights}")
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=args.lr_factor, 
            patience=args.lr_patience,
            verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_delta
        )
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=f'runs/{args.experiment_name}')
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(self.args.epochs):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch results
            logger.info(f'Epoch {epoch+1}/{self.args.epochs}:')
            logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            logger.info(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
                logger.info(f'  New best validation accuracy: {val_acc:.2f}%')
            
            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                logger.info(f'Early stopping triggered after epoch {epoch+1}')
                break
        
        # Training completed
        total_time = time.time() - start_time
        logger.info(f'Training completed in {total_time/3600:.2f} hours')
        logger.info(f'Best validation accuracy: {best_val_acc:.2f}%')
        
        # Close TensorBoard writer
        self.writer.close()
        
        return self.history
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'args': self.args,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.args.checkpoint_dir, 'resnet_cbam_sar.pth')
            torch.save(checkpoint, best_path)
            logger.info(f'Best model saved to {best_path}')


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ResNet-CBAM for SAR Landcover Classification')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='dataset', 
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=5, 
                       help='Number of landcover classes')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained ResNet50 weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30, 
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                       help='Weight decay for regularization')
    parser.add_argument('--use_class_weights', action='store_true', 
                       help='Use class weights for imbalanced dataset')
    
    # Scheduler arguments
    parser.add_argument('--lr_factor', type=float, default=0.5, 
                       help='Factor to reduce learning rate')
    parser.add_argument('--lr_patience', type=int, default=3, 
                       help='Patience for learning rate scheduler')
    
    # Early stopping arguments
    parser.add_argument('--early_stopping_patience', type=int, default=5, 
                       help='Patience for early stopping')
    parser.add_argument('--early_stopping_delta', type=float, default=0.001, 
                       help='Minimum change for early stopping')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', 
                       help='Directory to save model checkpoints')
    parser.add_argument('--experiment_name', type=str, 
                       default=f'resnet_cbam_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='Name for this experiment')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function"""
    args = parse_arguments()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset_info = load_dataset_from_folders(args.data_dir)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_info, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    # Create model
    logger.info("Creating model...")
    model = create_model(num_classes=args.num_classes, pretrained=args.pretrained)
    model = model.to(device)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device, args)
    
    # Start training
    history = trainer.train()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()