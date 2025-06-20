#!/usr/bin/env python3
"""
Enhanced Species Classification Model - Version 3
Major improvements for better accuracy on family-level classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
import os
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AdvancedAugmentationDataset(Dataset):
    """Enhanced dataset with advanced augmentation techniques"""
    
    def __init__(self, csv_file, image_dir, transform=None, target_level='family', augment_factor=2):
        print(f"Loading enhanced dataset from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.target_level = target_level
        self.augment_factor = augment_factor  # How many augmented versions per image
        
        # Create label encoder
        self.label_encoder = LabelEncoder()
        labels = self.df[target_level].fillna('Unknown').astype(str)
        self.label_encoder.fit(labels)
        self.target_labels = self.label_encoder.transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
        
        # Filter to existing images
        self.valid_indices = []
        for idx in range(len(self.df)):
            image_path = self.image_dir / self.df.iloc[idx]['image_filename']
            if image_path.exists():
                self.valid_indices.append(idx)
        
        self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)
        self.target_labels = self.target_labels[self.valid_indices]
        
        # Calculate class weights for balancing
        self.class_counts = Counter(self.target_labels)
        self.class_weights = torch.FloatTensor([
            1.0 / self.class_counts[i] for i in range(self.num_classes)
        ])
        
        # Create sample weights for weighted sampling
        self.sample_weights = torch.DoubleTensor([
            1.0 / self.class_counts[label] for label in self.target_labels
        ])
        
        print(f"Dataset: {len(self.df)} images, {self.num_classes} classes")
        print(f"Class distribution: min={min(self.class_counts.values())}, max={max(self.class_counts.values())}")
    
    def __len__(self):
        return len(self.df) * self.augment_factor
    
    def __getitem__(self, idx):
        # Map augmented index back to original index
        original_idx = idx % len(self.df)
        
        # Get image path
        image_filename = self.df.iloc[original_idx]['image_filename']
        image_path = self.image_dir / image_filename
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        # Get target label
        target = torch.tensor(self.target_labels[original_idx], dtype=torch.long)
        
        return image, target

def create_advanced_transforms(image_size=224, is_training=True):
    """Create advanced data augmentation transforms"""
    
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.2), int(image_size * 1.2))),
            # Advanced geometric augmentations
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
            
            # Advanced color augmentations
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            
            # Normalization
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    return transform

class EnhancedSpeciesClassifier(nn.Module):
    """Enhanced classifier with better architecture"""
    
    def __init__(self, num_classes, backbone='efficientnet_v2_s', dropout=0.5, use_attention=True):
        super().__init__()
        
        # Load pre-trained backbone
        if backbone == 'efficientnet_v2_s':
            self.backbone = models.efficientnet_v2_s(weights='DEFAULT')
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'efficientnet_v2_m':
            self.backbone = models.efficientnet_v2_m(weights='DEFAULT')
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights='DEFAULT')
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Enhanced classifier head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.BatchNorm1d(feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(feature_dim // 4, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out

class AdvancedTrainer:
    """Enhanced trainer with modern techniques"""
    
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function with class weights
        train_dataset = train_loader.dataset
        if hasattr(train_dataset, 'class_weights'):
            class_weights = train_dataset.class_weights.to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Tracking variables
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # TensorBoard logging
        self.writer = SummaryWriter(f"runs/enhanced_training_{int(time.time())}")
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
            
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update progress bar
            current_acc = 100. * correct / total
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc='Validating'):
                data, targets = data.to(self.device), targets.to(self.device)
                
                with autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        val_loss /= len(self.val_loader)
        val_acc = 100. * correct / total
        
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/Train', self.train_losses[-1], epoch)
        self.writer.add_scalar('Loss/Val', val_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', self.train_accuracies[-1], epoch)
        self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
        self.writer.add_scalar('Learning_Rate', self.learning_rates[-1], epoch)
        
        # Check for improvement
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.patience_counter = 0
            self.save_best_model(epoch)
            print(f'  🎯 New best validation accuracy: {val_acc:.2f}%')
        else:
            self.patience_counter += 1
        
        print(f'  Train Loss: {self.train_losses[-1]:.4f}, Train Acc: {self.train_accuracies[-1]:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {self.learning_rates[-1]:.6f}')
        
        return val_loss, val_acc
    
    def save_best_model(self, epoch):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'epoch': epoch + 1,
            'config': self.config
        }, 'best_model.pth')
    
    def train(self, num_epochs, early_stopping_patience=20):
        print(f"\n🚀 Starting Enhanced Training")
        print(f"Total epochs: {num_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\n📊 Epoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Early stopping check
            if self.patience_counter >= early_stopping_patience:
                print(f"\n🔴 Early stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
                break
        
        self.writer.close()
        print(f"\n✅ Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")

def create_enhanced_config():
    """Create enhanced configuration for better performance"""
    return {
        "csv_file": "filtered_family_data.csv",
        "image_dir": "images",
        "batch_size": 16,  # Increased from 4
        "num_epochs": 200,  # More epochs with early stopping
        "early_stopping_patience": 25,  # More patience
        "backbone": "efficientnet_v2_s",
        "dropout": 0.5,
        "image_size": 256,  # Larger images
        "learning_rate": 1e-4,  # Better starting LR
        "weight_decay": 0.01,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 4,
        "target_level": "family",
        "augment_factor": 3,  # Triple augmentation
        "use_weighted_sampling": True,
        "cross_validation": True
    }

def train_with_cross_validation(config, n_folds=5):
    """Train with k-fold cross validation for robust results"""
    set_random_seeds(42)
    
    # Load data
    df = pd.read_csv(config['csv_file'])
    
    # Filter to existing images
    image_path = Path(config['image_dir'])
    valid_rows = []
    for idx, row in df.iterrows():
        if (image_path / row['image_filename']).exists():
            valid_rows.append(idx)
    
    df = df.iloc[valid_rows].reset_index(drop=True)
    
    # Prepare labels for stratification
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df[config['target_level']].astype(str))
    
    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
        print(f"\n{'='*60}")
        print(f"🔄 FOLD {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # Create fold-specific datasets
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        # Save temporary CSV files for this fold
        train_df.to_csv(f'fold_{fold}_train.csv', index=False)
        val_df.to_csv(f'fold_{fold}_val.csv', index=False)
        
        # Create transforms
        train_transform = create_advanced_transforms(config['image_size'], is_training=True)
        val_transform = create_advanced_transforms(config['image_size'], is_training=False)
        
        # Create datasets
        train_dataset = AdvancedAugmentationDataset(
            csv_file=f'fold_{fold}_train.csv',
            image_dir=config['image_dir'],
            transform=train_transform,
            target_level=config['target_level'],
            augment_factor=config['augment_factor']
        )
        
        val_dataset = AdvancedAugmentationDataset(
            csv_file=f'fold_{fold}_val.csv',
            image_dir=config['image_dir'],
            transform=val_transform,
            target_level=config['target_level'],
            augment_factor=1  # No augmentation for validation
        )
        
        # Create weighted sampler for balanced training
        if config.get('use_weighted_sampling', True):
            sampler = WeightedRandomSampler(
                weights=train_dataset.sample_weights,
                num_samples=len(train_dataset),
                replacement=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            sampler=sampler,
            shuffle=shuffle,
            num_workers=config['num_workers'],
            pin_memory=True if config['device'] == 'cuda' else False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True if config['device'] == 'cuda' else False
        )
        
        # Create model
        model = EnhancedSpeciesClassifier(
            num_classes=train_dataset.num_classes,
            backbone=config['backbone'],
            dropout=config['dropout']
        )
        
        print(f"Classes in this fold: {train_dataset.num_classes}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Create trainer
        trainer = AdvancedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=config['device']
        )
        
        # Train this fold
        trainer.train(
            num_epochs=config['num_epochs'],
            early_stopping_patience=config['early_stopping_patience']
        )
        
        fold_results.append(trainer.best_val_acc)
        
        # Clean up temporary files
        os.remove(f'fold_{fold}_train.csv')
        os.remove(f'fold_{fold}_val.csv')
        
        print(f"Fold {fold + 1} best accuracy: {trainer.best_val_acc:.2f}%")
    
    # Print cross-validation results
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    
    print(f"\n{'='*60}")
    print(f"🏆 CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Fold accuracies: {[f'{acc:.2f}%' for acc in fold_results]}")
    print(f"Mean accuracy: {mean_acc:.2f} ± {std_acc:.2f}%")
    print(f"Best fold: {max(fold_results):.2f}%")
    print(f"Worst fold: {min(fold_results):.2f}%")
    
    return fold_results, mean_acc, std_acc

def main():
    """Main training function with all improvements"""
    
    print("🚀 ENHANCED SPECIES CLASSIFICATION TRAINING")
    print("=" * 60)
    
    # Set seeds for reproducibility
    set_random_seeds(42)
    
    # Create enhanced configuration
    config = create_enhanced_config()
    
    print("📋 Enhanced Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\n🔧 Device: {config['device']}")
    if config['device'] == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check data availability
    if not os.path.exists(config['csv_file']):
        print(f"❌ CSV file not found: {config['csv_file']}")
        return
    
    if not os.path.exists(config['image_dir']):
        print(f"❌ Image directory not found: {config['image_dir']}")
        return
    
    # Run training with cross-validation
    if config.get('cross_validation', True):
        fold_results, mean_acc, std_acc = train_with_cross_validation(config)
    else:
        # Simple train-val split training (fallback)
        print("Running simple train-validation split...")
        # Implementation for simple split...
    
    print(f"\n✅ Training completed!")
    print(f"Expected performance: {mean_acc:.2f} ± {std_acc:.2f}%")

if __name__ == "__main__":
    main() 