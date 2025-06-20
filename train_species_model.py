#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
import numpy as np
from PIL import Image
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
from tqdm import tqdm

def mixup_data(x, y, alpha=1.0):
    """Apply mixup augmentation to batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation to batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    y_a, y_b = y, y[index]
    
    # Generate random bounding box
    W = x.size(3)
    H = x.size(2)
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda based on actual cut area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y_a, y_b, lam

class SpeciesDataset(Dataset):
    """Custom dataset for species images with hierarchical labels"""
    
    def __init__(self, csv_file, image_dir, transform=None, target_level='scientificName'):
        """
        Args:
            csv_file: Path to CSV with species data
            image_dir: Directory containing images
            transform: Image transformations
            target_level: Which taxonomic level to predict ('scientificName', 'genus', 'family', etc.)
        """
        print(f"Loading dataset from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.target_level = target_level
        
        # Create label encoders for all taxonomic levels
        self.label_encoders = {}
        self.taxonomic_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'scientificName']
        
        for level in self.taxonomic_levels:
            if level in self.df.columns:
                le = LabelEncoder()
                # Fill NaN with 'Unknown' and convert to string
                labels = self.df[level].fillna('Unknown').astype(str)
                le.fit(labels)
                self.label_encoders[level] = le
                print(f"  {level}: {len(le.classes_)} classes")
        
        # Set primary target
        if target_level not in self.label_encoders:
            raise ValueError(f"Target level '{target_level}' not found in data")
        
        self.target_labels = self.label_encoders[target_level].transform(
            self.df[target_level].fillna('Unknown').astype(str)
        )
        self.num_classes = len(self.label_encoders[target_level].classes_)
        
        print(f"Dataset loaded: {len(self.df)} images, {self.num_classes} {target_level} classes")
        
        # Check how many images actually exist
        self.valid_indices = []
        missing_count = 0
        
        for idx in range(len(self.df)):
            image_path = self.image_dir / self.df.iloc[idx]['image_filename']
            if image_path.exists():
                self.valid_indices.append(idx)
            else:
                missing_count += 1
        
        print(f"Found {len(self.valid_indices)} existing images, {missing_count} missing")
        
        # Update dataframe to only include rows with existing images
        self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)
        self.target_labels = self.target_labels[self.valid_indices]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image path
        image_filename = self.df.iloc[idx]['image_filename']
        image_path = self.image_dir / image_filename
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            # Return black image if loading fails
            print(f"Error loading {image_path}: {e}")
            image = torch.zeros(3, 224, 224)
        
        # Get target label
        target = torch.tensor(self.target_labels[idx], dtype=torch.long)
        
        # Get hierarchical labels (for potential hierarchical loss)
        hierarchical_labels = {}
        for level in self.taxonomic_levels:
            if level in self.df.columns and level in self.label_encoders:
                label_text = str(self.df.iloc[idx][level]) if pd.notna(self.df.iloc[idx][level]) else 'Unknown'
                hierarchical_labels[level] = torch.tensor(
                    self.label_encoders[level].transform([label_text])[0], 
                    dtype=torch.long
                )
        
        return image, target, hierarchical_labels

class EfficientNetSpeciesClassifier(nn.Module):
    """EfficientNet-based species classifier with optional hierarchical outputs"""
    
    def __init__(self, num_classes, backbone='efficientnet_v2_s', dropout=0.3, hierarchical=False, num_classes_dict=None):
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
        
        self.hierarchical = hierarchical
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            self.backbone,
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        
        if hierarchical and num_classes_dict:
            # Hierarchical classifiers
            self.classifiers = nn.ModuleDict()
            for level, num_classes_level in num_classes_dict.items():
                self.classifiers[level] = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(feature_dim // 2, num_classes_level)
                )
        else:
            # Simple classifier
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim // 2, num_classes)
            )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        if self.hierarchical:
            outputs = {}
            for level, classifier in self.classifiers.items():
                outputs[level] = classifier(features)
            return outputs
        else:
            return self.classifier(features)

class SpeciesTrainer:
    """Training pipeline for species classification"""
    
    def __init__(self, model, train_loader, val_loader, config, device='cuda', log_dir='runs'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Set up logging
        self.writer = SummaryWriter(log_dir)
        
        # Loss function with optional label smoothing
        if config.get('label_smoothing', 0) > 0:
            # Note: PyTorch's CrossEntropyLoss supports label_smoothing in newer versions
            try:
                self.criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
            except TypeError:
                print("⚠️  Label smoothing not supported in this PyTorch version, using standard CrossEntropyLoss")
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer with configurable learning rate and weight decay
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.get('learning_rate', 3e-4), 
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Apply better weight initialization
        self._initialize_weights()
        
        # Learning rate scheduler
        if config.get('use_cosine_schedule', False):
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=config.get('num_epochs', 100),
                eta_min=config.get('learning_rate', 3e-4) * 0.01
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=7
            )
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
    
    def _initialize_weights(self):
        """Apply better weight initialization to the classifier layers"""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (images, targets, hierarchical_labels) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Apply augmentation if enabled
            use_mixup = self.config.get('use_mixup', False) and np.random.rand() < 0.5
            use_cutmix = self.config.get('use_cutmix', False) and np.random.rand() < 0.5 and not use_mixup
            
            if use_mixup:
                images, targets_a, targets_b, lam = mixup_data(
                    images, targets, alpha=self.config.get('mixup_alpha', 0.2)
                )
                
                # Forward pass
                if self.model.hierarchical:
                    outputs = self.model(images)
                    loss = mixup_criterion(self.criterion, outputs['scientificName'], targets_a, targets_b, lam)
                else:
                    outputs = self.model(images)
                    loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                    
            elif use_cutmix:
                images, targets_a, targets_b, lam = cutmix_data(
                    images, targets, alpha=self.config.get('cutmix_alpha', 1.0)
                )
                
                # Forward pass
                if self.model.hierarchical:
                    outputs = self.model(images)
                    loss = mixup_criterion(self.criterion, outputs['scientificName'], targets_a, targets_b, lam)
                else:
                    outputs = self.model(images)
                    loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                    
            else:
                # Standard training
                if self.model.hierarchical:
                    outputs = self.model(images)
                    loss = self.criterion(outputs['scientificName'], targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            if self.model.hierarchical:
                _, predicted = outputs['scientificName'].max(1)
            else:
                _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            for images, targets, hierarchical_labels in progress_bar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                if self.model.hierarchical:
                    outputs = self.model(images)
                    loss = self.criterion(outputs['scientificName'], targets)
                    _, predicted = outputs['scientificName'].max(1)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                    _, predicted = outputs.max(1)
                
                # Statistics
                total_loss += loss.item()
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store for detailed analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_predictions, all_targets
    
    def train(self, num_epochs, early_stopping_patience=15):
        """Full training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, predictions, targets = self.validate(epoch)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Val', val_acc, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print epoch results
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Learning rate scheduling
            if self.config.get('use_cosine_schedule', False):
                self.scheduler.step()  # CosineAnnealingLR steps every epoch
            else:
                self.scheduler.step(val_loss)  # ReduceLROnPlateau steps on validation loss
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                self.save_model('best_model.pth')
                print(f'  New best validation accuracy: {val_acc:.2f}%')
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f'Early stopping after {epoch+1} epochs (no improvement for {early_stopping_patience} epochs)')
                break
            
            print()
        
        training_time = time.time() - start_time
        print(f'Training completed in {training_time/60:.1f} minutes')
        print(f'Best validation accuracy: {self.best_val_acc:.2f}%')
        
        self.writer.close()
    
    def save_model(self, filepath):
        """Save model and training info"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }, filepath)
        print(f'Model saved to {filepath}')

def create_data_transforms(image_size=224):
    """Create data augmentation transforms"""
    
    # Enhanced training transforms with more aggressive augmentation
    train_transform = transforms.Compose([
        transforms.Resize((int(image_size * 1.2), int(image_size * 1.2))),
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),  # Some species might benefit from this
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3))  # Random erasing
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def analyze_dataset(csv_file):
    """Analyze the dataset before training"""
    print("📊 DATASET ANALYSIS")
    print("=" * 50)
    
    df = pd.read_csv(csv_file)
    
    print(f"Total observations: {len(df):,}")
    print(f"Unique species: {df['scientificName'].nunique():,}")
    
    # Species distribution
    species_counts = df['scientificName'].value_counts()
    print(f"\nSpecies distribution:")
    print(f"  Min samples per species: {species_counts.min()}")
    print(f"  Max samples per species: {species_counts.max()}")
    print(f"  Mean samples per species: {species_counts.mean():.1f}")
    print(f"  Median samples per species: {species_counts.median():.1f}")
    
    # Check for class imbalance
    imbalance_ratio = species_counts.max() / species_counts.min()
    print(f"  Class imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 10:
        print("  ⚠️  High class imbalance detected - consider class weighting")
    
    # Top species
    print(f"\nTop 10 most common species:")
    for species, count in species_counts.head(10).items():
        print(f"  {species}: {count} samples")
    
    # Taxonomic diversity
    for level in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']:
        if level in df.columns:
            unique_count = df[level].nunique()
            print(f"Unique {level}: {unique_count}")
    
    return df

def create_train_val_split(df, test_size=0.2, random_state=42):
    """Create stratified train/validation split"""
    print(f"\n📋 CREATING TRAIN/VAL SPLIT")
    print("=" * 50)
    
    # Stratified split to maintain species distribution
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['scientificName'], 
        random_state=random_state
    )
    
    print(f"Training set: {len(train_df):,} observations")
    print(f"Validation set: {len(val_df):,} observations")
    print(f"Species in training: {train_df['scientificName'].nunique():,}")
    print(f"Species in validation: {val_df['scientificName'].nunique():,}")
    
    return train_df, val_df

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """Plot training curves"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🚀 SPECIES RECOGNITION AI TRAINING")
    print("=" * 50)
    
    # Load configuration from JSON file (prioritize v2 if exists)
    config_files = ['fixed_config.json', 'improved_config_v2.json', 'improved_config.json']
    config_file = None
    
    for cf in config_files:
        if os.path.exists(cf):
            config_file = cf
            break
    
    if not config_file:
        print("❌ No configuration file found!")
        print("Please ensure improved_config.json or improved_config_v2.json exists.")
        return
        
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded configuration from {config_file}")
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing {config_file}: {e}")
        return
    
    # Set device if not specified or if cuda not available
    if 'device' not in config:
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif config['device'] == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, using CPU")
        config['device'] = 'cpu'
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Check if image directory exists
    if not os.path.exists(config['image_dir']):
        print(f"❌ Image directory '{config['image_dir']}' not found!")
        print("You need to download the images first. Options:")
        print("1. Use the image download functionality in process_observations.py")
        print("2. Create a symbolic link to your existing image directory")
        print("3. Update the image_dir path in the config above")
        return
    
    # Analyze dataset
    df = analyze_dataset(config['csv_file'])
    
    # Create train/val split
    train_df, val_df = create_train_val_split(df)
    
    # Save splits for reproducibility
    train_df.to_csv('train_split.csv', index=False)
    val_df.to_csv('val_split.csv', index=False)
    print("💾 Saved train_split.csv and val_split.csv")
    
    # Create transforms
    train_transform, val_transform = create_data_transforms(config['image_size'])
    
    # Create datasets
    print(f"\n🔄 CREATING DATASETS")
    print("=" * 50)
    
    target_level = config.get('target_level', 'scientificName')
    
    train_dataset = SpeciesDataset(
        csv_file='train_split.csv',
        image_dir=config['image_dir'],
        transform=train_transform,
        target_level=target_level
    )
    
    val_dataset = SpeciesDataset(
        csv_file='val_split.csv',
        image_dir=config['image_dir'],
        transform=val_transform,
        target_level=target_level
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print(f"\n🤖 CREATING MODEL")
    print("=" * 50)
    
    model = EfficientNetSpeciesClassifier(
        num_classes=train_dataset.num_classes,
        backbone=config['backbone'],
        dropout=config.get('dropout', 0.3),
        hierarchical=False  # Set to True for hierarchical classification
    )
    
    print(f"Model: {config['backbone']}")
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = SpeciesTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config['device'],
        log_dir=f"runs/species_training_{int(time.time())}"
    )
    
    # Start training
    print(f"\n🎯 STARTING TRAINING")
    print("=" * 50)
    
    trainer.train(
        num_epochs=config['num_epochs'],
        early_stopping_patience=config.get('early_stopping_patience', 15)
    )
    
    # Plot training curves
    plot_training_curves(
        trainer.train_losses,
        trainer.val_losses,
        trainer.train_accuracies,
        trainer.val_accuracies
    )
    
    # Save label encoder
    with open(f'{target_level}_label_encoder.json', 'w') as f:
        label_mapping = {
            'classes': train_dataset.label_encoders[target_level].classes_.tolist(),
            'class_to_idx': {cls: idx for idx, cls in enumerate(train_dataset.label_encoders[target_level].classes_)}
        }
        json.dump(label_mapping, f, indent=2)
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Model saved as: best_model.pth")
    print(f"Label encoder saved as: {target_level}_label_encoder.json")
    print(f"Training curves saved as: training_curves.png")

if __name__ == "__main__":
    main() 