#!/usr/bin/env python3
"""
Quick launcher for the improved model
Tests improvements with a single fold first for speed
"""

import os
import sys
import time
import torch
from pathlib import Path

# Import our enhanced model
from improved_model_v3 import *

def quick_test():
    """Quick test with single train-val split"""
    print("🚀 QUICK TEST - IMPROVED SPECIES CLASSIFICATION")
    print("=" * 60)
    
    # Set seeds
    set_random_seeds(42)
    
    # Create config for quick test
    config = {
        "csv_file": "filtered_family_data.csv",
        "image_dir": "images", 
        "batch_size": 16,
        "num_epochs": 50,  # Reduced for quick test
        "early_stopping_patience": 15,
        "backbone": "efficientnet_v2_s",
        "dropout": 0.5,
        "image_size": 256,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 2,
        "target_level": "family",
        "augment_factor": 2,  # Reduced for speed
        "use_weighted_sampling": True
    }
    
    print("📋 Quick Test Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Check prerequisites
    if not os.path.exists(config['csv_file']):
        print(f"❌ CSV file not found: {config['csv_file']}")
        return
    
    if not os.path.exists(config['image_dir']):
        print(f"❌ Image directory not found: {config['image_dir']}")
        return
    
    # Load and split data
    print(f"\n📊 Loading data...")
    df = pd.read_csv(config['csv_file'])
    
    # Filter to existing images
    image_path = Path(config['image_dir'])
    valid_rows = []
    for idx, row in df.iterrows():
        if (image_path / row['image_filename']).exists():
            valid_rows.append(idx)
    
    df = df.iloc[valid_rows].reset_index(drop=True)
    print(f"Valid samples: {len(df)}")
    
    # Create simple train-val split
    from sklearn.model_selection import train_test_split
    labels = df[config['target_level']].astype(str)
    
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Save temporary splits
    train_df.to_csv('quick_train.csv', index=False)
    val_df.to_csv('quick_val.csv', index=False)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create transforms
    train_transform = create_advanced_transforms(config['image_size'], is_training=True)
    val_transform = create_advanced_transforms(config['image_size'], is_training=False)
    
    # Create datasets
    train_dataset = AdvancedAugmentationDataset(
        csv_file='quick_train.csv',
        image_dir=config['image_dir'],
        transform=train_transform,
        target_level=config['target_level'],
        augment_factor=config['augment_factor']
    )
    
    val_dataset = AdvancedAugmentationDataset(
        csv_file='quick_val.csv',
        image_dir=config['image_dir'],
        transform=val_transform,
        target_level=config['target_level'],
        augment_factor=1
    )
    
    print(f"Number of classes: {train_dataset.num_classes}")
    print(f"Training samples (with augmentation): {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=train_dataset.sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
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
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config['device']
    )
    
    # Train
    print(f"\n🎯 Starting training...")
    start_time = time.time()
    
    trainer.train(
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience']
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n✅ Quick test completed!")
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Improvement over previous 13%: +{trainer.best_val_acc - 13:.2f}%")
    
    # Clean up
    os.remove('quick_train.csv')
    os.remove('quick_val.csv')
    
    # Recommend next steps
    if trainer.best_val_acc > 25:  # Significant improvement
        print(f"\n🎉 Great improvement! Consider running full cross-validation:")
        print(f"python improved_model_v3.py")
    else:
        print(f"\n🤔 Modest improvement. Consider:")
        print(f"1. Increasing image_size to 384")
        print(f"2. Using efficientnet_v2_m backbone")
        print(f"3. More aggressive augmentation")
    
    return trainer.best_val_acc

def check_system():
    """Check system requirements and data availability"""
    print("🔍 SYSTEM CHECK")
    print("=" * 40)
    
    # Check Python packages
    required_packages = ['torch', 'torchvision', 'pandas', 'sklearn', 'PIL', 'tqdm']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} missing")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"⚠️  GPU not available, will use CPU (slower)")
    
    # Check data files
    if os.path.exists('filtered_family_data.csv'):
        print(f"✅ Training data available")
    else:
        print(f"❌ filtered_family_data.csv not found")
    
    if os.path.exists('images'):
        image_count = len(list(Path('images').glob('*.jpg')))
        print(f"✅ Image directory available ({image_count} images)")
    else:
        print(f"❌ images directory not found")
    
    print()

if __name__ == "__main__":
    # Check system first
    check_system()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        # Run full cross-validation
        print("Running full cross-validation training...")
        os.system('python improved_model_v3.py')
    else:
        # Run quick test
        accuracy = quick_test() 