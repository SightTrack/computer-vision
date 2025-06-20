#!/usr/bin/env python3

import pandas as pd
import torch
import json
import numpy as np
from pathlib import Path
import torch.nn as nn
from train_species_model import SpeciesDataset, EfficientNetSpeciesClassifier, create_data_transforms
from torch.utils.data import DataLoader

def debug_validation_issue():
    """Debug why validation accuracy is always 0"""
    
    print("🔍 DEBUGGING VALIDATION ACCURACY ISSUE")
    print("=" * 60)
    
    # 1. Check data splits
    print("1. DATA SPLITS ANALYSIS")
    print("-" * 30)
    try:
        train_df = pd.read_csv('train_split.csv')
        val_df = pd.read_csv('val_split.csv')
        print(f"✅ Train samples: {len(train_df):,}")
        print(f"✅ Val samples: {len(val_df):,}")
        print(f"✅ Train species: {train_df['scientificName'].nunique():,}")
        print(f"✅ Val species: {val_df['scientificName'].nunique():,}")
        
        # Check species overlap
        train_species = set(train_df['scientificName'])
        val_species = set(val_df['scientificName'])
        overlap = train_species & val_species
        print(f"✅ Species overlap: {len(overlap):,} / {len(val_species):,}")
        
        if len(overlap) != len(val_species):
            print("❌ PROBLEM: Some validation species not in training set!")
            missing = val_species - train_species
            print(f"   Missing species: {list(missing)[:5]}...")
            
    except Exception as e:
        print(f"❌ Data split error: {e}")
        return
    
    # 2. Check label encoder
    print(f"\n2. LABEL ENCODER ANALYSIS")
    print("-" * 30)
    try:
        with open('species_label_encoder.json', 'r') as f:
            label_data = json.load(f)
        classes = label_data['classes']
        print(f"✅ Number of classes in encoder: {len(classes):,}")
        print(f"✅ First few classes: {classes[:3]}")
        print(f"✅ Last few classes: {classes[-3:]}")
        
        # Check if all val species are in encoder
        encoder_species = set(classes)
        missing_from_encoder = val_species - encoder_species
        if missing_from_encoder:
            print(f"❌ PROBLEM: {len(missing_from_encoder)} val species missing from encoder!")
            print(f"   Examples: {list(missing_from_encoder)[:3]}")
            
    except Exception as e:
        print(f"❌ Label encoder error: {e}")
        return
    
    # 3. Test dataset loading
    print(f"\n3. DATASET LOADING TEST")
    print("-" * 30)
    try:
        # Load config
        with open('improved_config_v2.json', 'r') as f:
            config = json.load(f)
            
        # Create transforms
        _, val_transform = create_data_transforms(config['image_size'])
        
        # Create validation dataset
        val_dataset = SpeciesDataset(
            csv_file='val_split.csv',
            image_dir=config['image_dir'],
            transform=val_transform,
            target_level='scientificName'
        )
        
        print(f"✅ Dataset created with {len(val_dataset)} samples")
        print(f"✅ Number of classes: {val_dataset.num_classes}")
        
        # Test first few samples
        print(f"\n   Testing first 3 samples:")
        for i in range(min(3, len(val_dataset))):
            try:
                image, target, hierarchical = val_dataset[i]
                print(f"   Sample {i}: image shape {image.shape}, target {target}, type {type(target)}")
                print(f"                target value: {target.item()}, valid range: 0-{val_dataset.num_classes-1}")
                
                if target.item() < 0 or target.item() >= val_dataset.num_classes:
                    print(f"   ❌ PROBLEM: Target {target.item()} out of range!")
                    
            except Exception as e:
                print(f"   ❌ Error loading sample {i}: {e}")
                
    except Exception as e:
        print(f"❌ Dataset loading error: {e}")
        return
    
    # 4. Test model loading and prediction
    print(f"\n4. MODEL PREDICTION TEST")
    print("-" * 30)
    try:
        # Load model
        checkpoint = torch.load('best_model.pth', map_location='cpu')
        
        # Create model
        model = EfficientNetSpeciesClassifier(
            num_classes=val_dataset.num_classes,
            backbone=config['backbone'],
            dropout=config.get('dropout', 0.3),
            hierarchical=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Model loaded successfully")
        
        # Test prediction on first batch
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        with torch.no_grad():
            for batch_idx, (images, targets, hierarchical_labels) in enumerate(val_loader):
                print(f"   Batch {batch_idx}:")
                print(f"     Images shape: {images.shape}")
                print(f"     Targets: {targets}")
                print(f"     Target range: {targets.min().item()} to {targets.max().item()}")
                
                # Forward pass
                outputs = model(images)
                print(f"     Output shape: {outputs.shape}")
                print(f"     Output range: {outputs.min().item():.4f} to {outputs.max().item():.4f}")
                
                # Get predictions
                _, predicted = outputs.max(1)
                print(f"     Predictions: {predicted}")
                print(f"     Prediction range: {predicted.min().item()} to {predicted.max().item()}")
                
                # Check accuracy calculation
                correct = predicted.eq(targets).sum().item()
                total = targets.size(0)
                accuracy = 100. * correct / total
                print(f"     Batch accuracy: {accuracy:.2f}% ({correct}/{total})")
                
                # Check if predictions are always the same
                unique_preds = torch.unique(predicted)
                print(f"     Unique predictions: {len(unique_preds)} values: {unique_preds}")
                
                if len(unique_preds) == 1:
                    print(f"   ❌ PROBLEM: Model always predicts class {unique_preds[0].item()}!")
                
                break  # Only test first batch
                
    except Exception as e:
        print(f"❌ Model prediction error: {e}")
        return
    
    # 5. Check training history
    print(f"\n5. TRAINING HISTORY ANALYSIS")
    print("-" * 30)
    try:
        train_accs = checkpoint['train_accuracies']
        val_accs = checkpoint['val_accuracies']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        
        print(f"✅ Training epochs: {len(train_accs)}")
        print(f"✅ Train accuracies: {train_accs}")
        print(f"✅ Val accuracies: {val_accs}")
        print(f"✅ Train losses: {[f'{x:.4f}' for x in train_losses]}")
        print(f"✅ Val losses: {[f'{x:.4f}' for x in val_losses]}")
        
        if all(acc == 0 for acc in val_accs):
            print(f"❌ CONFIRMED: All validation accuracies are 0!")
            
        if all(acc == train_accs[0] for acc in train_accs):
            print(f"❌ PROBLEM: Training accuracy not improving!")
            
    except Exception as e:
        print(f"❌ Training history error: {e}")

if __name__ == "__main__":
    debug_validation_issue() 