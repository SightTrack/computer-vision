#!/usr/bin/env python3
"""
Model Improvement Script - Various strategies to enhance species classification performance
"""

import torch
import itertools
import json
from train_species_model import *
import time

class ModelImprover:
    """Class to help improve model performance through various strategies"""
    
    def __init__(self, base_config):
        self.base_config = base_config
        self.results = []
    
    def hyperparameter_search(self, param_grid, max_experiments=10):
        """
        Perform hyperparameter search to find better configurations
        
        Args:
            param_grid: Dictionary of parameters to search
            max_experiments: Maximum number of experiments to run
        """
        print("🔍 HYPERPARAMETER SEARCH")
        print("=" * 50)
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        all_combinations = list(itertools.product(*values))
        
        # Limit experiments
        if len(all_combinations) > max_experiments:
            import random
            random.shuffle(all_combinations)
            all_combinations = all_combinations[:max_experiments]
        
        print(f"Running {len(all_combinations)} experiments...")
        
        best_accuracy = 0.0
        best_config = None
        
        for i, combination in enumerate(all_combinations, 1):
            # Create experiment config
            experiment_config = self.base_config.copy()
            for key, value in zip(keys, combination):
                experiment_config[key] = value
            
            print(f"\n📊 Experiment {i}/{len(all_combinations)}")
            print(f"Config: {dict(zip(keys, combination))}")
            
            try:
                # Run training with this configuration
                accuracy = self._train_with_config(experiment_config, experiment_id=i)
                
                # Track results
                result = {
                    'experiment_id': i,
                    'config': dict(zip(keys, combination)),
                    'accuracy': accuracy,
                    'timestamp': time.time()
                }
                self.results.append(result)
                
                # Check if best
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_config = experiment_config
                    print(f"🎯 New best accuracy: {accuracy:.2f}%")
                
            except Exception as e:
                print(f"❌ Experiment {i} failed: {e}")
                continue
        
        # Save results
        with open('hyperparameter_search_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n🏆 BEST RESULTS")
        print(f"Best accuracy: {best_accuracy:.2f}%")
        print(f"Best config: {best_config}")
        
        return best_config, best_accuracy
    
    def _train_with_config(self, config, experiment_id):
        """Train model with given configuration and return validation accuracy"""
        
        # Load data (same splits for fair comparison)
        if os.path.exists('train_split.csv') and os.path.exists('val_split.csv'):
            train_df = pd.read_csv('train_split.csv')
            val_df = pd.read_csv('val_split.csv')
        else:
            df = pd.read_csv(config['csv_file'])
            train_df, val_df = create_train_val_split(df)
        
        # Create transforms
        train_transform, val_transform = create_data_transforms(config['image_size'])
        
        # Create datasets
        train_dataset = SpeciesDataset(
            csv_file='train_split.csv',
            image_dir=config['image_dir'],
            transform=train_transform,
            target_level='scientificName'
        )
        
        val_dataset = SpeciesDataset(
            csv_file='val_split.csv',
            image_dir=config['image_dir'],
            transform=val_transform,
            target_level='scientificName'
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
        
        # Create model
        model = EfficientNetSpeciesClassifier(
            num_classes=train_dataset.num_classes,
            backbone=config['backbone'],
            dropout=config.get('dropout', 0.3),
            hierarchical=False
        )
        
        # Create trainer with custom parameters
        trainer = SpeciesTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=config['device'],
            log_dir=f"runs/experiment_{experiment_id}_{int(time.time())}"
        )
        
        # Customize optimizer if specified
        if 'learning_rate' in config:
            trainer.optimizer = optim.AdamW(
                model.parameters(), 
                lr=config['learning_rate'], 
                weight_decay=config.get('weight_decay', 1e-4)
            )
        
        # Train with early stopping
        trainer.train(
            num_epochs=config.get('num_epochs', 50),  # Shorter for hyperparameter search
            early_stopping_patience=config.get('early_stopping_patience', 10)
        )
        
        return trainer.best_val_acc

def suggest_improvements():
    """Suggest specific improvement strategies based on common issues"""
    
    print("💡 MODEL IMPROVEMENT SUGGESTIONS")
    print("=" * 50)
    
    improvements = {
        "1. Hyperparameter Tuning": {
            "description": "Optimize learning rate, batch size, and model architecture",
            "impact": "High",
            "effort": "Medium",
            "details": [
                "Try different learning rates: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]",
                "Experiment with batch sizes: [16, 32, 64, 128]",
                "Test different backbones: efficientnet_v2_s, efficientnet_v2_m, resnet50",
                "Adjust dropout rates: [0.1, 0.3, 0.5]"
            ]
        },
        
        "2. Data Augmentation": {
            "description": "Enhance training data with more aggressive augmentations",
            "impact": "High", 
            "effort": "Low",
            "details": [
                "Add CutMix and MixUp augmentations",
                "Increase rotation angles and color jitter",
                "Add random erasing and grayscale conversion",
                "Use AutoAugment or RandAugment policies"
            ]
        },
        
        "3. Advanced Training Techniques": {
            "description": "Use modern training strategies",
            "impact": "Medium-High",
            "effort": "Medium",
            "details": [
                "Implement focal loss for class imbalance",
                "Use cosine annealing learning rate schedule",
                "Add label smoothing",
                "Try gradient accumulation for larger effective batch size"
            ]
        },
        
        "4. Model Architecture": {
            "description": "Experiment with different model architectures",
            "impact": "Medium-High",
            "effort": "Medium",
            "details": [
                "Try newer models: ConvNeXt, Swin Transformer, EfficientNetV2-L",
                "Implement ensemble of multiple models",
                "Add attention mechanisms",
                "Use hierarchical classification"
            ]
        },
        
        "5. Data Quality": {
            "description": "Improve the training dataset",
            "impact": "Very High",
            "effort": "High",
            "details": [
                "Collect more training data, especially for rare species",
                "Remove mislabeled or poor-quality images",
                "Balance the dataset across species",
                "Add diverse image conditions (lighting, angles, backgrounds)"
            ]
        }
    }
    
    for strategy, info in improvements.items():
        print(f"\n{strategy}")
        print(f"Impact: {info['impact']} | Effort: {info['effort']}")
        print(f"Description: {info['description']}")
        print("Details:")
        for detail in info['details']:
            print(f"  • {detail}")

def create_improved_config():
    """Create an improved configuration based on best practices"""
    
    improved_config = {
        # Data
        'csv_file': 'ai_training_data.csv',
        'image_dir': 'images',
        
        # Training parameters
        'batch_size': 32,  # Good balance for most GPUs
        'num_epochs': 100,
        'early_stopping_patience': 15,
        
        # Model
        'backbone': 'efficientnet_v2_s',  # Good balance of speed/accuracy
        'dropout': 0.4,  # Slightly higher for better regularization
        'image_size': 224,
        
        # Optimization
        'learning_rate': 1e-4,  # Conservative but effective
        'weight_decay': 1e-4,
        'use_cosine_schedule': True,
        'label_smoothing': 0.1,
        
        # System
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        
        # Advanced techniques
        'use_mixup': True,
        'mixup_alpha': 0.2,
        'use_cutmix': True,
        'cutmix_alpha': 1.0,
    }
    
    return improved_config

def main():
    print("🚀 MODEL IMPROVEMENT TOOLKIT")
    print("=" * 50)
    
    # Base configuration
    base_config = {
        'csv_file': 'ai_training_data.csv',
        'image_dir': 'images',
        'batch_size': 32,
        'num_epochs': 30,  # Shorter for experiments
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'early_stopping_patience': 10
    }
    
    # Show improvement suggestions
    suggest_improvements()
    
    print(f"\n" + "="*50)
    choice = input("\nWhat would you like to do?\n"
                  "1. Quick hyperparameter search\n"
                  "2. Show improved config\n"
                  "3. Manual training with custom config\n"
                  "Choice (1-3): ")
    
    if choice == "1":
        # Quick hyperparameter search
        param_grid = {
            'learning_rate': [1e-5, 3e-5, 1e-4, 3e-4],
            'batch_size': [16, 32, 64],
            'backbone': ['efficientnet_v2_s', 'resnet50'],
            'dropout': [0.3, 0.4, 0.5],
            'image_size': [224, 256]
        }
        
        improver = ModelImprover(base_config)
        best_config, best_acc = improver.hyperparameter_search(param_grid, max_experiments=5)
        
        print(f"\n💾 Save best config for future use:")
        with open('best_config.json', 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"Saved to: best_config.json")
        
    elif choice == "2":
        improved_config = create_improved_config()
        print(f"\n📋 IMPROVED CONFIGURATION:")
        print(json.dumps(improved_config, indent=2))
        
        save_config = input("\nSave this config? (y/n): ")
        if save_config.lower() == 'y':
            with open('improved_config.json', 'w') as f:
                json.dump(improved_config, f, indent=2)
            print("Saved to: improved_config.json")
            print("You can modify train_species_model.py to use this config")
    
    elif choice == "3":
        print(f"\n📝 To manually improve the model:")
        print(f"1. Edit the 'config' dictionary in train_species_model.py")
        print(f"2. Try the suggestions shown above")
        print(f"3. Run: python train_species_model.py")
        print(f"4. Compare results with previous runs")

if __name__ == "__main__":
    main() 