#!/usr/bin/env python3
"""
SightTrack AI - Main Training Script
Professional training script for species classification model
"""

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.model import create_model
from src.dataset import create_dataloaders
from src.trainer import SpeciesTrainer
from src.utils import setup_logging, set_random_seed, get_device_info


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def update_config_paths(config: Dict[str, Any]) -> None:
    """Update configuration paths to be absolute."""
    # Update data paths
    config["data"]["csv_file"] = str(Path(config["data"]["csv_file"]).resolve())
    config["data"]["image_dir"] = str(Path(config["data"]["image_dir"]).resolve())
    
    # Update output paths
    for path_key in config["paths"]:
        config["paths"][path_key] = str(Path(config["paths"][path_key]).resolve())
        Path(config["paths"][path_key]).mkdir(parents=True, exist_ok=True)


def validate_data_files(config: Dict[str, Any]) -> None:
    """Validate that required data files exist."""
    csv_file = Path(config["data"]["csv_file"])
    image_dir = Path(config["data"]["image_dir"])
    
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Check if image directory has images
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {image_dir}")
    
    print(f"✓ Data validation passed")
    print(f"  - CSV file: {csv_file}")
    print(f"  - Image directory: {image_dir}")
    print(f"  - Number of images: {len(image_files)}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="SightTrack AI - Species Classification Training")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/model_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", 
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Update paths and create directories
    update_config_paths(config)
    
    # Set up logging
    setup_logging(
        log_file=Path(config["paths"]["logs_dir"]) / "training.log",
        verbose=args.debug or config["logging"]["verbose"]
    )
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    config["system"]["device"] = device
    
    # Print system information
    print("=" * 60)
    print("SightTrack AI - Species Classification Training")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    print(f"Device: {device}")
    
    device_info = get_device_info()
    for key, value in device_info.items():
        print(f"{key}: {value}")
    
    print("=" * 60)
    
    # Validate data files
    print("Validating data files...")
    try:
        validate_data_files(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Data validation failed: {e}")
        print("\nTo fix this:")
        print("1. Run ./scripts/download_data.sh to download the dataset")
        print("2. Ensure your CSV file and images are in the correct locations")
        print("3. Update the paths in your config file if needed")
        sys.exit(1)
    
    # Create data loaders
    print("Creating data loaders...")
    try:
        train_loader, val_loader, dataset = create_dataloaders(
            config, 
            validation_split=config["training"]["validation_split"]
        )
        print(f"✓ Training samples: {len(train_loader.dataset)}")
        print(f"✓ Validation samples: {len(val_loader.dataset)}")
        print(f"✓ Number of classes: {dataset.num_classes}")
        print(f"✓ Target level: {dataset.target_level}")
    except Exception as e:
        print(f"❌ Failed to create data loaders: {e}")
        sys.exit(1)
    
    # Create model
    print("Creating model...")
    try:
        model = create_model(config)
        model_info = model.get_model_info()
        print(f"✓ Model: {model_info['backbone']}")
        print(f"✓ Parameters: {model_info['total_parameters']:,}")
        print(f"✓ Model size: {model_info['model_size_mb']:.1f} MB")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        sys.exit(1)
    
    # Save label encoder
    label_encoder_path = Path(config["paths"]["model_save_dir"]) / "label_encoder.json"
    dataset.save_label_encoder(str(label_encoder_path))
    
    # Create trainer
    print("Initializing trainer...")
    try:
        trainer = SpeciesTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
    except Exception as e:
        print(f"❌ Failed to create trainer: {e}")
        sys.exit(1)
    
    # Start training
    print("Starting training...")
    print("=" * 60)
    
    try:
        results = trainer.train()
        
        print("=" * 60)
        print("Training completed successfully!")
        print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
        print(f"Final epoch: {results['final_epoch']}")
        print(f"Model saved to: {config['paths']['model_save_dir']}")
        print(f"Logs saved to: {config['paths']['logs_dir']}")
        
        # Save training results
        results_file = Path(config["paths"]["results_dir"]) / "training_results.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        print(f"Results saved to: {results_file}")
        
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)
    
    print("=" * 60)


if __name__ == "__main__":
    main() 