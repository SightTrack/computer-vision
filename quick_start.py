#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    print("🔍 CHECKING REQUIREMENTS...")
    
    required_packages = [
        'torch', 'torchvision', 'pandas', 'numpy', 'PIL', 
        'sklearn', 'matplotlib', 'tqdm', 'requests'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            if package == 'PIL':
                try:
                    __import__('Pillow')
                except ImportError:
                    missing.append('Pillow')
            else:
                missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All requirements installed!")
        return True

def check_files():
    """Check if required files exist"""
    print("\n📁 CHECKING FILES...")
    
    required_files = {
        'ai_training_data.csv': 'Processed training data',
        'train_species_model.py': 'Training script',
        'inference.py': 'Inference script',
        'download_images.py': 'Image download script'
    }
    
    all_good = True
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"✅ {filename} - {description}")
        else:
            print(f"❌ {filename} - {description} (MISSING)")
            all_good = False
    
    return all_good

def check_gpu():
    """Check GPU availability"""
    print("\n🖥️  CHECKING GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  No GPU available - will use CPU (slower)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def download_sample_images():
    """Download sample images for quick testing"""
    print("\n📥 DOWNLOADING SAMPLE IMAGES...")
    
    try:
        # Run the download script with option 2 (minimal dataset)
        result = subprocess.run([
            sys.executable, 'download_images.py'
        ], input='2\n', text=True, capture_output=True)
        
        if result.returncode == 0:
            print("✅ Images downloaded successfully!")
            return True
        else:
            print(f"❌ Download failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error downloading images: {e}")
        return False

def run_training():
    """Start the training process"""
    print("\n🚀 STARTING TRAINING...")
    
    # Check if we have minimal dataset
    if os.path.exists('minimal_training_data.csv'):
        print("Using minimal dataset for quick training...")
        
        # Modify the training script to use minimal dataset
        training_cmd = [
            sys.executable, '-c', '''
import sys
sys.path.append('.')

# Import and modify the training script
from train_species_model import *

# Override the config to use minimal dataset
def main():
    config = {
        "csv_file": "minimal_training_data.csv",
        "image_dir": "images",
        "batch_size": 16,  # Smaller for quick training
        "num_epochs": 20,  # Fewer epochs for testing
        "learning_rate": 3e-4,
        "image_size": 224,
        "backbone": "efficientnet_v2_s",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 2,
        "early_stopping_patience": 5
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    if not os.path.exists(config["image_dir"]):
        print(f"❌ Image directory '{config['image_dir']}' not found!")
        return
    
    # Run the training
    df = analyze_dataset(config["csv_file"])
    train_df, val_df = create_train_val_split(df)
    train_df.to_csv("train_split.csv", index=False)
    val_df.to_csv("val_split.csv", index=False)
    
    train_transform, val_transform = create_data_transforms(config["image_size"])
    
    train_dataset = SpeciesDataset(
        csv_file="train_split.csv",
        image_dir=config["image_dir"],
        transform=train_transform,
        target_level="scientificName"
    )
    
    val_dataset = SpeciesDataset(
        csv_file="val_split.csv",
        image_dir=config["image_dir"],
        transform=val_transform,
        target_level="scientificName"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )
    
    model = EfficientNetSpeciesClassifier(
        num_classes=train_dataset.num_classes,
        backbone=config["backbone"],
        dropout=0.3,
        hierarchical=False
    )
    
    trainer = SpeciesTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config["device"],
        log_dir=f"runs/quick_training_{int(time.time())}"
    )
    
    trainer.train(
        num_epochs=config["num_epochs"],
        early_stopping_patience=config["early_stopping_patience"]
    )
    
    # Save label encoder
    with open("species_label_encoder.json", "w") as f:
        label_mapping = {
            "classes": train_dataset.label_encoders["scientificName"].classes_.tolist(),
            "class_to_idx": {cls: idx for idx, cls in enumerate(train_dataset.label_encoders["scientificName"].classes_)}
        }
        json.dump(label_mapping, f, indent=2)
    
    print(f"\\n✅ TRAINING COMPLETE!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
'''
        ]
        
        try:
            subprocess.run(training_cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
            return False
    else:
        print("❌ No training data available. Run download first.")
        return False

def test_inference():
    """Test the trained model"""
    print("\n🔍 TESTING INFERENCE...")
    
    if not os.path.exists('best_model.pth'):
        print("❌ No trained model found (best_model.pth)")
        return False
    
    if not os.path.exists('species_label_encoder.json'):
        print("❌ No label encoder found (species_label_encoder.json)")
        return False
    
    # Find a test image
    image_dir = Path('images')
    if image_dir.exists():
        test_images = list(image_dir.glob('*.jpg'))[:5]
        if test_images:
            print(f"Testing on {len(test_images)} sample images...")
            
            for img_path in test_images:
                try:
                    cmd = [
                        sys.executable, 'inference.py',
                        '--model', 'best_model.pth',
                        '--labels', 'species_label_encoder.json',
                        '--image', str(img_path),
                        '--top_k', '3'
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"✅ {img_path.name}: {result.stdout.strip()}")
                    else:
                        print(f"❌ {img_path.name}: {result.stderr.strip()}")
                        
                except Exception as e:
                    print(f"❌ Error testing {img_path}: {e}")
            
            return True
        else:
            print("❌ No images found for testing")
            return False
    else:
        print("❌ Images directory not found")
        return False

def main():
    print("🚀 SPECIES RECOGNITION AI - QUICK START")
    print("=" * 60)
    print("This script will guide you through the complete training process")
    print()
    
    # Step 1: Check requirements
    if not check_requirements():
        print("\n❌ Please install requirements first: pip install -r requirements.txt")
        return
    
    # Step 2: Check files
    if not check_files():
        print("\n❌ Missing required files. Make sure you're in the right directory.")
        return
    
    # Step 3: Check GPU
    has_gpu = check_gpu()
    
    # Step 4: Check if we have images
    if not os.path.exists('images') or not any(Path('images').glob('*.jpg')):
        print("\n📥 No images found. Let's download some...")
        
        choice = input("Download sample images for quick training? (y/n): ").strip().lower()
        if choice == 'y':
            # Run the download script interactively
            print("Running download script...")
            try:
                subprocess.run([sys.executable, 'download_images.py'], check=True)
            except subprocess.CalledProcessError:
                print("❌ Download failed")
                return
        else:
            print("❌ Cannot proceed without images")
            return
    else:
        image_count = len(list(Path('images').glob('*.jpg')))
        print(f"✅ Found {image_count} images")
    
    # Step 5: Start training
    print("\n" + "="*60)
    print("READY TO TRAIN!")
    print("="*60)
    
    choice = input("Start training now? (y/n): ").strip().lower()
    if choice == 'y':
        if run_training():
            print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
            
            # Step 6: Test inference
            choice = input("\nTest the trained model? (y/n): ").strip().lower()
            if choice == 'y':
                test_inference()
            
            print("\n✅ ALL DONE!")
            print("Your AI model is ready to use. Check these files:")
            print("  - best_model.pth (trained model)")
            print("  - species_label_encoder.json (class mappings)")
            print("  - training_curves.png (training progress)")
            print("\nUse inference.py to make predictions on new images!")
            
        else:
            print("❌ Training failed")
    else:
        print("Training cancelled. Run 'python train_species_model.py' when ready.")

if __name__ == "__main__":
    main() 