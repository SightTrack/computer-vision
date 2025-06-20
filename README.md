# SightTrack AI - Species Classification

A professional deep learning system for automated species classification using computer vision. Built with PyTorch and EfficientNet for robust performance on AWS EC2.

## 🎯 Overview

SightTrack AI is a state-of-the-art species classification system that uses EfficientNet-based neural networks to identify wildlife species from photographs. The system is designed for easy deployment on AWS EC2 instances with comprehensive automation scripts.

### Key Features

- **Professional Architecture**: Clean, modular codebase with separation of concerns
- **EC2 Ready**: One-command setup for AWS EC2 deployment
- **Automated Data Pipeline**: Scripts for downloading and processing iNaturalist dataset
- **Advanced Training**: Mixed precision, data augmentation, and early stopping
- **Easy Inference**: Simple prediction scripts for single images or batch processing
- **Comprehensive Logging**: TensorBoard integration and detailed progress tracking

## 🚀 Quick Start on EC2

### Step 1: Launch EC2 Instance

1. Launch an EC2 instance (recommended: `g4dn.xlarge` or larger for GPU training)
2. Use Ubuntu 20.04 or 22.04 LTS AMI
3. Configure security groups to allow SSH access
4. Connect to your instance via SSH

### Step 2: Setup Environment

```bash
# Upload project files to EC2 instance
scp -r sighttrack-ai ubuntu@your-ec2-ip:/home/ubuntu/

# SSH into EC2 instance
ssh ubuntu@your-ec2-ip

# Navigate to project directory
cd /home/ubuntu/sighttrack-ai

# Run setup script (this will install all dependencies)
chmod +x scripts/setup_ec2.sh
./scripts/setup_ec2.sh
```

### Step 3: Activate Environment

```bash
# Activate virtual environment
source venv/bin/activate

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 4: Download Dataset

```bash
# Download and prepare the iNaturalist GBIF dataset
./scripts/download_data.sh
```

The script will automatically download the iNaturalist GBIF observations dataset directly from iNaturalist and process it for training.

### Step 5: Start Training

```bash
# Train the model with default configuration
python train.py

# Or with custom configuration
python train.py --config config/model_config.yaml --device cuda
```

### Step 6: Make Predictions

```bash
# Predict single image
python predict.py path/to/image.jpg

# Batch prediction
python predict.py path/to/images/ --batch --output results.json
```

## 📁 Project Structure

```
sighttrack-ai/
├── config/
│   └── model_config.yaml          # Training configuration
├── src/
│   ├── model.py                   # Model architecture
│   ├── dataset.py                 # Data loading and preprocessing
│   ├── trainer.py                 # Training logic
│   └── utils.py                   # Utility functions
├── scripts/
│   ├── setup_ec2.sh              # EC2 setup script
│   └── download_data.sh           # Data download script
├── data/
│   ├── raw/                       # Raw downloaded data
│   ├── processed/                 # Processed CSV files
│   └── images/                    # Image files
├── models/                        # Saved model checkpoints
├── logs/                         # Training logs
├── results/                      # Training results
├── train.py                      # Main training script
├── predict.py                    # Prediction script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## ⚙️ Configuration

The main configuration is in `config/model_config.yaml`. Key settings:

### Model Settings
```yaml
model:
  backbone: "efficientnet_v2_s"    # Model architecture
  dropout: 0.5                     # Dropout rate
  image_size: 224                  # Input image size
  pretrained: true                 # Use pretrained weights
```

### Training Settings
```yaml
training:
  batch_size: 16                   # Batch size
  num_epochs: 100                  # Maximum epochs
  learning_rate: 0.0001            # Learning rate
  early_stopping_patience: 15     # Early stopping patience
```

### Data Settings
```yaml
data:
  target_level: "family"           # Classification level (species, genus, family)
  csv_file: "data/processed/species_data.csv"
  image_dir: "data/images"
```

## 🎯 Advanced Usage

### Custom Training

```bash
# Train with specific parameters
python train.py \
    --config config/model_config.yaml \
    --device cuda \
    --debug

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_50.pth
```

### Prediction Options

```bash
# Single image with top 3 predictions
python predict.py image.jpg --top-k 3

# Batch processing with custom model
python predict.py images/ \
    --batch \
    --model models/custom_model.pth \
    --output predictions.json

# CPU-only prediction
python predict.py image.jpg --device cpu
```

### Model Evaluation

```bash
# View training progress
tensorboard --logdir logs/

# Check model performance
python -c "
import torch
checkpoint = torch.load('models/best_model.pth', map_location='cpu')
print(f'Best validation accuracy: {checkpoint[\"best_val_acc\"]:.4f}')
"
```

## 🛠️ Development

### Local Development Setup

```bash
# Clone repository
git clone <your-repo-url>
cd sighttrack-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Create directory structure
python -c "
from pathlib import Path
from src.utils import create_directory_structure
create_directory_structure(Path('.'))
"
```

### Code Quality

The codebase follows professional Python standards:

- **Type Hints**: Full type annotation support
- **Documentation**: Comprehensive docstrings
- **Error Handling**: Robust error handling and validation
- **Logging**: Structured logging throughout
- **Configuration**: YAML-based configuration management

## 📊 Performance

### Model Specifications

- **Architecture**: EfficientNet-V2-S (default)
- **Parameters**: ~20M parameters
- **Input Size**: 224x224 RGB images
- **Training Time**: ~2-4 hours on g4dn.xlarge
- **Inference Speed**: ~50ms per image on GPU

### Hardware Requirements

#### Minimum (CPU Training)
- 4 CPU cores
- 8GB RAM
- 20GB storage

#### Recommended (GPU Training)
- AWS g4dn.xlarge or larger
- NVIDIA GPU with 8GB+ VRAM
- 16GB+ RAM
- 50GB+ storage

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size in config
   # Edit config/model_config.yaml
   training:
     batch_size: 8  # Reduce from 16
   ```

2. **Missing Dependencies**
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt --force-reinstall
   ```

3. **Data Download Issues**
   ```bash
   # Check Kaggle credentials
   cat ~/.kaggle/kaggle.json
   
   # Manual dataset setup
   python -c "
   import pandas as pd
   from pathlib import Path
   # Create sample data...
   "
   ```

4. **Model Loading Errors**
   ```bash
   # Check model compatibility
   python -c "
   import torch
   model = torch.load('models/best_model.pth', map_location='cpu')
   print('Model keys:', model.keys())
   "
   ```

### Performance Optimization

- **Batch Size**: Adjust based on GPU memory
- **Image Size**: Reduce for faster training
- **Mixed Precision**: Enabled by default for speed
- **Data Loading**: Increase `num_workers` for faster I/O

## 📈 Monitoring

### Training Progress

- **TensorBoard**: `tensorboard --logdir logs/`
- **Log Files**: Check `logs/training.log`
- **Model Checkpoints**: Saved in `checkpoints/`

### System Monitoring

```bash
# GPU utilization
nvidia-smi

# System resources
htop

# Disk usage
df -h
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Built with PyTorch and Torchvision
- Uses EfficientNet architecture from Google Research
- Inspired by modern computer vision research
- Dataset from iNaturalist community

## 📞 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Include system specifications and error logs

---

**Happy Species Classifying! 🦅🌿🦋** 