# SightTrack AI - Complete EC2 Deployment Guide

This guide provides detailed, step-by-step instructions for deploying SightTrack AI on AWS EC2 with better hardware for species classification training.

## 📋 Prerequisites

1. **AWS Account** with EC2 access
2. **Kaggle Account** for dataset access
3. **SSH Client** (Terminal on Mac/Linux, PuTTY on Windows)
4. **Project Files** uploaded to your EC2 instance

## 🚀 Step-by-Step Deployment

### Step 1: Launch EC2 Instance

1. **Login to AWS Console**
   - Go to [AWS Console](https://console.aws.amazon.com/)
   - Navigate to EC2 service

2. **Launch Instance**
   ```
   AMI: Ubuntu Server 22.04 LTS (HVM), SSD Volume Type
   Instance Type: g4dn.xlarge (recommended) or g4dn.2xlarge (for faster training)
   Key Pair: Create new or use existing
   Storage: 100 GB gp3
   Security Group: Allow SSH (port 22) from your IP
   ```

3. **Instance Specifications**
   ```
   g4dn.xlarge:
   - 4 vCPUs
   - 16 GB RAM
   - 1x NVIDIA T4 GPU (16 GB GPU memory)
   - $0.526/hour (on-demand pricing)
   
   g4dn.2xlarge:
   - 8 vCPUs
   - 32 GB RAM
   - 1x NVIDIA T4 GPU (16 GB GPU memory)
   - $0.752/hour (on-demand pricing)
   ```

4. **Wait for Instance Launch**
   - Instance state should be "running"
   - Note the public IP address

### Step 2: Upload Project Files

From your local machine where you have the project:

```bash
# Create a tar archive of the project (excluding old files)
tar -czf sighttrack-ai.tar.gz \
    --exclude='*.pth' \
    --exclude='best_model.pth' \
    --exclude='dataset' \
    --exclude='images' \
    --exclude='runs' \
    --exclude='.git' \
    --exclude='__pycache__' \
    .

# Upload to EC2 instance
scp -i your-key.pem sighttrack-ai.tar.gz ubuntu@YOUR_EC2_IP:/home/ubuntu/

# Or use rsync for selective sync
rsync -avz -e "ssh -i your-key.pem" \
    --exclude='*.pth' \
    --exclude='dataset' \
    --exclude='images' \
    . ubuntu@YOUR_EC2_IP:/home/ubuntu/sighttrack-ai/
```

### Step 3: Connect to EC2 and Setup

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# Extract project files (if using tar)
cd /home/ubuntu
tar -xzf sighttrack-ai.tar.gz

# Navigate to project directory
cd sighttrack-ai

# Make setup script executable and run it
chmod +x scripts/setup_ec2.sh
./scripts/setup_ec2.sh
```

**Wait for setup to complete** (this may take 10-15 minutes). The script will:
- Update system packages
- Install Python 3.10 and dependencies
- Install CUDA toolkit (if GPU detected)
- Create virtual environment
- Install PyTorch with CUDA support
- Install all Python dependencies

### Step 4: Verify Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Check installations
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check GPU
nvidia-smi

# Expected output should show:
# - Python 3.10.x
# - PyTorch 2.1.x
# - CUDA Available: True
# - GPU details in nvidia-smi
```

### Step 5: Download and Prepare Dataset

```bash
# Still in the sighttrack-ai directory with venv activated
./scripts/download_data.sh
```

This script will:
- Download the iNaturalist GBIF dataset directly from iNaturalist
- Extract and process the species observation data
- Create sample data if needed
- Verify the setup

### Step 6: Configure Training

Edit the configuration if needed:
```bash
nano config/model_config.yaml
```

Key settings for EC2:
```yaml
training:
  batch_size: 32        # Can increase on g4dn.xlarge
  num_epochs: 50        # Reduce for testing
  learning_rate: 0.0001

system:
  device: "auto"        # Will use CUDA
  num_workers: 4        # Good for g4dn.xlarge
  pin_memory: true
  mixed_precision: true # Faster training
```

### Step 7: Start Training

```bash
# Start training with default config
python train.py

# Or with custom settings
python train.py --config config/model_config.yaml --device cuda --debug

# Run in background (recommended for long training)
nohup python train.py > training.log 2>&1 &

# Monitor training
tail -f training.log
```

### Step 8: Monitor Training Progress

**Option 1: TensorBoard**
```bash
# In another terminal session
ssh -i your-key.pem ubuntu@YOUR_EC2_IP
cd sighttrack-ai
source venv/bin/activate
tensorboard --logdir logs --host 0.0.0.0 --port 6006

# Access at: http://YOUR_EC2_IP:6006
# (Make sure port 6006 is open in security group)
```

**Option 2: Log Files**
```bash
# Training progress
tail -f logs/training.log

# System monitoring
watch -n 1 nvidia-smi
htop
```

### Step 9: Test Model

After training completes:

```bash
# Test prediction on sample image
python predict.py data/images/sample_0.jpg --top-k 5

# Batch prediction
python predict.py data/images/ --batch --output predictions.json

# Check results
python -c "
import torch
checkpoint = torch.load('models/best_model.pth', map_location='cpu')
print(f'Best validation accuracy: {checkpoint[\"best_val_acc\"]:.4f}')
print(f'Training completed at epoch: {checkpoint[\"epoch\"]}')
"
```

## 🔧 Troubleshooting Common Issues

### Issue 1: CUDA Out of Memory
```bash
# Edit config to reduce batch size
nano config/model_config.yaml
# Change batch_size from 32 to 16 or 8

# Or use gradient accumulation
# training:
#   batch_size: 8
#   gradient_accumulation_steps: 4  # Effective batch size = 32
```

### Issue 2: Slow Data Loading
```bash
# Check disk I/O
iostat -x 1

# Reduce num_workers if high I/O wait
# system:
#   num_workers: 2  # Reduce from 4
```

### Issue 3: SSH Connection Timeout
```bash
# Keep connection alive during training
ssh -i your-key.pem -o ServerAliveInterval=60 ubuntu@YOUR_EC2_IP

# Use screen or tmux for persistent sessions
sudo apt-get install screen
screen -S training
# Run training commands
# Detach: Ctrl+A, D
# Reattach: screen -r training
```

### Issue 4: Insufficient Storage
```bash
# Check disk usage
df -h

# Clean up if needed
rm -rf data/raw/*.zip  # Remove downloaded archives
rm -rf logs/old_runs   # Remove old tensorboard logs

# Extend EBS volume if needed (through AWS console)
```

## 📊 Cost Optimization

### Training Time Estimates
```
g4dn.xlarge (16GB GPU):
- Small dataset (1K images): 1-2 hours = $0.50-1.00
- Medium dataset (10K images): 4-8 hours = $2.00-4.00
- Large dataset (100K images): 12-24 hours = $6.00-12.00

g4dn.2xlarge (16GB GPU):
- 30-50% faster training
- Higher cost: $0.752/hour
```

### Cost Saving Tips

1. **Use Spot Instances**
   ```bash
   # Save 70% on costs but risk interruption
   # Good for experimentation
   ```

2. **Stop Instance When Not Training**
   ```bash
   # Stop (not terminate) to save costs
   # EBS storage: $0.10/GB/month
   ```

3. **Use Smaller Instance for Development**
   ```bash
   # t3.medium for code development: $0.0416/hour
   # Switch to GPU instance only for training
   ```

## 🚀 Production Deployment

For production inference deployment:

1. **Create AMI** from trained instance
2. **Use smaller instance** (t3.medium) for CPU inference
3. **Setup auto-scaling** for multiple predictions
4. **Use S3** for model storage
5. **Setup CloudWatch** for monitoring

## 📝 Checklist

Before starting training, verify:

- [ ] EC2 instance is running with GPU
- [ ] All dependencies installed correctly
- [ ] Kaggle credentials configured
- [ ] Dataset downloaded and processed
- [ ] Configuration file updated
- [ ] Training starts without errors
- [ ] TensorBoard accessible (optional)
- [ ] Monitoring tools working

## 🆘 Emergency Procedures

### If Training Fails
```bash
# Check last error
tail -50 training.log

# Restart from checkpoint
python train.py --resume checkpoints/latest_checkpoint.pth

# Start with smaller configuration
# Edit config: reduce batch_size, image_size, num_epochs
```

### If Instance Becomes Unresponsive
```bash
# Force reboot from AWS Console
# Check CloudWatch logs
# Verify EBS volume attachment
```

### Data Recovery
```bash
# Models saved in: models/
# Logs saved in: logs/
# Checkpoints in: checkpoints/

# Backup important files
tar -czf backup.tar.gz models/ logs/ checkpoints/ config/
scp backup.tar.gz local-machine:/backup/
```

---

**You're now ready to train state-of-the-art species classification models on AWS EC2! 🚀**

For additional support, check the main README.md or create an issue in the repository. 