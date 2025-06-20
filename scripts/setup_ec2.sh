#!/bin/bash

# SightTrack AI - EC2 Setup Script
# Professional setup for species classification on AWS EC2

set -e  # Exit on any error

echo "=================================================="
echo "SightTrack AI - EC2 Setup Script"
echo "Setting up species classification environment"
echo "=================================================="

# Update system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    python3.10-venv \
    git \
    wget \
    curl \
    unzip \
    htop \
    tree \
    nvidia-smi \
    build-essential

# Install CUDA if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing CUDA toolkit..."
    sudo apt-get install -y nvidia-cuda-toolkit
    echo "CUDA installation completed."
else
    echo "No NVIDIA GPU detected. Proceeding with CPU-only setup."
fi

# Create project directory
PROJECT_DIR="/home/ubuntu/sighttrack-ai"
echo "Setting up project directory at $PROJECT_DIR..."
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Create directory structure
echo "Creating directory structure..."
mkdir -p {data/{raw,processed,images},models,logs,checkpoints,results,scripts}

# Clone or copy project files (assuming they're already uploaded)
echo "Project files should be uploaded to this directory."

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Please upload project files first."
fi

# Install PyTorch with CUDA support if available
if command -v nvidia-smi &> /dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch CPU-only version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Set up environment variables
echo "Setting up environment variables..."
echo "export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH" >> ~/.bashrc
echo "export SIGHTTRACK_HOME=$PROJECT_DIR" >> ~/.bashrc

# Create .env file for local settings
cat > .env << EOF
# SightTrack AI Environment Configuration
SIGHTTRACK_HOME=$PROJECT_DIR
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
EOF

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.sh

echo "=================================================="
echo "EC2 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Run 'source venv/bin/activate' to activate the environment"
echo "2. Run './scripts/download_data.sh' to download the iNaturalist GBIF dataset"
echo "3. Run 'python train.py' to start training"
echo ""
echo "System Information:"
echo "- Python: $(python --version)"
echo "- PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "- CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if command -v nvidia-smi &> /dev/null; then
    echo "- GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo "==================================================" 