#!/bin/bash

# SightTrack AI - EC2 Setup Script
# Professional setup for species classification on AWS EC2

set -e  # Exit on any error

echo "=================================================="
echo "SightTrack AI - EC2 Setup Script"
echo "Setting up species classification environment"
echo "=================================================="

# Detect Ubuntu version
UBUNTU_VERSION=$(lsb_release -rs)
echo "Detected Ubuntu version: $UBUNTU_VERSION"

# Update system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Determine Python version and packages based on Ubuntu version
if [[ "$UBUNTU_VERSION" == "24.04" ]]; then
    PYTHON_VERSION="python3.12"
    PYTHON_DEV="python3.12-dev"
    PYTHON_VENV="python3.12-venv"
    NVIDIA_UTILS="nvidia-utils-535"
elif [[ "$UBUNTU_VERSION" == "22.04" ]]; then
    PYTHON_VERSION="python3.10"
    PYTHON_DEV="python3.10-dev"
    PYTHON_VENV="python3.10-venv"
    NVIDIA_UTILS="nvidia-utils-535"
else
    PYTHON_VERSION="python3"
    PYTHON_DEV="python3-dev"
    PYTHON_VENV="python3-venv"
    NVIDIA_UTILS="nvidia-utils-535"
fi

echo "Using Python version: $PYTHON_VERSION"

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    $PYTHON_VERSION \
    python3-pip \
    $PYTHON_DEV \
    $PYTHON_VENV \
    git \
    wget \
    curl \
    unzip \
    htop \
    tree \
    build-essential \
    software-properties-common

# Install NVIDIA drivers and CUDA if NVIDIA GPU is available
if lspci | grep -i nvidia > /dev/null; then
    echo "NVIDIA GPU detected. Installing NVIDIA drivers..."
    
    # Install NVIDIA drivers
    sudo apt-get install -y $NVIDIA_UTILS
    
    # Install CUDA toolkit
    echo "Installing CUDA toolkit..."
    sudo apt-get install -y nvidia-cuda-toolkit
    
    echo "NVIDIA and CUDA installation completed."
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

# Create Python virtual environment
echo "Creating Python virtual environment..."
$PYTHON_VERSION -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Installing basic dependencies..."
    pip install torch torchvision torchaudio pandas numpy scikit-learn pillow opencv-python matplotlib seaborn tqdm requests tensorboard pyyaml timm
fi

# Install PyTorch with CUDA support if available
if lspci | grep -i nvidia > /dev/null && command -v nvidia-smi &> /dev/null; then
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
echo "- Python: $($PYTHON_VERSION --version)"
echo "- PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed yet')"
echo "- CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
if command -v nvidia-smi &> /dev/null; then
    echo "- GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU info not available"
fi
echo "==================================================" 