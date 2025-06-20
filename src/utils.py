"""
SightTrack AI - Utilities Module
Common functions and utilities for species classification
"""

import os
import sys
import random
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def set_random_seed(seed: int = 42):
    """
    Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: Optional[Path] = None, verbose: bool = False):
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional path to log file
        verbose: Whether to enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed device information.
    
    Returns:
        Dictionary with device information
    """
    info = {}
    
    # CPU information
    info["CPU"] = os.cpu_count()
    
    # CUDA information
    info["CUDA Available"] = torch.cuda.is_available()
    
    if torch.cuda.is_available():
        info["CUDA Version"] = torch.version.cuda
        info["GPU Count"] = torch.cuda.device_count()
        
        # Get GPU details
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            info[f"GPU {i}"] = f"{gpu_name} ({gpu_memory:.1f} GB)"
    
    # PyTorch information
    info["PyTorch Version"] = torch.__version__
    
    return info


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes value to human readable string.
    
    Args:
        bytes_value: Value in bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def check_gpu_memory() -> Dict[str, float]:
    """
    Check GPU memory usage.
    
    Returns:
        Dictionary with memory information in GB
    """
    if not torch.cuda.is_available():
        return {}
    
    memory_info = {}
    
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
        free_memory = total_memory - allocated_memory
        
        memory_info[f"GPU_{i}"] = {
            "total": total_memory,
            "allocated": allocated_memory,
            "free": free_memory
        }
    
    return memory_info


def create_directory_structure(base_path: Path):
    """
    Create the standard directory structure for the project.
    
    Args:
        base_path: Base project directory
    """
    directories = [
        "data/raw",
        "data/processed",
        "data/images",
        "models",
        "logs",
        "checkpoints",
        "results",
        "scripts",
        "src",
        "config"
    ]
    
    for directory in directories:
        (base_path / directory).mkdir(parents=True, exist_ok=True)
    
    print(f"Directory structure created at: {base_path}")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params
    }


def save_model_summary(model: torch.nn.Module, save_path: Path):
    """
    Save model summary to file.
    
    Args:
        model: PyTorch model
        save_path: Path to save summary
    """
    param_info = count_parameters(model)
    
    summary = f"""
Model Summary
=============

Architecture: {model.__class__.__name__}
Total Parameters: {param_info['total']:,}
Trainable Parameters: {param_info['trainable']:,}
Non-trainable Parameters: {param_info['non_trainable']:,}

Model Size: {param_info['total'] * 4 / (1024**2):.1f} MB (assuming float32)

Model Structure:
{model}
"""
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write(summary)
    
    print(f"Model summary saved to: {save_path}")


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Progress meter for training."""
    
    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches: int):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_sections = ["model", "training", "data", "system", "paths", "logging"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate model config
    model_config = config["model"]
    required_model_keys = ["backbone", "dropout", "image_size", "pretrained"]
    
    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(f"Missing required model config key: {key}")
    
    # Validate training config
    training_config = config["training"]
    required_training_keys = ["batch_size", "num_epochs", "learning_rate"]
    
    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"Missing required training config key: {key}")
    
    # Validate data config
    data_config = config["data"]
    required_data_keys = ["csv_file", "image_dir", "target_level"]
    
    for key in required_data_keys:
        if key not in data_config:
            raise ValueError(f"Missing required data config key: {key}")
    
    return True 