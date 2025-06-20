"""
SightTrack AI - Training Module
Professional implementation for species classification training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm
import json


class MixUpLoss:
    """MixUp augmentation and loss calculation."""
    
    def __init__(self, criterion, alpha: float = 0.2):
        self.criterion = criterion
        self.alpha = alpha
    
    def mixup_data(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply MixUp augmentation."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def calculate_loss(self, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        """Calculate MixUp loss."""
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)


class SpeciesTrainer:
    """
    Professional trainer for species classification model.
    
    Supports mixed precision training, advanced optimizers, and comprehensive logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to use for training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup training components
        self.criterion = self._setup_criterion()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.scaler = GradScaler() if config["system"]["mixed_precision"] else None
        
        # Setup MixUp if enabled
        self.mixup = None
        if config["training"]["use_mixup"]:
            self.mixup = MixUpLoss(self.criterion, config["training"]["mixup_alpha"])
        
        # Setup logging
        self.writer = None
        if config["logging"]["use_tensorboard"]:
            log_dir = Path(config["paths"]["logs_dir"]) / f"run_{int(time.time())}"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(log_dir))
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        print(f"Trainer initialized on {device}")
        print(f"Model: {model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _setup_criterion(self) -> nn.Module:
        """Setup loss criterion with label smoothing."""
        if self.config["training"]["label_smoothing"] > 0:
            return nn.CrossEntropyLoss(
                label_smoothing=self.config["training"]["label_smoothing"]
            )
        else:
            return nn.CrossEntropyLoss()
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on configuration."""
        optimizer_name = self.config["training"]["optimizer"].lower()
        lr = self.config["training"]["learning_rate"]
        weight_decay = self.config["training"]["weight_decay"]
        
        if optimizer_name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_name = self.config["training"]["scheduler"].lower()
        
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config["training"]["num_epochs"]
            )
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=30, 
                gamma=0.1
            )
        elif scheduler_name == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=10
            )
        else:
            return None
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images, targets = images.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Apply MixUp if enabled
            if self.mixup and np.random.random() < 0.5:
                images, targets_a, targets_b, lam = self.mixup.mixup_data(images, targets)
                
                if self.scaler:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.mixup.calculate_loss(outputs, targets_a, targets_b, lam)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.mixup.calculate_loss(outputs, targets_a, targets_b, lam)
                    loss.backward()
                    self.optimizer.step()
            else:
                if self.scaler:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
            
            # Log to tensorboard
            if self.writer and batch_idx % self.config["logging"]["log_every_n_steps"] == 0:
                step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), step)
                self.writer.add_scalar('Train/BatchAcc', correct / total, step)
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validating"):
                images, targets = images.to(self.device), targets.to(self.device)
                
                if self.scaler:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = Path(filepath).parent / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training history and results
        """
        print(f"Starting training for {self.config['training']['num_epochs']} epochs")
        
        for epoch in range(self.config["training"]["num_epochs"]):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
                self.writer.add_scalar('Train/EpochAcc', train_acc, epoch)
                self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)
                self.writer.add_scalar('Val/EpochAcc', val_acc, epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Check for best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Print progress
            print(f"Epoch {epoch + 1}/{self.config['training']['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Best Val Acc: {self.best_val_acc:.4f}")
            print("-" * 50)
            
            # Save checkpoint
            if (epoch + 1) % self.config["logging"]["save_model_every_n_epochs"] == 0:
                checkpoint_dir = Path(self.config["paths"]["checkpoints_dir"])
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                self.save_checkpoint(str(checkpoint_path), is_best)
            
            # Early stopping
            if self.early_stopping_counter >= self.config["training"]["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        model_dir = Path(self.config["paths"]["model_save_dir"])
        model_dir.mkdir(parents=True, exist_ok=True)
        final_path = model_dir / "final_model.pth"
        self.save_checkpoint(str(final_path))
        
        # Close tensorboard writer
        if self.writer:
            self.writer.close()
        
        # Return training results
        results = {
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "final_epoch": self.current_epoch + 1,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs
        }
        
        return results 