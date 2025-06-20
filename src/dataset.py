"""
SightTrack AI - Dataset Module
Professional implementation for species image dataset handling
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json
from typing import Tuple, Optional, Dict, Any
import warnings


class SpeciesDataset(Dataset):
    """
    Custom dataset for species images with taxonomic labels.
    
    Supports various taxonomic levels and advanced data augmentation.
    """
    
    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        transform: Optional[transforms.Compose] = None,
        target_level: str = "family"
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_file: Path to CSV file with species data
            image_dir: Directory containing images
            transform: Image transformations
            target_level: Taxonomic level for classification
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.target_level = target_level
        
        # Load and validate data
        self.df = self._load_and_validate_data(csv_file)
        
        # Create label encoder
        self.label_encoder = self._create_label_encoder()
        
        # Filter to existing images
        self._filter_existing_images()
        
        print(f"Dataset loaded: {len(self.df)} images, {self.num_classes} {target_level} classes")
    
    def _load_and_validate_data(self, csv_file: str) -> pd.DataFrame:
        """Load and validate the CSV data."""
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} records from {csv_file}")
        except Exception as e:
            raise ValueError(f"Error loading CSV file {csv_file}: {e}")
        
        # Validate required columns
        required_columns = ["image_filename", self.target_level]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with missing target labels
        initial_len = len(df)
        df = df.dropna(subset=[self.target_level])
        df = df[df[self.target_level] != ""]
        
        print(f"Removed {initial_len - len(df)} rows with missing {self.target_level} labels")
        
        return df.reset_index(drop=True)
    
    def _create_label_encoder(self) -> LabelEncoder:
        """Create and fit label encoder for target level."""
        label_encoder = LabelEncoder()
        labels = self.df[self.target_level].astype(str)
        label_encoder.fit(labels)
        
        self.num_classes = len(label_encoder.classes_)
        self.class_names = label_encoder.classes_.tolist()
        
        return label_encoder
    
    def _filter_existing_images(self):
        """Filter dataset to only include rows with existing images."""
        valid_indices = []
        missing_count = 0
        
        for idx in range(len(self.df)):
            image_path = self.image_dir / self.df.iloc[idx]["image_filename"]
            if image_path.exists():
                valid_indices.append(idx)
            else:
                missing_count += 1
        
        if missing_count > 0:
            print(f"Warning: {missing_count} images not found on disk")
        
        # Update dataframe
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        
        # Update target labels
        self.target_labels = self.label_encoder.transform(
            self.df[self.target_level].astype(str)
        )
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset."""
        # Get image path
        image_filename = self.df.iloc[idx]["image_filename"]
        image_path = self.image_dir / image_filename
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            warnings.warn(f"Error loading {image_path}: {e}")
            # Return a black image if loading fails
            if self.transform:
                image = self.transform(Image.new("RGB", (224, 224), (0, 0, 0)))
            else:
                image = torch.zeros(3, 224, 224)
        
        # Get target label
        target = torch.tensor(self.target_labels[idx], dtype=torch.long)
        
        return image, target
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset handling."""
        from collections import Counter
        
        class_counts = Counter(self.target_labels)
        total_samples = len(self.target_labels)
        
        weights = []
        for i in range(self.num_classes):
            weight = total_samples / (self.num_classes * class_counts.get(i, 1))
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def save_label_encoder(self, filepath: str):
        """Save label encoder to file."""
        label_data = {
            "classes": self.class_names,
            "class_to_idx": {cls: idx for idx, cls in enumerate(self.class_names)},
            "target_level": self.target_level
        }
        
        with open(filepath, "w") as f:
            json.dump(label_data, f, indent=2)
        
        print(f"Label encoder saved to {filepath}")


def create_transforms(
    image_size: int = 224,
    is_training: bool = True,
    use_advanced_augmentation: bool = True
) -> transforms.Compose:
    """
    Create image transformations for training or validation.
    
    Args:
        image_size: Target image size
        is_training: Whether to apply training augmentations
        use_advanced_augmentation: Whether to use advanced augmentation
        
    Returns:
        Composed transformations
    """
    if is_training:
        transform_list = [
            transforms.Resize((int(image_size * 1.2), int(image_size * 1.2))),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                image_size, 
                scale=(0.8, 1.0), 
                ratio=(0.9, 1.1)
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
        ]
        
        if use_advanced_augmentation:
            transform_list.extend([
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ], p=0.2),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        if use_advanced_augmentation:
            transform_list.append(
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.33))
            )
    
    else:
        transform_list = [
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    
    return transforms.Compose(transform_list)


def create_dataloaders(
    config: Dict[str, Any],
    validation_split: float = 0.2
) -> Tuple[DataLoader, DataLoader, SpeciesDataset]:
    """
    Create training and validation data loaders.
    
    Args:
        config: Configuration dictionary
        validation_split: Fraction of data to use for validation
        
    Returns:
        Tuple of (train_loader, val_loader, full_dataset)
    """
    data_config = config["data"]
    training_config = config["training"]
    model_config = config["model"]
    system_config = config["system"]
    
    # Create transforms
    train_transform = create_transforms(
        image_size=model_config["image_size"],
        is_training=True,
        use_advanced_augmentation=training_config["use_advanced_augmentation"]
    )
    
    val_transform = create_transforms(
        image_size=model_config["image_size"],
        is_training=False
    )
    
    # Load full dataset
    full_dataset = SpeciesDataset(
        csv_file=data_config["csv_file"],
        image_dir=data_config["image_dir"],
        transform=None,  # We'll set this later
        target_level=data_config["target_level"]
    )
    
    # Update config with actual number of classes
    config["model"]["num_classes"] = full_dataset.num_classes
    
    # Create train/val split
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=validation_split,
        random_state=42,
        stratify=full_dataset.target_labels
    )
    
    # Create subset datasets
    train_df = full_dataset.df.iloc[train_indices].reset_index(drop=True)
    val_df = full_dataset.df.iloc[val_indices].reset_index(drop=True)
    
    train_dataset = SpeciesDataset(
        csv_file=data_config["csv_file"],
        image_dir=data_config["image_dir"],
        transform=train_transform,
        target_level=data_config["target_level"]
    )
    train_dataset.df = train_df
    train_dataset.target_labels = full_dataset.target_labels[train_indices]
    
    val_dataset = SpeciesDataset(
        csv_file=data_config["csv_file"],
        image_dir=data_config["image_dir"],
        transform=val_transform,
        target_level=data_config["target_level"]
    )
    val_dataset.df = val_df
    val_dataset.target_labels = full_dataset.target_labels[val_indices]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=system_config["num_workers"],
        pin_memory=system_config["pin_memory"],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=system_config["num_workers"],
        pin_memory=system_config["pin_memory"]
    )
    
    return train_loader, val_loader, full_dataset 