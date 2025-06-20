import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class BiodiversityDataset(Dataset):
    """Custom dataset for biodiversity images with hierarchical labels"""
    
    def __init__(self, df, image_dir, transform=None, target_level='species'):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.target_level = target_level
        
        # Create label encoders for different taxonomic levels
        self.label_encoders = {}
        self.hierarchical_labels = {}
        
        for level in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
            if level in df.columns:
                le = LabelEncoder()
                labels = df[level].fillna('Unknown').astype(str)
                self.hierarchical_labels[level] = le.fit_transform(labels)
                self.label_encoders[level] = le
        
        # Set primary target
        self.labels = self.hierarchical_labels[target_level]
        self.num_classes = len(self.label_encoders[target_level].classes_)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image path (adjust based on your data structure)
        image_path = f"{self.image_dir}/{self.df.iloc[idx]['image_filename']}"
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except:
            # Handle missing images with placeholder
            image = torch.zeros(3, 224, 224)
        
        # Get all hierarchical labels for this sample
        hierarchical_targets = {}
        for level in self.hierarchical_labels:
            hierarchical_targets[level] = torch.tensor(self.hierarchical_labels[level][idx])
        
        return image, hierarchical_targets, self.labels[idx]

class HierarchicalSpeciesClassifier(nn.Module):
    """Hierarchical classifier for species identification"""
    
    def __init__(self, num_classes_dict, backbone='efficientnet_v2_s', dropout=0.3):
        super().__init__()
        
        # Load pre-trained backbone
        if backbone == 'efficientnet_v2_s':
            self.backbone = models.efficientnet_v2_s(pretrained=True)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            self.backbone,
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        
        # Hierarchical classifiers
        self.classifiers = nn.ModuleDict()
        for level, num_classes in num_classes_dict.items():
            self.classifiers[level] = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim // 2, num_classes)
            )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        outputs = {}
        for level, classifier in self.classifiers.items():
            outputs[level] = classifier(features)
        
        return outputs

class SpeciesClassificationTrainer:
    """Training pipeline for species classification"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function with class weighting for imbalanced data
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, hierarchical_targets, species_labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            species_labels = species_labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate hierarchical loss (weighted combination)
            loss = 0
            weights = {'kingdom': 0.1, 'phylum': 0.15, 'class': 0.2, 'order': 0.25, 
                      'family': 0.3, 'genus': 0.5, 'species': 1.0}
            
            for level in outputs:
                if level in hierarchical_targets:
                    targets = hierarchical_targets[level].to(self.device)
                    level_loss = self.criterion(outputs[level], targets)
                    loss += weights.get(level, 1.0) * level_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy for species level
            _, predicted = outputs['species'].max(1)
            total += species_labels.size(0)
            correct += predicted.eq(species_labels).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, hierarchical_targets, species_labels in self.val_loader:
                images = images.to(self.device)
                species_labels = species_labels.to(self.device)
                
                outputs = self.model(images)
                
                # Calculate loss
                loss = 0
                weights = {'kingdom': 0.1, 'phylum': 0.15, 'class': 0.2, 'order': 0.25, 
                          'family': 0.3, 'genus': 0.5, 'species': 1.0}
                
                for level in outputs:
                    if level in hierarchical_targets:
                        targets = hierarchical_targets[level].to(self.device)
                        level_loss = self.criterion(outputs[level], targets)
                        loss += weights.get(level, 1.0) * level_loss
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = outputs['species'].max(1)
                total += species_labels.size(0)
                correct += predicted.eq(species_labels).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total

def prepare_data_transforms():
    """Data augmentation and preprocessing transforms"""
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_balanced_dataset(df, min_samples=10, max_samples=1000):
    """Create a balanced dataset by sampling species"""
    
    species_counts = df['scientificName'].value_counts()
    
    # Filter species with enough samples
    valid_species = species_counts[species_counts >= min_samples].index
    df_filtered = df[df['scientificName'].isin(valid_species)]
    
    # Balance dataset by sampling from each species
    balanced_dfs = []
    for species in valid_species:
        species_df = df_filtered[df_filtered['scientificName'] == species]
        sample_size = min(len(species_df), max_samples)
        sampled_df = species_df.sample(n=sample_size, random_state=42)
        balanced_dfs.append(sampled_df)
    
    return pd.concat(balanced_dfs, ignore_index=True)

# Example usage
if __name__ == "__main__":
    # Load your processed GBIF data
    df = pd.read_csv("processed_gbif_data.csv")
    
    # Create balanced dataset
    balanced_df = create_balanced_dataset(df)
    
    # Split data
    train_df, val_df = train_test_split(balanced_df, test_size=0.2, 
                                       stratify=balanced_df['scientificName'], 
                                       random_state=42)
    
    # Prepare transforms
    train_transform, val_transform = prepare_data_transforms()
    
    # Create datasets
    train_dataset = BiodiversityDataset(train_df, "images/", train_transform)
    val_dataset = BiodiversityDataset(val_df, "images/", val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Get number of classes for each taxonomic level
    num_classes_dict = {}
    for level in train_dataset.label_encoders:
        num_classes_dict[level] = len(train_dataset.label_encoders[level].classes_)
    
    # Create model
    model = HierarchicalSpeciesClassifier(num_classes_dict)
    
    # Train model
    trainer = SpeciesClassificationTrainer(model, train_loader, val_loader)
    
    # Training loop
    for epoch in range(50):
        train_loss, train_acc = trainer.train_epoch()
        val_loss, val_acc = trainer.validate()
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        trainer.scheduler.step(val_loss)