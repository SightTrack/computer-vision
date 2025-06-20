"""
SightTrack AI - Species Classification Model
Professional implementation of EfficientNet-based species classifier
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any


class SpeciesClassifier(nn.Module):
    """
    EfficientNet-based species classifier for taxonomic classification.
    
    Supports multiple backbone architectures and hierarchical classification.
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = "efficientnet_v2_s",
        dropout: float = 0.5,
        pretrained: bool = True
    ):
        """
        Initialize the species classifier.
        
        Args:
            num_classes: Number of output classes
            backbone: Backbone architecture name
            dropout: Dropout probability
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Initialize backbone
        self.backbone = self._create_backbone(backbone, pretrained)
        feature_dim = self._get_feature_dim(backbone)
        
        # Classification head
        self.classifier = self._create_classifier_head(feature_dim, num_classes, dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """Create the backbone network."""
        weights = 'DEFAULT' if pretrained else None
        
        if backbone == "efficientnet_v2_s":
            model = models.efficientnet_v2_s(weights=weights)
            model.classifier = nn.Identity()
        elif backbone == "efficientnet_v2_m":
            model = models.efficientnet_v2_m(weights=weights)
            model.classifier = nn.Identity()
        elif backbone == "efficientnet_v2_l":
            model = models.efficientnet_v2_l(weights=weights)
            model.classifier = nn.Identity()
        elif backbone == "resnet50":
            model = models.resnet50(weights=weights)
            model.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        return model
    
    def _get_feature_dim(self, backbone: str) -> int:
        """Get the feature dimension for the backbone."""
        feature_dims = {
            "efficientnet_v2_s": 1280,
            "efficientnet_v2_m": 1280,
            "efficientnet_v2_l": 1280,
            "resnet50": 2048
        }
        return feature_dims[backbone]
    
    def _create_classifier_head(
        self, 
        feature_dim: int, 
        num_classes: int, 
        dropout: float
    ) -> nn.Module:
        """Create the classification head."""
        return nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.BatchNorm1d(feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(feature_dim // 4, num_classes)
        )
    
    def _initialize_weights(self):
        """Initialize classifier weights using Xavier initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        features = self.backbone(x)
        return self.classifier(features)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "backbone": self.backbone_name,
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }


def create_model(config: Dict[str, Any]) -> SpeciesClassifier:
    """
    Create a species classifier model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized SpeciesClassifier model
    """
    model_config = config["model"]
    
    model = SpeciesClassifier(
        num_classes=model_config["num_classes"],
        backbone=model_config["backbone"],
        dropout=model_config["dropout"],
        pretrained=model_config["pretrained"]
    )
    
    return model


def load_trained_model(
    model_path: str, 
    config: Dict[str, Any], 
    device: str = "cpu"
) -> SpeciesClassifier:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        config: Model configuration
        device: Device to load model on
        
    Returns:
        Loaded model in evaluation mode
    """
    model = create_model(config)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model = model.to(device)
    model.eval()
    
    return model 