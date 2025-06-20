import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import argparse
import numpy as np
import os
from pathlib import Path
import time

class EfficientNetSpeciesClassifier(nn.Module):
    """EfficientNet-based species classifier with optional hierarchical outputs"""
    
    def __init__(self, num_classes, backbone='efficientnet_v2_s', dropout=0.3, hierarchical=False, num_classes_dict=None):
        super().__init__()
        
        # Load pre-trained backbone
        if backbone == 'efficientnet_v2_s':
            self.backbone = models.efficientnet_v2_s(weights='DEFAULT')
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'efficientnet_v2_m':
            self.backbone = models.efficientnet_v2_m(weights='DEFAULT')
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights='DEFAULT')
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.hierarchical = hierarchical
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            self.backbone,
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        
        if hierarchical and num_classes_dict:
            # Hierarchical classifiers
            self.classifiers = nn.ModuleDict()
            for level, num_classes_level in num_classes_dict.items():
                self.classifiers[level] = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(feature_dim // 2, num_classes_level)
                )
        else:
            # Simple classifier
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim // 2, num_classes)
            )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        if self.hierarchical:
            outputs = {}
            for level, classifier in self.classifiers.items():
                outputs[level] = classifier(features)
            return outputs
        else:
            return self.classifier(features)

class SpeciesPredictor:
    """Class for making species predictions on images"""
    
    def __init__(self, model_path, label_encoder_path, device='auto', backbone='efficientnet_v2_s'):
        """
        Initialize the species predictor
        
        Args:
            model_path: Path to saved model (.pth file)
            label_encoder_path: Path to label encoder JSON
            device: Device to use ('auto', 'cuda', 'cpu')
            backbone: Model backbone architecture
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load label encoder
        print(f"Loading label encoder from {label_encoder_path}...")
        with open(label_encoder_path, 'r') as f:
            label_data = json.load(f)
        
        self.classes = label_data['classes']
        self.class_to_idx = label_data['class_to_idx']
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.num_classes = len(self.classes)
        
        print(f"Loaded {self.num_classes} species classes")
        
        # Initialize model
        print(f"Loading model from {model_path}...")
        self.model = EfficientNetSpeciesClassifier(
            num_classes=self.num_classes,
            backbone=backbone,
            dropout=0.3,
            hierarchical=False
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully!")
        if 'best_val_acc' in checkpoint:
            print(f"Model's best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    
    def preprocess_image(self, image_path):
        """Load and preprocess an image"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor
        
        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {e}")
    
    def predict(self, image_path, top_k=5):
        """
        Predict species for a single image
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
            
        Returns:
            List of (species_name, confidence_score) tuples
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            start_time = time.time()
            
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            
            prediction_time = time.time() - start_time
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes))
        
        # Convert to species names and confidence scores
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            species_name = self.idx_to_class[idx.item()]
            confidence = prob.item()
            predictions.append((species_name, confidence))
        
        return predictions, prediction_time
    
    def predict_batch(self, image_paths, top_k=5):
        """
        Predict species for multiple images
        
        Args:
            image_paths: List of image file paths
            top_k: Number of top predictions to return for each image
            
        Returns:
            List of prediction results for each image
        """
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths, 1):
            try:
                predictions, pred_time = self.predict(image_path, top_k)
                results.append({
                    'image_path': image_path,
                    'predictions': predictions,
                    'prediction_time': pred_time,
                    'success': True
                })
                print(f"[{i}/{len(image_paths)}] ✅ {os.path.basename(image_path)}")
                
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'success': False
                })
                print(f"[{i}/{len(image_paths)}] ❌ {os.path.basename(image_path)}: {e}")
        
        return results
    
    def get_model_info(self):
        """Get information about the loaded model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'sample_classes': self.classes[:10]  # Show first 10 classes as sample
        }

def print_predictions(image_path, predictions, prediction_time):
    """Pretty print predictions for a single image"""
    print(f"\n🔍 Predictions for: {os.path.basename(image_path)}")
    print(f"⏱️  Prediction time: {prediction_time*1000:.1f}ms")
    print("-" * 60)
    
    for i, (species, confidence) in enumerate(predictions, 1):
        confidence_pct = confidence * 100
        bar_length = int(confidence * 20)  # Scale to 20 chars
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        print(f"{i:2d}. {species:<30} {confidence_pct:6.2f}% [{bar}]")

def main():
    parser = argparse.ArgumentParser(description='Predict species from images using trained model')
    parser.add_argument('image_path', help='Path to image file or directory')
    parser.add_argument('--model', default='best_model.pth', help='Path to trained model')
    parser.add_argument('--labels', default='species_label_encoder.json', help='Path to label encoder')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    parser.add_argument('--backbone', default='efficientnet_v2_s', help='Model backbone')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions to show')
    parser.add_argument('--batch', action='store_true', help='Process directory of images')
    parser.add_argument('--save-results', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Check if model and label files exist
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("Make sure you've trained a model first using train_species_model.py")
        return
    
    if not os.path.exists(args.labels):
        print(f"❌ Label encoder file not found: {args.labels}")
        print("Make sure you have the species_label_encoder.json file from training")
        return
    
    # Initialize predictor
    try:
        predictor = SpeciesPredictor(
            model_path=args.model,
            label_encoder_path=args.labels,
            device=args.device,
            backbone=args.backbone
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Show model info
    model_info = predictor.get_model_info()
    print(f"\n📊 Model Information:")
    print(f"   Classes: {model_info['num_classes']:,}")
    print(f"   Parameters: {model_info['total_parameters']:,}")
    print(f"   Device: {model_info['device']}")
    print(f"   Sample classes: {', '.join(model_info['sample_classes'][:5])}...")
    
    # Process images
    if args.batch or os.path.isdir(args.image_path):
        # Batch processing
        if os.path.isdir(args.image_path):
            # Get all image files from directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(args.image_path).glob(f'*{ext}'))
                image_files.extend(Path(args.image_path).glob(f'*{ext.upper()}'))
            
            image_paths = [str(p) for p in image_files]
            
            if not image_paths:
                print(f"❌ No image files found in {args.image_path}")
                return
            
            print(f"Found {len(image_paths)} images")
        else:
            print("❌ For batch processing, provide a directory path")
            return
        
        # Process all images
        results = predictor.predict_batch(image_paths, args.top_k)
        
        # Print results
        print(f"\n🎯 BATCH PREDICTION RESULTS")
        print("=" * 70)
        
        successful_predictions = 0
        total_time = 0
        
        for result in results:
            if result['success']:
                successful_predictions += 1
                total_time += result['prediction_time']
                print_predictions(result['image_path'], result['predictions'], result['prediction_time'])
            else:
                print(f"\n❌ Failed: {os.path.basename(result['image_path'])}")
                print(f"   Error: {result['error']}")
        
        print(f"\n📈 Summary:")
        print(f"   Successful predictions: {successful_predictions}/{len(results)}")
        if successful_predictions > 0:
            print(f"   Average prediction time: {(total_time/successful_predictions)*1000:.1f}ms")
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"   Results saved to: {args.save_results}")
    
    else:
        # Single image processing
        if not os.path.exists(args.image_path):
            print(f"❌ Image file not found: {args.image_path}")
            return
        
        try:
            predictions, prediction_time = predictor.predict(args.image_path, args.top_k)
            print_predictions(args.image_path, predictions, prediction_time)
            
            # Save results if requested
            if args.save_results:
                result = {
                    'image_path': args.image_path,
                    'predictions': [(species, float(conf)) for species, conf in predictions],
                    'prediction_time': prediction_time
                }
                with open(args.save_results, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n💾 Results saved to: {args.save_results}")
                
        except Exception as e:
            print(f"❌ Error predicting: {e}")

if __name__ == "__main__":
    main() 