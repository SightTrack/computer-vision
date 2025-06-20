#!/usr/bin/env python3
"""
SightTrack AI - Prediction Script
Professional inference script for species classification
"""

import os
import sys
import yaml
import torch
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image
import torchvision.transforms as transforms

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.model import load_trained_model


class SpeciesPredictor:
    """Professional species prediction class."""
    
    def __init__(self, model_path: str, config_path: str, label_encoder_path: str, device: str = "auto"):
        """
        Initialize the species predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model configuration
            label_encoder_path: Path to label encoder JSON
            device: Device to use for inference
        """
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load label encoder
        with open(label_encoder_path, 'r') as f:
            label_data = json.load(f)
        
        self.class_names = label_data["classes"]
        self.class_to_idx = label_data["class_to_idx"]
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_names)
        self.target_level = label_data["target_level"]
        
        # Update config with actual number of classes
        self.config["model"]["num_classes"] = self.num_classes
        
        # Load model
        self.model = load_trained_model(model_path, self.config, self.device)
        
        # Create image transforms
        self.transform = self._create_transform()
        
        print(f"Model loaded successfully!")
        print(f"Target level: {self.target_level}")
        print(f"Number of classes: {self.num_classes}")
    
    def _create_transform(self) -> transforms.Compose:
        """Create image preprocessing transforms."""
        image_size = self.config["model"]["image_size"]
        
        return transforms.Compose([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess an image."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            raise ValueError(f"Error processing image {image_path}: {e}")
    
    def predict(self, image_path: str, top_k: int = 5) -> Tuple[List[Tuple[str, float]], float]:
        """
        Predict species for a single image.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of (predictions, inference_time)
        """
        import time
        
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            start_time = time.time()
            
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
            inference_time = time.time() - start_time
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes))
        
        # Convert to species names and confidence scores
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            species_name = self.idx_to_class[idx.item()]
            confidence = prob.item()
            predictions.append((species_name, confidence))
        
        return predictions, inference_time
    
    def predict_batch(self, image_paths: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Predict species for multiple images.
        
        Args:
            image_paths: List of image file paths
            top_k: Number of top predictions per image
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                predictions, inference_time = self.predict(image_path, top_k)
                
                result = {
                    "image_path": image_path,
                    "predictions": predictions,
                    "inference_time": inference_time,
                    "status": "success"
                }
            except Exception as e:
                result = {
                    "image_path": image_path,
                    "predictions": [],
                    "inference_time": 0.0,
                    "status": "error",
                    "error": str(e)
                }
            
            results.append(result)
        
        return results


def print_predictions(image_path: str, predictions: List[Tuple[str, float]], inference_time: float):
    """Print prediction results in a formatted way."""
    print(f"\nImage: {image_path}")
    print(f"Inference time: {inference_time:.3f}s")
    print("-" * 50)
    
    for i, (species_name, confidence) in enumerate(predictions, 1):
        print(f"{i:2d}. {species_name:<30} {confidence:.3f} ({confidence*100:.1f}%)")


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="SightTrack AI - Species Classification Prediction")
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to image file or directory of images"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model configuration"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="models/label_encoder.json",
        help="Path to label encoder"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to show"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results (JSON format)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process directory of images in batch mode"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    print("Validating input files...")
    
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)
    
    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    if not Path(args.labels).exists():
        print(f"❌ Label encoder file not found: {args.labels}")
        sys.exit(1)
    
    # Initialize predictor
    print("Loading model...")
    try:
        predictor = SpeciesPredictor(
            model_path=args.model,
            config_path=args.config,
            label_encoder_path=args.labels,
            device=args.device
        )
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    print("✓ Model loaded successfully")
    print("=" * 60)
    
    # Process images
    if args.batch:
        # Batch processing
        image_dir = Path(args.image_path)
        if not image_dir.is_dir():
            print(f"❌ Directory not found: {image_dir}")
            sys.exit(1)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = [
            str(p) for p in image_dir.iterdir() 
            if p.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            print(f"❌ No image files found in {image_dir}")
            sys.exit(1)
        
        print(f"Processing {len(image_paths)} images...")
        results = predictor.predict_batch(image_paths, args.top_k)
        
        # Print results
        successful = 0
        failed = 0
        
        for result in results:
            if result["status"] == "success":
                print_predictions(
                    result["image_path"], 
                    result["predictions"], 
                    result["inference_time"]
                )
                successful += 1
            else:
                print(f"\n❌ Failed to process {result['image_path']}: {result['error']}")
                failed += 1
        
        print("=" * 60)
        print(f"Batch processing completed:")
        print(f"✓ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        # Save results to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")
    
    else:
        # Single image processing
        image_path = Path(args.image_path)
        if not image_path.exists():
            print(f"❌ Image file not found: {image_path}")
            sys.exit(1)
        
        try:
            predictions, inference_time = predictor.predict(str(image_path), args.top_k)
            print_predictions(str(image_path), predictions, inference_time)
            
            # Save result if requested
            if args.output:
                result = {
                    "image_path": str(image_path),
                    "predictions": predictions,
                    "inference_time": inference_time,
                    "status": "success"
                }
                
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResult saved to: {args.output}")
        
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main() 