#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import argparse
import os
from pathlib import Path
import pandas as pd

# Import the model from training script
from train_species_model import EfficientNetSpeciesClassifier

class SpeciesPredictor:
    """Species prediction from trained model"""
    
    def __init__(self, model_path, label_encoder_path, device='auto'):
        """
        Args:
            model_path: Path to trained model (.pth file)
            label_encoder_path: Path to label encoder JSON
            device: 'cuda', 'cpu', or 'auto'
        """
        self.device = self._setup_device(device)
        
        # Load label encoder
        print(f"Loading label encoder from {label_encoder_path}...")
        with open(label_encoder_path, 'r') as f:
            label_data = json.load(f)
            self.classes = label_data['classes']
            self.class_to_idx = label_data['class_to_idx']
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        print(f"Loaded {len(self.classes)} species classes")
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = EfficientNetSpeciesClassifier(
            num_classes=len(self.classes),
            backbone='efficientnet_v2_s',  # Make sure this matches training
            dropout=0.3,
            hierarchical=False
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Best validation accuracy during training: {checkpoint.get('best_val_acc', 'Unknown'):.2f}%")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _setup_device(self, device):
        """Setup computation device"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            device = 'cpu'
        return torch.device(device)
    
    def predict_single_image(self, image_path, top_k=5):
        """
        Predict species for a single image
        
        Args:
            image_path: Path to image file
            top_k: Return top K predictions
            
        Returns:
            List of (species, confidence) tuples
        """
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {e}")
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        # Convert to species names
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            species = self.idx_to_class[idx.item()]
            confidence = prob.item()
            predictions.append((species, confidence))
        
        return predictions
    
    def predict_batch(self, image_paths, top_k=5):
        """
        Predict species for multiple images
        
        Args:
            image_paths: List of image paths
            top_k: Return top K predictions for each image
            
        Returns:
            Dictionary mapping image_path -> predictions
        """
        results = {}
        
        print(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            try:
                predictions = self.predict_single_image(image_path, top_k)
                results[image_path] = predictions
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results[image_path] = None
        
        return results
    
    def predict_from_csv(self, csv_path, image_dir, output_csv=None, top_k=5):
        """
        Predict species for images listed in a CSV file
        
        Args:
            csv_path: Path to CSV with image filenames
            image_dir: Directory containing images
            output_csv: Path to save results (optional)
            top_k: Return top K predictions
            
        Returns:
            DataFrame with predictions
        """
        df = pd.read_csv(csv_path)
        image_dir = Path(image_dir)
        
        if 'image_filename' not in df.columns:
            raise ValueError("CSV must contain 'image_filename' column")
        
        print(f"Loaded {len(df)} images from CSV")
        
        # Predict for each image
        predictions_data = []
        
        for idx, row in df.iterrows():
            image_path = image_dir / row['image_filename']
            
            if not image_path.exists():
                print(f"Image not found: {image_path}")
                continue
            
            try:
                predictions = self.predict_single_image(str(image_path), top_k)
                
                # Create row for each prediction
                for rank, (species, confidence) in enumerate(predictions):
                    pred_row = row.copy()
                    pred_row['predicted_species'] = species
                    pred_row['confidence'] = confidence
                    pred_row['rank'] = rank + 1
                    predictions_data.append(pred_row)
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} images")
        
        # Create results DataFrame
        results_df = pd.DataFrame(predictions_data)
        
        if output_csv:
            results_df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")
        
        return results_df

def main():
    parser = argparse.ArgumentParser(description='Species Recognition Inference')
    parser.add_argument('--model', required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--labels', required=True, help='Path to label encoder JSON')
    parser.add_argument('--image', help='Single image to classify')
    parser.add_argument('--csv', help='CSV file with image filenames')
    parser.add_argument('--image_dir', help='Directory containing images (for CSV mode)')
    parser.add_argument('--output', help='Output CSV file (for CSV mode)')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='Computation device')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.csv:
        parser.error("Must specify either --image or --csv")
    
    if args.csv and not args.image_dir:
        parser.error("--image_dir required when using --csv")
    
    # Create predictor
    predictor = SpeciesPredictor(args.model, args.labels, args.device)
    
    if args.image:
        # Single image prediction
        print(f"\n🔍 PREDICTING SPECIES FOR: {args.image}")
        print("=" * 60)
        
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return
        
        try:
            predictions = predictor.predict_single_image(args.image, args.top_k)
            
            print(f"Top {len(predictions)} predictions:")
            for rank, (species, confidence) in enumerate(predictions, 1):
                print(f"  {rank}. {species} ({confidence*100:.2f}%)")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif args.csv:
        # Batch prediction from CSV
        print(f"\n🔍 PREDICTING SPECIES FOR CSV: {args.csv}")
        print("=" * 60)
        
        if not os.path.exists(args.csv):
            print(f"❌ CSV not found: {args.csv}")
            return
            
        if not os.path.exists(args.image_dir):
            print(f"❌ Image directory not found: {args.image_dir}")
            return
        
        try:
            results_df = predictor.predict_from_csv(
                args.csv, 
                args.image_dir, 
                args.output, 
                args.top_k
            )
            
            print(f"\n✅ PREDICTION COMPLETE!")
            print(f"Processed {len(results_df[results_df['rank'] == 1])} images")
            print(f"Total predictions: {len(results_df)}")
            
            if args.output:
                print(f"Results saved to: {args.output}")
            
            # Show sample results
            print(f"\nSample predictions:")
            sample_df = results_df[results_df['rank'] == 1].head(10)
            for _, row in sample_df.iterrows():
                print(f"  {row['image_filename']}: {row['predicted_species']} ({row['confidence']*100:.1f}%)")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 