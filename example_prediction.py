#!/usr/bin/env python3
"""
Example script showing how to use the SpeciesPredictor class programmatically
"""

from predict_species import SpeciesPredictor
import os

def main():
    # Example usage of the SpeciesPredictor
    
    print("🔬 Species Prediction Example")
    print("=" * 40)
    
    # Check if required files exist
    model_path = 'best_model.pth'
    labels_path = 'species_label_encoder.json'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train a model first using train_species_model.py")
        return
    
    if not os.path.exists(labels_path):
        print(f"❌ Label encoder not found: {labels_path}")
        print("Please make sure you have the species_label_encoder.json file")
        return
    
    # Initialize the predictor
    try:
        predictor = SpeciesPredictor(
            model_path=model_path,
            label_encoder_path=labels_path,
            device='auto'  # Use CUDA if available, otherwise CPU
        )
        
        # Show model information
        model_info = predictor.get_model_info()
        print(f"\n📊 Model loaded successfully!")
        print(f"   Number of species classes: {model_info['num_classes']:,}")
        print(f"   Total parameters: {model_info['total_parameters']:,}")
        print(f"   Using device: {model_info['device']}")
        print(f"   Example species: {', '.join(model_info['sample_classes'][:3])}...")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Example 1: Single image prediction
    print(f"\n1️⃣ Single Image Prediction Example")
    print("-" * 40)
    
    # You would replace this with an actual image path
    example_image = "test_images/wolf.jpg"
    
    if os.path.exists(example_image):
        try:
            predictions, pred_time = predictor.predict(example_image, top_k=3)
            
            print(f"🔍 Predictions for: {os.path.basename(example_image)}")
            print(f"⏱️  Prediction time: {pred_time*1000:.1f}ms")
            print()
            
            for i, (species, confidence) in enumerate(predictions, 1):
                print(f"{i}. {species} ({confidence*100:.1f}% confidence)")
                
        except Exception as e:
            print(f"❌ Error predicting: {e}")
    else:
        print(f"ℹ️  Example image not found: {example_image}")
        print("   Replace with path to an actual image file")
    
    # Example 2: Batch prediction
    print(f"\n2️⃣ Batch Prediction Example")
    print("-" * 40)
    
    # Example image directory
    example_dir = "path/to/image/directory"
    
    if os.path.exists(example_dir) and os.path.isdir(example_dir):
        # Get image files from directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for file in os.listdir(example_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(example_dir, file))
        
        if image_files:
            print(f"Found {len(image_files)} images in directory")
            
            # Process up to 3 images as example
            sample_images = image_files[:3]
            results = predictor.predict_batch(sample_images, top_k=2)
            
            print(f"\nResults:")
            for result in results:
                if result['success']:
                    image_name = os.path.basename(result['image_path'])
                    top_prediction = result['predictions'][0]
                    print(f"  {image_name}: {top_prediction[0]} ({top_prediction[1]*100:.1f}%)")
                else:
                    print(f"  {os.path.basename(result['image_path'])}: Failed - {result['error']}")
        else:
            print(f"No image files found in {example_dir}")
    else:
        print(f"ℹ️  Example directory not found: {example_dir}")
        print("   Replace with path to a directory containing images")
    
    # Example 3: Custom confidence threshold
    print(f"\n3️⃣ Confidence Filtering Example")
    print("-" * 40)
    
    min_confidence = 0.5  # Only show predictions above 50% confidence
    
    print(f"Filtering predictions with confidence >= {min_confidence*100:.0f}%")
    print("This helps identify uncertain predictions that might need manual review")
    
    # In a real scenario, you would apply this filter to actual predictions:
    # for species, confidence in predictions:
    #     if confidence >= min_confidence:
    #         print(f"High confidence: {species} ({confidence*100:.1f}%)")
    #     else:
    #         print(f"Low confidence: {species} ({confidence*100:.1f}%) - needs review")
    
    print(f"\n✅ Example completed!")
    print(f"💡 Usage tips:")
    print(f"   • Use predictor.predict() for single images")
    print(f"   • Use predictor.predict_batch() for multiple images")
    print(f"   • Check confidence scores to identify uncertain predictions")
    print(f"   • Use top_k parameter to get multiple predictions per image")

if __name__ == "__main__":
    main() 