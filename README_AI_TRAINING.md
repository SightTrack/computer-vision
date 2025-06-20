# Species Recognition AI Training Guide

This guide shows you how to train an AI model to recognize plant and animal species using your processed iNaturalist/GBIF data.

## 📋 Prerequisites

You should have:
- ✅ A processed CSV file (`ai_training_data.csv`) with species data
- 🖼️ Downloaded images corresponding to the CSV entries
- 🐍 Python 3.8+ with GPU support (recommended)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Images

You need to download the actual images. You have a few options:

**Option A: Use the image download from your data processing script**
```bash
python process_observations.py
# When prompted, choose to download images
```

**Option B: Organize existing images**
```bash
# Create images directory and copy your images there
mkdir images
# Copy or link your images to the images/ directory
# Images should be named like: 12345.jpg (matching the image_filename in CSV)
```

**Option C: Download from iNaturalist URLs (if you have them)**
```bash
# You can modify the process_observations.py script to download images
# based on the observation IDs in your CSV
```

### 3. Train the Model

```bash
python train_species_model.py
```

The script will:
- 📊 Analyze your dataset
- 🔄 Create train/validation splits
- 🤖 Train an EfficientNet model
- 💾 Save the best model and training curves
- 📈 Show training progress with tensorboard

## 📁 File Structure

After training, you'll have:
```
sighttrack-ai/
├── ai_training_data.csv          # Your processed species data
├── images/                       # Directory with species images
├── train_species_model.py        # Main training script
├── inference.py                  # Prediction script
├── requirements.txt              # Python dependencies
├── best_model.pth               # Trained model weights
├── species_label_encoder.json   # Species class mappings
├── train_split.csv              # Training data split
├── val_split.csv                # Validation data split
├── training_curves.png          # Training progress plots
└── runs/                        # Tensorboard logs
```

## 🎯 Training Details

### Model Architecture
- **Backbone**: EfficientNet-V2-S (pre-trained on ImageNet)
- **Input Size**: 224×224 RGB images
- **Classes**: Number of unique species in your dataset
- **Transfer Learning**: Uses pre-trained weights for faster training

### Data Augmentation
- Random resized crops
- Random horizontal flips
- Random rotations (±10°)
- Color jittering (brightness, contrast, saturation, hue)
- Normalization with ImageNet statistics

### Training Configuration
```python
{
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 3e-4,
    'image_size': 224,
    'early_stopping_patience': 15,
    'device': 'cuda' if available else 'cpu'
}
```

## 📊 Monitoring Training

### Tensorboard
View real-time training progress:
```bash
tensorboard --logdir runs
```
Then open http://localhost:6006 in your browser.

### Training Output
The script prints progress including:
- Loss and accuracy for training/validation
- Learning rate scheduling
- Best model checkpoints
- Early stopping notifications

## 🔍 Using the Trained Model

### Single Image Prediction
```bash
python inference.py --model best_model.pth --labels species_label_encoder.json --image path/to/image.jpg
```

### Batch Prediction from CSV
```bash
python inference.py --model best_model.pth --labels species_label_encoder.json --csv test_images.csv --image_dir images/ --output predictions.csv
```

### Python API
```python
from inference import SpeciesPredictor

# Load predictor
predictor = SpeciesPredictor('best_model.pth', 'species_label_encoder.json')

# Predict single image
predictions = predictor.predict_single_image('image.jpg', top_k=5)
for species, confidence in predictions:
    print(f"{species}: {confidence*100:.2f}%")
```

## 📈 Expected Performance

### Dataset Size Guidelines
Based on your data, here's what to expect:

| Species Count | Samples/Species | Expected Accuracy | Quality |
|---------------|-----------------|-------------------|---------|
| 10-50         | 50+             | 85-95%           | Excellent |
| 50-200        | 20-50           | 75-85%           | Very Good |
| 200-500       | 10-20           | 65-75%           | Good |
| 500+          | 5-10            | 50-65%           | Challenging |

### Performance Factors
- **More samples per species** = Higher accuracy
- **More diverse images** = Better generalization
- **Higher image quality** = Better feature learning
- **Balanced dataset** = More consistent performance

## 🛠️ Customization

### Modify Training Parameters
Edit the `config` dictionary in `train_species_model.py`:

```python
config = {
    'batch_size': 32,           # Reduce if out of memory
    'num_epochs': 100,          # Increase for more training
    'backbone': 'efficientnet_v2_s',  # Try 'efficientnet_v2_m' or 'resnet50'
    'image_size': 224,          # Increase for higher resolution
    'early_stopping_patience': 15,
}
```

### Hierarchical Classification
Enable multi-level classification (genus, family, etc.):

```python
model = EfficientNetSpeciesClassifier(
    num_classes=train_dataset.num_classes,
    hierarchical=True,  # Enable hierarchical mode
    num_classes_dict=num_classes_dict
)
```

### Data Augmentation
Modify transforms in `create_data_transforms()`:

```python
# Add more aggressive augmentation
transforms.RandomRotation(degrees=30),      # Increase rotation
transforms.RandomPerspective(distortion_scale=0.2),  # Add perspective
transforms.GaussianBlur(kernel_size=3),     # Add blur
```

## 🚨 Troubleshooting

### Common Issues

**Out of Memory Error**
```bash
# Reduce batch size
config['batch_size'] = 16  # or 8

# Reduce number of workers
config['num_workers'] = 2  # or 0
```

**Low Accuracy**
- Increase training epochs
- Add more data augmentation
- Use a larger model backbone
- Collect more training data

**Images Not Found**
- Check that images are in the specified directory
- Verify image filenames match CSV entries
- Ensure images are in supported formats (JPG, PNG)

### Performance Optimization

**GPU Training**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage
nvidia-smi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0
```

**CPU Training**
If no GPU available, the training will use CPU (slower but works):
```python
config['device'] = 'cpu'
config['num_workers'] = 0  # Reduce for CPU
config['batch_size'] = 16  # Smaller batches for CPU
```

## 📝 Next Steps

After training:

1. **Evaluate Model**: Test on held-out data
2. **Deploy Model**: Integrate into your application
3. **Improve Data**: Collect more samples for underperforming species
4. **Fine-tune**: Adjust hyperparameters based on results
5. **Ensemble**: Combine multiple models for better accuracy

## 🆘 Support

If you encounter issues:

1. Check that all file paths are correct
2. Verify image formats are supported
3. Ensure sufficient disk space and memory
4. Review the error messages for specific guidance

For advanced features like hierarchical classification or custom architectures, modify the model classes in `train_species_model.py`.

## 📊 Dataset Statistics

Your current dataset (`ai_training_data.csv`):
- **Total observations**: 42,576
- **Unique species**: Run the training script to see detailed analysis
- **Recommended**: Check class balance and add more samples for rare species if needed

Happy training! 🎯🚀 