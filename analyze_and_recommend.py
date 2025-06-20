#!/usr/bin/env python3

import pandas as pd
import numpy as np
from collections import Counter
import json
from pathlib import Path

def analyze_data_issues(csv_file='ai_training_data.csv', image_dir='images'):
    """Analyze dataset issues and provide recommendations"""
    
    print("🔍 DEEP DATASET ANALYSIS")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv(csv_file)
    image_path = Path(image_dir)
    
    # Check image availability
    existing_images = 0
    missing_images = 0
    
    for idx, row in df.iterrows():
        img_path = image_path / row['image_filename']
        if img_path.exists():
            existing_images += 1
        else:
            missing_images += 1
    
    print(f"📊 Image Availability:")
    print(f"  Total observations: {len(df):,}")
    print(f"  Existing images: {existing_images:,} ({existing_images/len(df)*100:.1f}%)")
    print(f"  Missing images: {missing_images:,} ({missing_images/len(df)*100:.1f}%)")
    print()
    
    # Taxonomic level analysis
    taxonomic_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'scientificName']
    
    print("🏗️ TAXONOMIC HIERARCHY ANALYSIS:")
    level_stats = {}
    
    for level in taxonomic_levels:
        if level in df.columns:
            unique_count = df[level].nunique()
            level_stats[level] = {
                'unique_count': unique_count,
                'avg_samples': len(df) / unique_count,
                'min_samples': df[level].value_counts().min(),
                'max_samples': df[level].value_counts().max()
            }
            
            print(f"  {level.capitalize()}:")
            print(f"    Unique: {unique_count:,}")
            print(f"    Avg samples/class: {level_stats[level]['avg_samples']:.1f}")
            print(f"    Min samples: {level_stats[level]['min_samples']}")
            print(f"    Max samples: {level_stats[level]['max_samples']}")
    
    print()
    
    # Recommend best taxonomic level
    print("🎯 RECOMMENDATIONS:")
    print()
    
    print("1. **CRITICAL: Address Missing Images**")
    print(f"   - Only {existing_images/len(df)*100:.1f}% of images exist!")
    print("   - Download missing images or filter dataset to existing images only")
    print()
    
    print("2. **Change Target Level**")
    print("   Current: scientificName (7,988 classes)")
    
    # Find best level based on data availability
    best_level = None
    best_score = 0
    
    for level, stats in level_stats.items():
        if stats['avg_samples'] >= 10 and stats['min_samples'] >= 3:
            score = stats['avg_samples'] / stats['unique_count'] * 1000  # Prefer fewer classes with more samples
            if score > best_score:
                best_score = score
                best_level = level
    
    if best_level:
        print(f"   Recommended: {best_level} ({level_stats[best_level]['unique_count']} classes)")
        print(f"   - Avg samples/class: {level_stats[best_level]['avg_samples']:.1f}")
    else:
        print("   Recommended: genus or family (filter to classes with ≥5 samples)")
    
    print()
    print("3. **Training Strategy Changes**")
    print("   - Use much lower learning rate (0.00001)")
    print("   - Increase dropout to 0.5")
    print("   - Enable data augmentation (mixup + cutmix)")
    print("   - Freeze backbone for first 10 epochs")
    print("   - Add class weighting for imbalance")
    print("   - Use cosine learning rate schedule")
    print()
    
    # Create filtered dataset recommendation
    print("4. **Create Filtered Dataset**")
    
    # Filter to existing images only
    df_filtered = df[df['image_filename'].apply(lambda x: (image_path / x).exists())]
    
    # For each taxonomic level, show what we'd get with minimum sample filtering
    for min_samples in [3, 5, 10]:
        print(f"\n   With ≥{min_samples} samples per class:")
        
        for level in ['family', 'genus', 'scientificName']:
            if level in df_filtered.columns:
                class_counts = df_filtered[level].value_counts()
                valid_classes = class_counts[class_counts >= min_samples]
                
                if len(valid_classes) > 0:
                    df_level_filtered = df_filtered[df_filtered[level].isin(valid_classes.index)]
                    
                    print(f"     {level}: {len(valid_classes)} classes, {len(df_level_filtered)} samples")
                    print(f"       Avg: {len(df_level_filtered)/len(valid_classes):.1f} samples/class")
    
    print()
    print("5. **Hardware Recommendations**")
    print("   - Training on CPU will be very slow")
    print("   - Consider using GPU if available")
    print("   - Reduce batch size to 4-8 for limited memory")
    
    return level_stats

def create_filtered_dataset(csv_file='ai_training_data.csv', image_dir='images', 
                          target_level='family', min_samples=5, output_file='filtered_training_data.csv'):
    """Create a filtered dataset with only existing images and sufficient samples per class"""
    
    print(f"\n🔧 Creating filtered dataset...")
    
    df = pd.read_csv(csv_file)
    image_path = Path(image_dir)
    
    # Filter to existing images
    df_filtered = df[df['image_filename'].apply(lambda x: (image_path / x).exists())]
    print(f"After image filtering: {len(df_filtered)} samples")
    
    # Filter to classes with sufficient samples
    if target_level in df_filtered.columns:
        class_counts = df_filtered[target_level].value_counts()
        valid_classes = class_counts[class_counts >= min_samples]
        
        df_final = df_filtered[df_filtered[target_level].isin(valid_classes.index)]
        
        print(f"After class filtering (≥{min_samples} samples): {len(df_final)} samples")
        print(f"Number of {target_level} classes: {len(valid_classes)}")
        print(f"Average samples per class: {len(df_final)/len(valid_classes):.1f}")
        
        # Save filtered dataset
        df_final.to_csv(output_file, index=False)
        print(f"Saved filtered dataset to: {output_file}")
        
        return output_file
    else:
        print(f"Error: {target_level} not found in dataset")
        return None

if __name__ == "__main__":
    # Analyze current dataset
    stats = analyze_data_issues()
    
    # Create filtered dataset
    filtered_file = create_filtered_dataset(
        target_level='family',
        min_samples=5,
        output_file='filtered_family_data.csv'
    )
    
    if filtered_file:
        print(f"\n✅ Next steps:")
        print(f"1. Use the filtered dataset: {filtered_file}")
        print(f"2. Update config to use 'family' as target_level")
        print(f"3. Restart training with improved_config.json") 