#!/usr/bin/env python3

import pandas as pd
import requests
import os
from pathlib import Path
import time
from tqdm import tqdm
import concurrent.futures
from urllib.parse import urlparse
import hashlib

def download_image(args):
    """Download a single image with error handling"""
    observation_id, image_dir, timeout, retry_attempts = args
    
    # iNaturalist image URL pattern
    # Most iNaturalist images follow this pattern: https://inaturalist-open-data.s3.amazonaws.com/photos/{id}/large.{ext}
    # Or: https://static.inaturalist.org/photos/{id}/large.{ext}
    
    possible_urls = [
        f"https://inaturalist-open-data.s3.amazonaws.com/photos/{observation_id}/large.jpg",
        f"https://inaturalist-open-data.s3.amazonaws.com/photos/{observation_id}/medium.jpg",
        f"https://static.inaturalist.org/photos/{observation_id}/large.jpg",
        f"https://static.inaturalist.org/photos/{observation_id}/medium.jpg",
        f"https://inaturalist-open-data.s3.amazonaws.com/photos/{observation_id}/original.jpg",
        f"https://static.inaturalist.org/photos/{observation_id}/original.jpg"
    ]
    
    filepath = image_dir / f"{observation_id}.jpg"
    
    # Skip if already exists
    if filepath.exists():
        return f"exists:{observation_id}"
    
    # Try each URL
    for url in possible_urls:
        for attempt in range(retry_attempts):
            try:
                response = requests.get(url, timeout=timeout, stream=True)
                if response.status_code == 200:
                    # Check if it's actually an image
                    content_type = response.headers.get('content-type', '')
                    if 'image' in content_type:
                        with open(filepath, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        return f"success:{observation_id}"
                    else:
                        # Not an image, try next URL
                        break
                elif response.status_code == 404:
                    # Not found, try next URL
                    break
                else:
                    # Other error, retry
                    if attempt < retry_attempts - 1:
                        time.sleep(0.5)
                        continue
                    else:
                        break
            except Exception as e:
                if attempt < retry_attempts - 1:
                    time.sleep(0.5)
                    continue
                else:
                    # Last attempt failed
                    break
    
    return f"failed:{observation_id}"

def download_sample_images(csv_file, image_dir, max_images=1000, max_workers=10):
    """Download a sample of images for testing"""
    
    print(f"📥 DOWNLOADING SAMPLE IMAGES FOR TESTING")
    print("=" * 50)
    
    df = pd.read_csv(csv_file)
    print(f"Total observations in CSV: {len(df):,}")
    
    # Sample random observations
    sample_df = df.sample(n=min(max_images, len(df)), random_state=42)
    print(f"Downloading {len(sample_df)} sample images...")
    
    # Create image directory
    image_dir = Path(image_dir)
    image_dir.mkdir(exist_ok=True)
    
    # Prepare download arguments
    download_args = [
        (row['id'], image_dir, 10, 2)  # id, image_dir, timeout, retry_attempts
        for _, row in sample_df.iterrows()
    ]
    
    # Download with progress bar
    successful = 0
    failed = 0
    exists = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(download_image, download_args),
            total=len(download_args),
            desc="Downloading images"
        ))
    
    # Count results
    for result in results:
        if result.startswith("success:"):
            successful += 1
        elif result.startswith("failed:"):
            failed += 1
        elif result.startswith("exists:"):
            exists += 1
    
    print(f"\n📊 DOWNLOAD RESULTS:")
    print(f"  ✅ Successfully downloaded: {successful}")
    print(f"  ♻️  Already existed: {exists}")
    print(f"  ❌ Failed to download: {failed}")
    print(f"  📁 Images saved to: {image_dir}")
    
    total_available = successful + exists
    print(f"\n🎯 Ready for training with {total_available} images!")
    
    return total_available

def create_minimal_dataset(csv_file, image_dir, output_csv, min_species=50, max_per_species=20):
    """Create a minimal dataset for quick training/testing"""
    
    print(f"🔄 CREATING MINIMAL DATASET")
    print("=" * 50)
    
    df = pd.read_csv(csv_file)
    
    # Select top species by count
    species_counts = df['scientificName'].value_counts()
    top_species = species_counts.head(min_species).index
    
    print(f"Selected top {len(top_species)} species")
    
    # Sample from each species
    minimal_data = []
    for species in top_species:
        species_df = df[df['scientificName'] == species]
        sample_size = min(len(species_df), max_per_species)
        sampled = species_df.sample(n=sample_size, random_state=42)
        minimal_data.append(sampled)
    
    minimal_df = pd.concat(minimal_data, ignore_index=True)
    
    print(f"Minimal dataset: {len(minimal_df)} observations, {minimal_df['scientificName'].nunique()} species")
    
    # Save minimal CSV
    minimal_df.to_csv(output_csv, index=False)
    print(f"Saved to: {output_csv}")
    
    # Download images for minimal dataset
    sample_ids = minimal_df['id'].tolist()
    download_args = [
        (obs_id, Path(image_dir), 10, 2)
        for obs_id in sample_ids
    ]
    
    print(f"Downloading {len(download_args)} images for minimal dataset...")
    
    # Create image directory
    Path(image_dir).mkdir(exist_ok=True)
    
    # Download
    successful = 0
    failed = 0
    exists = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(
            executor.map(download_image, download_args),
            total=len(download_args),
            desc="Downloading minimal dataset"
        ))
    
    for result in results:
        if result.startswith("success:"):
            successful += 1
        elif result.startswith("failed:"):
            failed += 1
        elif result.startswith("exists:"):
            exists += 1
    
    print(f"\n📊 MINIMAL DATASET RESULTS:")
    print(f"  ✅ Successfully downloaded: {successful}")
    print(f"  ♻️  Already existed: {exists}")
    print(f"  ❌ Failed to download: {failed}")
    
    return successful + exists

def main():
    print("🚀 SPECIES RECOGNITION IMAGE DOWNLOADER")
    print("=" * 50)
    
    # Configuration
    csv_file = 'ai_training_data.csv'
    image_dir = 'images'
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        print("Make sure you've run the data processing script first!")
        return
    
    print("Choose an option:")
    print("1. Download sample images for testing (1000 images)")
    print("2. Create minimal dataset for quick training (50 species, ~1000 images)")
    print("3. Download all images (this will take a very long time!)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        download_sample_images(csv_file, image_dir, max_images=1000)
        
    elif choice == '2':
        total_images = create_minimal_dataset(
            csv_file, 
            image_dir, 
            'minimal_training_data.csv',
            min_species=50,
            max_per_species=20
        )
        
        if total_images > 0:
            print(f"\n🎯 MINIMAL DATASET READY!")
            print(f"To train with this dataset, use:")
            print(f"python train_species_model.py")
            print(f"# Then edit the config to use 'minimal_training_data.csv'")
        
    elif choice == '3':
        print("⚠️  WARNING: This will attempt to download ~42,000 images!")
        print("This could take several hours and use significant bandwidth.")
        confirm = input("Are you sure? (type 'yes' to continue): ").strip().lower()
        
        if confirm == 'yes':
            # For full download, we'd need to process in batches
            print("Starting full download... (you can interrupt with Ctrl+C)")
            
            df = pd.read_csv(csv_file)
            batch_size = 1000
            total_batches = (len(df) + batch_size - 1) // batch_size
            
            total_successful = 0
            total_failed = 0
            total_exists = 0
            
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]
                print(f"\nProcessing batch {i//batch_size + 1}/{total_batches}")
                
                download_args = [
                    (row['id'], Path(image_dir), 10, 2)
                    for _, row in batch_df.iterrows()
                ]
                
                Path(image_dir).mkdir(exist_ok=True)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    results = list(tqdm(
                        executor.map(download_image, download_args),
                        total=len(download_args),
                        desc=f"Batch {i//batch_size + 1}"
                    ))
                
                # Count batch results
                batch_successful = sum(1 for r in results if r.startswith("success:"))
                batch_failed = sum(1 for r in results if r.startswith("failed:"))
                batch_exists = sum(1 for r in results if r.startswith("exists:"))
                
                total_successful += batch_successful
                total_failed += batch_failed
                total_exists += batch_exists
                
                print(f"Batch results: {batch_successful} success, {batch_exists} existed, {batch_failed} failed")
                print(f"Total so far: {total_successful + total_exists} images available")
                
                # Brief pause between batches
                time.sleep(1)
            
            print(f"\n🎉 FULL DOWNLOAD COMPLETE!")
            print(f"Total images available: {total_successful + total_exists}")
        else:
            print("Download cancelled")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 