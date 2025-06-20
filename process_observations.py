import pandas as pd
import zipfile
import os
from collections import Counter
import matplotlib.pyplot as plt
import requests
from urllib.parse import urlparse
import time
from pathlib import Path

# Extract and explore the DwC-A structure
def explore_dwca(archive_path):
    """Explore Darwin Core Archive structure"""
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        # List all files in archive
        file_list = zip_ref.namelist()
        print("Files in archive:")
        for file in file_list:
            print(f"  {file}")
        
        # Extract metadata
        if 'meta.xml' in file_list:
            with zip_ref.open('meta.xml') as meta_file:
                print("\nMetadata structure:")
                print(meta_file.read().decode('utf-8')[:1000])

def debug_csv_structure(archive_path):
    """Debug CSV file structure to understand delimiter and encoding"""
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        occurrence_files = [f for f in zip_ref.namelist() if f.lower() in ['observations.csv', 'occurrence.txt', 'occurrence.csv']]
        
        if occurrence_files:
            print(f"\nDebugging file: {occurrence_files[0]}")
            with zip_ref.open(occurrence_files[0]) as f:
                # Read first few lines as raw text
                first_lines = []
                for i in range(5):
                    try:
                        line = f.readline().decode('utf-8').strip()
                        if line:
                            first_lines.append(line)
                    except:
                        try:
                            f.seek(0)
                            line = f.readline().decode('latin-1').strip()
                            if line:
                                first_lines.append(line)
                        except:
                            break
                
                print("First few lines of the file:")
                for i, line in enumerate(first_lines):
                    print(f"Line {i+1}: {line[:200]}...")  # Show first 200 chars
                    
                # Analyze delimiters
                if first_lines:
                    header = first_lines[0]
                    comma_count = header.count(',')
                    tab_count = header.count('\t')
                    semicolon_count = header.count(';')
                    pipe_count = header.count('|')
                    
                    print(f"\nDelimiter analysis of header:")
                    print(f"  Commas: {comma_count}")
                    print(f"  Tabs: {tab_count}")
                    print(f"  Semicolons: {semicolon_count}")
                    print(f"  Pipes: {pipe_count}")
                    
                    # Suggest best delimiter
                    delimiters = {'comma': comma_count, 'tab': tab_count, 'semicolon': semicolon_count, 'pipe': pipe_count}
                    best_delimiter = max(delimiters, key=delimiters.get)
                    print(f"  Suggested delimiter: {best_delimiter}")

# Load occurrence data (main species observations)
def load_occurrence_data(archive_path, sample_size=10000):
    """Load and sample occurrence data"""
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        # Find the observations file (iNaturalist uses observations.csv)
        occurrence_files = [f for f in zip_ref.namelist() if f.lower() in ['observations.csv', 'occurrence.txt', 'occurrence.csv']]
        
        if occurrence_files:
            print(f"Found observations file: {occurrence_files[0]}")
            
            with zip_ref.open(occurrence_files[0]) as f:
                # Try different delimiters and reading options
                read_kwargs = {
                    'low_memory': False,
                    'encoding': 'utf-8'
                }
                
                # Only add nrows if sample_size is specified
                if sample_size is not None:
                    read_kwargs['nrows'] = sample_size
                
                try:
                    # First try comma-separated (most common for CSV)
                    df = pd.read_csv(f, sep=',', **read_kwargs)
                    if len(df.columns) > 5:  # If we got multiple columns, this worked
                        print(f"Successfully loaded {len(df):,} observations using comma delimiter")
                        return df
                except Exception as e:
                    print(f"Comma delimiter failed: {e}")
                    pass
                
                # Reset file pointer
                f.seek(0)
                try:
                    # Try tab-separated
                    df = pd.read_csv(f, sep='\t', **read_kwargs)
                    if len(df.columns) > 5:
                        print(f"Successfully loaded {len(df):,} observations using tab delimiter")
                        return df
                except Exception as e:
                    print(f"Tab delimiter failed: {e}")
                    pass
                
                # Reset file pointer
                f.seek(0)
                try:
                    # Try auto-detection
                    df = pd.read_csv(f, **read_kwargs)
                    if len(df.columns) > 5:
                        print(f"Successfully loaded {len(df):,} observations using auto-detection")
                        return df
                except Exception as e:
                    print(f"Auto-detection failed: {e}")
                    pass
                
                # Reset file pointer and try with different encoding
                f.seek(0)
                try:
                    read_kwargs['encoding'] = 'latin-1'
                    df = pd.read_csv(f, sep=',', **read_kwargs)
                    print(f"Successfully loaded {len(df):,} observations using latin-1 encoding")
                    return df
                except Exception as e:
                    print(f"Error reading file with latin-1: {e}")
                    return None
        else:
            print("No occurrence file found in archive!")
            print("Available files:")
            for file in zip_ref.namelist():
                print(f"  - {file}")
    return None

# Analyze species distribution
def analyze_species_distribution(df):
    """Analyze species distribution in dataset"""
    
    # Key columns for classification (updated for your actual data structure)
    important_cols = ['scientificName', 'kingdom', 'phylum', 'class', 'order', 
                     'family', 'genus', 'taxonRank', 'decimalLatitude', 'decimalLongitude']
    
    available_cols = [col for col in important_cols if col in df.columns]
    print(f"Available taxonomic columns: {available_cols}")
    
    if not available_cols:
        print("Warning: No expected taxonomic columns found!")
        print(f"Actual columns in data: {list(df.columns)}")
        return df
    
    # Species distribution
    if 'scientificName' in df.columns:
        species_counts = df['scientificName'].value_counts()
        print(f"\nTotal unique species: {len(species_counts)}")
        print(f"Top 10 most observed species:")
        print(species_counts.head(10))
        
        # Distribution by kingdom
        if 'kingdom' in df.columns:
            kingdom_counts = df['kingdom'].value_counts()
            print(f"\nKingdom distribution:")
            print(kingdom_counts)
        
        # Distribution by taxon rank
        if 'taxonRank' in df.columns:
            rank_counts = df['taxonRank'].value_counts()
            print(f"\nTaxon rank distribution:")
            print(rank_counts)
    
    return df[available_cols] if available_cols else df

# Filter high-quality observations
def filter_quality_data(df):
    """Filter for research-grade observations suitable for ML"""
    
    # Remove observations without proper species identification
    quality_filters = {
        'has_scientific_name': df['scientificName'].notna(),
        'research_grade': df.get('qualityGrade', 'research') == 'research',
        'has_coordinates': df['decimalLatitude'].notna() & df['decimalLongitude'].notna(),
        'has_media': df.get('hasMedia', True) == True,  # Assuming photos exist
    }
    
    filtered_df = df.copy()
    for filter_name, condition in quality_filters.items():
        if condition.any():
            before_count = len(filtered_df)
            filtered_df = filtered_df[condition]
            after_count = len(filtered_df)
            print(f"{filter_name}: {before_count} → {after_count} observations")
    
    return filtered_df

# Create hierarchical labels for multi-level classification
def create_hierarchical_labels(df):
    """Create hierarchical classification labels"""
    
    # Create species-level labels (most specific)
    species_labels = df['scientificName'].fillna('Unknown')
    
    # Create higher-level labels for hierarchical classification
    hierarchical_labels = {
        'kingdom': df.get('kingdom', 'Unknown'),
        'phylum': df.get('phylum', 'Unknown'), 
        'class': df.get('class', 'Unknown'),
        'order': df.get('order', 'Unknown'),
        'family': df.get('family', 'Unknown'),
        'genus': df.get('genus', 'Unknown'),
        'species': species_labels
    }
    
    return hierarchical_labels

# Process and save the cleaned dataset
def process_and_save_dataset(archive_path, output_csv="processed_gbif_data.csv", 
                           chunk_size=100000, max_total_rows=None):
    """Process the full GBIF dataset and save to CSV"""
    
    processed_chunks = []
    total_processed = 0
    
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        occurrence_files = [f for f in zip_ref.namelist() if f.lower() in ['observations.csv', 'occurrence.txt', 'occurrence.csv']]
        
        if not occurrence_files:
            print("No occurrence file found!")
            return None
            
        print(f"Processing {occurrence_files[0]}...")
        
        with zip_ref.open(occurrence_files[0]) as f:
            # Determine the correct delimiter by testing a small sample first
            test_sample = f.read(1000).decode('utf-8')
            f.seek(0)
            
            delimiter = ','  # default
            if test_sample.count('\t') > test_sample.count(','):
                delimiter = '\t'
            
            print(f"Using delimiter: {'tab' if delimiter == '\t' else 'comma'}")
            
            # Process in chunks to handle large files
            chunk_num = 0
            
            try:
                for chunk in pd.read_csv(f, sep=delimiter, chunksize=chunk_size, low_memory=False, encoding='utf-8'):
                    chunk_num += 1
                    print(f"Processing chunk {chunk_num}, rows: {len(chunk)}")
                    
                    # Filter for quality observations
                    quality_chunk = filter_quality_observations(chunk)
                    
                    if len(quality_chunk) > 0:
                        # Add image filename column (construct from GBIF data)
                        quality_chunk = add_image_info(quality_chunk)
                        processed_chunks.append(quality_chunk)
                        total_processed += len(quality_chunk)
                        
                        print(f"  Kept {len(quality_chunk)} quality observations")
                        print(f"  Total processed so far: {total_processed}")
                    
                    # Stop if we've hit the max rows limit
                    if max_total_rows and total_processed >= max_total_rows:
                        print(f"Reached maximum rows limit: {max_total_rows}")
                        break
            except Exception as e:
                print(f"Error processing chunks: {e}")
                # Try with different encoding
                f.seek(0)
                try:
                    for chunk in pd.read_csv(f, sep=delimiter, chunksize=chunk_size, low_memory=False, encoding='latin-1'):
                        chunk_num += 1
                        print(f"Processing chunk {chunk_num}, rows: {len(chunk)}")
                        
                        # Filter for quality observations
                        quality_chunk = filter_quality_observations(chunk)
                        
                        if len(quality_chunk) > 0:
                            # Add image filename column (construct from GBIF data)
                            quality_chunk = add_image_info(quality_chunk)
                            processed_chunks.append(quality_chunk)
                            total_processed += len(quality_chunk)
                            
                            print(f"  Kept {len(quality_chunk)} quality observations")
                            print(f"  Total processed so far: {total_processed}")
                        
                        # Stop if we've hit the max rows limit
                        if max_total_rows and total_processed >= max_total_rows:
                            print(f"Reached maximum rows limit: {max_total_rows}")
                            break
                except Exception as e2:
                    print(f"Error with latin-1 encoding: {e2}")
                    return None
    
    if processed_chunks:
        # Combine all chunks
        final_df = pd.concat(processed_chunks, ignore_index=True)
        
        # Final cleaning and balancing
        final_df = balance_species_data(final_df)
        
        # Save to CSV
        final_df.to_csv(output_csv, index=False)
        print(f"\nSaved {len(final_df)} observations to {output_csv}")
        print(f"Unique species: {final_df['scientificName'].nunique()}")
        
        return final_df
    else:
        print("No quality data found!")
        return None

def filter_quality_observations(df):
    """Enhanced quality filtering for iNaturalist ML training"""
    
    print(f"Input observations: {len(df)}")
    
    # Apply quality filters based on your actual column structure
    quality_mask = (
        # Must have scientific name
        df['scientificName'].notna() & 
        (df['scientificName'] != '')
    )
    
    # Must be species level (not genus, subspecies, etc.)
    if 'taxonRank' in df.columns:
        quality_mask = quality_mask & (df['taxonRank'] == 'species')
    
    # Must have coordinates
    quality_mask = quality_mask & df['decimalLatitude'].notna() & df['decimalLongitude'].notna()
    
    # Filter out captive specimens if column exists
    if 'captive' in df.columns:
        # Handle both boolean and string values for captive
        if df['captive'].dtype == 'object':  # String values
            quality_mask = quality_mask & (
                df['captive'].isna() | 
                df['captive'].str.lower().isin(['wild', 'no', 'false', ''])
            )
        else:  # Boolean values
            quality_mask = quality_mask & (df['captive'].isna() | (df['captive'] == False))
    
    # Additional quality filters for image recognition
    # Research grade quality (if available)
    if 'qualityGrade' in df.columns:
        quality_mask = quality_mask & (df['qualityGrade'] == 'research')
    
    # Must have images (check if media exists)
    if 'hasMedia' in df.columns:
        quality_mask = quality_mask & (df['hasMedia'] == True)
    
    result = df[quality_mask].copy()
    print(f"After quality filtering: {len(result)} observations")
    print(f"  Species rank: {(df['taxonRank'] == 'species').sum() if 'taxonRank' in df.columns else 'N/A'}")
    print(f"  Has coordinates: {(df['decimalLatitude'].notna() & df['decimalLongitude'].notna()).sum()}")
    print(f"  Not captive: {(df['captive'].isna() | (df['captive'] == False)).sum() if 'captive' in df.columns else 'N/A'}")
    print(f"  Research grade: {(df['qualityGrade'] == 'research').sum() if 'qualityGrade' in df.columns else 'N/A'}")
    
    return result

def add_image_info(df):
    """Add image filename from iNaturalist observation ID"""
    
    # Use iNaturalist observation ID for image filenames
    df['image_filename'] = df['id'].astype(str) + '.jpg'
    df['observation_id'] = df['id']
    
    return df

def balance_species_data(df, min_samples=5, max_samples=500):
    """Balance the dataset by species"""
    
    print("Balancing species data...")
    
    # Count observations per species
    species_counts = df['scientificName'].value_counts()
    print(f"Species before filtering: {len(species_counts)}")
    
    # Filter species with enough samples
    valid_species = species_counts[species_counts >= min_samples].index
    df_filtered = df[df['scientificName'].isin(valid_species)]
    
    print(f"Species after min sample filter ({min_samples}): {len(valid_species)}")
    
    # Sample from each species to balance
    balanced_dfs = []
    for species in valid_species:
        species_df = df_filtered[df_filtered['scientificName'] == species]
        sample_size = min(len(species_df), max_samples)
        
        if sample_size >= min_samples:
            sampled_df = species_df.sample(n=sample_size, random_state=42)
            balanced_dfs.append(sampled_df)
    
    if balanced_dfs:
        result_df = pd.concat(balanced_dfs, ignore_index=True)
        print(f"Final balanced dataset: {len(result_df)} observations, {result_df['scientificName'].nunique()} species")
        return result_df
    else:
        return df

def filter_for_image_recognition(df, min_observations_per_species=20, 
                                max_observations_per_species=1000,
                                geographic_filter=None):
    """Advanced filtering specifically for image recognition training"""
    
    print(f"\n=== FILTERING FOR IMAGE RECOGNITION ===")
    print(f"Starting with {len(df)} observations")
    
    # 1. Remove observations with problematic species names
    valid_species_mask = (
        df['scientificName'].notna() & 
        (df['scientificName'] != '') &
        (~df['scientificName'].str.contains(r'sp\.|hybrid|x |×', case=False, na=False)) &  # Remove hybrids and unidentified
        (df['scientificName'].str.count(' ') >= 1)  # Must have genus + species (binomial)
    )
    df = df[valid_species_mask]
    print(f"After species name filtering: {len(df)} observations")
    
    # 2. Geographic filtering (optional)
    if geographic_filter:
        lat_min, lat_max, lon_min, lon_max = geographic_filter
        geo_mask = (
            (df['decimalLatitude'] >= lat_min) & 
            (df['decimalLatitude'] <= lat_max) &
            (df['decimalLongitude'] >= lon_min) & 
            (df['decimalLongitude'] <= lon_max)
        )
        df = df[geo_mask]
        print(f"After geographic filtering: {len(df)} observations")
    
    # 3. Filter species by observation count
    species_counts = df['scientificName'].value_counts()
    valid_species = species_counts[
        (species_counts >= min_observations_per_species) & 
        (species_counts <= max_observations_per_species)
    ].index
    
    df = df[df['scientificName'].isin(valid_species)]
    print(f"After species count filtering ({min_observations_per_species}-{max_observations_per_species}): {len(df)} observations")
    print(f"Valid species: {len(valid_species)}")
    
    # 4. Balance dataset by sampling from each species
    balanced_dfs = []
    target_samples = min(max_observations_per_species, 200)  # Cap at reasonable number
    
    for species in valid_species:
        species_df = df[df['scientificName'] == species]
        sample_size = min(len(species_df), target_samples)
        sampled_df = species_df.sample(n=sample_size, random_state=42)
        balanced_dfs.append(sampled_df)
    
    if balanced_dfs:
        result_df = pd.concat(balanced_dfs, ignore_index=True)
        print(f"Final balanced dataset: {len(result_df)} observations, {result_df['scientificName'].nunique()} species")
        
        # Add derived columns for ML
        result_df = add_ml_features(result_df)
        
        return result_df
    else:
        return pd.DataFrame()

def add_ml_features(df):
    """Add features useful for machine learning"""
    
    # Image filename (assuming iNaturalist structure)
    df['image_filename'] = df['id'].astype(str) + '.jpg'
    df['observation_id'] = df['id']
    
    # Taxonomic hierarchy for hierarchical classification
    for level in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']:
        if level not in df.columns:
            df[level] = 'Unknown'
    
    # Extract genus and species separately
    df['genus_species'] = df['scientificName'].str.split(' ', n=1, expand=True)[0]
    df['species_epithet'] = df['scientificName'].str.split(' ', n=1, expand=True)[1]
    
    # Geographic features that might be useful
    df['latitude_rounded'] = df['decimalLatitude'].round(1)
    df['longitude_rounded'] = df['decimalLongitude'].round(1)
    
    # Date features (if eventDate exists)
    if 'eventDate' in df.columns:
        df['eventDate'] = pd.to_datetime(df['eventDate'], errors='coerce')
        df['year'] = df['eventDate'].dt.year
        df['month'] = df['eventDate'].dt.month
        df['day_of_year'] = df['eventDate'].dt.dayofyear
    
    return df

def extract_image_urls(archive_path, observation_ids=None):
    """Extract image URLs from media.csv for downloading"""
    
    media_urls = {}
    
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        if 'media.csv' in zip_ref.namelist():
            print("Extracting image URLs from media.csv...")
            
            with zip_ref.open('media.csv') as f:
                # Read media file
                media_df = pd.read_csv(f)
                
                # Filter for images only
                if 'type' in media_df.columns:
                    image_mask = media_df['type'].str.contains('image', case=False, na=False)
                    media_df = media_df[image_mask]
                
                # Filter for specific observation IDs if provided
                if observation_ids is not None:
                    if 'CoreId' in media_df.columns:
                        media_df = media_df[media_df['CoreId'].isin(observation_ids)]
                    elif 'id' in media_df.columns:
                        media_df = media_df[media_df['id'].isin(observation_ids)]
                
                # Extract URLs
                if 'identifier' in media_df.columns:
                    for _, row in media_df.iterrows():
                        obs_id = row.get('CoreId', row.get('id'))
                        url = row['identifier']
                        if obs_id and url:
                            if obs_id not in media_urls:
                                media_urls[obs_id] = []
                            media_urls[obs_id].append(url)
    
    print(f"Found image URLs for {len(media_urls)} observations")
    return media_urls

def download_images(media_urls, image_dir="images", max_images_per_species=None, delay=0.1):
    """Download images for the filtered observations"""
    
    Path(image_dir).mkdir(exist_ok=True)
    
    downloaded = 0
    failed = 0
    
    print(f"Starting image download to {image_dir}/")
    
    for obs_id, urls in media_urls.items():
        # Take first image URL for each observation
        url = urls[0] if urls else None
        if not url:
            continue
            
        filename = f"{obs_id}.jpg"
        filepath = Path(image_dir) / filename
        
        # Skip if already exists
        if filepath.exists():
            downloaded += 1
            continue
        
        try:
            response = requests.get(url, timeout=10, stream=True)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded += 1
                
                if downloaded % 100 == 0:
                    print(f"Downloaded {downloaded} images...")
            else:
                failed += 1
                
        except Exception as e:
            failed += 1
            if failed % 50 == 0:
                print(f"Failed downloads: {failed}")
        
        # Rate limiting
        time.sleep(delay)
    
    print(f"Download complete: {downloaded} successful, {failed} failed")
    return downloaded, failed

def create_ml_ready_dataset(archive_path, output_dir="ml_dataset", 
                          min_species_samples=20, max_species_samples=200,
                          download_images_flag=True):
    """Create a complete ML-ready dataset with images and labels"""
    
    print("=== CREATING ML-READY DATASET ===")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    image_dir = Path(output_dir) / "images"
    
    # Step 1: Process observations
    print("Step 1: Processing observations...")
    df = load_occurrence_data(archive_path, sample_size=None)  # Load all data
    
    if df is None:
        print("Failed to load data!")
        return None
    
    # Step 2: Filter for ML
    df_ml = filter_for_image_recognition(
        df, 
        min_observations_per_species=min_species_samples,
        max_observations_per_species=max_species_samples
    )
    
    if len(df_ml) == 0:
        print("No valid data after filtering!")
        return None
    
    # Step 3: Save metadata
    metadata_file = Path(output_dir) / "metadata.csv"
    df_ml.to_csv(metadata_file, index=False)
    print(f"Saved metadata to {metadata_file}")
    
    # Step 4: Extract and download images
    if download_images_flag:
        print("Step 2: Extracting image URLs...")
        media_urls = extract_image_urls(archive_path, df_ml['id'].tolist())
        
        print("Step 3: Downloading images...")
        downloaded, failed = download_images(media_urls, str(image_dir))
        
        # Update metadata to only include observations with downloaded images
        available_images = [f.stem for f in image_dir.glob("*.jpg")]
        df_final = df_ml[df_ml['id'].astype(str).isin(available_images)]
        
        df_final.to_csv(metadata_file, index=False)
        print(f"Final dataset: {len(df_final)} observations with images")
    
    # Step 5: Create class mapping
    create_class_mappings(df_ml, output_dir)
    
    # Step 6: Dataset statistics
    print_dataset_statistics(df_ml)
    
    return df_ml

def create_class_mappings(df, output_dir):
    """Create class mappings for different taxonomic levels"""
    
    mappings = {}
    
    for level in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'scientificName']:
        if level in df.columns:
            unique_classes = sorted(df[level].dropna().unique())
            class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
            mappings[level] = {
                'classes': unique_classes,
                'class_to_idx': class_to_idx,
                'num_classes': len(unique_classes)
            }
    
    # Save mappings
    import json
    mapping_file = Path(output_dir) / "class_mappings.json"
    with open(mapping_file, 'w') as f:
        # Convert to JSON-serializable format
        json_mappings = {}
        for level, mapping in mappings.items():
            json_mappings[level] = {
                'classes': mapping['classes'],
                'class_to_idx': mapping['class_to_idx'],
                'num_classes': mapping['num_classes']
            }
        json.dump(json_mappings, f, indent=2)
    
    print(f"Saved class mappings to {mapping_file}")
    return mappings

def print_dataset_statistics(df):
    """Print comprehensive dataset statistics"""
    
    print(f"\n=== DATASET STATISTICS ===")
    print(f"Total observations: {len(df)}")
    print(f"Unique species: {df['scientificName'].nunique()}")
    
    # Taxonomic distribution
    for level in ['kingdom', 'phylum', 'class', 'order', 'family']:
        if level in df.columns:
            count = df[level].nunique()
            print(f"Unique {level}: {count}")
    
    # Species distribution
    species_counts = df['scientificName'].value_counts()
    print(f"\nSpecies distribution:")
    print(f"  Min observations per species: {species_counts.min()}")
    print(f"  Max observations per species: {species_counts.max()}")
    print(f"  Mean observations per species: {species_counts.mean():.1f}")
    print(f"  Median observations per species: {species_counts.median():.1f}")
    
    print(f"\nTop 10 most observed species:")
    print(species_counts.head(10))
    
    # Geographic distribution
    if 'decimalLatitude' in df.columns and 'decimalLongitude' in df.columns:
        print(f"\nGeographic range:")
        print(f"  Latitude: {df['decimalLatitude'].min():.2f} to {df['decimalLatitude'].max():.2f}")
        print(f"  Longitude: {df['decimalLongitude'].min():.2f} to {df['decimalLongitude'].max():.2f}")
    
    # Temporal distribution
    if 'eventDate' in df.columns:
        df['eventDate'] = pd.to_datetime(df['eventDate'], errors='coerce')
        valid_dates = df['eventDate'].dropna()
        if len(valid_dates) > 0:
            print(f"\nTemporal range:")
            print(f"  From: {valid_dates.min()}")
            print(f"  To: {valid_dates.max()}")
            
            yearly_counts = valid_dates.dt.year.value_counts().sort_index()
            print(f"  Most active years: {yearly_counts.tail(3).to_dict()}")

def analyze_data_availability(archive_path, sample_size=50000):
    """Analyze how much data is available before filtering"""
    
    print("=== DATA AVAILABILITY ANALYSIS ===")
    
    # Load a larger sample to get better estimates
    df = load_occurrence_data(archive_path, sample_size=sample_size)
    if df is None:
        print("Failed to load data!")
        return None
    
    print(f"Analyzing sample of {len(df)} observations...")
    
    # Basic counts
    total_obs = len(df)
    species_with_names = df['scientificName'].notna().sum()
    species_level = (df['taxonRank'] == 'species').sum() if 'taxonRank' in df.columns else 0
    research_grade = (df['qualityGrade'] == 'research').sum() if 'qualityGrade' in df.columns else total_obs
    has_coords = (df['decimalLatitude'].notna() & df['decimalLongitude'].notna()).sum()
    not_captive = (df['captive'].isna() | (df['captive'] == False)).sum() if 'captive' in df.columns else total_obs
    
    print(f"\nQuality Analysis:")
    print(f"  Total observations: {total_obs:,}")
    print(f"  With species names: {species_with_names:,} ({species_with_names/total_obs*100:.1f}%)")
    print(f"  Species level: {species_level:,} ({species_level/total_obs*100:.1f}%)")
    print(f"  Research grade: {research_grade:,} ({research_grade/total_obs*100:.1f}%)")
    print(f"  With coordinates: {has_coords:,} ({has_coords/total_obs*100:.1f}%)")
    print(f"  Not captive: {not_captive:,} ({not_captive/total_obs*100:.1f}%)")
    
    # Apply basic quality filters
    quality_mask = (
        df['scientificName'].notna() & 
        (df['scientificName'] != '')
    )
    
    if 'taxonRank' in df.columns:
        quality_mask = quality_mask & (df['taxonRank'] == 'species')
    
    if 'qualityGrade' in df.columns:
        quality_mask = quality_mask & (df['qualityGrade'] == 'research')
    
    quality_mask = quality_mask & df['decimalLatitude'].notna() & df['decimalLongitude'].notna()
    
    if 'captive' in df.columns:
        quality_mask = quality_mask & (df['captive'].isna() | (df['captive'] == False))
    
    quality_df = df[quality_mask]
    print(f"  After basic quality filtering: {len(quality_df):,} ({len(quality_df)/total_obs*100:.1f}%)")
    
    # Species distribution analysis
    if len(quality_df) > 0:
        species_counts = quality_df['scientificName'].value_counts()
        print(f"\nSpecies Distribution:")
        print(f"  Total unique species: {len(species_counts):,}")
        print(f"  Species with 1 observation: {(species_counts == 1).sum():,}")
        print(f"  Species with 2-4 observations: {((species_counts >= 2) & (species_counts <= 4)).sum():,}")
        print(f"  Species with 5-9 observations: {((species_counts >= 5) & (species_counts <= 9)).sum():,}")
        print(f"  Species with 10-19 observations: {((species_counts >= 10) & (species_counts <= 19)).sum():,}")
        print(f"  Species with 20+ observations: {(species_counts >= 20).sum():,}")
        print(f"  Species with 50+ observations: {(species_counts >= 50).sum():,}")
        print(f"  Species with 100+ observations: {(species_counts >= 100).sum():,}")
        
        # Estimate dataset sizes for different thresholds
        print(f"\nEstimated Dataset Sizes:")
        for min_samples in [5, 10, 20, 50, 100]:
            valid_species = species_counts[species_counts >= min_samples]
            if len(valid_species) > 0:
                # Estimate with balancing (cap at 200 samples per species)
                estimated_samples = sum(min(count, 200) for count in valid_species)
                print(f"  Min {min_samples:3d} samples/species: {len(valid_species):4,} species, ~{estimated_samples:6,} total images")
        
        # Kingdom/taxonomic analysis
        if 'kingdom' in quality_df.columns:
            kingdom_counts = quality_df['kingdom'].value_counts()
            print(f"\nTaxonomic Distribution:")
            for kingdom, count in kingdom_counts.head(5).items():
                if pd.notna(kingdom):
                    kingdom_species = quality_df[quality_df['kingdom'] == kingdom]['scientificName'].nunique()
                    print(f"  {kingdom}: {count:,} observations, {kingdom_species:,} species")
    
    return quality_df

def estimate_full_dataset_size(archive_path):
    """Estimate the size of the full dataset without loading it all"""
    
    print("\n=== FULL DATASET SIZE ESTIMATION ===")
    
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        occurrence_files = [f for f in zip_ref.namelist() if f.lower() in ['observations.csv', 'occurrence.txt', 'occurrence.csv']]
        
        if occurrence_files:
            file_info = zip_ref.getinfo(occurrence_files[0])
            file_size_mb = file_info.file_size / (1024 * 1024)
            
            print(f"Observations file: {occurrence_files[0]}")
            print(f"File size: {file_size_mb:.1f} MB")
            
            # Estimate number of rows (rough estimate: ~500-1000 bytes per row for CSV)
            estimated_rows = file_info.file_size // 750  # Conservative estimate
            print(f"Estimated total observations: {estimated_rows:,}")
            
            # Check if there's a media file
            if 'media.csv' in zip_ref.namelist():
                media_info = zip_ref.getinfo('media.csv')
                media_size_mb = media_info.file_size / (1024 * 1024)
                estimated_media_rows = media_info.file_size // 200  # Media entries are smaller
                print(f"Media file size: {media_size_mb:.1f} MB")
                print(f"Estimated media entries: {estimated_media_rows:,}")
            
            return estimated_rows
    
    return None

def check_data_requirements(species_count, samples_per_species, target_accuracy=0.85):
    """Check if the dataset meets requirements for good model performance"""
    
    print(f"\n=== DATA REQUIREMENTS CHECK ===")
    
    total_samples = species_count * samples_per_species
    
    # General guidelines for image classification
    print(f"Dataset: {species_count:,} species × {samples_per_species} samples = {total_samples:,} total images")
    
    # Training data requirements
    if species_count <= 10:
        recommended_samples = 100
        quality = "Excellent"
    elif species_count <= 50:
        recommended_samples = 50
        quality = "Very Good"
    elif species_count <= 100:
        recommended_samples = 30
        quality = "Good"
    elif species_count <= 500:
        recommended_samples = 20
        quality = "Acceptable"
    else:
        recommended_samples = 15
        quality = "Challenging"
    
    print(f"\nRecommendations for {species_count:,} species:")
    print(f"  Recommended samples per species: {recommended_samples}")
    print(f"  Your samples per species: {samples_per_species}")
    print(f"  Expected quality: {quality}")
    
    if samples_per_species >= recommended_samples:
        print(f"  ✅ You have sufficient data for good performance")
    elif samples_per_species >= recommended_samples * 0.7:
        print(f"  ⚠️  You have adequate data, but consider getting more")
    else:
        print(f"  ❌ You may need more data for reliable performance")
    
    # Model complexity recommendations
    if total_samples < 1000:
        print(f"\nModel Recommendations:")
        print(f"  - Use transfer learning (pre-trained models)")
        print(f"  - Start with simple models (EfficientNet-B0)")
        print(f"  - Use heavy data augmentation")
        print(f"  - Consider few-shot learning techniques")
    elif total_samples < 10000:
        print(f"\nModel Recommendations:")
        print(f"  - Transfer learning strongly recommended")
        print(f"  - Medium complexity models (EfficientNet-B2)")
        print(f"  - Moderate data augmentation")
    else:
        print(f"\nModel Recommendations:")
        print(f"  - Transfer learning or training from scratch")
        print(f"  - Can use larger models (EfficientNet-B4+)")
        print(f"  - Standard data augmentation")
    
    return samples_per_species >= recommended_samples

def suggest_optimal_parameters(df_analysis):
    """Suggest optimal parameters based on data analysis"""
    
    if df_analysis is None or len(df_analysis) == 0:
        return None
    
    print(f"\n=== PARAMETER RECOMMENDATIONS ===")
    
    species_counts = df_analysis['scientificName'].value_counts()
    
    # Find optimal balance between species count and samples per species
    scenarios = []
    for min_samples in [5, 10, 15, 20, 30, 50, 100]:
        valid_species = species_counts[species_counts >= min_samples]
        if len(valid_species) > 0:
            avg_samples = min(valid_species.mean(), 200)  # Cap at 200
            total_images = len(valid_species) * min(avg_samples, 200)
            scenarios.append({
                'min_samples': min_samples,
                'species_count': len(valid_species),
                'avg_samples': avg_samples,
                'total_images': total_images,
                'quality_score': len(valid_species) * min(avg_samples, 100) / 1000  # Balanced score
            })
    
    if scenarios:
        print("Scenario Analysis:")
        print("Min Samples | Species | Avg Samples | Total Images | Quality Score")
        print("-" * 65)
        for scenario in scenarios:
            print(f"{scenario['min_samples']:11d} | {scenario['species_count']:7,} | {scenario['avg_samples']:11.1f} | {scenario['total_images']:12,} | {scenario['quality_score']:11.1f}")
        
        # Find best scenario
        best_scenario = max(scenarios, key=lambda x: x['quality_score'])
        print(f"\n🎯 RECOMMENDED PARAMETERS:")
        print(f"   Minimum samples per species: {best_scenario['min_samples']}")
        print(f"   Expected species count: {best_scenario['species_count']:,}")
        print(f"   Expected total images: {best_scenario['total_images']:,}")
        
        # Additional recommendations
        if best_scenario['species_count'] < 50:
            print(f"   🔹 Small dataset - focus on quality and use transfer learning")
        elif best_scenario['species_count'] < 200:
            print(f"   🔹 Medium dataset - good for specialized models")
        else:
            print(f"   🔹 Large dataset - can train robust models")
        
        return best_scenario
    
    return None

def load_observations_from_csv(csv_path, sample_size=10000):
    """Load observations from a standalone CSV file (not in ZIP)"""
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return None
    
    print(f"Loading observations from: {csv_path}")
    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")
    
    read_kwargs = {
        'low_memory': False,
        'encoding': 'utf-8'
    }
    
    # Only add nrows if sample_size is specified
    if sample_size is not None:
        read_kwargs['nrows'] = sample_size
    
    # Try different delimiters
    try:
        # First try comma-separated
        df = pd.read_csv(csv_path, sep=',', **read_kwargs)
        if len(df.columns) > 5:
            print(f"Successfully loaded {len(df):,} observations using comma delimiter")
            return df
    except Exception as e:
        print(f"Comma delimiter failed: {e}")
    
    try:
        # Try tab-separated
        df = pd.read_csv(csv_path, sep='\t', **read_kwargs)
        if len(df.columns) > 5:
            print(f"Successfully loaded {len(df):,} observations using tab delimiter")
            return df
    except Exception as e:
        print(f"Tab delimiter failed: {e}")
    
    try:
        # Try auto-detection
        df = pd.read_csv(csv_path, **read_kwargs)
        if len(df.columns) > 5:
            print(f"Successfully loaded {len(df):,} observations using auto-detection")
            return df
    except Exception as e:
        print(f"Auto-detection failed: {e}")
    
    try:
        # Try latin-1 encoding
        read_kwargs['encoding'] = 'latin-1'
        df = pd.read_csv(csv_path, sep=',', **read_kwargs)
        print(f"Successfully loaded {len(df):,} observations using latin-1 encoding")
        return df
    except Exception as e:
        print(f"Error reading file with latin-1: {e}")
        return None

def detect_file_type_and_load(file_path, sample_size=10000):
    """Auto-detect file type and load observations accordingly"""
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.zip':
        print("Detected ZIP archive, loading from DwC-A...")
        return load_occurrence_data(file_path, sample_size)
    elif file_ext in ['.csv', '.txt']:
        print("Detected CSV/TXT file, loading directly...")
        return load_observations_from_csv(file_path, sample_size)
    else:
        print(f"Unsupported file type: {file_ext}")
        print("Supported formats: .zip (DwC-A), .csv, .txt")
        return None

def analyze_data_availability_simple(df):
    """Simplified data analysis that works with already loaded data"""
    
    print(f"Analyzing {len(df):,} observations...")
    
    # First, let's see what columns we actually have
    print(f"\nAvailable columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")
    
    # Basic counts
    total_obs = len(df)
    species_with_names = df['scientificName'].notna().sum() if 'scientificName' in df.columns else 0
    species_level = (df['taxonRank'] == 'species').sum() if 'taxonRank' in df.columns else 0
    research_grade = (df['qualityGrade'] == 'research').sum() if 'qualityGrade' in df.columns else total_obs
    has_coords = (df['decimalLatitude'].notna() & df['decimalLongitude'].notna()).sum() if 'decimalLatitude' in df.columns and 'decimalLongitude' in df.columns else 0
    not_captive = (df['captive'].isna() | (df['captive'] == False)).sum() if 'captive' in df.columns else total_obs
    
    print(f"\nQuality Analysis:")
    print(f"  Total observations: {total_obs:,}")
    
    # Check each filter step by step
    print(f"\nStep-by-step filtering:")
    
    # Step 1: Scientific names
    if 'scientificName' in df.columns:
        has_name = df['scientificName'].notna() & (df['scientificName'] != '')
        print(f"  1. Has scientific name: {has_name.sum():,} / {total_obs:,} ({has_name.sum()/total_obs*100:.1f}%)")
        remaining_df = df[has_name]
    else:
        print(f"  1. ❌ No 'scientificName' column found!")
        return None
    
    # Step 2: Taxon rank (optional)
    if 'taxonRank' in df.columns:
        species_mask = remaining_df['taxonRank'] == 'species'
        print(f"  2. Species rank: {species_mask.sum():,} / {len(remaining_df):,} ({species_mask.sum()/len(remaining_df)*100:.1f}%)")
        
        # Show what ranks we have
        rank_counts = remaining_df['taxonRank'].value_counts()
        print(f"     Available ranks: {dict(rank_counts.head())}")
        
        # Be flexible - if very few species rank, don't filter by this
        if species_mask.sum() < len(remaining_df) * 0.1:  # Less than 10% are species rank
            print(f"     ⚠️  Very few 'species' rank observations, skipping this filter")
        else:
            remaining_df = remaining_df[species_mask]
    else:
        print(f"  2. No taxonRank column - skipping species filter")
    
    # Step 3: Research grade (optional)
    if 'qualityGrade' in df.columns:
        research_mask = remaining_df['qualityGrade'] == 'research'
        print(f"  3. Research grade: {research_mask.sum():,} / {len(remaining_df):,} ({research_mask.sum()/len(remaining_df)*100:.1f}%)")
        
        # Show what quality grades we have
        quality_counts = remaining_df['qualityGrade'].value_counts()
        print(f"     Available grades: {dict(quality_counts.head())}")
        
        # Be flexible - if very few research grade, don't filter by this
        if research_mask.sum() < len(remaining_df) * 0.1:
            print(f"     ⚠️  Very few 'research' grade observations, skipping this filter")
        else:
            remaining_df = remaining_df[research_mask]
    else:
        print(f"  3. No qualityGrade column - skipping research filter")
    
    # Step 4: Coordinates (more flexible)
    has_lat = 'decimalLatitude' in remaining_df.columns and remaining_df['decimalLatitude'].notna()
    has_lon = 'decimalLongitude' in remaining_df.columns and remaining_df['decimalLongitude'].notna()
    
    if has_lat.any() and has_lon.any():
        coord_mask = has_lat & has_lon
        print(f"  4. Has coordinates: {coord_mask.sum():,} / {len(remaining_df):,} ({coord_mask.sum()/len(remaining_df)*100:.1f}%)")
        
        # Be more flexible with coordinates
        if coord_mask.sum() < len(remaining_df) * 0.5:  # Less than 50% have coordinates
            print(f"     ⚠️  Many observations lack coordinates, but proceeding...")
        
        remaining_df = remaining_df[coord_mask]
    else:
        print(f"  4. ❌ Missing coordinate columns")
        print(f"     Has decimalLatitude: {'decimalLatitude' in remaining_df.columns}")
        print(f"     Has decimalLongitude: {'decimalLongitude' in remaining_df.columns}")
    
    # Step 5: Captive (optional)
    if 'captive' in remaining_df.columns:
        # Handle both boolean and string values for captive
        captive_values = remaining_df['captive'].unique()
        print(f"     Captive values found: {captive_values}")
        
        # More flexible captive filtering
        if remaining_df['captive'].dtype == 'object':  # String values
            not_captive_mask = (
                remaining_df['captive'].isna() | 
                (remaining_df['captive'].str.lower().isin(['wild', 'no', 'false', '']) if remaining_df['captive'].notna().any() else True)
            )
        else:  # Boolean values
            not_captive_mask = remaining_df['captive'].isna() | (remaining_df['captive'] == False)
        
        print(f"  5. Not captive: {not_captive_mask.sum():,} / {len(remaining_df):,} ({not_captive_mask.sum()/len(remaining_df)*100:.1f}%)")
        remaining_df = remaining_df[not_captive_mask]
    else:
        print(f"  5. No captive column - assuming all wild")
    
    print(f"\nFinal result: {len(remaining_df):,} observations remain")
    
    # Species distribution analysis
    if len(remaining_df) > 0:
        species_counts = remaining_df['scientificName'].value_counts()
        print(f"\nSpecies Distribution:")
        print(f"  Total unique species: {len(species_counts):,}")
        
        # Show distribution at different thresholds
        for threshold in [1, 5, 10, 20, 50]:
            count = (species_counts >= threshold).sum()
            print(f"  Species with {threshold}+ observations: {count:,}")
        
        # Show top species
        print(f"\nTop 10 most observed species:")
        for species, count in species_counts.head(10).items():
            print(f"  {species}: {count:,} observations")
        
        # Show sample of data
        print(f"\nSample of remaining data:")
        sample_cols = ['scientificName']
        if 'taxonRank' in remaining_df.columns:
            sample_cols.append('taxonRank')
        if 'qualityGrade' in remaining_df.columns:
            sample_cols.append('qualityGrade')
        if 'decimalLatitude' in remaining_df.columns:
            sample_cols.append('decimalLatitude')
        
        print(remaining_df[sample_cols].head())
    else:
        print("\n❌ NO DATA REMAINING - Diagnosis:")
        print("This usually happens when:")
        print("1. Wrong file format or delimiter")
        print("2. Missing required columns")
        print("3. All data fails quality filters")
        print("4. Column names are different than expected")
        
        print(f"\nFirst few rows of raw data:")
        print(df.head())
    
    return remaining_df if len(remaining_df) > 0 else None

def create_ai_training_csv(file_path, min_species_samples=20, max_species_samples=200, output_file="ai_training_data.csv"):
    """Create a clean CSV file optimized for AI training"""
    
    print(f"\n🔄 PROCESSING DATA FOR AI TRAINING...")
    
    # Load the full dataset
    print("Loading full dataset...")
    df = detect_file_type_and_load(file_path, sample_size=None)  # Load all data
    
    if df is None:
        print("❌ Failed to load dataset")
        return None
    
    print(f"Loaded {len(df):,} total observations")
    
    # Apply quality filters for AI training
    print("Applying quality filters...")
    df_filtered = filter_for_image_recognition(
        df, 
        min_observations_per_species=min_species_samples,
        max_observations_per_species=max_species_samples
    )
    
    if len(df_filtered) == 0:
        print("❌ No data remaining after filtering")
        return None
    
    # Prepare columns for AI training
    print("Preparing columns for AI training...")
    
    # Essential columns for AI training
    essential_columns = [
        'id',                    # Unique identifier
        'scientificName',        # Species label (target)
        'decimalLatitude',       # Location
        'decimalLongitude',      # Location
        'image_filename'         # Image file reference
    ]
    
    # Optional taxonomic hierarchy columns
    taxonomic_columns = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    
    # Optional metadata columns
    metadata_columns = ['eventDate', 'year', 'month', 'day_of_year', 'taxonRank']
    
    # Build final column list
    final_columns = []
    
    # Add essential columns
    for col in essential_columns:
        if col in df_filtered.columns:
            final_columns.append(col)
        else:
            print(f"⚠️  Missing essential column: {col}")
    
    # Add available taxonomic columns
    for col in taxonomic_columns:
        if col in df_filtered.columns:
            final_columns.append(col)
    
    # Add available metadata columns
    for col in metadata_columns:
        if col in df_filtered.columns:
            final_columns.append(col)
    
    # Create final dataset
    df_final = df_filtered[final_columns].copy()
    
    # Add AI-specific columns
    df_final['species_id'] = pd.Categorical(df_final['scientificName']).codes
    df_final['species_count'] = df_final.groupby('scientificName')['scientificName'].transform('count')
    
    # Sort by species name for consistency
    df_final = df_final.sort_values(['scientificName', 'id'])
    
    # Save to CSV
    print(f"Saving to {output_file}...")
    df_final.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n📋 AI TRAINING CSV CREATED:")
    print(f"   File: {output_file}")
    print(f"   Observations: {len(df_final):,}")
    print(f"   Species: {df_final['scientificName'].nunique():,}")
    print(f"   Columns: {len(df_final.columns)}")
    print(f"   Average per species: {len(df_final) / df_final['scientificName'].nunique():.1f}")
    
    print(f"\n📊 COLUMN BREAKDOWN:")
    for col in df_final.columns:
        non_null = df_final[col].notna().sum()
        print(f"   {col}: {non_null:,} non-null ({non_null/len(df_final)*100:.1f}%)")
    
    print(f"\n🔬 SPECIES SAMPLE:")
    species_sample = df_final['scientificName'].value_counts().head(5)
    for species, count in species_sample.items():
        print(f"   {species}: {count} observations")
    
    return df_final

def quick_data_diagnostic(file_path):
    """Quick diagnostic to understand data structure and common issues"""
    
    print("🔍 QUICK DIAGNOSTIC...")
    
    try:
        # Load just a small sample for quick diagnosis
        df_sample = detect_file_type_and_load(file_path, sample_size=1000)
        
        if df_sample is None:
            print("❌ Could not load any data")
            return False
        
        print(f"✅ Loaded {len(df_sample)} sample observations")
        print(f"📋 Found {len(df_sample.columns)} columns")
        
        # Check for common column name variations
        column_mappings = {
            'scientificName': ['scientificName', 'scientific_name', 'species', 'taxon', 'name'],
            'decimalLatitude': ['decimalLatitude', 'decimal_latitude', 'latitude', 'lat', 'y'],
            'decimalLongitude': ['decimalLongitude', 'decimal_longitude', 'longitude', 'lon', 'lng', 'x'],
            'taxonRank': ['taxonRank', 'taxon_rank', 'rank', 'level'],
            'qualityGrade': ['qualityGrade', 'quality_grade', 'quality', 'grade'],
            'id': ['id', 'observation_id', 'observationId', 'gbifID']
        }
        
        print(f"\n🔍 Column mapping analysis:")
        found_columns = {}
        
        for standard_name, variations in column_mappings.items():
            found = None
            for variation in variations:
                if variation in df_sample.columns:
                    found = variation
                    break
            
            if found:
                print(f"  ✅ {standard_name}: found as '{found}'")
                found_columns[standard_name] = found
            else:
                print(f"  ❌ {standard_name}: not found")
                print(f"     Looked for: {variations}")
        
        # Show actual columns for reference
        print(f"\n📝 All available columns:")
        for i, col in enumerate(df_sample.columns):
            non_null = df_sample[col].notna().sum()
            print(f"  {i+1:2d}. {col} ({non_null}/{len(df_sample)} non-null)")
        
        # Check data quality with found columns
        if 'scientificName' in found_columns:
            species_col = found_columns['scientificName']
            valid_species = df_sample[species_col].notna() & (df_sample[species_col] != '')
            print(f"\n🧬 Species data quality:")
            print(f"  Valid species names: {valid_species.sum()}/{len(df_sample)}")
            
            if valid_species.sum() > 0:
                unique_species = df_sample[valid_species][species_col].nunique()
                print(f"  Unique species in sample: {unique_species}")
                
                # Show sample species
                species_sample = df_sample[valid_species][species_col].value_counts().head(5)
                print(f"  Sample species:")
                for species, count in species_sample.items():
                    print(f"    {species}: {count}")
        
        # Show sample of actual data
        print(f"\n📊 Sample data:")
        display_cols = list(df_sample.columns)[:5]  # First 5 columns
        print(df_sample[display_cols].head(3).to_string())
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # Get the dataset file path from user
    print("=== SPECIES DETECTION AI TRAINING DATA PROCESSOR ===")
    print("This script processes iNaturalist data for AI model training")
    
    # Ask for file path
    default_path = "./dataset/gbif-observations-dwca.zip"
    archive_path = input(f"Enter path to your observations file (default: {default_path}): ").strip()
    if not archive_path:
        archive_path = default_path
    
    # Check if file exists
    if not os.path.exists(archive_path):
        print(f"❌ File not found: {archive_path}")
        print("\nPlease check:")
        print("1. The file path is correct")
        print("2. The file exists")
        print("3. You have read permissions")
        
        # List files in current directory to help user
        print(f"\nFiles in current directory:")
        try:
            for file in os.listdir('.'):
                if file.endswith(('.zip', '.csv', '.txt')):
                    print(f"  - {file}")
        except:
            pass
        exit(1)
    
    print(f"Using file: {archive_path}")
    
    # STEP 0: Quick diagnostic
    print("\n" + "="*60)
    print("STEP 0: QUICK DIAGNOSTIC")
    print("="*60)
    
    diagnostic_ok = quick_data_diagnostic(archive_path)
    if not diagnostic_ok:
        print("❌ Diagnostic failed - cannot proceed")
        exit(1)
    
    # Ask if user wants to continue with full analysis
    continue_analysis = input("\n🤔 Continue with full analysis? (y/n): ").strip().lower()
    if continue_analysis != 'y':
        print("Analysis stopped by user")
        exit(0)
    
    # STEP 1: Check file structure
    print("\n" + "="*60)
    print("STEP 1: ANALYZING FILE STRUCTURE")
    print("="*60)
    
    # Detect file type and use appropriate debug function
    file_ext = os.path.splitext(archive_path)[1].lower()
    
    if file_ext == '.zip':
        debug_csv_structure(archive_path)
        estimated_total = estimate_full_dataset_size(archive_path)
    else:
        print(f"Detected {file_ext} file")
        file_size_mb = os.path.getsize(archive_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.1f} MB")
        estimated_total = None
    
    # STEP 2: Load and analyze data
    print(f"\n" + "="*60)
    print("STEP 2: LOADING AND ANALYZING DATA")
    print("="*60)
    
    # Use larger sample for analysis if dataset is big
    analysis_sample_size = min(100000, estimated_total // 10) if estimated_total else 50000
    print(f"Loading {analysis_sample_size:,} observations for analysis...")
    
    # Load data with proper error handling
    try:
        df_analysis = detect_file_type_and_load(archive_path, sample_size=analysis_sample_size)
        
        if df_analysis is None:
            print("❌ Failed to load data file")
            print("\nTroubleshooting:")
            print("1. Check if the file is corrupted")
            print("2. Ensure it's a valid CSV or ZIP file")
            print("3. Try with a smaller sample size")
            exit(1)
        
        if len(df_analysis) == 0:
            print("❌ Loaded empty dataset")
            print("The file was read but contains no data")
            exit(1)
            
        print(f"✅ Successfully loaded {len(df_analysis):,} observations")
        print(f"Columns found: {len(df_analysis.columns)}")
        
        # Quick column check
        required_cols = ['scientificName', 'decimalLatitude', 'decimalLongitude']
        missing_cols = [col for col in required_cols if col not in df_analysis.columns]
        
        if missing_cols:
            print(f"⚠️  Missing important columns: {missing_cols}")
            print("Available columns:")
            for i, col in enumerate(df_analysis.columns):
                print(f"  {i+1:2d}. {col}")
        else:
            print("✅ Found all required columns for AI training")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("\nThis could be due to:")
        print("1. Unsupported file format")
        print("2. Corrupted file")
        print("3. Memory limitations")
        print("4. Encoding issues")
        exit(1)
    
    # STEP 3: Comprehensive analysis
    if df_analysis is not None and len(df_analysis) > 0:
        print(f"\n" + "="*60)
        print("STEP 3: DATA QUALITY ANALYSIS")
        print("="*60)
        
        try:
            # Analyze data availability 
            df_filtered = analyze_data_availability_simple(df_analysis)
            
            if df_filtered is None or len(df_filtered) == 0:
                print("❌ No usable data found after filtering")
                print("Try lowering quality requirements or check your data source")
                exit(1)
            
            # Get parameter recommendations
            best_scenario = suggest_optimal_parameters(df_filtered)
            
            if best_scenario:
                # Check ML requirements
                check_data_requirements(
                    best_scenario['species_count'], 
                    int(best_scenario['avg_samples'])
                )
                
                print(f"\n" + "="*60)
                print("STEP 4: CREATE AI TRAINING DATASET")
                print("="*60)
                
                # Ask user to proceed
                proceed = input(f"\n🚀 Create CSV dataset for AI training? (y/n): ").strip().lower()
                
                if proceed == 'y':
                    # Use recommended parameters
                    min_samples = best_scenario['min_samples']
                    max_samples = 200
                    
                    print(f"\nCreating AI training dataset with:")
                    print(f"  - Minimum {min_samples} observations per species")
                    print(f"  - Maximum {max_samples} observations per species")
                    print(f"  - Expected {best_scenario['species_count']:,} species")
                    print(f"  - Expected {best_scenario['total_images']:,} total observations")
                    
                    # Create the dataset
                    final_df = create_ai_training_csv(
                        archive_path,
                        min_species_samples=min_samples,
                        max_species_samples=max_samples
                    )
                    
                    if final_df is not None and len(final_df) > 0:
                        print(f"\n✅ AI TRAINING DATASET CREATED!")
                        print(f"   📄 CSV file: ./ai_training_data.csv")
                        print(f"   📊 {len(final_df):,} observations")
                        print(f"   🔬 {final_df['scientificName'].nunique():,} species")
                        print(f"   📈 {len(final_df) / final_df['scientificName'].nunique():.1f} avg observations per species")
                        print(f"\n🎯 Ready for AI model training!")
                    else:
                        print("❌ Failed to create training dataset")
                else:
                    print("Dataset creation cancelled")
            else:
                print("❌ Could not determine optimal parameters")
                
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)