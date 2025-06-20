#!/usr/bin/env python3

import pandas as pd
import zipfile
import os
from pathlib import Path
import gc  # Garbage collection

def process_large_dataset_in_chunks(file_path, 
                                   output_file="ai_training_data.csv",
                                   chunk_size=10000,
                                   min_species_samples=2,
                                   max_species_samples=100,
                                   max_total_rows=100000):
    """
    Memory-efficient processing of large datasets by chunking
    """
    
    print("🔄 MEMORY-EFFICIENT PROCESSING...")
    print(f"Target output: {output_file}")
    print(f"Chunk size: {chunk_size:,} rows")
    print(f"Min samples per species: {min_species_samples}")
    print(f"Max samples per species: {max_species_samples}")
    print(f"Max total output rows: {max_total_rows:,}")
    
    # Track species counts across chunks
    species_counts = {}
    species_samples = {}  # Store samples for each species
    total_processed = 0
    total_kept = 0
    chunk_num = 0
    
    # Open output file for writing
    output_path = Path(output_file)
    wrote_header = False
    
    try:
        # Open the ZIP file and observations CSV
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            with zip_ref.open('observations.csv') as f:
                print("📂 Processing file in chunks...")
                
                # Process in chunks
                for chunk in pd.read_csv(f, chunksize=chunk_size, low_memory=False):
                    chunk_num += 1
                    total_processed += len(chunk)
                    
                    print(f"\n🔄 Processing chunk {chunk_num} ({len(chunk):,} rows)")
                    print(f"   Total processed so far: {total_processed:,}")
                    
                    # Apply basic quality filters
                    filtered_chunk = apply_basic_filters(chunk)
                    
                    if len(filtered_chunk) == 0:
                        print("   ⚠️  No data survived filtering in this chunk")
                        continue
                    
                    print(f"   ✅ {len(filtered_chunk):,} rows passed basic filters")
                    
                    # Process species in this chunk
                    chunk_species_data = []
                    
                    for species in filtered_chunk['scientificName'].unique():
                        species_rows = filtered_chunk[filtered_chunk['scientificName'] == species]
                        
                        # Update global species count
                        if species not in species_counts:
                            species_counts[species] = 0
                            species_samples[species] = []
                        
                        # Add samples for this species (up to limit)
                        current_count = species_counts[species]
                        available_slots = max_species_samples - current_count
                        
                        if available_slots > 0:
                            # Take up to available_slots samples
                            new_samples = species_rows.head(available_slots)
                            species_samples[species].append(new_samples)
                            species_counts[species] += len(new_samples)
                            chunk_species_data.append(new_samples)
                    
                    # Combine chunk data
                    if chunk_species_data:
                        chunk_output = pd.concat(chunk_species_data, ignore_index=True)
                        total_kept += len(chunk_output)
                        
                        print(f"   📝 Keeping {len(chunk_output):,} rows from this chunk")
                        
                        # Prepare columns for output
                        chunk_output = prepare_columns_for_ai(chunk_output)
                        
                        # Write to CSV (append mode)
                        chunk_output.to_csv(
                            output_path, 
                            mode='w' if not wrote_header else 'a',
                            header=not wrote_header,
                            index=False
                        )
                        wrote_header = True
                        
                        print(f"   💾 Written to CSV. Total kept: {total_kept:,}")
                    
                    # Check if we've reached the limit
                    if total_kept >= max_total_rows:
                        print(f"\n🛑 Reached maximum output rows ({max_total_rows:,})")
                        break
                    
                    # Memory cleanup
                    del chunk, filtered_chunk
                    if chunk_species_data:
                        del chunk_species_data
                    gc.collect()
                    
                    # Progress update
                    if chunk_num % 10 == 0:
                        print(f"\n📊 Progress Update:")
                        print(f"   Chunks processed: {chunk_num}")
                        print(f"   Total rows processed: {total_processed:,}")
                        print(f"   Total rows kept: {total_kept:,}")
                        print(f"   Unique species found: {len(species_counts):,}")
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return None
    
    print(f"\n✅ PROCESSING COMPLETE!")
    
    # Final filtering by species count
    if total_kept > 0:
        print(f"📋 Applying final species count filter...")
        final_df = apply_species_count_filter(output_file, min_species_samples)
        
        print(f"\n🎉 FINAL RESULTS:")
        print(f"   📄 Output file: {output_file}")
        print(f"   📊 Total observations: {len(final_df):,}")
        print(f"   🔬 Unique species: {final_df['scientificName'].nunique():,}")
        print(f"   📈 Avg observations per species: {len(final_df) / final_df['scientificName'].nunique():.1f}")
        
        return final_df
    else:
        print("❌ No data was processed")
        return None

def apply_basic_filters(df):
    """Apply basic quality filters to a chunk"""
    
    # Start with all rows
    mask = pd.Series(True, index=df.index)
    
    # Must have scientific name
    if 'scientificName' in df.columns:
        mask &= df['scientificName'].notna() & (df['scientificName'] != '')
    else:
        return pd.DataFrame()  # No scientific name column
    
    # Must be species level (if column exists)
    if 'taxonRank' in df.columns:
        mask &= df['taxonRank'] == 'species'
    
    # Must have coordinates
    if 'decimalLatitude' in df.columns and 'decimalLongitude' in df.columns:
        mask &= df['decimalLatitude'].notna() & df['decimalLongitude'].notna()
    
    # Not captive (handle string values)
    if 'captive' in df.columns:
        if df['captive'].dtype == 'object':
            mask &= df['captive'].str.lower().isin(['wild', 'no', 'false', '']) | df['captive'].isna()
        else:
            mask &= (df['captive'] == False) | df['captive'].isna()
    
    # Remove problematic species names
    mask &= ~df['scientificName'].str.contains(r'sp\.|hybrid|x |×', case=False, na=False)
    mask &= df['scientificName'].str.count(' ') >= 1  # Must have genus + species
    
    return df[mask].copy()

def prepare_columns_for_ai(df):
    """Prepare columns for AI training"""
    
    # Essential columns
    essential_cols = ['id', 'scientificName', 'decimalLatitude', 'decimalLongitude']
    
    # Optional columns
    optional_cols = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'taxonRank', 'eventDate']
    
    # Build final column list
    final_cols = []
    for col in essential_cols:
        if col in df.columns:
            final_cols.append(col)
    
    for col in optional_cols:
        if col in df.columns:
            final_cols.append(col)
    
    # Create output dataframe
    result = df[final_cols].copy()
    
    # Add AI-specific columns
    result['image_filename'] = result['id'].astype(str) + '.jpg'
    
    return result

def apply_species_count_filter(csv_file, min_samples):
    """Apply final species count filter to the CSV file"""
    
    print(f"🔍 Loading final CSV for species filtering...")
    
    # Load the CSV
    df = pd.read_csv(csv_file)
    
    # Count samples per species
    species_counts = df['scientificName'].value_counts()
    valid_species = species_counts[species_counts >= min_samples].index
    
    print(f"   Species before filter: {len(species_counts):,}")
    print(f"   Species after filter: {len(valid_species):,}")
    
    # Filter and save
    final_df = df[df['scientificName'].isin(valid_species)].copy()
    
    # Add species ID for AI training
    final_df['species_id'] = pd.Categorical(final_df['scientificName']).codes
    
    # Save final version
    final_df.to_csv(csv_file, index=False)
    
    return final_df

def estimate_memory_usage(file_path, chunk_size=10000):
    """Estimate memory usage for processing"""
    
    print("🧮 ESTIMATING MEMORY USAGE...")
    
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        with zip_ref.open('observations.csv') as f:
            # Read a small sample to estimate
            sample = pd.read_csv(f, nrows=1000)
            
            # Estimate memory per row
            memory_per_row = sample.memory_usage(deep=True).sum() / len(sample)
            chunk_memory_mb = (memory_per_row * chunk_size) / (1024 * 1024)
            
            print(f"   Estimated memory per row: {memory_per_row:.0f} bytes")
            print(f"   Estimated memory per chunk: {chunk_memory_mb:.1f} MB")
            print(f"   Recommended chunk size: {chunk_size:,} rows")
            
            if chunk_memory_mb > 500:  # More than 500MB per chunk
                recommended_chunk = int(500 * 1024 * 1024 / memory_per_row)
                print(f"   ⚠️  Large chunks detected. Consider reducing to: {recommended_chunk:,} rows")
                return recommended_chunk
            
            return chunk_size

def main():
    print("=== MEMORY-EFFICIENT AI TRAINING DATA PROCESSOR ===")
    print("This processes large datasets without running out of memory")
    
    # Get file path
    default_path = "./dataset/gbif-observations-dwca.zip"
    file_path = input(f"Enter path to your data file (default: {default_path}): ").strip()
    if not file_path:
        file_path = default_path
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    # Get parameters with memory-friendly defaults
    print(f"\n📋 CONFIGURATION:")
    
    chunk_size = input("Chunk size (default 5000 for memory efficiency): ").strip()
    chunk_size = int(chunk_size) if chunk_size else 5000
    
    min_samples = input("Min samples per species (default 2): ").strip()
    min_samples = int(min_samples) if min_samples else 2
    
    max_samples = input("Max samples per species (default 50): ").strip()
    max_samples = int(max_samples) if max_samples else 50
    
    max_total = input("Max total output rows (default 50000): ").strip()
    max_total = int(max_total) if max_total else 50000
    
    # Estimate memory usage
    recommended_chunk = estimate_memory_usage(file_path, chunk_size)
    if recommended_chunk != chunk_size:
        use_recommended = input(f"Use recommended chunk size of {recommended_chunk:,}? (y/n): ").strip().lower()
        if use_recommended == 'y':
            chunk_size = recommended_chunk
    
    print(f"\n🚀 STARTING PROCESSING...")
    print(f"   File: {file_path}")
    print(f"   Chunk size: {chunk_size:,}")
    print(f"   Min samples per species: {min_samples}")
    print(f"   Max samples per species: {max_samples}")
    print(f"   Max total output: {max_total:,}")
    
    # Process the dataset
    result = process_large_dataset_in_chunks(
        file_path,
        chunk_size=chunk_size,
        min_species_samples=min_samples,
        max_species_samples=max_samples,
        max_total_rows=max_total
    )
    
    if result is not None:
        print(f"\n🎯 SUCCESS! Your AI training dataset is ready.")
        print(f"   File: ai_training_data.csv")
        print(f"   Ready for machine learning!")
    else:
        print(f"\n❌ Processing failed")

if __name__ == "__main__":
    main() 