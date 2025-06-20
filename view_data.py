import pandas as pd
import zipfile

def inspect_inaturalist_archive(archive_path):
    """Inspect the structure of your iNaturalist GBIF archive"""
    
    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
        print("=== ARCHIVE CONTENTS ===")
        for file in zip_ref.namelist():
            info = zip_ref.getinfo(file)
            size_mb = info.file_size / (1024 * 1024)
            print(f"{file}: {size_mb:.1f} MB")
        
        print("\n=== OBSERVATIONS.CSV STRUCTURE ===")
        
        # Load first few rows to understand structure
        with zip_ref.open('observations.csv') as f:
            df_sample = pd.read_csv(f, nrows=1000)
            
            print(f"Shape: {df_sample.shape}")
            print(f"Columns ({len(df_sample.columns)}):")
            for i, col in enumerate(df_sample.columns):
                print(f"  {i+1:2d}. {col}")
            
            print(f"\n=== SAMPLE DATA ===")
            print(df_sample.head(3))
            
            # Check key columns for species classification
            key_columns = ['scientificName', 'kingdom', 'phylum', 'class', 'order', 
                          'family', 'genus', 'species', 'taxonRank', 'quality_grade',
                          'qualityGrade', 'latitude', 'longitude', 'id']
            
            print(f"\n=== KEY COLUMNS CHECK ===")
            for col in key_columns:
                if col in df_sample.columns:
                    non_null = df_sample[col].notna().sum()
                    unique_vals = df_sample[col].nunique()
                    print(f"✓ {col}: {non_null}/{len(df_sample)} non-null, {unique_vals} unique")
                    if col in ['taxonRank', 'quality_grade', 'qualityGrade']:
                        print(f"  Values: {df_sample[col].value_counts().head(3).to_dict()}")
                else:
                    print(f"✗ {col}: NOT FOUND")
            
            # Species distribution
            if 'scientificName' in df_sample.columns:
                species_counts = df_sample['scientificName'].value_counts()
                print(f"\n=== SPECIES DISTRIBUTION (SAMPLE) ===")
                print(f"Total species in sample: {len(species_counts)}")
                print("Top 10 species:")
                print(species_counts.head(10))
                
                # Check for empty/null species names
                null_species = df_sample['scientificName'].isna().sum()
                empty_species = (df_sample['scientificName'] == '').sum()
                print(f"Null species names: {null_species}")
                print(f"Empty species names: {empty_species}")
        
        # Check media.csv for image information
        print(f"\n=== MEDIA.CSV STRUCTURE ===")
        if 'media.csv' in zip_ref.namelist():
            with zip_ref.open('media.csv') as f:
                media_df = pd.read_csv(f, nrows=100)
                print(f"Media shape: {media_df.shape}")
                print(f"Media columns: {list(media_df.columns)}")
                if 'type' in media_df.columns:
                    print(f"Media types: {media_df['type'].value_counts().to_dict()}")
                print("Sample media entries:")
                print(media_df.head(3))
        
        return df_sample

def create_processing_plan(df_sample):
    """Create a processing plan based on the actual data structure"""
    
    print(f"\n=== PROCESSING RECOMMENDATIONS ===")
    
    # Determine the correct column names
    species_col = None
    if 'scientificName' in df_sample.columns:
        species_col = 'scientificName'
    elif 'scientific_name' in df_sample.columns:
        species_col = 'scientific_name'
    
    rank_col = None
    if 'taxonRank' in df_sample.columns:
        rank_col = 'taxonRank'
    elif 'rank' in df_sample.columns:
        rank_col = 'rank'
    
    quality_col = None
    if 'quality_grade' in df_sample.columns:
        quality_col = 'quality_grade'
    elif 'qualityGrade' in df_sample.columns:
        quality_col = 'qualityGrade'
    
    lat_col = 'latitude' if 'latitude' in df_sample.columns else 'decimalLatitude'
    lon_col = 'longitude' if 'longitude' in df_sample.columns else 'decimalLongitude'
    
    id_col = None
    if 'id' in df_sample.columns:
        id_col = 'id'
    elif 'observationID' in df_sample.columns:
        id_col = 'observationID'
    
    print(f"Species column: {species_col}")
    print(f"Rank column: {rank_col}")
    print(f"Quality column: {quality_col}")
    print(f"Coordinates: {lat_col}, {lon_col}")
    print(f"ID column: {id_col}")
    
    # Estimate filtering results
    if species_col and rank_col and quality_col:
        species_mask = df_sample[species_col].notna() & (df_sample[species_col] != '')
        rank_mask = df_sample[rank_col] == 'species'
        quality_mask = df_sample[quality_col] == 'research'
        
        print(f"\nFiltering estimates (from sample of {len(df_sample)}):")
        print(f"  Has species name: {species_mask.sum()}")
        print(f"  Species rank: {rank_mask.sum()}")
        print(f"  Research grade: {quality_mask.sum()}")
        print(f"  All filters: {(species_mask & rank_mask & quality_mask).sum()}")
    
    return {
        'species_col': species_col,
        'rank_col': rank_col,
        'quality_col': quality_col,
        'lat_col': lat_col,
        'lon_col': lon_col,
        'id_col': id_col
    }

if __name__ == "__main__":
    archive_path = "./dataset/gbif-observations-dwca.zip"
    
    # Inspect the archive
    df_sample = inspect_inaturalist_archive(archive_path)
    
    # Create processing plan
    column_mapping = create_processing_plan(df_sample)
    
    print(f"\n=== NEXT STEPS ===")
    print("1. Update the processing script with the correct column names")
    print("2. Run the full processing pipeline")
    print("3. Download images using the media.csv file")
    print("4. Train your species classification model")