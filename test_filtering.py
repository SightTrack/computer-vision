import pandas as pd
import zipfile

def test_filtering():
    with zipfile.ZipFile('./dataset/gbif-observations-dwca.zip', 'r') as zip_ref:
        with zip_ref.open('observations.csv') as f:
            df = pd.read_csv(f, nrows=1000)
            
    print('=== FILTERING ANALYSIS ===')
    print(f'Sample size: {len(df)}')
    print()

    # Check taxonRank values
    print('taxonRank values:')
    print(df['taxonRank'].value_counts())
    print()

    # Check captive values  
    print('captive values:')
    print(df['captive'].value_counts())
    print()

    # Check if there's a qualityGrade column
    if 'qualityGrade' in df.columns:
        print('qualityGrade values:')
        print(df['qualityGrade'].value_counts())
    else:
        print('No qualityGrade column found')
    print()

    # Check coordinates
    coord_check = df['decimalLatitude'].notna() & df['decimalLongitude'].notna()
    print(f'Has coordinates: {coord_check.sum()}/{len(df)}')
    print()

    # Apply filters step by step
    print('=== STEP BY STEP FILTERING ===')
    remaining = df.copy()
    print(f'Start: {len(remaining)}')

    # Step 1: scientific name
    has_name = remaining['scientificName'].notna() & (remaining['scientificName'] != '')
    remaining = remaining[has_name]
    print(f'After scientific name filter: {len(remaining)}')

    # Step 2: taxon rank
    if 'taxonRank' in remaining.columns:
        species_only = remaining['taxonRank'] == 'species'
        print(f'Species rank observations: {species_only.sum()}/{len(remaining)}')
        remaining = remaining[species_only]
        print(f'After species filter: {len(remaining)}')

    # Step 3: coordinates
    coord_mask = remaining['decimalLatitude'].notna() & remaining['decimalLongitude'].notna()
    remaining = remaining[coord_mask]
    print(f'After coordinate filter: {len(remaining)}')

    # Step 4: captive
    if 'captive' in remaining.columns:
        not_captive = remaining['captive'].isna() | (remaining['captive'] == False)
        print(f'Not captive: {not_captive.sum()}/{len(remaining)}')
        remaining = remaining[not_captive]
        print(f'After captive filter: {len(remaining)}')

    print(f'Final remaining: {len(remaining)}')
    
    # Show sample of remaining data
    if len(remaining) > 0:
        print('\nSample of remaining data:')
        print(remaining[['scientificName', 'taxonRank', 'captive']].head())

if __name__ == "__main__":
    test_filtering() 