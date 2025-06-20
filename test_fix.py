#!/usr/bin/env python3

import pandas as pd
import zipfile
from process_observations import analyze_data_availability_simple, detect_file_type_and_load

def test_fixed_filtering():
    print("🧪 TESTING FIXED FILTERING...")
    
    # Load sample data
    df = detect_file_type_and_load('./dataset/gbif-observations-dwca.zip', sample_size=1000)
    
    if df is None:
        print("❌ Failed to load data")
        return
    
    print(f"✅ Loaded {len(df)} observations")
    
    # Test the fixed analysis
    result = analyze_data_availability_simple(df)
    
    if result is not None and len(result) > 0:
        print(f"\n🎉 SUCCESS! Fixed filtering returned {len(result)} observations")
        print(f"Species found: {result['scientificName'].nunique()}")
        print(f"Top species: {list(result['scientificName'].value_counts().head(3).index)}")
    else:
        print("❌ Still no data after filtering")

if __name__ == "__main__":
    test_fixed_filtering() 