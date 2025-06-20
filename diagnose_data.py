import pandas as pd
import zipfile
import os

def diagnose_data_file(file_path):
    """Simple diagnostic to understand data structure"""
    
    print("🔍 DIAGNOSING DATA FILE...")
    print(f"File: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    file_ext = os.path.splitext(file_path)[1].lower()
    print(f"File type: {file_ext}")
    
    try:
        # Try to load a small sample
        if file_ext == '.zip':
            print("Loading from ZIP archive...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                files = zip_ref.namelist()
                print(f"Files in ZIP: {files}")
                
                # Find observations file
                obs_files = [f for f in files if f.lower() in ['observations.csv', 'occurrence.txt', 'occurrence.csv']]
                if obs_files:
                    print(f"Found observations file: {obs_files[0]}")
                    
                    with zip_ref.open(obs_files[0]) as f:
                        # Try different delimiters
                        for delimiter in [',', '\t']:
                            try:
                                f.seek(0)
                                df = pd.read_csv(f, sep=delimiter, nrows=100, low_memory=False)
                                if len(df.columns) > 3:
                                    print(f"✅ Successfully read with '{delimiter}' delimiter")
                                    print(f"Shape: {df.shape}")
                                    break
                            except Exception as e:
                                print(f"Failed with '{delimiter}': {e}")
                                continue
                else:
                    print("❌ No observations file found in ZIP")
                    return
                    
        else:
            print("Loading CSV file directly...")
            # Try different delimiters
            for delimiter in [',', '\t']:
                try:
                    df = pd.read_csv(file_path, sep=delimiter, nrows=100, low_memory=False)
                    if len(df.columns) > 3:
                        print(f"✅ Successfully read with '{delimiter}' delimiter")
                        print(f"Shape: {df.shape}")
                        break
                except Exception as e:
                    print(f"Failed with '{delimiter}': {e}")
                    continue
        
        # Analyze the loaded data
        if 'df' in locals():
            print(f"\n📋 COLUMNS FOUND ({len(df.columns)}):")
            for i, col in enumerate(df.columns):
                non_null = df[col].notna().sum()
                print(f"  {i+1:2d}. {col} ({non_null}/{len(df)} non-null)")
            
            # Check for key columns
            key_columns = ['scientificName', 'decimalLatitude', 'decimalLongitude', 'id']
            print(f"\n🔍 KEY COLUMN CHECK:")
            for col in key_columns:
                if col in df.columns:
                    print(f"  ✅ {col}: FOUND")
                else:
                    print(f"  ❌ {col}: MISSING")
                    # Look for similar column names
                    similar = [c for c in df.columns if col.lower() in c.lower() or c.lower() in col.lower()]
                    if similar:
                        print(f"      Similar: {similar}")
            
            # Show sample data
            print(f"\n📊 SAMPLE DATA (first 3 rows):")
            try:
                # Show first few columns only
                display_cols = list(df.columns)[:6]
                print(df[display_cols].head(3).to_string())
            except:
                print("Could not display sample data")
            
            # Check for species data specifically
            species_cols = [col for col in df.columns if 'scientific' in col.lower() or 'species' in col.lower() or 'name' in col.lower()]
            if species_cols:
                print(f"\n🧬 SPECIES-RELATED COLUMNS:")
                for col in species_cols:
                    unique_count = df[col].nunique()
                    non_null = df[col].notna().sum()
                    print(f"  {col}: {non_null} non-null, {unique_count} unique values")
                    
                    # Show sample values
                    if non_null > 0:
                        sample_values = df[col].dropna().head(3).tolist()
                        print(f"    Sample: {sample_values}")
        else:
            print("❌ Could not load any data")
            
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Get file path
    default_path = "./dataset/gbif-observations-dwca.zip"
    file_path = input(f"Enter path to your data file (default: {default_path}): ").strip()
    if not file_path:
        file_path = default_path
    
    diagnose_data_file(file_path) 