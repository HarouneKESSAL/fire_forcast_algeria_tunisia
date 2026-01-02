import pandas as pd
import os

# Paths
csv_path = "/run/media/swift/MISO_EFI/DATA/SOIL/processed/HWSD2_LAYERS_D1.csv"
output_dir = "/run/media/swift/MISO_EFI/DATA/SOIL/processed"
output_path = os.path.join(output_dir, "layer_1_final.csv")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

try:
    # Load the CSV file
    df_d1 = pd.read_csv(csv_path)
    print(f'✓ CSV file loaded: {df_d1.shape}')
    
    # Define columns to keep
    id_columns = ["ID", "HWSD2_SMU_ID", "WISE30s_SMU_ID", "HWSD1_SMU_ID", 
                  "COVERAGE", "SEQUENCE", "SHARE"]
    
    soil_characteristics = ["COARSE", "SAND", "SILT", "CLAY", "TEXTURE_USDA", "TEXTURE_SOTER", 
                            "BULK", "REF_BULK", "ORG_CARBON", "PH_WATER", "TOTAL_N", "CN_RATIO", 
                            "CEC_SOIL", "CEC_CLAY", "CEC_EFF", "TEB", "BSAT", "ALUM_SAT", 
                            "ESP", "TCARBON_EQ", "GYPSUM", "ELEC_COND"]
    
    # Filter to available columns only
    available_columns = [col for col in (id_columns + soil_characteristics) 
                         if col in df_d1.columns]
    
    # Create final dataframe
    df_final = df_d1[available_columns]
    
    # Save to CSV
    df_final.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\n✅ SUCCESS: Data saved to '{output_path}'")
    print(f"📊 Shape: {df_final.shape[0]} rows × {df_final.shape[1]} columns")
    print(f"📁 File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # Show column information
    print(f"\n📋 Columns saved ({len(available_columns)} total):")
    for i, col in enumerate(available_columns, 1):
        col_type = "ID" if col in id_columns else "Soil Characteristic"
        print(f"  {i:2d}. {col:20s} [{col_type}]")
    
    # Show sample
    print(f"\n👀 First 3 rows:")
    print(df_final.head(3).to_string())
    
except FileNotFoundError:
    print(f"❌ ERROR: Input file not found: {csv_path}")
    print(f"   Please check if the file exists at this location.")
except PermissionError:
    print(f"❌ ERROR: Permission denied when writing to: {output_path}")
    print(f"   Check your write permissions for this directory.")
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {e}")