import pandas as pd
import os

# File paths
csv_file_path = r"C:\Users\anis\Desktop\DATA\DATA_CLEANED\processed\fire_with_negatives.csv"
parquet_file_path = r"C:\Users\anis\Desktop\DATA\DATA_CLEANED\processed\fire_with_negatives.parquet"

try:
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    print("Original data shape:", df.shape)

    # Count rows before filtering
    original_count = len(df)
    rows_with_type_2 = len(df[df['type'] == 2])
    rows_with_null_type = df['type'].isnull().sum()

    print(f"Rows with type == 2: {rows_with_type_2}")
    print(f"Rows with null type: {rows_with_null_type}")
    print(f"Rows with type != 2 (non-null): {original_count - rows_with_type_2 - rows_with_null_type}")

    # Keep rows where type == 2 OR type is null
    df_filtered = df[(df['type'] == 2) | (df['type'].isnull())]

    print("Filtered data shape:", df_filtered.shape)
    print(f"Removed {original_count - len(df_filtered)} rows")

    # Override the original CSV file
    df_filtered.to_csv(csv_file_path, index=False)
    print(f"CSV file has been overridden: {csv_file_path}")

    # Update the Parquet file
    df_filtered.to_parquet(parquet_file_path, index=False)
    print(f"Parquet file has been updated: {parquet_file_path}")

    # Show the distribution of 'type' in the filtered data
    print("\nValue counts in filtered 'type' column:")
    print(df_filtered['type'].value_counts(dropna=False))

except FileNotFoundError:
    print(f"Error: File not found at {csv_file_path}")
except KeyError:
    print("Error: 'type' column not found in the CSV file")
except Exception as e:
    print(f"Error processing file: {e}")