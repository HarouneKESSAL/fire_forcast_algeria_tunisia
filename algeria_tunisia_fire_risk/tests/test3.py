from pathlib import Path
import pandas as pd


def check_merged_dataset():
    base = Path('C:/Users/anis/Desktop/DATA')
    paths = [
        base / 'processed' / 'merged_dataset.parquet',
        base / 'processed' / 'merged_dataset.csv',
        base / 'DATA_CLEANED' / 'processed' / 'merged_dataset.parquet',
        base / 'DATA_CLEANED' / 'processed' / 'merged_dataset.csv',
        base / 'merged_dataset.parquet',
        base / 'merged_dataset.csv',
    ]

    for p in paths:
        if p.exists():
            print(f'\nFILE: {p}')
            try:
                if p.suffix == '.parquet':
                    df = pd.read_parquet(p)
                else:
                    df = pd.read_csv(p)

                print(f' rows: {len(df)}')
                print(f' columns: {list(df.columns)}')

                # Check for fire column
                if 'fire' in df.columns:
                    print(' fire value_counts:')
                    print(df['fire'].value_counts(dropna=False))

                    # Check data types and unique values
                    print(f" fire dtype: {df['fire'].dtype}")
                    print(f" fire unique values: {sorted(df['fire'].dropna().unique())}")

                    # Check if there are any non-fire (0) values
                    non_fire_count = (df['fire'] == 0).sum() if 0 in df['fire'].values else 0
                    fire_count = (df['fire'] == 1).sum() if 1 in df['fire'].values else 0
                    print(f" fire=1 count: {fire_count}")
                    print(f" fire=0 count: {non_fire_count}")

                    # Check for any other values
                    other_values = [v for v in df['fire'].unique() if v not in [0, 1] and pd.notna(v)]
                    if other_values:
                        print(f" other fire values: {other_values}")
                        for val in other_values:
                            count = (df['fire'] == val).sum()
                            print(f"  value {val}: {count} rows")

                else:
                    print(" no 'fire' column found")
                    # Look for alternative columns
                    fire_like_cols = [col for col in df.columns if 'fire' in col.lower() or 'label' in col.lower()]
                    if fire_like_cols:
                        print(f" potential label columns: {fire_like_cols}")
                        for col in fire_like_cols:
                            print(f"  {col} unique values: {sorted(df[col].dropna().unique())}")

                # Show first few rows to understand data structure
                print("\nFirst 3 rows:")
                print(df.head(3))

            except Exception as e:
                print(f" error reading file: {e}")

    # If no files found, list what's available
    if not any(p.exists() for p in paths):
        print("No merged dataset files found. Available files:")
        for pattern in ['*.parquet', '*.csv']:
            for file in base.rglob(pattern):
                if 'merged' in file.name.lower() or 'fire' in file.name.lower():
                    print(f"  {file}")


if __name__ == "__main__":
    check_merged_dataset()