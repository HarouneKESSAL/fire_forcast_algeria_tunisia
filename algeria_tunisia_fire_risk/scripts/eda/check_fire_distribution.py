import pandas as pd, pathlib, json
PATH = pathlib.Path(r'C:\Users\anis\Desktop\DATA\DATA_CLEANED\processed\merged_dataset.csv')
print('Loading:', PATH)
if not PATH.exists():
    raise SystemExit('File not found: ' + str(PATH))
# Load
df = pd.read_csv(PATH)
print('Rows:', len(df), 'Cols:', len(df.columns))
if 'fire' in df.columns:
    vc = df['fire'].value_counts(dropna=False)
    print('Fire value_counts:\n', vc)
    print('Fire dtype:', df['fire'].dtype)
    print('Unique fire values:', sorted(df['fire'].dropna().unique()))
    invalid = (~df['fire'].isin([0,1])) & df['fire'].notna()
    print('Invalid fire label rows:', int(invalid.sum()))
else:
    print("No 'fire' column present!")
# Basic coordinate sanity
issues = {}
if 'latitude' in df.columns:
    lat_num = pd.to_numeric(df['latitude'], errors='coerce')
    issues['bad_lat'] = int((~lat_num.between(-90,90)).sum())
if 'longitude' in df.columns:
    lon_num = pd.to_numeric(df['longitude'], errors='coerce')
    issues['bad_lon'] = int((~lon_num.between(-180,180)).sum())
print('Integrity issues:', json.dumps(issues))
print('Head:\n', df.head(3).to_string())
