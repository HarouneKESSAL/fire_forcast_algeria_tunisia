import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = Path('f:/DATA/models')
MODEL_PATH = MODEL_DIR / 'random_forest_tomek_enn.pkl'
META_PATH = MODEL_DIR / 'random_forest_tomek_enn_meta.json'


def load_model_and_meta():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError('Model or meta not found. Run train_export_rf.py first.')
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


def predict_from_csv(input_csv: str, output_csv: str):
    model, meta = load_model_and_meta()
    thr = float(meta.get('threshold', 0.77))
    feature_cols = meta.get('features')

    df = pd.read_csv(input_csv)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required feature columns: {missing}')

    X = df[feature_cols].values
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= thr).astype(int)

    out = df.copy()
    out['fire_proba'] = proba
    out['fire_pred'] = pred
    pd.DataFrame(out).to_csv(output_csv, index=False)
    print('Predictions written to:', output_csv)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict fire using trained RF model')
    parser.add_argument('--input', help='Input CSV with feature columns', default='f:/DATA/DATA_CLEANED/processed/engineered_features_scaled.csv')
    parser.add_argument('--output', help='Output CSV path', default='f:/DATA/results/predictions.csv')
    args = parser.parse_args()

    # Ensure output directory exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    predict_from_csv(args.input, args.output)
