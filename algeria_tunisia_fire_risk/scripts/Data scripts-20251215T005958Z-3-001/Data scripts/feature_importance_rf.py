import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

MODEL_DIR = Path('f:/DATA/models')
MODEL_PATH = MODEL_DIR / 'random_forest_tomek_enn.pkl'
META_PATH = MODEL_DIR / 'random_forest_tomek_enn_meta.json'
RESULTS_PATH = Path('f:/DATA/results/rf_feature_importance.csv')

if __name__ == '__main__':
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError('Model or meta not found. Run train_export_rf.py first.')
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    feature_cols = meta.get('features')

    importances = model.feature_importances_
    df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
    df.sort_values('importance', ascending=False, inplace=True)
    df.to_csv(RESULTS_PATH, index=False)
    print('Feature importances saved to:', RESULTS_PATH)
