import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
import joblib

DATA_PATH = Path('f:/DATA/DATA_CLEANED/processed/engineered_features_scaled.csv')
RESULTS_JSON = Path('f:/DATA/results/rf_tomek_enn_threshold_tuning.json')
MODEL_DIR = Path('f:/DATA/models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / 'random_forest_tomek_enn.pkl'
META_PATH = MODEL_DIR / 'random_forest_tomek_enn_meta.json'

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_threshold(default_thr: float = 0.77) -> float:
    if RESULTS_JSON.exists():
        data = json.loads(RESULTS_JSON.read_text())
        agg = data.get('aggregated', {})
        thr = float(agg.get('threshold_mean', default_thr))
        return thr
    return default_thr


def tomek_enn_clean(X, y):
    tl = TomekLinks()
    X_tl, y_tl = tl.fit_resample(X, y)
    enn = EditedNearestNeighbours(n_neighbors=3)
    X_clean, y_clean = enn.fit_resample(X_tl, y_tl)
    return X_clean, y_clean


def train_and_export():
    df = pd.read_csv(DATA_PATH)
    if 'fire' not in df.columns:
        raise ValueError('Target column "fire" not found in dataset')

    y = df['fire'].astype(int).values
    X = df.drop(columns=['fire']).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    X_train_clean, y_train_clean = tomek_enn_clean(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf.fit(X_train_clean, y_train_clean)

    # Evaluate on holdout with tuned threshold
    thr = load_threshold()
    y_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= thr).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    joblib.dump(rf, MODEL_PATH)
    meta = {
        'threshold': thr,
        'random_state': RANDOM_STATE,
        'test_size': TEST_SIZE,
        'features': [c for c in df.columns if c != 'fire'],
        'classification_report': report,
        'confusion_matrix': cm
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print('Model saved to:', MODEL_PATH)
    print('Meta saved to:', META_PATH)
    print('Confusion matrix:', cm)
    print('F1:', report['weighted avg']['f1-score'])


if __name__ == '__main__':
    train_and_export()
