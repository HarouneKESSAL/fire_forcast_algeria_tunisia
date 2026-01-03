#!/usr/bin/env python3
"""
Generate balanced dataset using ONLY Tomek Links + ENN (undersampling).
Standard approach: No pre-reduction, applies cleaning directly.

This creates the clean dataset for both unsupervised and supervised models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
import warnings
import time

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_PATH = Path('f:/DATA/DATA_CLEANED/processed/engineered_features_scaled.csv')
OUTPUT_PATH = Path('f:/DATA/DATA_CLEANED/processed/engineered_features_tomek_enn_balanced.csv')
LOG_PATH = Path('f:/DATA/DATA_CLEANED/processed/tomek_enn_generation_log.txt')

# ============================================================================
# MAIN
# ============================================================================

print("="*80)
print("GENERATING BALANCED DATASET: TOMEK + ENN (STANDARD)")
print("="*80)

start_time = time.time()

# Load data
print("\n[1] Loading raw data...")
df = pd.read_csv(INPUT_PATH)
print(f"    Loaded: {df.shape[0]:,} samples, {df.shape[1]} features")

# Separate features and labels
if 'fire' in df.columns:
    X = df.drop('fire', axis=1)
    y = df['fire']
    class_dist = y.value_counts().to_dict()
    print(f"    Class distribution: {class_dist}")
    print(f"    Fire rate: {y.mean():.2%}")
else:
    print("    ERROR: 'fire' column not found!")
    exit(1)

original_samples = X.shape[0]

# ============================================================================
# STEP 1: TOMEK LINKS
# ============================================================================

print(f"\n[2] Applying Tomek Links (remove borderline samples)...")
print(f"    Before: {X.shape[0]} samples")

tomek = TomekLinks(sampling_strategy='majority', n_jobs=-1)
X_tomek, y_tomek = tomek.fit_resample(X, y)

removed_tomek = X.shape[0] - X_tomek.shape[0]
print(f"    After Tomek: {X_tomek.shape[0]} samples (removed {removed_tomek})")
print(f"    Fire: {(y_tomek == 1).sum()} ({(y_tomek == 1).mean():.2%})")

# ============================================================================
# STEP 2: ENN (remove noisy samples)
# ============================================================================

print(f"\n[3] Applying ENN (Edited Nearest Neighbours - remove noisy samples)...")
print(f"    Before: {X_tomek.shape[0]} samples")

enn = EditedNearestNeighbours(n_neighbors=3, n_jobs=-1)
X_balanced, y_balanced = enn.fit_resample(X_tomek, y_tomek)

removed_enn = X_tomek.shape[0] - X_balanced.shape[0]
total_removed = original_samples - X_balanced.shape[0]

print(f"    After ENN: {X_balanced.shape[0]} samples (removed {removed_enn})")
print(f"    Fire: {(y_balanced == 1).sum()} ({(y_balanced == 1).mean():.2%})")

# ============================================================================
# STEP 3: SAVE RESULTS
# ============================================================================

print(f"\n[4] Saving balanced dataset...")

# Reconstruct dataframe
df_balanced = pd.DataFrame(X_balanced, columns=X.columns)
df_balanced['fire'] = y_balanced.values

# Save
df_balanced.to_csv(OUTPUT_PATH, index=False)
print(f"    Saved to: {OUTPUT_PATH}")
print(f"    Shape: {df_balanced.shape}")
print(f"    Fire column: {df_balanced['fire'].value_counts().to_dict()}")

# ============================================================================
# SAVE LOG
# ============================================================================

# Save log with UTF-8 encoding
with open(LOG_PATH, 'w', encoding='utf-8') as f:
    elapsed = time.time() - start_time
    final_dist = np.bincount(y_balanced)
    f.write(f"""Tomek + ENN Balancing Log
Generated: {pd.Timestamp.now()}

METHOD: Undersampling (Tomek Links + ENN) - NO SMOTE OVERSAMPLING
Standard approach: No pre-reduction, applies cleaning directly

ORIGINAL DATASET:
- Samples: {X.shape[0]:,}
- Fire: {(y == 1).sum():,} ({(y == 1).mean():.2%})
- No fire: {(y == 0).sum():,} ({(y == 0).mean():.2%})

AFTER TOMEK LINKS:
- Samples: {X_tomek.shape[0]:,}
- Removed: {removed_tomek:,}
- Fire: {(y_tomek == 1).sum():,} ({(y_tomek == 1).mean():.2%})
- No fire: {(y_tomek == 0).sum():,} ({(y_tomek == 0).mean():.2%})

AFTER ENN:
- Samples: {X_balanced.shape[0]:,}
- Removed: {removed_enn:,}
- Fire: {(y_balanced == 1).sum():,} ({(y_balanced == 1).mean():.2%})
- No fire: {(y_balanced == 0).sum():,} ({(y_balanced == 0).mean():.2%})

FINAL DATASET:
- Total removed: {total_removed:,} ({(total_removed/original_samples)*100:.2%})
- Output: {OUTPUT_PATH}

REASONING:
- Tomek Links: Removes borderline majority samples (ambiguous cases)
- ENN: Removes noisy/misclassified samples from both classes
- Result: Cleaner dataset with better cluster separation
- NO oversampling: Preserves real data distribution

USAGE:
Use this cleaned dataset for:
- Unsupervised clustering (K-Means, DBSCAN, CLARANS)
- Supervised models (Decision Tree, Random Forest, KNN)
""")

print(f"    Log saved: {LOG_PATH}")

# ============================================================================
# SUMMARY
# ============================================================================

elapsed = time.time() - start_time
final_dist = np.bincount(y_balanced)

print("\n" + "="*80)
print("SUCCESS")
print("="*80)
print(f"""
Original:  {original_samples:,} samples ({class_dist[0]:,} / {class_dist[1]:,})
Cleaned:   {X_balanced.shape[0]:,} samples ({final_dist[0]:,} / {final_dist[1]:,})
Removed:   {total_removed:,} ({(total_removed/original_samples)*100:.2%})

Output: {OUTPUT_PATH.name}
Time: {elapsed:.1f} seconds

Ready to use with all models
""")
