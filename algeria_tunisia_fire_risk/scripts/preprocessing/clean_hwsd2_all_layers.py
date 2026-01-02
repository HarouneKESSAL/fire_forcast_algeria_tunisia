#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧹 HWSD2 — Clean All Soil Layer CSVs (D1–D7)
--------------------------------------------
Cleans each HWSD2_LAYERS_D*.csv:
 - Keeps relevant numeric columns for soil modeling
 - Converts columns to numeric types
 - Removes duplicates & rows without SMU_ID
 - Saves cleaned versions to /processed/cleaned/
"""

import os
import pandas as pd
from pathlib import Path

# === CONFIG ===
BASE = Path("/home/swift/Desktop/DATA/SOIL/processed")
OUT_DIR = BASE / "cleaned"
OUT_DIR.mkdir(exist_ok=True)

# Soil attributes to keep (important for rasterization)
COLUMNS_TO_KEEP = [
    "HWSD2_SMU_ID", "LAYER", "COARSE", "SAND", "SILT", "CLAY",
    "TEXTURE_USDA", "TEXTURE_SOTER", "BULK", "REF_BULK", "ORG_CARBON",
    "PH_WATER", "TOTAL_N", "CN_RATIO", "CEC_SOIL", "CEC_CLAY",
    "CEC_EFF", "TEB", "BSAT", "ALUM_SAT", "ESP", "TCARBON_EQ",
    "GYPSUM", "ELEC_COND"
]

def clean_layer(csv_path):
    print(f"🧾 Cleaning {csv_path.name} ...")
    df = pd.read_csv(csv_path, low_memory=False)

    # --- Keep only relevant columns ---
    cols = [c for c in COLUMNS_TO_KEEP if c in df.columns]
    df = df[cols].copy()

    # --- Drop invalid rows ---
    df = df.dropna(subset=["HWSD2_SMU_ID"])
    df = df.drop_duplicates(subset=["HWSD2_SMU_ID", "LAYER"], keep="first")

    # --- Convert numeric columns ---
    num_cols = [c for c in df.columns if c not in ["HWSD2_SMU_ID", "LAYER", "TEXTURE_USDA", "TEXTURE_SOTER"]]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # --- Fill missing values ---
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean(numeric_only=True))

    out_path = OUT_DIR / csv_path.name.replace(".csv", "_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Saved cleaned: {out_path.name} ({len(df)} rows)")

def main():
    csvs = sorted(BASE.glob("HWSD2_LAYERS_D*.csv"))
    if not csvs:
        print("❌ No HWSD2 CSVs found in", BASE)
        return

    for csv in csvs:
        clean_layer(csv)

    print("\n🎉 All layers cleaned successfully! Files saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
