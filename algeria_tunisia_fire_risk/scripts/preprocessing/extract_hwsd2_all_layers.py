#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📤 Extracts all soil layers (D1–D7) from HWSD2.mdb into separate CSVs.
Each CSV can then be processed by your raster generation script.
"""

import subprocess, io
import pandas as pd
from pathlib import Path

BASE = Path("/home/swift/Desktop/DATA")
MDB = BASE / "SOIL" / "HWSD2" / "data" / "HWSD2.mdb"
OUTDIR = BASE / "SOIL" / "processed"
OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"📘 Reading HWSD2_LAYERS table from {MDB.name}...")

    # Export the full HWSD2_LAYERS table
    p = subprocess.run(["mdb-export", MDB.as_posix(), "HWSD2_LAYERS"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"⚠️ MDB export failed: {p.stderr}")

    df = pd.read_csv(io.StringIO(p.stdout), low_memory=False)
    print(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns.")
    print("🔍 Unique layers detected:", df["LAYER"].unique().tolist())

    # Export each LAYER (D1–D7)
    for layer in sorted(df["LAYER"].unique(), key=lambda x: int(x[1:])):  # sort by numeric part
        subset = df[df["LAYER"] == layer].copy()
        out_path = OUTDIR / f"HWSD2_LAYERS_{layer}.csv"
        subset.to_csv(out_path, index=False)
        print(f"🖨️ Exported {layer}: {len(subset):,} rows → {out_path.name}")

    print("\n✅ All layers exported successfully!")

if __name__ == "__main__":
    main()
