"""
DeepLabCut label conversion script
====================================
Converts CollectedData_*.csv files under labeled-data/ into the
CollectedData_*.h5 format expected by DeepLabCut.

Usage:
    python scripts/csv_to_h5.py

Why is this needed?
-------------------
DeepLabCut reads label data from .h5 (HDF5/pandas) files.
The .csv file contains the same data in a portable, human-readable form.
Only the .csv is committed to Git; the .h5 is generated from it by this script.
"""

import pathlib
import sys
import pandas as pd


def convert(csv_path: pathlib.Path) -> None:
    h5_path = csv_path.with_suffix(".h5")

    # DLC format: first 3 rows are a MultiIndex header (scorer/bodyparts/coords)
    #             first 3 columns are a MultiIndex index (dir/video/filename)
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=[0, 1, 2])

    df.to_hdf(h5_path, key="df_with_missing", mode="w")
    print(f"[OK] {h5_path}  ({len(df)} frame)")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/csv_to_h5.py /absolute/path/to/CollectedData_furkan.csv")
        sys.exit(1)

    csv_path = pathlib.Path(sys.argv[1]).resolve()

    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    convert(csv_path)
    print("\nDone. You can now run 'deeplabcut.create_training_dataset(...)'.")


if __name__ == "__main__":
    main()
