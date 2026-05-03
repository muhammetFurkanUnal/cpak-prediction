"""
Headless DLC setup + training script
=====================================
Builds the project folder from scratch and starts training. Intended for SSH/headless use.

Prerequisites:
  - pip install deeplabcut[pytorch] pandas tables
  - Images must be in preprocessing/knee-preop-prepped/

Usage:
  python scripts/train_headless.py            # setup + train
  python scripts/train_headless.py --setup-only  # only create project folder
"""

import argparse
import shutil
import pathlib
import sys
import os

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT   = pathlib.Path(__file__).resolve().parent.parent
PROJECT_DIR = REPO_ROOT / "kneeap-furkan-2026-04-29"
LABELED_DIR = PROJECT_DIR / "labeled-data" / "video"
IMAGES_SRC  = REPO_ROOT / "preprocessing" / "knee-preop-prepped"
ASSETS      = REPO_ROOT / "assets" / "kneeap"
CSV_SRC     = ASSETS / "CollectedData_furkan.csv"
CONFIG_SRC  = ASSETS / "config.yaml"
CONFIG_DST  = PROJECT_DIR / "config.yaml"

# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_project():
    print(f"[1/4] Creating project folder: {PROJECT_DIR}")
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "videos").mkdir(exist_ok=True)
    (PROJECT_DIR / "dlc-models").mkdir(exist_ok=True)
    (PROJECT_DIR / "evaluation-results").mkdir(exist_ok=True)
    (PROJECT_DIR / "training-datasets").mkdir(exist_ok=True)

    print(f"[2/4] Copying config.yaml and updating project_path")
    config_text = CONFIG_SRC.read_text()
    import re
    config_text = re.sub(
        r"project_path:.*",
        f"project_path: {PROJECT_DIR}",
        config_text,
    )
    CONFIG_DST.write_text(config_text)
    print(f"   project_path → {PROJECT_DIR}")

    print(f"[3/4] Copying images {IMAGES_SRC} → {LABELED_DIR}")
    if not IMAGES_SRC.exists():
        print(f"   ERROR: {IMAGES_SRC} not found!", file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(CSV_SRC, header=[0, 1, 2], index_col=[0, 1, 2])
    labeled_frames = [idx[2] for idx in df.index]

    copied = 0
    missing = []
    for fname in labeled_frames:
        src = IMAGES_SRC / fname
        dst = LABELED_DIR / fname
        if dst.exists():
            copied += 1
            continue
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing.append(fname)

    print(f"   {copied} images copied, {len(missing)} missing")
    if missing:
        print("   Missing:", missing[:5], "..." if len(missing) > 5 else "")

    print(f"[4/4] Writing CSV and H5 → {LABELED_DIR}")
    csv_dst = LABELED_DIR / "CollectedData_furkan.csv"
    shutil.copy2(CSV_SRC, csv_dst)

    h5_dst = LABELED_DIR / "CollectedData_furkan.h5"
    df.to_hdf(h5_dst, key="df_with_missing", mode="w")
    print(f"   H5 written: {h5_dst}")

    print(f"\nSetup complete. Config: {CONFIG_DST}")
    return str(CONFIG_DST)


# ── Training ──────────────────────────────────────────────────────────────────

def shuffle_exists(shuffle: int) -> bool:
    td = PROJECT_DIR / "training-datasets"
    return any(td.rglob(f"*shuffle{shuffle}*")) if td.exists() else False


def train(config_path: str, shuffle: int = 1, epochs: int = 1000, max_snapshots: int = 5):
    if not shuffle_exists(shuffle):
        print(f"\nERROR: shuffle {shuffle} does not exist.")
        print("Create it first:  python scripts/shuffles_headless.py create")
        sys.exit(1)

    import deeplabcut
    print(f"\n── train_network ({epochs} epochs) ──────────────────────────────────")
    # NOTE: PyTorch backend ignores maxiters; epochs= is the authoritative kwarg
    # and it overwrites pytorch_config.yaml on disk via update_model_cfg.
    deeplabcut.train_network(
        config_path,
        shuffle=shuffle,
        trainingsetindex=0,
        epochs=epochs,
        save_epochs=max(1, epochs // 20),
        max_snapshots_to_keep=max_snapshots,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-only", action="store_true",
                        help="Only create project folder, do not train")
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of epochs to train (default: 1000)")
    args = parser.parse_args()

    config_path = setup_project()

    if args.setup_only:
        print("\nSetup complete. Run without --setup-only to start training.")
        return

    train(config_path, shuffle=args.shuffle, epochs=args.epochs)


if __name__ == "__main__":
    main()
