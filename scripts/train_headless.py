"""
Headless DLC setup + training script
=====================================
SSH ortamında, sıfırdan proje klasörünü oluşturur ve eğitimi başlatır.

Ön koşullar:
  - pip install deeplabcut[pytorch] pandas tables
  - Resimler preprocessing/knee-preop-prepped/ altında olmalı

Kullanım:
  python scripts/train_headless.py            # setup + eğitim
  python scripts/train_headless.py --setup-only  # sadece klasör kur
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
    print(f"[1/4] Proje klasörü oluşturuluyor: {PROJECT_DIR}")
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "videos").mkdir(exist_ok=True)
    (PROJECT_DIR / "dlc-models").mkdir(exist_ok=True)
    (PROJECT_DIR / "evaluation-results").mkdir(exist_ok=True)
    (PROJECT_DIR / "training-datasets").mkdir(exist_ok=True)

    print(f"[2/4] config.yaml kopyalanıyor ve project_path güncelleniyor")
    config_text = CONFIG_SRC.read_text()
    # project_path satırını mevcut makineye göre güncelle
    import re
    config_text = re.sub(
        r"project_path:.*",
        f"project_path: {PROJECT_DIR}",
        config_text,
    )
    CONFIG_DST.write_text(config_text)
    print(f"   project_path → {PROJECT_DIR}")

    print(f"[3/4] Resimler {IMAGES_SRC} → {LABELED_DIR}")
    if not IMAGES_SRC.exists():
        print(f"   HATA: {IMAGES_SRC} bulunamadı!", file=sys.stderr)
        sys.exit(1)

    # CSV'deki frame isimlerini oku; sadece etiketli resimleri kopyala
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

    print(f"   {copied} resim kopyalandı, {len(missing)} eksik")
    if missing:
        print("   Eksik resimler:", missing[:5], "..." if len(missing) > 5 else "")

    print(f"[4/4] CSV ve H5 oluşturuluyor → {LABELED_DIR}")
    csv_dst = LABELED_DIR / "CollectedData_furkan.csv"
    shutil.copy2(CSV_SRC, csv_dst)

    h5_dst = LABELED_DIR / "CollectedData_furkan.h5"
    df.to_hdf(h5_dst, key="df_with_missing", mode="w")
    print(f"   H5 yazıldı: {h5_dst}")

    print(f"\nKurulum tamam. Config: {CONFIG_DST}")
    return str(CONFIG_DST)


# ── Training ──────────────────────────────────────────────────────────────────

def train(config_path: str, shuffle: int = 1, max_snapshots: int = 5):
    import deeplabcut

    print("\n── create_training_dataset ──────────────────────────────────────────")
    deeplabcut.create_training_dataset(config_path, num_shuffles=1, Shuffles=[shuffle])

    print("\n── train_network ────────────────────────────────────────────────────")
    deeplabcut.train_network(
        config_path,
        shuffle=shuffle,
        trainingsetindex=0,
        max_snapshots_to_keep=max_snapshots,
        displayiters=500,
        saveiters=2000,
        maxiters=200_000,
        allow_growth=True,
    )

    print("\n── evaluate_network ─────────────────────────────────────────────────")
    deeplabcut.evaluate_network(config_path, shuffle=[shuffle], plotting=False)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup-only", action="store_true",
                        help="Sadece proje klasörünü kur, eğitimi başlatma")
    parser.add_argument("--shuffle", type=int, default=1)
    args = parser.parse_args()

    config_path = setup_project()

    if args.setup_only:
        print("\nKurulum tamamlandı. Eğitim için '--setup-only' olmadan tekrar çalıştır.")
        return

    train(config_path, shuffle=args.shuffle)


if __name__ == "__main__":
    main()
