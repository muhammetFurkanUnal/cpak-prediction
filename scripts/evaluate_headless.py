"""
Headless DLC evaluate script
==============================
Kullanım:
  python scripts/evaluate_headless.py --list
  python scripts/evaluate_headless.py --snapshot-index -1   # son snapshot
  python scripts/evaluate_headless.py --snapshot-index 2    # 3. snapshot (0'dan başlar)
"""

import argparse
import pathlib
import re

REPO_ROOT   = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "kneeap-furkan-2026-04-29" / "config.yaml"


def find_snapshots(shuffle: int) -> list[pathlib.Path]:
    dlc_models = REPO_ROOT / "kneeap-furkan-2026-04-29" / "dlc-models"
    if not dlc_models.exists():
        return []
    # tüm .pt dosyalarını bul
    all_pts = sorted(dlc_models.rglob("*.pt"),
                     key=lambda p: int(m.group()) if (m := re.search(r"(\d+)", p.stem)) else 0)
    if shuffle is not None:
        all_pts = [p for p in all_pts if f"shuffle{shuffle}" in str(p)]
    return all_pts


def list_snapshots(shuffle: int):
    dlc_models = REPO_ROOT / "kneeap-furkan-2026-04-29" / "dlc-models"
    print(f"Aranan dizin: {dlc_models}")
    if not dlc_models.exists():
        print("dlc-models dizini yok — eğitim tamamlandı mı?")
        return
    snapshots = find_snapshots(shuffle)
    if not snapshots:
        # dizin var ama .pt yok — ne var göster
        all_files = list(dlc_models.rglob("*"))
        print(f"dlc-models altında {len(all_files)} dosya var, .pt yok.")
        for f in all_files[:20]:
            print(f"  {f.relative_to(dlc_models)}")
        return
    print(f"{'Index':<6} {'İterasyon':<12} {'Dosya'}")
    print("-" * 50)
    for i, s in enumerate(snapshots):
        iters = re.search(r"(\d+)", s.stem).group()
        marker = " ← last" if i == len(snapshots) - 1 else ""
        print(f"{i:<6} {iters:<12} {s.name}{marker}")


def set_snapshot_index(index: int):
    text = CONFIG_PATH.read_text()
    text = re.sub(r"snapshotindex:.*", f"snapshotindex: {index}", text)
    CONFIG_PATH.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--list", action="store_true", help="Mevcut snapshot'ları listele")
    parser.add_argument("--snapshot-index", type=int, default=None,
                        help="Hangi snapshot eval edilsin (0=ilk, -1=son, 2=3. snapshot)")
    args = parser.parse_args()

    if args.list:
        list_snapshots(args.shuffle)
        return

    if args.snapshot_index is not None:
        print(f"snapshotindex → {args.snapshot_index} olarak ayarlandı")
        set_snapshot_index(args.snapshot_index)

    import deeplabcut

    config_path = str(CONFIG_PATH)
    print(f"Config: {config_path}")
    print(f"Shuffle: {args.shuffle}\n")

    MPLBACKEND = "Agg"
    import matplotlib
    matplotlib.use(MPLBACKEND)

    deeplabcut.evaluate_network(
        config_path,
        Shuffles=[args.shuffle],
        plotting=True,
    )


if __name__ == "__main__":
    main()
