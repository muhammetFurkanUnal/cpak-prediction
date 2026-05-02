"""
Headless DLC evaluate script
==============================
Kullanım:
  python scripts/evaluate_headless.py --list
  python scripts/evaluate_headless.py                        # best snapshot
  python scripts/evaluate_headless.py --snapshot-index 4    # 5. snapshot (0'dan başlar)
"""

import argparse
import contextlib
import pathlib
import re

REPO_ROOT   = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "kneeap-furkan-2026-04-29" / "config.yaml"
TRAIN_DIR   = (REPO_ROOT / "kneeap-furkan-2026-04-29" / "dlc-models-pytorch"
               / "iteration-0" / "kneeapApr29-trainset95shuffle1" / "train")


def find_snapshots() -> list[pathlib.Path]:
    if not TRAIN_DIR.exists():
        return []
    pts = [p for p in TRAIN_DIR.glob("*.pt") if re.search(r"\d+", p.stem)]
    return sorted(pts, key=lambda p: p.stem)  # DLC ile aynı: alfabetik


def list_snapshots():
    snapshots = find_snapshots()
    if not snapshots:
        print("Snapshot bulunamadı.")
        return
    print(f"{'Index':<6} {'Dosya'}")
    print("-" * 40)
    for i, s in enumerate(snapshots):
        best = " ★ best (DLC default)" if "best" in s.stem else ""
        print(f"{i:<6} {s.name}{best}")


@contextlib.contextmanager
def hide_best_snapshot():
    """DLC'nin snapshot-best-* dosyasını otomatik seçmesini geçici olarak engeller."""
    best_files = list(TRAIN_DIR.glob("snapshot-best-*.pt"))
    hidden = []
    for f in best_files:
        tmp = f.with_suffix(".pt.hidden")
        f.rename(tmp)
        hidden.append((tmp, f))
        print(f"  [geçici gizlendi] {f.name}")
    try:
        yield
    finally:
        for tmp, original in hidden:
            tmp.rename(original)
            print(f"  [geri yüklendi] {original.name}")


def set_snapshot_index(index: int):
    text = CONFIG_PATH.read_text()
    text = re.sub(r"snapshotindex:.*", f"snapshotindex: {index}", text)
    CONFIG_PATH.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--snapshot-index", type=int, default=None,
                        help="0=ilk, -1=son, 4=5. snapshot (best hariç, alfabetik sıra)")
    args = parser.parse_args()

    if args.list:
        list_snapshots()
        return

    import matplotlib
    matplotlib.use("Agg")
    import deeplabcut

    config_path = str(CONFIG_PATH)

    if args.snapshot_index is not None:
        snapshots = find_snapshots()
        non_best = [s for s in snapshots if "best" not in s.stem]
        try:
            chosen = non_best[args.snapshot_index]
        except IndexError:
            print(f"Geçersiz index. 0–{len(non_best)-1} arasında bir değer gir.")
            return
        print(f"Seçilen snapshot: {chosen.name}")
        # DLC'nin alfabetik listesinde bu snapshot'ın index'ini bul
        dlc_index = snapshots.index(chosen)
        set_snapshot_index(dlc_index)
        with hide_best_snapshot():
            deeplabcut.evaluate_network(config_path, Shuffles=[args.shuffle], plotting=True)
    else:
        # varsayılan: DLC kendi seçsin (snapshot-best-*)
        set_snapshot_index(-1)
        deeplabcut.evaluate_network(config_path, Shuffles=[args.shuffle], plotting=True)


if __name__ == "__main__":
    main()
