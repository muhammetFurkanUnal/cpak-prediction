"""
Headless DLC evaluate script
==============================
Usage:
  python scripts/evaluate_headless.py --list
  python scripts/evaluate_headless.py                        # best snapshot
  python scripts/evaluate_headless.py --snapshot-index 4    # 5th snapshot (0-based)
"""

import argparse
import contextlib
import pathlib
import re

REPO_ROOT   = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "kneeap-furkan-2026-04-29" / "config.yaml"
DLC_MODELS  = REPO_ROOT / "kneeap-furkan-2026-04-29" / "dlc-models-pytorch"


def get_train_dir(shuffle: int) -> pathlib.Path | None:
    matches = list(DLC_MODELS.glob(f"**/iteration-*/*shuffle{shuffle}/train"))
    return matches[0] if matches else None


def find_snapshots(shuffle: int) -> list[pathlib.Path]:
    train_dir = get_train_dir(shuffle)
    if not train_dir:
        return []
    pts = [p for p in train_dir.glob("*.pt") if re.search(r"\d+", p.stem)]
    # DLC sorts numerically by iteration; "best" is special-cased to last.
    def key(p: pathlib.Path):
        n = int(re.search(r"(\d+)", p.stem).group())
        is_best = "best" in p.stem
        return (is_best, n)
    return sorted(pts, key=key)


def list_snapshots(shuffle: int):
    snapshots = find_snapshots(shuffle)
    if not snapshots:
        print(f"No snapshots found for shuffle {shuffle}.")
        return
    print(f"{'Index':<6} {'File'}")
    print("-" * 40)
    for i, s in enumerate(snapshots):
        best = " ★ best (DLC default)" if "best" in s.stem else ""
        print(f"{i:<6} {s.name}{best}")


@contextlib.contextmanager
def hide_best_snapshot(shuffle: int):
    """Temporarily hides snapshot-best-* so DLC is forced to use a numbered snapshot."""
    train_dir = get_train_dir(shuffle)
    best_files = list(train_dir.glob("snapshot-best-*.pt")) if train_dir else []
    hidden = []
    for f in best_files:
        tmp = f.with_suffix(".pt.hidden")
        f.rename(tmp)
        hidden.append((tmp, f))
        print(f"  [temporarily hidden] {f.name}")
    try:
        yield
    finally:
        for tmp, original in hidden:
            tmp.rename(original)
            print(f"  [restored] {original.name}")


def set_snapshot_index(index: int):
    text = CONFIG_PATH.read_text()
    text = re.sub(r"snapshotindex:.*", f"snapshotindex: {index}", text)
    CONFIG_PATH.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--snapshot-index", type=int, default=None,
                        help="0=first, -1=last, 4=5th snapshot (excludes best, alphabetical order)")
    args = parser.parse_args()

    if args.list:
        list_snapshots(args.shuffle)
        return

    import matplotlib
    matplotlib.use("Agg")
    import deeplabcut

    config_path = str(CONFIG_PATH)

    if args.snapshot_index is not None:
        snapshots = find_snapshots(args.shuffle)
        non_best = [s for s in snapshots if "best" not in s.stem]
        try:
            chosen = non_best[args.snapshot_index]
        except IndexError:
            print(f"Invalid index. Use a value between 0 and {len(non_best)-1}.")
            return
        print(f"Selected snapshot: {chosen.name}")
        dlc_index = snapshots.index(chosen)
        set_snapshot_index(dlc_index)
        with hide_best_snapshot(args.shuffle):
            deeplabcut.evaluate_network(config_path, Shuffles=[args.shuffle], plotting=True)
    else:
        # default: let DLC pick (snapshot-best-*)
        set_snapshot_index(-1)
        deeplabcut.evaluate_network(config_path, Shuffles=[args.shuffle], plotting=True)


if __name__ == "__main__":
    main()
