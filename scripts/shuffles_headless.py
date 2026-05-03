"""
Manage DeepLabCut shuffles (training datasets).

Usage:
  python scripts/shuffles_headless.py list
  python scripts/shuffles_headless.py create            # auto-pick next index
  python scripts/shuffles_headless.py create 3          # create shuffle 3
  python scripts/shuffles_headless.py delete 2
"""

import argparse
import pathlib
import re
import shutil
import sys

REPO_ROOT   = pathlib.Path(__file__).resolve().parent.parent
PROJECT_DIR = REPO_ROOT / "kneeap-furkan-2026-04-29"
CONFIG_PATH = PROJECT_DIR / "config.yaml"


def find_existing_shuffles() -> list[int]:
    td = PROJECT_DIR / "training-datasets"
    if not td.exists():
        return []
    indices = set()
    for path in td.rglob("*shuffle*"):
        m = re.search(r"shuffle(\d+)", path.name)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def list_shuffles():
    shuffles = find_existing_shuffles()
    if not shuffles:
        print("No shuffles exist yet.")
        return
    print("Existing shuffles:")
    for idx in shuffles:
        # also check if model weights exist
        model_dir = PROJECT_DIR / "dlc-models-pytorch"
        has_model = any(model_dir.rglob(f"*shuffle{idx}*")) if model_dir.exists() else False
        marker = " (trained)" if has_model else ""
        print(f"  shuffle {idx}{marker}")


def delete_shuffle(idx: int):
    deleted = []
    for base in ("training-datasets", "dlc-models-pytorch", "dlc-models"):
        root = PROJECT_DIR / base
        if not root.exists():
            continue
        for path in root.rglob(f"*shuffle{idx}*"):
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            deleted.append(path)
    if deleted:
        print(f"Deleted shuffle {idx} ({len(deleted)} entries):")
        for p in deleted:
            print(f"  - {p.relative_to(PROJECT_DIR)}")
    else:
        print(f"No data found for shuffle {idx}")


def create_shuffle(idx: int | None):
    # Ensure project folder is set up
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from train_headless import setup_project
    setup_project()

    existing = find_existing_shuffles()
    if idx is None:
        idx = max(existing) + 1 if existing else 1
        print(f"Auto-picked next index: {idx}")
    elif idx in existing:
        print(f"Shuffle {idx} already exists. Delete it first or pick another index.")
        return

    import deeplabcut
    print(f"\n── create_training_dataset (shuffle {idx}) ──────────────────────────")
    deeplabcut.create_training_dataset(str(CONFIG_PATH), num_shuffles=1, Shuffles=[idx])
    print(f"\nShuffle {idx} created.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List existing shuffles")

    p_create = sub.add_parser("create", help="Create a new shuffle")
    p_create.add_argument("index", type=int, nargs="?", help="Shuffle index (auto if omitted)")

    p_delete = sub.add_parser("delete", help="Delete a shuffle")
    p_delete.add_argument("index", type=int)

    args = parser.parse_args()

    if args.cmd == "list":
        list_shuffles()
    elif args.cmd == "create":
        create_shuffle(args.index)
    elif args.cmd == "delete":
        delete_shuffle(args.index)


if __name__ == "__main__":
    main()
