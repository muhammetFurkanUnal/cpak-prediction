"""
Headless DLC evaluate script
==============================
Kullanım:
  python scripts/evaluate_headless.py
  python scripts/evaluate_headless.py --shuffle 1
"""

import argparse
import pathlib

REPO_ROOT   = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "kneeap-furkan-2026-04-29" / "config.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle", type=int, default=1)
    args = parser.parse_args()

    import deeplabcut

    config_path = str(CONFIG_PATH)
    print(f"Config: {config_path}")
    print(f"Shuffle: {args.shuffle}\n")

    deeplabcut.evaluate_network(
        config_path,
        Shuffles=[args.shuffle],
        plotting=True,
    )


if __name__ == "__main__":
    main()
