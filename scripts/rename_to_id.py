#!/usr/bin/env python3
"""Rename every file inside a directory to its leading ID only.

Given a directory, each file named like ``9108134.Seq1.Ser1.Img1.png`` is
renamed to ``9108134.png`` (everything before the first dot is kept as the ID,
the original extension is preserved). Subdirectories are traversed recursively.

Example:
    python scripts/rename_to_id.py /path/to/dir
    python scripts/rename_to_id.py /path/to/dir --dry-run
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Klasördeki tüm dosyaları baştaki ID'ye göre yeniden adlandır.")
    parser.add_argument("directory", help="Dosyaların bulunduğu klasör")
    parser.add_argument("--dry-run", action="store_true", help="Sadece ne yapılacağını göster, dosyayı değiştirme")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Klasör bulunamadı: {args.directory}", file=sys.stderr)
        sys.exit(1)

    for root, _dirs, files in os.walk(args.directory):
        for name in files:
            # İlk nokta ID'nin sonu, son nokta uzantının başı
            file_id = name.split(".", 1)[0]
            ext = os.path.splitext(name)[1]  # ör. ".png"
            new_name = file_id + ext

            if new_name == name:
                continue

            src = os.path.join(root, name)
            dst = os.path.join(root, new_name)
            if os.path.exists(dst):
                print(f"ATLANDI (hedef zaten var): {name} -> {new_name}", file=sys.stderr)
                continue

            print(f"{src} -> {new_name}")
            if not args.dry_run:
                os.rename(src, dst)


if __name__ == "__main__":
    main()
