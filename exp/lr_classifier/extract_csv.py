"""Convert tagimg JSON output into a CSV of (image, label) pairs.

Tag mapping:
    fibula_left  -> l
    fibula_right -> r

Images carrying both tags get the label "l,r"; images without any mapped
tag are skipped.

Example:
    python3 extract_csv.py ../../preprocessing/knee-preop-prepped/tags.json fibula.csv
"""

import argparse
import csv
import json
from pathlib import Path


TAG_TO_CODE = {
    "fibula_left": "l",
    "fibula_right": "r",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path, help="tags.json input")
    parser.add_argument("csv_path", type=Path, help="output CSV")
    args = parser.parse_args()

    data = json.loads(args.json_path.read_text())
    tags: dict[str, list[str]] = data.get("tags", {})

    with args.csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        for name, ts in tags.items():
            codes = [TAG_TO_CODE[t] for t in ts if t in TAG_TO_CODE]
            if not codes:
                continue
            writer.writerow([name, ",".join(codes)])


if __name__ == "__main__":
    main()
