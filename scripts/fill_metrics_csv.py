#!/usr/bin/env python3
"""
metrics.csv — model measurements in the metrics.xlsx layout (cpak only)
=======================================================================

Mirrors ``data/metrics.xlsx`` (the ground-truth table) row-for-row, but the
metric columns (MPTA, LDFA, their difference/sum, alignment, joint type and the
CPAK classification) hold the MODEL's measurements instead of the hand-measured
ground truth, so the two files can be compared side by side.

* Only cpak rows are filled — i.e. GRAFİ = 1 (UZUNLUK / full-length film), for
  both 4000-series and long-ID patients. These come from the cpak model's cached
  ``notebooks/out/inference/<model>/orthopedic_metrics.json``
  (MPTA = tibia_mech_angle_inter, LDFA = femur_mech_angle_notch).
* kneeAP rows (GRAFİ = 2, knee close-ups) are intentionally LEFT BLANK.
* A cpak row whose image is missing from the JSON is also left blank.

Angles are kept at one decimal to match the xlsx; the alignment / joint /
CPAK-type thresholds are the same ones the ground-truth table uses.

Usage
-----
    .venv/bin/python scripts/fill_metrics_csv.py \\
        --cpak-model sh14-ep1000 \\
        --output data/metrics.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import openpyxl

# Reuse the classification / key-parsing helpers already proven in the
# prediction pipeline so metrics.csv stays consistent with it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpak_predict_to_csv import (  # noqa: E402
    INFERENCE_DIR,
    PROJECT_ROOT,
    alignment,
    cpak_type,
    hasta_to_key,
    joint,
    load_cpak,
    route,
)

DEFAULT_CPAK_MODEL = "sh14-ep1000"
DEFAULT_XLSX = PROJECT_ROOT / "data" / "metrics.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "metrics.csv"

# The metric columns that the model fills (indices 2..8 of the 9-column layout).
N_COLS = 9


def cpak_json(model_name: str) -> Path:
    return INFERENCE_DIR / model_name / "orthopedic_metrics.json"


def read_template(xlsx_path: Path):
    """Return (header9, rows) — only the real rows (stops at the first fully
    empty row), each truncated/padded to the 9-column metrics layout."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.active
    header = None
    rows = []
    for raw in ws.iter_rows(values_only=True):
        cells = list(raw[:N_COLS]) + [None] * max(0, N_COLS - len(raw))
        if header is None:
            header = ["" if c is None else str(c) for c in cells]
            continue
        if all(c is None or str(c).strip() == "" for c in cells):
            break  # trailing empty rows — stop
        rows.append(cells)
    return header, rows


def fmt(x: float) -> str:
    """One-decimal, with -0.0 normalised to 0.0."""
    v = round(x, 1)
    return f"{v + 0.0:.1f}"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Build metrics.csv (model measurements, cpak only) from cached inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cpak-model", default=DEFAULT_CPAK_MODEL)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    cpak_path = cpak_json(args.cpak_model)
    for p in (cpak_path, args.xlsx):
        if not p.exists():
            sys.exit(f"Required file not found: {p}")

    cpak_preds = load_cpak(cpak_path)
    header, rows = read_template(args.xlsx)
    print(f"cpak model : {args.cpak_model}  ({cpak_path.relative_to(PROJECT_ROOT)})")
    print(f"  {len(cpak_preds)} full-leg entries in JSON.")
    print(f"template   : {args.xlsx.relative_to(PROJECT_ROOT)}  ({len(rows)} rows)")

    out_rows = [header]
    filled = no_image = blank = 0

    for cells in rows:
        hasta = "" if cells[0] is None else str(cells[0]).strip()
        grafi = "" if cells[1] is None else str(cells[1]).strip()
        if grafi.endswith(".0"):
            grafi = grafi[:-2]  # 1.0 -> 1

        out = [hasta, grafi] + [""] * (N_COLS - 2)
        key = hasta_to_key(hasta) if hasta else None
        model = route(key[0], grafi) if key else None

        if model != "cpak":          # kneeAP / unroutable -> leave blank
            blank += 1
        elif key not in cpak_preds:  # cpak row but no image -> leave blank
            no_image += 1
        else:
            mpta_raw, ldfa_raw = cpak_preds[key]
            mpta = round(mpta_raw, 1)
            ldfa = round(ldfa_raw, 1)
            diff = round(mpta - ldfa, 1)
            total = round(mpta + ldfa, 1)
            align = alignment(diff)
            jnt = joint(total)
            out[2:9] = [
                fmt(mpta), fmt(ldfa), fmt(diff), fmt(total),
                align, jnt, cpak_type(align, jnt),
            ]
            filled += 1

        out_rows.append(out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out_rows)

    print("\nDone.")
    print(f"  cpak filled : {filled}  (no image: {no_image})")
    print(f"  left blank  : {blank}  (kneeAP / unroutable)")
    print(f"\nSaved: {args.output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
