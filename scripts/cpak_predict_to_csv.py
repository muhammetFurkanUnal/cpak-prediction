#!/usr/bin/env python3
"""
CPAK / kneeAP prediction → CSV
==============================

Reads the ground-truth table (``data/cpak-grnd-truth.csv``) and, for each row,
fills in the metrics its radiograph implies (MPTA, LDFA, their difference/sum,
alignment, joint type and the CPAK classification), writing everything to a NEW
csv file. The original file is never touched — each run reads the original and
rebuilds the output from scratch.

The angles are NOT recomputed here. Inference was already run and its results
cached as JSON under ``notebooks/out/inference/<model-name>/``; this script just
reads those files, so it is cheap to run repeatedly:
* cpak   → ``sh14-ep1000/orthopedic_metrics.json``
           (MPTA = tibia_mech_angle_inter, LDFA = femur_mech_angle_notch).
* kneeap → ``kneeap-sh4-ep1000/kneeap_angles.json``
           (MPTA = ampta, LDFA = aldfa).
The JSONs are keyed by image filename (e.g. ``4000.r.png``) and the cpak file
already covers both 4000-series and long-ID full-leg images.

Row → model routing
-------------------
* 4000-series, GRAFİ = 1 (UZUNLUK / full-length film)
      → cpak model. Mechanical angles.
* 4000-series, GRAFİ = 2 (AP / knee close-up)
      → kneeap model. Anatomical angles (the only thing a knee close-up can
        give, since the mechanical axis is off-frame).
* long-ID patients, GRAFİ = 1
      → cpak model.
* long-ID patients, GRAFİ = 2
      → left blank (no knee close-up images exist for these patients).

Both metric paths match the production API exactly
(``api/main.py`` → ``notebooks.lib.{cpak_inference,kneeap_inference}``), which is
what produced the cached JSON files in the first place.

Usage
-----
    .venv/bin/python scripts/cpak_predict_to_csv.py \\
        --cpak-model sh14-ep1000 \\
        --kneeap-model kneeap-sh4-ep1000 \\
        --output ./cpak-grnd-truth-predicted.csv

All three are optional; defaults are the model names above and an output file
written into the current working directory.
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CPAK_MODEL   = "sh14-ep1000"
DEFAULT_KNEEAP_MODEL = "kneeap-sh4-ep1000"
DEFAULT_OUTPUT_NAME  = "cpak-grnd-truth-predicted.csv"

INFERENCE_DIR = PROJECT_ROOT / "notebooks" / "out" / "inference"
INPUT_CSV     = PROJECT_ROOT / "data" / "cpak-grnd-truth.csv"


def cpak_json(model_name: str) -> Path:
    """Cached cpak results for a model (no model is loaded — read directly)."""
    return INFERENCE_DIR / model_name / "orthopedic_metrics.json"


def kneeap_json(model_name: str) -> Path:
    """Cached kneeap (AP) results for a model."""
    return INFERENCE_DIR / model_name / "kneeap_angles.json"

# New columns appended to every row (blank where no prediction applies).
PRED_HEADERS = [
    "MPTA_pred", "LDFA_pred",
    "MPTA-LDFA_pred", "MPTA+LDFA_pred",
    "DİZİLİM_pred", "EKLEM_pred", "SINIFLAMA_pred",
    "MPTA_pred_raw", "LDFA_pred_raw",
    "model_pred",
]


# ── Classification (same thresholds the ground-truth table uses) ────────────────
def alignment(diff: int) -> str:
    """DİZİLİM from MPTA-LDFA:  <-2 varus / [-2,2] nötr / >2 valgus."""
    if diff < -2:
        return "varus"
    if diff > 2:
        return "valgus"
    return "nötr"


def joint(total: int) -> str:
    """EKLEM from MPTA+LDFA:  <177 apex distal / [177,183] nötr / >183 apex proksimal."""
    if total < 177:
        return "apex distal"
    if total > 183:
        return "apex proksimal"
    return "nötr"


# CPAK 3×3 grid → Tip 1..9
_CPAK_GRID = {
    ("varus",  "apex distal"):    "Tip 1",
    ("nötr",   "apex distal"):    "Tip 2",
    ("valgus", "apex distal"):    "Tip 3",
    ("varus",  "nötr"):           "Tip 4",
    ("nötr",   "nötr"):           "Tip 5",
    ("valgus", "nötr"):           "Tip 6",
    ("varus",  "apex proksimal"): "Tip 7",
    ("nötr",   "apex proksimal"): "Tip 8",
    ("valgus", "apex proksimal"): "Tip 9",
}


def cpak_type(align: str, jnt: str) -> str:
    return _CPAK_GRID[(align, jnt)]


def round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


# ── Key parsing ──────────────────────────────────────────────────────────────────
# A key is (patient_id, side) e.g. ("4000", "r") or ("10105873454", "l").
_KEY_RE = re.compile(r"^(\d+)([lr])$")


def _normalize(text: str) -> str:
    """Strip dots/spaces and lowercase: '4000.r' / ' 4096L' / '10105873454 r ' → '4000r'..."""
    return re.sub(r"[.\s]", "", text).lower()


def img_to_key(filename: str):
    """Image filename → (pid, side). Handles '4000.r.png' and '10105873454 r .png'."""
    stem = filename[:-4] if filename.lower().endswith(".png") else filename
    m = _KEY_RE.match(_normalize(stem))
    return (m.group(1), m.group(2)) if m else None


def hasta_to_key(value: str):
    """HASTA cell → (pid, side) or None."""
    m = _KEY_RE.match(_normalize(value))
    return (m.group(1), m.group(2)) if m else None


def is_4xxx(pid: str) -> bool:
    return len(pid) == 4 and pid.startswith("4")


# ── Cached inference results ──────────────────────────────────────────────────────
def load_cpak(json_path: Path) -> dict:
    """cpak orthopedic_metrics.json → {(pid, side): (mpta_raw, ldfa_raw)} (mechanical)."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    preds = {}
    for fname, m in data.items():
        key = img_to_key(fname)
        if key is None:
            continue
        preds[key] = (m["tibia_mech_angle_inter"], m["femur_mech_angle_notch"])
    return preds


def load_kneeap(json_path: Path) -> dict:
    """kneeap kneeap_angles.json → {(pid, side): (mpta_raw, ldfa_raw)} (anatomical)."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    preds = {}
    for fname, m in data.items():
        key = img_to_key(fname)
        if key is None:
            continue
        preds[key] = (m["ampta"], m["aldfa"])
    return preds


# ── Row routing ───────────────────────────────────────────────────────────────────
def route(pid: str, grafi: str):
    """Return the model name that should fill this row, or None to leave it blank."""
    if is_4xxx(pid):
        if grafi == "1":
            return "cpak"
        if grafi == "2":
            return "kneeap"
        return None
    # long-ID
    return "cpak" if grafi == "1" else None


# ── CLI ─────────────────────────────────────────────────────────────────────────
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill the CPAK ground-truth CSV from cached inference results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cpak-model", default=DEFAULT_CPAK_MODEL,
        help="model name (folder under notebooks/out/inference/) for full-leg rows; "
             "reads its orthopedic_metrics.json",
    )
    parser.add_argument(
        "--kneeap-model", default=DEFAULT_KNEEAP_MODEL,
        help="model name (folder under notebooks/out/inference/) for AP knee rows; "
             "reads its kneeap_angles.json",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output CSV path; if a directory, the file is written inside it. "
             f"Default: ./{DEFAULT_OUTPUT_NAME} in the current directory",
    )
    return parser.parse_args(argv)


def resolve_output(output: Path | None) -> Path:
    """Decide the output CSV path. Default: cwd/DEFAULT_OUTPUT_NAME; a dir → dir/name."""
    if output is None:
        return Path.cwd() / DEFAULT_OUTPUT_NAME
    if output.is_dir() or str(output).endswith(("/", "\\")):
        return output / DEFAULT_OUTPUT_NAME
    return output


# ── Main ─────────────────────────────────────────────────────────────────────────
def main(argv=None) -> None:
    args = parse_args(argv)

    cpak_path = cpak_json(args.cpak_model)
    kneeap_path = kneeap_json(args.kneeap_model)
    output_csv = resolve_output(args.output)

    for p in (cpak_path, kneeap_path, INPUT_CSV):
        if not p.exists():
            sys.exit(f"Required file not found: {p}")

    print(f"cpak model     : {args.cpak_model}  ({cpak_path.relative_to(PROJECT_ROOT)})")
    print(f"kneeap model   : {args.kneeap_model}  ({kneeap_path.relative_to(PROJECT_ROOT)})")
    cpak_preds = load_cpak(cpak_path)
    print(f"  {len(cpak_preds)} full-leg entries.")
    kneeap_preds = load_kneeap(kneeap_path)
    print(f"  {len(kneeap_preds)} close-up entries.\n")

    preds_by_model = {"cpak": cpak_preds, "kneeap": kneeap_preds}

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    header = rows[0]
    n_cols = len(header)  # pad data rows to this width so predicted cols stay aligned
    out_rows = [header + PRED_HEADERS]

    # tallies: per model "filled" and "no image", plus blanks
    filled = {"cpak": 0, "kneeap": 0}
    no_image = {"cpak": 0, "kneeap": 0}
    blank = 0
    agree = {"cpak": [0, 0], "kneeap": [0, 0]}  # [match, total] vs gt SINIFLAMA

    for row in rows[1:]:
        row = list(row)
        if len(row) < n_cols:
            row += [""] * (n_cols - len(row))
        pred_cols = [""] * len(PRED_HEADERS)

        hasta = row[0].strip() if row else ""
        grafi = row[1].strip() if len(row) > 1 else ""
        key = hasta_to_key(hasta) if hasta else None

        if key is None:
            out_rows.append(row + pred_cols)
            continue

        model = route(key[0], grafi)
        if model is None:
            blank += 1
            out_rows.append(row + pred_cols)
            continue

        if key not in preds_by_model[model]:
            no_image[model] += 1
            out_rows.append(row + pred_cols)
            continue

        mpta_raw, ldfa_raw = preds_by_model[model][key]
        mpta = round_half_up(mpta_raw)
        ldfa = round_half_up(ldfa_raw)
        diff = mpta - ldfa
        total = mpta + ldfa
        align = alignment(diff)
        jnt = joint(total)
        tip = cpak_type(align, jnt)

        pred_cols = [
            str(mpta), str(ldfa),
            str(diff), str(total),
            align, jnt, tip,
            f"{mpta_raw:.2f}", f"{ldfa_raw:.2f}",
            model,
        ]
        filled[model] += 1

        gt_tip = row[8].strip() if len(row) > 8 else ""  # ground-truth SINIFLAMA
        if gt_tip:
            agree[model][1] += 1
            if gt_tip == tip:
                agree[model][0] += 1

        out_rows.append(row + pred_cols)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out_rows)

    print("Done.")
    print(f"  filled — cpak  : {filled['cpak']}  (no entry: {no_image['cpak']})")
    print(f"  filled — kneeap: {filled['kneeap']}  (no entry: {no_image['kneeap']})")
    print(f"  left blank     : {blank}")
    for model in ("cpak", "kneeap"):
        match, tot = agree[model]
        if tot:
            print(f"  CPAK type vs GT ({model:<6}): {match}/{tot} ({100.0*match/tot:.1f}%)")
    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()
