#!/usr/bin/env python3
"""
CPAK / kneeAP prediction → CSV
==============================

Reads the ground-truth table (``data/cpak-grnd-truth.csv``), runs the model that
fits each row's radiograph, computes the same metrics the table records (MPTA,
LDFA, their difference/sum, alignment, joint type and the CPAK classification),
and writes everything to a NEW csv file. The original file is never touched —
each run reads the original and rebuilds the output from scratch.

Row → model routing
-------------------
* 4000-series, GRAFİ = 1 (UZUNLUK / full-length film)
      → cpak model on the full-leg crop in ``preprocessing/preop-prepped``.
        LDFA = femur_mech_angle_notch, MPTA = tibia_mech_angle_inter (mechanical).
* 4000-series, GRAFİ = 2 (AP / knee close-up)
      → kneeap model (kneeap-sh4-ep1000) on ``preprocessing/knee-preop-prepped``.
        LDFA = aldfa, MPTA = ampta (anatomical — the only thing a knee close-up
        can give, since the mechanical axis is off-frame).
* long-ID patients, GRAFİ = 1
      → cpak model on the full-leg crop in ``preprocessing/long-id-prep``.
* long-ID patients, GRAFİ = 2
      → left blank (no knee close-up images exist for these patients).

Both models / metric paths match the production API exactly
(``api/main.py`` → ``notebooks.lib.{cpak_inference,kneeap_inference}``).

Usage
-----
    .venv/bin/python scripts/cpak_predict_to_csv.py
"""

import csv
import math
import re
import sys
from pathlib import Path

import cv2

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # so `notebooks.lib` is importable

from notebooks.lib import cpak_inference, kneeap_inference  # noqa: E402

MODELS_DIR   = PROJECT_ROOT / "notebooks" / "out" / "models"
CPAK_MODEL   = MODELS_DIR / "sh14-ep1000.onnx"
KNEEAP_MODEL = MODELS_DIR / "kneeap-sh4-ep1000.onnx"

# cpak (full-leg) image sources
PREOP_PREPPED_DIR = PROJECT_ROOT / "preprocessing" / "preop-prepped"    # 4000-series
LONGID_PREP_DIR   = PROJECT_ROOT / "preprocessing" / "long-id-split-prepped"     # long-ID
# kneeap (knee close-up) image source
KNEE_PREPPED_DIR  = PROJECT_ROOT / "preprocessing" / "knee-preop-prepped"  # 4000-series

INPUT_CSV  = PROJECT_ROOT / "data" / "cpak-grnd-truth.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "cpak-grnd-truth-predicted.csv"

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


# ── Inference ───────────────────────────────────────────────────────────────────
def _session(model_path: Path):
    import onnxruntime as ort
    return ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def run_cpak(image_dirs) -> dict:
    """cpak full-leg model → {(pid, side): (mpta_raw, ldfa_raw)} (mechanical angles)."""
    session = _session(CPAK_MODEL)
    preds = {}
    for image_dir in image_dirs:
        for path in sorted(Path(image_dir).glob("*.png")):
            key = img_to_key(path.name)
            if key is None:
                continue
            img = cv2.imread(str(path))
            if img is None:
                print(f"  ! could not read {path}")
                continue
            heatmap, offset = cpak_inference.get_predictions(session, img)
            coords = cpak_inference.extract_coordinates(heatmap, offset)
            m = cpak_inference.compute_orthopedic_metrics(coords)
            preds[key] = (m["tibia_mech_angle_inter"], m["femur_mech_angle_notch"])
    return preds


def run_kneeap(image_dir) -> dict:
    """kneeap close-up model → {(pid, side): (mpta_raw, ldfa_raw)} (anatomical angles)."""
    session = _session(KNEEAP_MODEL)
    preds = {}
    for path in sorted(Path(image_dir).glob("*.png")):
        key = img_to_key(path.name)
        if key is None:
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"  ! could not read {path}")
            continue
        heatmap, offset = kneeap_inference.get_predictions(session, img)
        coords = kneeap_inference.extract_coordinates(heatmap, offset)
        m = kneeap_inference.compute_kneeap_metrics(coords)
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


# ── Main ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    for p in (CPAK_MODEL, KNEEAP_MODEL, INPUT_CSV):
        if not p.exists():
            sys.exit(f"Required file not found: {p}")

    print(f"cpak model   : {CPAK_MODEL.name}")
    print(f"kneeap model : {KNEEAP_MODEL.name}")
    print("Running cpak inference (full-leg: preop-prepped + long-id-prep)...")
    cpak_preds = run_cpak([PREOP_PREPPED_DIR, LONGID_PREP_DIR])
    print(f"  {len(cpak_preds)} full-leg images.")
    print("Running kneeap inference (knee close-ups: knee-preop-prepped)...")
    kneeap_preds = run_kneeap(KNEE_PREPPED_DIR)
    print(f"  {len(kneeap_preds)} close-up images.\n")

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

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(out_rows)

    print("Done.")
    print(f"  filled — cpak  : {filled['cpak']}  (no image: {no_image['cpak']})")
    print(f"  filled — kneeap: {filled['kneeap']}  (no image: {no_image['kneeap']})")
    print(f"  left blank     : {blank}")
    for model in ("cpak", "kneeap"):
        match, tot = agree[model]
        if tot:
            print(f"  CPAK type vs GT ({model:<6}): {match}/{tot} ({100.0*match/tot:.1f}%)")
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
