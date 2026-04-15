"""
Test / evaluation module for cpak inference results.

Converts test.ipynb logic into reusable functions.

Typical usage
-------------
from notebooks.lib.test import run_evaluation

run_evaluation(
    truth_json_path="path/to/angles.json",
    inference_json_path="path/to/orthopedic_metrics.json",
    output_folder="path/to/out/test/model-name",
)
"""

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# File name normalizers
# ---------------------------------------------------------------------------

def truth_file_name_normalizer(filename: str) -> str:
    """
    Normalizes ground-truth JSON keys to a common format.

    Example
    -------
    "4075_L_İŞARETLENMİŞ-img-00000-00000.jpg"  →  "4075_l"
    """
    parts = filename.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1].lower()}"
    return filename


def inference_file_name_normalizer(filename: str) -> str:
    """
    Normalizes inference JSON keys to a common format.

    Example
    -------
    "4000.l.png"  →  "4000_l"
    """
    name_without_extension = Path(filename).stem
    return name_without_extension.replace(".", "_")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_truth_angles(truth_json_path: str) -> dict:
    """
    Loads and normalizes the ground-truth angles JSON.

    Returns
    -------
    dict  –  {"4075_l": {"femur": "86.34", "tibia": "83.32"}, ...}
    """
    with open(truth_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return {truth_file_name_normalizer(k): v for k, v in raw.items()}


def load_inference_angles(inference_json_path: str) -> dict:
    """
    Loads and normalizes the orthopedic_metrics.json produced by infer_images().

    Returns
    -------
    dict  –  {
        "4075_l": {
            "femur": {"femur_ax_middle": 87.52, "femur_notch": 86.57},
            "tibia": {"tibia_ax_middle": 82.77, "tibia_inter": 82.10}
        }, ...
    }
    """
    with open(inference_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    result = {}
    for key, metrics in raw.items():
        new_key = inference_file_name_normalizer(key)
        result[new_key] = {
            "femur": {
                "femur_ax_middle": round(metrics["femur_mech_angle_ax_middle"], 2),
                "femur_notch": metrics["femur_mech_angle_notch"],
            },
            "tibia": {
                "tibia_ax_middle": round(metrics["tibia_mech_angle_ax_middle"], 2),
                "tibia_inter": metrics["tibia_mech_angle_inter"],
            },
        }
    return result


# ---------------------------------------------------------------------------
# Comparison DataFrame
# ---------------------------------------------------------------------------

def create_comparison_df(gt_dict: dict, pred_dict: dict, output_folder: str) -> pd.DataFrame:
    """
    Joins ground truth and prediction dicts on common keys,
    computes per-sample absolute errors, and saves a CSV.

    Columns
    -------
    id, f_gt, f_ax_pred, f_notch_pred, f_ax_err, f_notch_err,
    t_gt, t_ax_pred, t_inter_pred, t_ax_err, t_inter_err
    """
    os.makedirs(output_folder, exist_ok=True)

    common_keys = sorted(set(gt_dict) & set(pred_dict))
    data = []

    for key in common_keys:
        row = {"id": key}

        # Femur
        try:
            f_gt = float(gt_dict[key].get("femur"))
        except (ValueError, TypeError):
            f_gt = None

        f_ax = pred_dict[key]["femur"].get("femur_ax_middle")
        f_notch = pred_dict[key]["femur"].get("femur_notch")

        row.update({
            "f_gt": f_gt,
            "f_ax_pred": f_ax,
            "f_notch_pred": f_notch,
            "f_ax_err": abs(f_gt - f_ax) if (f_gt is not None and f_ax is not None) else None,
            "f_notch_err": abs(f_gt - f_notch) if (f_gt is not None and f_notch is not None) else None,
        })

        # Tibia
        try:
            t_gt = float(gt_dict[key].get("tibia"))
        except (ValueError, TypeError):
            t_gt = None

        t_ax = pred_dict[key]["tibia"].get("tibia_ax_middle")
        t_inter = pred_dict[key]["tibia"].get("tibia_inter")

        row.update({
            "t_gt": t_gt,
            "t_ax_pred": t_ax,
            "t_inter_pred": t_inter,
            "t_ax_err": abs(t_gt - t_ax) if (t_gt is not None and t_ax is not None) else None,
            "t_inter_err": abs(t_gt - t_inter) if (t_gt is not None and t_inter is not None) else None,
        })

        data.append(row)

    df = pd.DataFrame(data)
    csv_path = os.path.join(output_folder, "angles.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"CSV saved: {csv_path}")
    return df


# ---------------------------------------------------------------------------
# Metrics (MAE / RMSE / R²)
# ---------------------------------------------------------------------------

def calculate_and_save_metrics(df: pd.DataFrame, output_folder: str) -> dict:
    """
    Computes MAE, RMSE, and R² for every bone × method combination
    and saves the result to metrics.json.
    """
    os.makedirs(output_folder, exist_ok=True)

    configs = [
        ("f", "femur", ["ax_pred", "notch_pred"]),
        ("t", "tibia", ["ax_pred", "inter_pred"]),
    ]

    metrics = {}
    for prefix, bone, suffixes in configs:
        bone_results = {}
        y_true = df[f"{prefix}_gt"]

        for suffix in suffixes:
            method = "axial" if "ax" in suffix else ("notch" if "notch" in suffix else "intercondylar")
            y_pred = df[f"{prefix}_{suffix}"]

            mask = y_true.notnull() & y_pred.notnull()
            yt, yp = y_true[mask], y_pred[mask]

            if len(yt) > 0:
                bone_results[method] = {
                    "MAE": round(float(mean_absolute_error(yt, yp)), 3),
                    "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 3),
                    "R2": round(float(r2_score(yt, yp)), 3),
                    "sample_count": int(len(yt)),
                }
            else:
                bone_results[method] = "No valid data"

        metrics[bone] = bone_results

    file_path = os.path.join(output_folder, "metrics.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"Metrics saved: {file_path}")
    return metrics


# ---------------------------------------------------------------------------
# Visualization – regression + residual plots
# ---------------------------------------------------------------------------

def visualize_performance(df: pd.DataFrame, output_folder: str) -> None:
    """
    Saves regression and error-density plots for femur and tibia.
    """
    os.makedirs(output_folder, exist_ok=True)
    sns.set_theme(style="whitegrid")

    configs = [
        ("f", "Femur", ["ax_pred", "notch_pred"]),
        ("t", "Tibia", ["ax_pred", "inter_pred"]),
    ]

    for prefix, bone_name, methods in configs:
        plt.figure(figsize=(16, 7))

        # Regression plot
        plt.subplot(1, 2, 1)
        for method_suffix in methods:
            label = "Axial" if "ax" in method_suffix else ("Notch" if "notch" in method_suffix else "Intercondylar")
            sns.regplot(
                data=df,
                x=f"{prefix}_gt",
                y=f"{prefix}_{method_suffix}",
                label=label,
                scatter_kws={"alpha": 0.5},
                line_kws={"linestyle": "--"},
            )

        gt_col = df[f"{prefix}_gt"]
        min_val, max_val = gt_col.min(), gt_col.max()
        plt.plot([min_val, max_val], [min_val, max_val], color="black", lw=2, label="Perfect Match")
        plt.title(f"{bone_name} - Method Alignment Comparison")
        plt.xlabel("Ground Truth (Degrees)")
        plt.ylabel("Model Prediction (Degrees)")
        plt.legend()

        # Error density plot
        plt.subplot(1, 2, 2)
        for method_suffix in methods:
            label = "Axial" if "ax" in method_suffix else ("Notch" if "notch" in method_suffix else "Intercondylar")
            errors = df[f"{prefix}_gt"] - df[f"{prefix}_{method_suffix}"]
            sns.kdeplot(errors, fill=True, label=f"{label} Error")

        plt.axvline(x=0, color="red", linestyle="--", label="Zero Error")
        plt.title(f"{bone_name} - Error Density (Residuals)")
        plt.xlabel("Error (Actual - Predicted)")
        plt.ylabel("Density")
        plt.legend()

        plt.tight_layout()
        save_path = os.path.join(output_folder, f"{bone_name.lower()}_comparison_graphs.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Graph saved: {save_path}")


# ---------------------------------------------------------------------------
# Error distribution (0.5° bins)
# ---------------------------------------------------------------------------

def error_distribution(df: pd.DataFrame, output_folder: str) -> dict:
    """
    Buckets absolute errors into 0.5° bins and saves distribution.json.
    Returns the full report dict.
    """
    os.makedirs(output_folder, exist_ok=True)

    bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, np.inf]
    labels = [
        "0.0 - 0.5", "0.5 - 1.0", "1.0 - 1.5", "1.5 - 2.0",
        "2.0 - 2.5", "2.5 - 3.0", "3.0 - 3.5", "3.5 - 4.0", "> 4.0",
    ]

    configs = [
        ("f", "Femur", ["ax_pred", "notch_pred"]),
        ("t", "Tibia", ["ax_pred", "inter_pred"]),
    ]

    final_report = {}
    df = df.copy()

    for prefix, bone_name, methods in configs:
        bone_data = {}
        for method_suffix in methods:
            method_key = "axial" if "ax" in method_suffix else ("notch" if "notch" in method_suffix else "intercondylar")
            abs_err_col = f"{prefix}_{method_key}_abs_err"
            range_col = f"{prefix}_{method_key}_range"

            df[abs_err_col] = (df[f"{prefix}_gt"] - df[f"{prefix}_{method_suffix}"]).abs()
            df[range_col] = pd.cut(df[abs_err_col], bins=bins, labels=labels, include_lowest=True)

            method_distribution = {}
            for label in labels:
                matched_ids = df[df[range_col] == label]["id"].tolist()
                method_distribution[label] = {
                    "count": len(matched_ids),
                    "percentage": f"{(len(matched_ids) / len(df)) * 100:.1f}%",
                    "samples": matched_ids,
                }
            bone_data[method_key] = method_distribution
        final_report[bone_name] = bone_data

    output_path = os.path.join(output_folder, "distribution.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)
    print(f"Distribution saved: {output_path}")
    return final_report


def distribution_plot(data: dict, output_folder: str) -> None:
    """
    Saves a 2×2 bar-chart grid from a distribution report dict.
    """
    os.makedirs(output_folder, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle("Distribution of Samples by Bone and View Type", fontsize=20, fontweight="bold")

    plot_configs = [
        ("Femur", "axial", axes[0, 0], "#3498db"),
        ("Femur", "notch", axes[0, 1], "#e74c3c"),
        ("Tibia", "axial", axes[1, 0], "#2ecc71"),
        ("Tibia", "intercondylar", axes[1, 1], "#f1c40f"),
    ]

    for bone, view, ax, color in plot_configs:
        sub_data = data[bone][view]
        bar_labels = list(sub_data.keys())
        counts = [v["count"] for v in sub_data.values()]

        bars = ax.bar(bar_labels, counts, color=color, edgecolor="black", alpha=0.8)
        ax.set_title(f"{bone} - {view.capitalize()}", fontsize=15, fontweight="bold")
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, yval,
                    ha="center", va="bottom", fontweight="bold")

    save_path = os.path.join(output_folder, "distribution_plot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Distribution plot saved: {save_path}")


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    truth_json_path: str,
    inference_json_path: str,
    output_folder: str,
) -> dict:
    """
    Full evaluation pipeline.

    Parameters
    ----------
    truth_json_path     : path to ground-truth angles.json
    inference_json_path : path to orthopedic_metrics.json (output of infer_images)
    output_folder       : directory where all outputs are saved

    Returns
    -------
    dict with keys "metrics" and "distribution"
    """
    truth_angles = load_truth_angles(truth_json_path)
    inference_angles = load_inference_angles(inference_json_path)

    df = create_comparison_df(truth_angles, inference_angles, output_folder)
    metrics = calculate_and_save_metrics(df, output_folder)
    visualize_performance(df, output_folder)
    dist = error_distribution(df, output_folder)
    distribution_plot(dist, output_folder)

    return {"metrics": metrics, "distribution": dist}
