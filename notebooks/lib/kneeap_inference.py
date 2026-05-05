import cv2
import numpy as np
from pathlib import Path
import json
import os
import math
from typing import TypedDict, Tuple, List

import onnxruntime as ort

Point = Tuple[float, float]


# ── Bodypart indices (0-based; in config they are labeled '1'..'30' = 1-based) ──
# Femur
NOTCH            = 0    # '1'  intercondylar notch
F_LAT_JOINT      = 1    # '2'  femur lateral condyle (joint side)
F_LAT_EDGE       = 2    # '3'  femur lateral condyle (outer edge)
F_LAT_CHAIN      = [1, 5, 3, 6, 4, 7, 2]    # 2 → 6 → 4 → 7 → 5 → 8 → 3
F_MED_JOINT      = 8    # '9'  femur medial condyle (joint side)
F_MED_EDGE       = 9    # '10' femur medial condyle (outer edge)
F_MED_CHAIN      = [8, 12, 10, 13, 11, 14, 9]   # 9 → 13 → 11 → 14 → 12 → 15 → 10
# Tibia
T_INTER_LAT      = 15   # '16' tibial intercondylar (lateral spike side)
T_LAT_JOINT      = 16   # '17' tibial lateral plateau (joint side)
T_LAT_EDGE       = 17   # '18' tibial lateral plateau (outer edge)
T_LAT_CHAIN      = [16, 20, 18, 21, 19, 22, 17]  # 17 → 21 → 19 → 22 → 20 → 23 → 18
T_INTER_MED      = 23   # '24' tibial intercondylar (medial spike side)
T_MED_EDGE       = 24   # '25' tibial medial plateau (outer edge)
T_MED_CHAIN      = [23, 27, 25, 28, 26, 29, 24]  # 24 → 28 → 26 → 29 → 27 → 30 → 25


class KneeMetrics(TypedDict):
    # femur landmarks
    femur_notch: Point
    femur_lateral_joint: Point
    femur_medial_joint: Point
    femur_joint_mid: Point
    # tibia landmarks
    tibia_intercondylar_mid: Point
    tibia_lateral_joint: Point
    tibia_medial_joint: Point
    tibia_joint_mid: Point
    # angles
    jlca: float            # joint line convergence angle


# ── Geometry helpers ──────────────────────────────────────────────────────────

def calculate_center(points: List[Point]) -> Point:
    if not points:
        return (0.0, 0.0)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    return (sx / len(points), sy / len(points))


def calculate_vector_angle(v1: Point, v2: Point) -> float:
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    m1 = math.hypot(*v1)
    m2 = math.hypot(*v2)
    if m1 == 0 or m2 == 0:
        return 0.0
    val = max(-1.0, min(1.0, dot / (m1 * m2)))
    angle = math.degrees(math.acos(val))
    return angle if angle <= 90.0 else 180.0 - angle


# ── Orthopedic metrics (knee AP) ──────────────────────────────────────────────

def compute_kneeap_metrics(coords: List[Point]) -> KneeMetrics:
    femur_notch         = tuple(coords[NOTCH])
    femur_lateral_joint = tuple(coords[F_LAT_JOINT])
    femur_medial_joint  = tuple(coords[F_MED_JOINT])
    femur_joint_mid     = (
        (femur_lateral_joint[0] + femur_medial_joint[0]) / 2.0,
        (femur_lateral_joint[1] + femur_medial_joint[1]) / 2.0,
    )

    tibia_intercondylar_mid = (
        (coords[T_INTER_LAT][0] + coords[T_INTER_MED][0]) / 2.0,
        (coords[T_INTER_LAT][1] + coords[T_INTER_MED][1]) / 2.0,
    )
    tibia_lateral_joint = tuple(coords[T_LAT_JOINT])
    tibia_medial_joint  = tuple(coords[T_INTER_MED])  # '24' is the medial joint side
    tibia_joint_mid     = (
        (tibia_lateral_joint[0] + tibia_medial_joint[0]) / 2.0,
        (tibia_lateral_joint[1] + tibia_medial_joint[1]) / 2.0,
    )

    femur_joint_vec = (
        femur_medial_joint[0] - femur_lateral_joint[0],
        femur_medial_joint[1] - femur_lateral_joint[1],
    )
    tibia_joint_vec = (
        tibia_medial_joint[0] - tibia_lateral_joint[0],
        tibia_medial_joint[1] - tibia_lateral_joint[1],
    )
    jlca = calculate_vector_angle(femur_joint_vec, tibia_joint_vec)

    return {
        "femur_notch": femur_notch,
        "femur_lateral_joint": femur_lateral_joint,
        "femur_medial_joint": femur_medial_joint,
        "femur_joint_mid": femur_joint_mid,
        "tibia_intercondylar_mid": tibia_intercondylar_mid,
        "tibia_lateral_joint": tibia_lateral_joint,
        "tibia_medial_joint": tibia_medial_joint,
        "tibia_joint_mid": tibia_joint_mid,
        "jlca": jlca,
    }


# ── Drawing ───────────────────────────────────────────────────────────────────

CHAIN_GROUPS = [
    ("femur_lateral",  F_LAT_CHAIN, (255, 200, 0)),
    ("femur_medial",   F_MED_CHAIN, (255, 100, 0)),
    ("tibia_lateral",  T_LAT_CHAIN, (0, 200, 255)),
    ("tibia_medial",   T_MED_CHAIN, (0, 100, 255)),
]


def _chain_midpoints(lat_chain, med_chain, coords):
    """Pair-wise midpoints between lateral and medial chain points."""
    return [
        ((coords[li][0] + coords[mi][0]) / 2.0,
         (coords[li][1] + coords[mi][1]) / 2.0)
        for li, mi in zip(lat_chain, med_chain)
    ]


def _fit_anatomical_axis(midpoints):
    """Least-squares line through midpoints. Returns unit-direction (vx, vy) and a point (x0, y0)."""
    pts = np.array(midpoints, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    return float(vx), float(vy), float(x0), float(y0)


def draw_lines(image, metrics: KneeMetrics, coords: List[Point]):
    vis = image.copy()
    radius = 1

    def pt(p): return (int(round(p[0])), int(round(p[1])))
    def dot(p, color, r=radius): cv2.circle(vis, pt(p), r, color, -1)
    def line(a, b, color, thick=1): cv2.line(vis, pt(a), pt(b), color, thick)
    def text(s, p, color):
        cv2.putText(vis, s, pt(p), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # condyle outline chains
    for _name, chain, color in CHAIN_GROUPS:
        for a, b in zip(chain[:-1], chain[1:]):
            line(coords[a], coords[b], color)
        for idx in chain:
            dot(coords[idx], color)

    # notch + intercondylar dots
    dot(metrics["femur_notch"], (255, 0, 0))
    dot(coords[T_INTER_LAT], (0, 0, 255))
    dot(coords[T_INTER_MED], (0, 0, 255))

    # joint lines
    line(metrics["femur_lateral_joint"], metrics["femur_medial_joint"], (0, 255, 0), 2)
    line(metrics["tibia_lateral_joint"], metrics["tibia_medial_joint"], (0, 255, 0), 2)

    # anatomical axes — least-squares fit through 4 farthest-from-joint lateral/medial midpoints,
    # extended from those midpoints down/up to just past the joint reference (notch / intercondylar mid).
    for lat_chain, med_chain, ref_point, take_top, axis_color in [
        (F_LAT_CHAIN, F_MED_CHAIN, metrics["femur_notch"],            True,  (170, 90, 170)),  # muted purple
        (T_LAT_CHAIN, T_MED_CHAIN, metrics["tibia_intercondylar_mid"], False, (90, 170, 170)),  # muted teal
    ]:
        midpoints = _chain_midpoints(lat_chain, med_chain, coords)
        midpoints.sort(key=lambda p: p[1])  # ascending y (top of image first)
        chosen = midpoints[:4] if take_top else midpoints[-4:]
        for m in chosen:
            dot(m, axis_color, r=2)
        if len(chosen) >= 2:
            vx, vy, x0, y0 = _fit_anatomical_axis(chosen)
            def proj(p): return (p[0] - x0) * vx + (p[1] - y0) * vy
            t_mids = [proj(m) for m in chosen]
            t_ref  = proj(ref_point)
            t_mean = sum(t_mids) / len(t_mids)
            direction = 1.0 if t_ref >= t_mean else -1.0
            t_far = min(t_mids) if direction > 0 else max(t_mids)
            overshoot = abs(t_ref - t_mean) * 0.08
            t1 = t_far
            t2 = t_ref + direction * overshoot
            p1 = (x0 + vx * t1, y0 + vy * t1)
            p2 = (x0 + vx * t2, y0 + vy * t2)
            line(p1, p2, axis_color, 1)

    # JLCA label near tibia joint mid
    # label_pos = (metrics["tibia_joint_mid"][0] + 8, metrics["tibia_joint_mid"][1])
    # text(f"JLCA {metrics['jlca']:.2f}", label_pos, (0, 255, 255))

    return vis


# ── Inference pipeline ────────────────────────────────────────────────────────

def get_predictions(session, image):
    """Raw output from model. Returns (heatmap, offset)."""
    if len(image.shape) == 2 or image.shape[2] == 1:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    x = (x - mean) / std
    x = np.expand_dims(x, axis=0).astype(np.float32)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: x})
    heatmap = outputs[0][0]
    offset  = outputs[1][0]
    return heatmap, offset


def visualize_heatmap(heatmap, output_path="heatmap_output.jpg"):
    combined = np.max(heatmap, axis=0)
    combined = np.clip(combined, 0, 1.0)
    gray = (combined * 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    cv2.imwrite(output_path, color)
    return color


def extract_coordinates(heatmap, offset, stride=8, locref_stdev=7.2801, debug=False):
    """Translate (heatmap, offset) → list of (x, y, confidence)."""
    num_joints = heatmap.shape[0]
    out = []
    for i in range(num_joints):
        joint_hm = heatmap[i, :, :]
        _, conf, _, max_loc = cv2.minMaxLoc(joint_hm)
        x_map, y_map = max_loc
        x_base = (x_map + 0.5) * stride
        y_base = (y_map + 0.5) * stride
        ox = offset[i * 2,     y_map, x_map] * locref_stdev
        oy = offset[i * 2 + 1, y_map, x_map] * locref_stdev
        if debug:
            out.append((x_base + offset[i*2, y_map, x_map],
                        y_base + offset[i*2+1, y_map, x_map], conf))
        else:
            out.append((x_base + ox, y_base + oy, conf))
    return out


def visualize_predictions(image, coords, stride=8, threshold=0.5, grid=False):
    vis = image.copy()
    h, w = vis.shape[:2]
    if grid:
        for x in range(stride, w, stride):
            cv2.line(vis, (x, 0), (x, h), (100, 100, 100), 1)
        for y in range(stride, h, stride):
            cv2.line(vis, (0, y), (w, y), (100, 100, 100), 1)

    for i, (x, y, prob) in enumerate(coords):
        if prob > threshold:
            ix, iy = int(round(x)), int(round(y))
            cv2.circle(vis, (ix, iy), 1, (0, 255, 0), -1)
            cv2.putText(vis, str(i + 1), (ix + 4, iy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return vis


# ── Top-level entry points ────────────────────────────────────────────────────

def infer_single_image(onnx_path, image_path, output_path="output.jpg"):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)

    img = cv2.imread(image_path)
    if img is None:
        print("Image not found.")
        return

    heatmap, offset = get_predictions(session, img)
    coords = extract_coordinates(heatmap, offset)
    result = visualize_predictions(img, coords, threshold=0.0)
    cv2.imwrite(output_path, result)
    print(f"Saved: {output_path}")
    return heatmap, offset, coords


def infer_images(
    onnx_path,
    input_folder,
    output_folder,
    inference_json_name="inference_results.json",
    metrics_json_name="kneeap_metrics.json",
    landmarks=False,
):
    session = ort.InferenceSession(
        onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    os.makedirs(output_folder, exist_ok=True)

    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = sorted(str(p) for p in Path(input_folder).iterdir()
                         if p.suffix.lower() in valid_ext)

    all_results = {}
    all_metrics = {}

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        heatmap, offset = get_predictions(session, img)
        coords = extract_coordinates(heatmap, offset)
        metrics = compute_kneeap_metrics(coords)

        name = os.path.basename(path)
        all_results[name] = [
            {"joint_id": i + 1, "x": float(x), "y": float(y), "conf": float(c)}
            for i, (x, y, c) in enumerate(coords)
        ]
        all_metrics[name] = metrics

        result_img = (visualize_predictions(img, coords, threshold=0.0)
                      if landmarks else draw_lines(img, metrics, coords))
        cv2.imwrite(os.path.join(output_folder, name), result_img)
        print(f"Processed: {name}")

    with open(os.path.join(output_folder, inference_json_name), 'w') as f:
        json.dump(all_results, f)
    with open(os.path.join(output_folder, metrics_json_name), 'w') as f:
        json.dump(all_metrics, f)

    print(f"Finished. Saved to: {output_folder}")
