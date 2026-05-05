import cv2
import numpy as np
from pathlib import Path
import json

import os
import sys

import onnxruntime as ort
import math
import cv2
from typing import TypedDict, Tuple, List

Point = Tuple[float, float]

class OrthopedicMetrics(TypedDict):
    # femur head
    multi_center: Point
    femur_head: Point
    # femur joint
    femur_lateral: Point
    femur_medial: Point
    femur_ax_middle: Point
    femur_notch: Point
    # tibia joint
    tibia_lateral: Point
    tibia_medial: Point
    tibia_intercondiler: Point  # small spikes in the middle of tibia
    tibia_ax_middle: Point
    # ankle
    ankle_lateral: Point
    ankle_medial: Point
    ankle_ax_middle: Point
    ankle_model_middle: Point
    final_ankle_middle: Point
    # angles
    femur_mech_angle_ax_middle: float
    femur_mech_angle_notch: float
    tibia_mech_angle_ax_middle: float
    tibia_mech_angle_inter: float

def calculate_circle_center(points: List[Point]) -> Point:
    if len(points) < 3:
        return (0.0, 0.0)

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    A = np.c_[x, y, np.ones(len(x))]
    b = x**2 + y**2

    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    center_x = c[0] / 2.0
    center_y = c[1] / 2.0

    return (center_x, center_y)


def calculate_vector_angle(v1: Point, v2: Point) -> float:
    dot_product = (v1[0] * v2[0]) + (v1[1] * v2[1])
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
        
    val = dot_product / (mag1 * mag2)
    val = max(-1.0, min(1.0, val))
    
    angle_deg = math.degrees(math.acos(val))
    
    return angle_deg if angle_deg <= 90.0 else 180.0 - angle_deg


def compute_orthopedic_metrics(coords: List[Point]) -> OrthopedicMetrics:
    # points can be learned from deeplabcut gui
    # however, deeplabcut gui is 1-indexed, coords is 0-indexed 


    labels = {
        # Femur Head
        "femur_head_3": 2,   # coords[2]
        "femur_head_4": 3,   # coords[3]
        "femur_head_5": 4,   # coords[4]
        "femur_head_6": 5,   # coords[5]
        "femur_head_7": 6,   # coords[6]
        "femur_head_center": 7,      # coords[7]

        # Femur Joint
        "femur_lateral": 8,       
        "femur_medial": 9,        
        "femur_notch": 10,       

        # Tibia Joint
        "tibia_lateral": 15,      
        "tibia_medial": 16,        
        "tibia_intercondiler": 21,  

        # Ankle 
        "ankle_lateral": 24,        
        "ankle_medial": 25,        
        "ankle_model_middle": 26    
    }

    # femur head
    multi_center = calculate_circle_center([
        coords[labels["femur_head_3"]], 
        coords[labels["femur_head_4"]], 
        coords[labels["femur_head_5"]], 
        coords[labels["femur_head_6"]], 
        coords[labels["femur_head_7"]]
        ])
    femur_head = ((multi_center[0] + 3.0 * coords[labels["femur_head_center"]][0]) / 4.0, (multi_center[1] + 3.0 * coords[labels["femur_head_center"]][1]) / 4.0)
    
    # femur joint
    femur_lateral = (float(coords[labels["femur_lateral"]][0]), float(coords[labels["femur_lateral"]][1]))
    femur_medial = (float(coords[labels["femur_medial"]][0]), float(coords[labels["femur_medial"]][1]))
    femur_ax_middle = ((femur_lateral[0] + femur_medial[0]) / 2.0, (femur_lateral[1] + femur_medial[1]) / 2.0)
    femur_notch = (float(coords[labels["femur_notch"]][0]), float(coords[labels["femur_notch"]][1]))

    # tibia joint
    tibia_lateral = (float(coords[labels["tibia_lateral"]][0]), float(coords[labels["tibia_lateral"]][1]))
    tibia_medial = (float(coords[labels["tibia_medial"]][0]), float(coords[labels["tibia_medial"]][1]))
    tibia_ax_middle = ((tibia_lateral[0] + tibia_medial[0]) / 2.0, (tibia_lateral[1] + tibia_medial[1]) / 2.0)
    tibia_intercondiler = (float(coords[labels["tibia_intercondiler"]][0]), float(coords[labels["tibia_intercondiler"]][1]))

    # ankle
    ankle_lateral = (float(coords[labels["ankle_lateral"]][0]), float(coords[labels["ankle_lateral"]][1]))
    ankle_medial = (float(coords[labels["ankle_medial"]][0]), float(coords[labels["ankle_medial"]][1]))
    ankle_ax_middle = ((ankle_lateral[0] + ankle_medial[0]) / 2.0, (ankle_lateral[1] + ankle_medial[1]) / 2.0)
    ankle_model_middle = (float(coords[labels["ankle_model_middle"]][0]), float(coords[labels["ankle_model_middle"]][1]))
    final_ankle_middle = ((ankle_ax_middle[0] + ankle_model_middle[0]) / 2.0, (ankle_ax_middle[1] + ankle_model_middle[1]) / 2.0)

    # femur angles
    femur_mech_vec_notch = (femur_notch[0] - femur_head[0], femur_notch[1] - femur_head[1])
    femur_joint_vec = (femur_medial[0] - femur_lateral[0], femur_medial[1] - femur_lateral[1])
    femur_mech_angle_notch = calculate_vector_angle(femur_mech_vec_notch, femur_joint_vec)

    # tibia angles
    tibia_mech_vec_inter = (tibia_intercondiler[0] - final_ankle_middle[0], tibia_intercondiler[1] - final_ankle_middle[1])
    tibia_joint_vec = (tibia_medial[0] - tibia_lateral[0], tibia_medial[1] - tibia_lateral[1])
    tibia_mech_angle_inter = calculate_vector_angle(tibia_mech_vec_inter, tibia_joint_vec)

    return {
        # femur head
        "multi_center": multi_center,
        "femur_head": femur_head,
        # femur joint
        "femur_lateral": femur_lateral,
        "femur_medial": femur_medial,
        "femur_notch": femur_notch,
        # tibia joint
        "tibia_lateral": tibia_lateral,
        "tibia_medial": tibia_medial,
        "tibia_intercondiler": tibia_intercondiler,
        # ankle
        "ankle_lateral": ankle_lateral,
        "ankle_medial": ankle_medial,
        "final_ankle_middle": final_ankle_middle,
        # angles
        "femur_mech_angle_notch": femur_mech_angle_notch,
        "tibia_mech_angle_inter": tibia_mech_angle_inter,
    }


def draw_lines(image, metrics: OrthopedicMetrics, coords: List[Point]):
    vis_img = image.copy()
    radius = 1

    def draw_pt(pt: Point, color: Tuple[int, int, int]):
        cv2.circle(vis_img, (int(round(pt[0])), int(round(pt[1]))), radius, color, -1)

    def draw_ln(pt1: Point, pt2: Point, color: Tuple[int, int, int]):
        cv2.line(vis_img, (int(round(pt1[0])), int(round(pt1[1]))), 
                 (int(round(pt2[0])), int(round(pt2[1]))), color, 1)

    def put_text(text: str, pt: Point, color: Tuple[int, int, int]):
        cv2.putText(vis_img, text, (int(round(pt[0])), int(round(pt[1]))), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # femur head
    draw_pt(metrics["femur_head"], (0, 255, 0))
    draw_pt(metrics["multi_center"], (255, 0, 0))

    # femur joint
    draw_pt(metrics["femur_lateral"], (0, 0, 255))
    draw_pt(metrics["femur_medial"], (0, 0, 255))
    draw_ln(metrics["femur_lateral"], metrics["femur_medial"], (0, 255, 0))
    draw_pt(metrics["femur_notch"], (255, 0, 0))

    # tibia joint
    draw_pt(metrics["tibia_lateral"], (0, 0, 255))
    draw_pt(metrics["tibia_medial"], (0, 0, 255))
    draw_ln(metrics["tibia_lateral"], metrics["tibia_medial"], (0, 255, 0))
    draw_pt(metrics["tibia_intercondiler"], (255, 0, 0))

    # ankle
    draw_pt(metrics["ankle_lateral"], (0, 0, 255))
    draw_pt(metrics["ankle_medial"], (0, 0, 255))
    draw_pt(metrics["final_ankle_middle"], (255, 255, 0))

    # LDFA: femur mechanical axis via notch
    draw_ln(metrics["femur_head"], metrics["femur_notch"], (0, 255, 255))

    # MPTA: tibia mechanical axis via intercondylar
    draw_ln(metrics["final_ankle_middle"], metrics["tibia_intercondiler"], (0, 255, 255))

    femur_text_x = metrics["femur_notch"][0] + 8
    femur_text_y = metrics["femur_notch"][1]
    tibia_text_x = metrics["tibia_intercondiler"][0] + 8
    tibia_text_y = metrics["tibia_intercondiler"][1]

    put_text(f"LDFA {metrics['femur_mech_angle_notch']:.2f}", (femur_text_x, femur_text_y), (0, 255, 255))
    put_text(f"MPTA {metrics['tibia_mech_angle_inter']:.2f}", (tibia_text_x, tibia_text_y), (0, 255, 255))
    

    return vis_img


def get_predictions(session, image):
    """
    Raw output from model.
    returns heatmap and offset.
    """
    # Ön İşleme (RGB Dönüşümü ve Normalizasyon)
    if len(image.shape) == 2 or image.shape[2] == 1:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_data = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    input_data = (input_data - mean) / std
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

    # Model Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})

    # Ham Çıktıları Ayır
    heatmap = outputs[0][0]
    offset = outputs[1][0]
    
    return (heatmap, offset)


def visualize_heatmap(heatmap, output_path="heatmap_output.jpg"):
	"""
	visualizes heatmap and saves to target path as .jpg
	"""
	# num_joints = heatmap.shape[0]
	# h_map, w_map = heatmap.shape[1:]

	# Tüm scoremaps'leri tek bir görüntüde birleştir (Maksimum olasılığı alarak)
	combined_heatmap = np.max(heatmap, axis=0)

	# 0-1 arası olasılık değerlerini 0-255 (grayscale) arasına çek
	# Bazı pikseller 1.0'dan büyük olabilir, o yüzden kırpıyoruz.
	heatmap_norm = np.clip(combined_heatmap, 0, 1.0)
	heatmap_gray = (heatmap_norm * 255).astype(np.uint8)

	# Isı haritasına renk katmak için "Jet" colormap uygulayalım
	# Düşük olasılıklar mavi, yüksek olasılıklar kırmızı olur.
	heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)

	cv2.imwrite(output_path, heatmap_color)

	return heatmap_color


def extract_coordinates(heatmap, offset, stride=8, locref_stdev=7.2801, debug=False):
    """
    translates heatmap and offset to x, y coordinates.
    returns a list of 3-tuples
    [xi, yi, confidence_i]
    """
    num_joints = heatmap.shape[0]
    predictions = []

    for i in range(num_joints):
        # Orijinal heatmap'i bozmamak için her eklemi ayrı bir değişkene alıyoruz
        joint_heatmap = heatmap[i, :, :]
        
        # 1. En yüksek olasılıklı hücreyi (bin) bul
        _, confidence, _, max_loc = cv2.minMaxLoc(joint_heatmap)
        x_map, y_map = max_loc

        # 2. Temel Konum (Stride ile çarpma)
        x_base = (x_map + 0.5) * stride
        y_base = (y_map + 0.5) * stride
        
        # 3. Hassas Düzeltme (Locref ofsetleri)
        # DeepLabCut offset formatı: [x0, y0, x1, y1, ...]
        offset_x = offset[i * 2, y_map, x_map] * locref_stdev
        offset_y = offset[i * 2 + 1, y_map, x_map] * locref_stdev
        
        x_final = x_base + offset_x
        y_final = y_base + offset_y

        if debug:
            x_base_deb = (x_map+0.5) * stride
            y_base_deb = (y_map+0.5) * stride

            offset_x_deb = offset[i * 2, y_map, x_map]
            offset_y_deb = offset[i * 2 + 1, y_map, x_map]
            
            x_final_deb = x_base_deb + offset_x_deb
            y_final_deb = y_base_deb + offset_y_deb

            predictions.append((x_final_deb, y_final_deb, confidence))
        else:
            predictions.append((x_final, y_final, confidence))
        
    return predictions


def visualize_predictions(image, coords, stride=8, threshold=0.5, grid=False):
    """
    Draw predicted points onto image
    """
    vis_img = image.copy()
    h, w = vis_img.shape[:2]
    
    if grid:
        for x in range(stride, w, stride):
            cv2.line(vis_img, (x, 0), (x, h), (100, 100, 100), 1)
            
        for y in range(stride, h, stride):
            cv2.line(vis_img, (0, y), (w, y), (100, 100, 100), 1)

    for i, (x, y, prob) in enumerate(coords):
        if prob > threshold:
            ix, iy = int(round(x)), int(round(y))
            
            grid_x = (ix // stride) * stride
            grid_y = (iy // stride) * stride
            
            if grid:
                cv2.rectangle(vis_img, (grid_x, grid_y), 
                            (grid_x + stride, grid_y + stride), 
                            (0, 165, 255), 1)

            radius = 1
            cv2.circle(vis_img, (ix, iy), radius, (0, 255, 0), -1)
            cv2.putText(vis_img, str(i), (ix + 4, iy - 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            
    return vis_img


def infer_single_image(onnx_path, image_path, output_path="output.jpg"):
    
	"""
    full inference pipeline.
	loads model, gets raw model predictions, translates them into coordinates,
    draws predicted points on image and saves as a new .jpg file
	"""

	# Oturumu bir kez başlat (Döngü dışı olması performans için kritiktir)
	providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
	session = ort.InferenceSession(onnx_path, providers=providers)

	# Resmi oku
	img = cv2.imread(image_path)
	if img is None:
		print("Resim bulunamadı!")
		return

	# Adım 1: Tahminleri al (Çıkarım)
	heatmap, offset = get_predictions(session, img)

	coords = extract_coordinates(heatmap, offset)

	# Adım 2: Tahminleri çiz (Görselleştirme)
	result_img = visualize_predictions(img, coords, threshold=0.0)

	# Kaydet
	cv2.imwrite(output_path, result_img)
	print(f"İşlem tamam! {output_path} kaydedildi.")

	return heatmap, offset, coords


def infer_images(
        onnx_path, 
        input_folder, 
        output_folder, 
        inference_json_name="inference_results.json", 
        metrics_json_name="orthopedic_metrics.json", 
        landmarks=False
    ):
    session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [str(p) for p in Path(input_folder).iterdir() if p.suffix.lower() in valid_extensions]
    image_paths.sort()

    all_results = {}
    all_metrics = {}

    for img_path_str in image_paths:
        img = cv2.imread(img_path_str)
        if img is None: 
            continue

        heatmap, offset = get_predictions(session, img)
        coords = extract_coordinates(heatmap, offset)
        
        metrics = compute_orthopedic_metrics(coords)

        img_name = os.path.basename(img_path_str)
        image_data = []
        
        for i, (x, y, conf) in enumerate(coords):
            image_data.append({
                "joint_id": i,
                "x": float(x),
                "y": float(y),
                "conf": float(conf)
            })

        all_results[img_name] = image_data
        all_metrics[img_name] = metrics

        if landmarks:
            result_img = visualize_predictions(img, coords, threshold=0.0)
        else:
            result_img = draw_lines(img, metrics, coords)
            
        cv2.imwrite(os.path.join(output_folder, img_name), result_img)
        print(f"Processed: {img_name}")

    with open(os.path.join(output_folder, inference_json_name), 'w') as f:
        json.dump(all_results, f)

    with open(os.path.join(output_folder, metrics_json_name), 'w') as f:
        json.dump(all_metrics, f)

    print(f"Finished. Saved to: {output_folder}")



def calculate_center(points):
    """
    Calculates the geometric center (centroid) of a list of (x, y) points.
    """
    if not points:
        return None
    
    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    
    center_x = sum_x / len(points)
    center_y = sum_y / len(points)
    
    return (center_x, center_y)




# -- FLOW --
#
#
# ┌──────────────────────────────────────────────────────────────────┐
# │                        GET_PREDICTIONS                           │
# └───────────────────────────────┬──────────────────────────────────┘
#                                 │
#                                 v
# ┌──────────────────────────────────────────────────────────────────┐
# │        EXTRACT_COORDINATES (Heatmap ➔ Coord. [x, y])            │
# └───────────────────────────────┬──────────────────────────────────┘
#                                 │
#                                 v
# ┌──────────────────────────────────────────────────────────────────┐
# │                   COMPUTE_ORTHOPEDIC_METRICS                     │
# └───────────────┬───────────────────────────────┬──────────────────┘
#                 │                               │
#                 v                               v
# ┌───────────────────────────────┐     ┌────────────────────────────┐
# │         DRAW_LINES            |     │   VISUALIZE_PREDICTIONS    │
# └───────────────┬───────────────┘     └───────────────┬────────────┘
