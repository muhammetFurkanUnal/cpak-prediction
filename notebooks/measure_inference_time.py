"""
Measure inference latency of the production landmark models.

Covers both production ONNX models:
  - CPAK    : sh14-ep1000.onnx        (ResNet-50, 27 keypoints, full-leg 1536x640)
  - kneeAP  : kneeap-sh4-ep1000.onnx  (ResNet-50, 34 keypoints, knee AP 464x267)

Times the AI-measurement path defined in the paper: from image (already loaded)
to angle output, server side only (no network). Per-image breakdown:
preprocess+model, model-only (session.run), coordinate extraction, and
orthopedic/knee-metric computation. Also reports end-to-end including image read.

Outputs (per model, under notebooks/out/inference/timing/):
  - <key>_inference_timing.json : config + per-image samples + aggregate stats
  - <key>_inference_timing.md   : human-readable summary table

Run:  .venv/bin/python notebooks/measure_inference_time.py
"""
import os
import sys
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "lib"))
import cpak_inference as cpak_lib      # noqa: E402
import kneeap_inference as kneeap_lib  # noqa: E402

OUT_DIR = ROOT / "notebooks/out/inference/timing"
WARMUP = 5          # discarded runs to trigger graph optimisation / provider warmup
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

MODELS = [
    {
        "key": "cpak",
        "onnx": ROOT / "notebooks/out/models/sh14-ep1000.onnx",
        "input_dir": ROOT / "preprocessing/long-id-split-prepped",
        "compute": cpak_lib.compute_orthopedic_metrics,
        "model_type": "DeepLabCut ResNet-50 (CPAK full-leg landmark model, 27 keypoints)",
        "input_note": (
            "94 single-leg full-length radiographs at production size 1536x640. The "
            "exact 47-image held-out test split is not persisted in the repo, so this "
            "measures per-image inference latency over the full single-leg set "
            "(representative; latency is size-driven, not split-driven)."
        ),
        "title": "CPAK Modeli Inference Süresi Ölçümü",
        "subtitle": "tek-bacak tam-boy grafisi",
    },
    {
        "key": "kneeap",
        "onnx": ROOT / "notebooks/out/models/kneeap-sh4-ep1000.onnx",
        "input_dir": ROOT / "kneeap-furkan-2026-04-29/labeled-data/video",
        "compute": kneeap_lib.compute_kneeap_metrics,
        "model_type": "DeepLabCut ResNet-50 (knee-AP landmark model, 34 keypoints)",
        "input_note": (
            "194 knee-AP radiographs at production size 464x267 (kneeap-sh4-ep1000 is "
            "the production shuffle per scripts/cpak_predict_to_csv.py). Measures "
            "per-image inference latency over the full knee-AP set."
        ),
        "title": "kneeAP Modeli Inference Süresi Ölçümü",
        "subtitle": "diz AP grafisi",
    },
]


def _preprocess(image):
    """Mirror get_predictions preprocessing (RGB + ImageNet norm). Same in both libs."""
    if len(image.shape) == 2 or image.shape[2] == 1:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    x = (x - mean) / std
    return np.expand_dims(x, axis=0).astype(np.float32)


def _agg(samples):
    s = sorted(samples)
    n = len(s)
    return {
        "n": n,
        "mean_ms": round(statistics.mean(s) * 1000, 3),
        "std_ms": round((statistics.stdev(s) if n > 1 else 0.0) * 1000, 3),
        "median_ms": round(statistics.median(s) * 1000, 3),
        "min_ms": round(s[0] * 1000, 3),
        "max_ms": round(s[-1] * 1000, 3),
        "p95_ms": round(s[min(n - 1, int(round(0.95 * (n - 1))))] * 1000, 3),
    }


def measure_model(cfg, hardware):
    if not cfg["onnx"].exists():
        print(f"  SKIP {cfg['key']}: model not found ({cfg['onnx']})")
        return None

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(cfg["onnx"]), providers=providers)
    active_providers = session.get_providers()
    input_name = session.get_inputs()[0].name
    extract = kneeap_lib.extract_coordinates  # identical impl to cpak's

    image_paths = sorted(
        str(p) for p in cfg["input_dir"].iterdir() if p.suffix.lower() in VALID_EXT
    )
    if not image_paths:
        print(f"  SKIP {cfg['key']}: no images in {cfg['input_dir']}")
        return None

    warm_x = _preprocess(cv2.imread(image_paths[0]))
    for _ in range(WARMUP):
        session.run(None, {input_name: warm_x})

    per_image = []
    t_read, t_pre, t_model, t_infer, t_extract, t_metrics, t_pipeline, t_e2e = (
        [], [], [], [], [], [], [], [])

    for path in image_paths:
        name = os.path.basename(path)
        c0 = perf_counter()
        img = cv2.imread(path)
        c1 = perf_counter()
        if img is None:
            continue
        x = _preprocess(img)
        c2 = perf_counter()
        outputs = session.run(None, {input_name: x})
        c3 = perf_counter()
        heatmap, offset = outputs[0][0], outputs[1][0]
        coords = extract(heatmap, offset)
        c4 = perf_counter()
        _ = cfg["compute"](coords)
        c5 = perf_counter()

        read, pre, model = c1 - c0, c2 - c1, c3 - c2
        ext, met = c4 - c3, c5 - c4
        infer = pre + model
        pipeline = pre + model + ext + met
        e2e = read + pipeline

        t_read.append(read); t_pre.append(pre); t_model.append(model)
        t_infer.append(infer); t_extract.append(ext); t_metrics.append(met)
        t_pipeline.append(pipeline); t_e2e.append(e2e)

        per_image.append({
            "image": name, "shape": list(img.shape),
            "read_ms": round(read * 1000, 3), "preprocess_ms": round(pre * 1000, 3),
            "model_ms": round(model * 1000, 3), "extract_ms": round(ext * 1000, 3),
            "metrics_ms": round(met * 1000, 3), "pipeline_ms": round(pipeline * 1000, 3),
            "e2e_ms": round(e2e * 1000, 3),
        })

    stages = {
        "read": _agg(t_read),
        "preprocess": _agg(t_pre),
        "model_only": _agg(t_model),
        "get_predictions": _agg(t_infer),
        "extract_coordinates": _agg(t_extract),
        "compute_metrics": _agg(t_metrics),
        "pipeline_ai_inference": _agg(t_pipeline),
        "end_to_end_incl_read": _agg(t_e2e),
    }

    result = {
        "model": cfg["onnx"].name,
        "model_type": cfg["model_type"],
        "input_dir": str(cfg["input_dir"].relative_to(ROOT)),
        "input_note": cfg["input_note"],
        "n_images": len(per_image),
        "warmup_runs": WARMUP,
        "requested_providers": providers,
        "active_providers": active_providers,
        "hardware": hardware,
        "hardware_note": (
            "Measured on this machine (Apple Silicon, no CUDA GPU). CUDA was requested "
            "but unavailable, so onnxruntime fell back to CPU. Production server uses "
            "CUDAExecutionProvider on GPU; those numbers would differ."
        ),
        "definitions": {
            "pipeline_ai_inference": "preprocess + model + extract_coordinates + compute_metrics (server-side, excludes disk read and network)",
            "end_to_end_incl_read": "pipeline + image read from disk",
            "model_only": "onnxruntime session.run only",
        },
        "stages_ms": stages,
        "per_image": per_image,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{cfg['key']}_inference_timing.json", "w") as f:
        json.dump(result, f, indent=2)

    pa, e2e, mo = (stages["pipeline_ai_inference"],
                   stages["end_to_end_incl_read"], stages["model_only"])
    md = f"""# {cfg['title']}

**Model:** `{cfg['onnx'].name}` — {cfg['model_type']}
**Görüntü sayısı:** {result['n_images']} {cfg['subtitle']} ({result['input_dir']})
**Warm-up:** {WARMUP} çalıştırma (ölçüme dahil değil)
**Aktif sağlayıcı:** {', '.join(active_providers)}
**Donanım:** {hardware['platform']} — {hardware['machine']}, onnxruntime {ort.__version__}

> ⚠️ Bu ölçüm bu makinede (Apple Silicon, CUDA yok) yapılmıştır; onnxruntime CPU'ya
> düşmüştür. Production sunucusu CUDA GPU kullanır ve süreleri farklı olur.

## Özet (görüntü başına)

| Aşama | Ortalama (ms) | Std (ms) | Medyan (ms) | Min (ms) | Maks (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Model (session.run) | {mo['mean_ms']} | {mo['std_ms']} | {mo['median_ms']} | {mo['min_ms']} | {mo['max_ms']} | {mo['p95_ms']} |
| **AI inference (server)** | **{pa['mean_ms']}** | **{pa['std_ms']}** | **{pa['median_ms']}** | **{pa['min_ms']}** | **{pa['max_ms']}** | **{pa['p95_ms']}** |
| Uçtan uca (+disk okuma) | {e2e['mean_ms']} | {e2e['std_ms']} | {e2e['median_ms']} | {e2e['min_ms']} | {e2e['max_ms']} | {e2e['p95_ms']} |

**AI inference süresi = {pa['mean_ms'] / 1000:.3f} ± {pa['std_ms'] / 1000:.3f} saniye/görüntü**
(ön işleme + model + koordinat çıkarımı + açı hesabı; disk okuma ve ağ hariç).

## Aşama kırılımı (ortalama, ms)

| Aşama | Ortalama (ms) |
|---|---:|
| Disk okuma (imread) | {stages['read']['mean_ms']} |
| Ön işleme | {stages['preprocess']['mean_ms']} |
| Model (session.run) | {stages['model_only']['mean_ms']} |
| Koordinat çıkarımı | {stages['extract_coordinates']['mean_ms']} |
| Açı/metrik hesabı | {stages['compute_metrics']['mean_ms']} |

Ham veriler: `{cfg['key']}_inference_timing.json`.
"""
    with open(OUT_DIR / f"{cfg['key']}_inference_timing.md", "w") as f:
        f.write(md)

    print(f"[{cfg['key']}] providers={active_providers} n={result['n_images']} "
          f"AI={pa['mean_ms']}±{pa['std_ms']}ms ({pa['mean_ms']/1000:.3f}s) "
          f"model-only={mo['mean_ms']}ms e2e={e2e['mean_ms']}ms")
    return result


def main():
    hardware = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
    }
    results = {}
    for cfg in MODELS:
        r = measure_model(cfg, hardware)
        if r:
            results[cfg["key"]] = {
                "model": r["model"],
                "n_images": r["n_images"],
                "ai_inference_ms": r["stages_ms"]["pipeline_ai_inference"],
                "model_only_ms": r["stages_ms"]["model_only"],
            }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({"hardware": hardware, "models": results}, f, indent=2)
    print(f"Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
