# CPAK — DeepLabCut Landmark Detection

Anatomical landmark detection on X-ray images using DeepLabCut (PyTorch backend).

## Requirements

- Python 3.12
- CUDA 12.1 compatible GPU
- ~3 GB disk space for dataset + model weights

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd cpak
```

### 2. Create a virtual environment and install dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Create a DeepLabCut project using the GUI

Launch the DeepLabCut GUI:

```bash
python -m deeplabcut
```

In the GUI:

1. Click **"Create New Project"**
2. Fill in the fields exactly as follows:
   - **Project name:** `cpak`
   - **Experimenter name:** `furkan`
   - **Working directory:** the `cpak/` folder you just cloned
3. Under **"Add videos"**, add any placeholder video file
4. Click **"Create"**

This will create a folder named `cpak-furkan-<today's date>/` inside the repo directory.

### 4. Replace the generated config.yaml

Copy the project config from `assets/` into the newly created project folder:

```bash
cp assets/config.yaml cpak-furkan-<date>/config.yaml
```

Then open `cpak-furkan-<date>/config.yaml` and update the two machine-specific fields:

```yaml
# Update this to match your actual project folder path:
project_path: /your/absolute/path/to/cpak/cpak-furkan-<date>

# Update this key to match the same project folder path:
video_sets:
  /your/absolute/path/to/cpak/cpak-furkan-<date>/../videos/video.mp4:
    crop: 0, 1008, 0, 4784
```

> The `video.mp4` file does not need to exist. Only the filename stem (`video`) matters — it determines which subfolder under `labeled-data/` DeepLabCut reads.

### 5. Download the dataset

Download the labeled image dataset from the link below and extract it:

**[https://drive.google.com/drive/folders/1rDl57FmFnuq-J0RgA1oLiwDOGc5kTwX1](#)**

Place the images in:

```
cpak-furkan-<date>/
  labeled-data/
    video/
      4000.l.png
      4000.r.png
      ...  (188 images total)
```

### 6. Copy the label file

```bash
cp assets/CollectedData_furkan.csv cpak-furkan-<date>/labeled-data/video/
```

### 7. Generate the H5 label file

DeepLabCut requires labels in `.h5` format. Generate it from the CSV:

```bash
python scripts/csv_to_h5.py /absolute/path/to/cpak-furkan-<date>/labeled-data/video/CollectedData_furkan.csv
```

You should see:

```
[OK] /absolute/path/to/cpak-furkan-<date>/labeled-data/video/CollectedData_furkan.h5  (188 frame)
```

### 8. Create the training dataset

In the GUI, go to the **"Create Training Dataset"** tab and click **"Create Training Dataset"**.

### 9. Train

In the GUI, go to the **"Train Network"** tab and click **"Train Network"**.

Training progress is saved under `cpak-furkan-<date>/dlc-models-pytorch/`.

---

## Evaluate

In the GUI, go to the **"Evaluate Network"** tab and click **"Evaluate Network"**.

---

## Inference

DeepLabCut is only used for training. Once training is complete, the model is exported to ONNX and all subsequent inference runs through custom code — no DeepLabCut dependency needed.

### 1. Locate the training outputs

After training, find these two files under the project folder:

```
cpak-furkan-<date>/dlc-models-pytorch/iteration-0/<shuffle>/train/
  ├── pytorch_config.yaml   ← model architecture
  └── snapshot-best-*.pt    ← trained weights
```

### 2. Export to ONNX

Open `notebooks/export-model.ipynb` and set the paths at the top of the notebook:

```python
pytorch_config_path = "/absolute/path/to/.../train/pytorch_config.yaml"
pt_path             = "/absolute/path/to/.../train/snapshot-best-*.pt"
onnx_file_path      = "notebooks/out/models/model.onnx"
```

Run all cells. This produces a standalone `model.onnx` file that no longer requires DeepLabCut or PyTorch to run.

### 3. Run inference

Open `notebooks/inference.ipynb` and set:

```python
onnx_path         = "notebooks/out/models/model.onnx"
input_folder_path = "/path/to/images/"
output_path       = "notebooks/out/inference/"
```

Run all cells. Outputs written to `output_path`:

| File | Contents |
|---|---|
| `inference_results.json` | Per-image joint coordinates (x, y, confidence) for all 27 keypoints |
| `orthopedic_metrics.json` | Computed femur / tibia / ankle mechanical angles |
| `*.jpg` | Input images with anatomical axes drawn on top |

### 4. Test against ground truth

Open `notebooks/test.ipynb` and set:

```python
truth_json_path     = "/path/to/ground_truth_angles.json"
inference_json_path = "notebooks/out/inference/orthopedic_metrics.json"
output_folder       = "notebooks/out/test/"
```

Ground truth JSON format:

```json
{
  "4075.l": { "femur": "86.34", "tibia": "83.32" },
  "4000.r": { "femur": "91.12", "tibia": "88.45" }
}
```

Run all cells. Outputs written to `output_folder`:

| File | Contents |
|---|---|
| `angles.csv` | Predicted vs. ground truth angles, per-image errors |
| `metrics.json` | MAE, RMSE, R² for femur and tibia (axial and notch/intercondylar methods) |
| `femur_comparison_graphs.png` | Regression plots and error distribution |
| `tibia_comparison_graphs.png` | Regression plots and error distribution |
| `distribution.json` | Sample counts grouped by error range (0–0.5°, 0.5–1.0°, …) |
| `distribution_plot.png` | Bar chart of error distribution |

---

## Project structure

```
cpak/
├── assets/
│   ├── config.yaml                  # DLC project config template
│   └── CollectedData_furkan.csv     # Landmark labels (188 frames × 27 keypoints)
├── notebooks/
│   ├── lib/inference.py             # Inference and metric computation library
│   ├── export-model.ipynb           # Export trained .pt → .onnx
│   ├── inference.ipynb              # Batch inference: images → coordinates + angles
│   └── test.ipynb                   # Evaluate predictions against ground truth
├── scripts/
│   └── csv_to_h5.py                 # Converts labels CSV → H5 (required by DLC)
├── preprocessing/
│   ├── methods.py
│   └── preprocess.ipynb
└── requirements.txt
```

Files generated locally (not in the repository):

```
cpak-furkan-<date>/
├── labeled-data/video/              # images (downloaded from Drive) + label files
├── training-datasets/               # generated by create_training_dataset()
└── dlc-models-pytorch/              # model snapshots generated by train_network()
```
