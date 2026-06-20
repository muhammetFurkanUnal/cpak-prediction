"""
Tüm dataset yollarının tek kaynağı (dev-gui).

Buradaki sabitler, uygulamanın veriyi diskte tam olarak nereden okuduğunu
gösterir. Yeni bir kaynak klasör veya model eklenince yalnızca burası ve
``registry.py`` güncellenir.
"""

from pathlib import Path

# dev-gui/ klasörünün bir üstü = proje kökü.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Ground truth ────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
GROUND_TRUTH_CSV = DATA_DIR / "cpak-grnd-truth.csv"

# ── Kaynak (ön-işlenmiş) görüntüler ─────────────────────────────────────────
# Hangi klasörün kullanılacağı (grafi, id-sınıfı) ile belirlenir:
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
SRC_FULLLEG_4XXX = PREPROCESSING_DIR / "preop-single-prepped"    # 4xxx,    grafi 1 (tüm bacak)
SRC_KNEE_4XXX = PREPROCESSING_DIR / "knee-preop-prepped"         # 4xxx,    grafi 2 (diz AP)
SRC_FULLLEG_LONG = PREPROCESSING_DIR / "long-id-split-prepped"   # uzun-id, grafi 1 (tüm bacak)

# ── Model çıktıları ─────────────────────────────────────────────────────────
INFERENCE_DIR = PROJECT_ROOT / "notebooks" / "out" / "inference"   # <model>/<pid>.<side>.png + metrik json'ları
TEST_DIR = PROJECT_ROOT / "notebooks" / "out" / "test"             # <model>/ (yalnız sh14'te var)
WEIGHTS_PT_DIR = PROJECT_ROOT / "notebooks" / "models"             # <model>/snapshot-1000.pt
WEIGHTS_ONNX_DIR = PROJECT_ROOT / "notebooks" / "out" / "models"   # <model>.onnx

# ── Ön yüz ──────────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).resolve().parent / "static"
