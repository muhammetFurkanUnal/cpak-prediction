"""
dev-gui FastAPI sunucusu — salt-okuma model çıktı gözden geçirici.

Yalnızca standart kütüphane + FastAPI/uvicorn kullanır. torch / onnxruntime /
cv2 / numpy KESİNLİKLE import edilmez; bu yüzden hafif ve hızlı başlar. Görüntüler
diskten ``FileResponse`` ile talep üzerine akıtılır, belleğe yüklenmez.

Mevcut canlı-inference API'sinden (``api/main.py``, port 8000) tamamen ayrıdır;
bu uygulama port 8050'de çalışır ve hiçbir dosyayı değiştirmez.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

import paths
from indexer import build_index

app = FastAPI(title="cpak dev-gui", description="Model çıktı gözden geçirici (salt-okuma).", version="0.1.0")

# İndeks açılışta bir kez kurulur ve bellekte tutulur.
INDEX = build_index()

app.mount("/static", StaticFiles(directory=str(paths.STATIC_DIR)), name="static")

_PNG_CACHE = {"Cache-Control": "public, max-age=86400"}


def _png(path: str) -> FileResponse:
    return FileResponse(path, media_type="image/png", headers=_PNG_CACHE)


# ── Sayfa ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    html = (paths.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    return {"status": "ok", **INDEX["stats"]}


# ── Veri (JSON) ─────────────────────────────────────────────────────────────
@app.get("/api/models")
def api_models():
    """Mevcut modeller + provenance (her modelin beslendiği dosyalar)."""
    return INDEX["models"]


@app.get("/api/images")
def api_images():
    """Grid için hafif liste — görüntü baytı veya metrik içermez."""
    return INDEX["image_list"]


@app.get("/api/images/{key}")
def api_image(key: str):
    """Tek görüntünün tam kaydı: ground-truth + model-başı çıktılar."""
    rec = INDEX["images"].get(key)
    if rec is None:
        raise HTTPException(status_code=404, detail="Görüntü bulunamadı.")
    return rec


@app.get("/api/models/{model}/images")
def api_model_images(model: str):
    """Bu modelin çıktı ürettiği anahtarlar (görüntü-görüntü gezinme için)."""
    lst = INDEX["model_images"].get(model)
    if lst is None:
        raise HTTPException(status_code=404, detail="Model bulunamadı.")
    return lst


@app.get("/api/models/{model}/test")
def api_model_test(model: str):
    """Test analizi paketi; yoksa nerede arandığını açıkça bildirir."""
    bundle = INDEX["tests"].get(model)
    if bundle is not None:
        return bundle
    return {"has_test": False, "looked_at": f"notebooks/out/test/{model}/"}


# ── Görüntüler (diskten akıtılır) ───────────────────────────────────────────
@app.get("/img/source/{key}")
def img_source(key: str):
    """Kaynak (ön-işlenmiş) görüntü."""
    path = INDEX["source_png"].get(key)
    if path is None:
        raise HTTPException(status_code=404, detail="Kaynak görüntü yok.")
    return _png(path)


@app.get("/img/inference/{model}/{key}")
def img_inference(model: str, key: str):
    """Bir modelin annotasyonlu (çizilmiş) çıktı görüntüsü."""
    path = INDEX["inference_png"].get(model, {}).get(key)
    if path is None:
        raise HTTPException(status_code=404, detail="Bu model için çıktı görüntüsü yok.")
    return _png(path)


_ALLOWED_CHARTS = {"distribution_plot", "femur_comparison_graphs", "tibia_comparison_graphs"}


@app.get("/img/test/{model}/{chart}")
def img_test(model: str, chart: str):
    """Test analizi grafiği (sabit beyaz liste)."""
    if chart not in _ALLOWED_CHARTS:
        raise HTTPException(status_code=404, detail="Bilinmeyen grafik.")
    path = paths.TEST_DIR / model / f"{chart}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Grafik yok.")
    return _png(str(path))
