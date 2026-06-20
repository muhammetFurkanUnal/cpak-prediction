"""
Veri indeksi: açılışta bir kez kurulur.

CSV ground-truth'u, kaynak (ön-işlenmiş) görüntüleri, her modelin inference
çıktılarını ve (varsa) test analizini tarar; hepsini ``(pid, side, grafi)``
bileşik anahtarında birleştirir. Hiçbir eksik veri uydurulmaz; bir şey diskte
yoksa ilgili alan None/boş kalır.

İndeks tamamen küçük skalerlerden ve yol string'lerinden oluşur (birkaç yüz
kayıt) → bellek tek haneli MB düzeyinde kalır. Görüntü baytları indekste
TUTULMAZ; sunucu onları diskten talep üzerine akıtır.
"""

import csv
import json
from pathlib import Path

import keys
import paths
from registry import GRAFI_KIND, GRAFI_LABEL, KIND_GRAFI, KIND_LABEL, MODELS


# ── Küçük yardımcılar ───────────────────────────────────────────────────────
def _rel(p) -> str:
    """Yolu proje köküne göre okunaklı biçimde göster (provenance için)."""
    try:
        return str(Path(p).resolve().relative_to(paths.PROJECT_ROOT))
    except Exception:
        return str(p)


def _num(v):
    """Yalnızca sayı ise döndür; liste/None/str ise None (açı alanları skalerdir)."""
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _cell(row, i):
    return row[i].strip() if i < len(row) else ""


def _cell_num(row, i):
    s = _cell(row, i)
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def _default_display(pid: str, side: str) -> str:
    return f"{pid}{side.upper()}"


# ── İndeks kurulumu ─────────────────────────────────────────────────────────
def build_index() -> dict:
    images: dict[str, dict] = {}          # ckey -> tam kayıt
    source_png: dict[str, str] = {}       # ckey -> kaynak görüntü yolu
    inference_png: dict[str, dict] = {}   # model -> {ckey -> annotasyonlu png yolu}

    # 1) Kaynak (ön-işlenmiş) görüntüler. Klasör (grafi, id-sınıfı) ile bellidir.
    for src_dir, grafi in (
        (paths.SRC_FULLLEG_4XXX, "1"),
        (paths.SRC_FULLLEG_LONG, "1"),
        (paths.SRC_KNEE_4XXX, "2"),
    ):
        if not src_dir.is_dir():
            continue
        for p in src_dir.glob("*.png"):
            k = keys.img_to_key(p.name)
            if not k:
                continue
            source_png.setdefault(keys.make_key(k[0], k[1], grafi), str(p))

    def ensure(ckey, pid, side, grafi) -> dict:
        rec = images.get(ckey)
        if rec is None:
            rec = {
                "key": ckey,
                "pid": pid,
                "side": side,
                "grafi": grafi,
                "id_class": "short" if keys.is_4xxx(pid) else "long",
                "display_id": _default_display(pid, side),
                "graphic_label": GRAFI_LABEL.get(grafi, grafi),
                "expected_model_kind": GRAFI_KIND.get(grafi),
                "has_source": ckey in source_png,
                "ground_truth": [],
                "outputs": {},
            }
            images[ckey] = rec
        return rec

    # 2) Modeller: inference çıktıları + görüntü-başı metrikler.
    models_info = []
    for cfg in MODELS:
        name, kind = cfg["name"], cfg["kind"]
        grafi = KIND_GRAFI[kind]
        inf_dir = paths.INFERENCE_DIR / name

        if not inf_dir.is_dir():
            models_info.append({
                "name": name, "kind": kind, "grafi": grafi,
                "graphic_label": KIND_LABEL[kind],
                "experimental": cfg["experimental"], "available": False,
                "output_count": 0, "has_test": False,
                "weights_present": False, "weights_path": None,
                "inference_dir": _rel(inf_dir), "metric_files": [],
                "angle_source": None, "test_dir": None,
            })
            continue

        metric_files = sorted(p.name for p in inf_dir.glob("*.json"))

        # Annotasyonlu çıktı PNG'leri → ckey eşlemesi.
        pngmap: dict[str, str] = {}
        for p in inf_dir.glob("*.png"):
            k = keys.img_to_key(p.name)
            if not k:
                continue
            ckey = keys.make_key(k[0], k[1], grafi)
            pngmap[ckey] = str(p)
            ensure(ckey, k[0], k[1], grafi)["outputs"].setdefault(name, {})["has_png"] = True
        inference_png[name] = pngmap

        # Görüntü-başı metrikler (yalnız bildirilen primary_metrics dosyasından).
        pm = inf_dir / cfg["primary_metrics"]
        raw = json.loads(pm.read_text(encoding="utf-8")) if pm.exists() else {}
        amap, emap = cfg["angle_map"], cfg["extra_map"]
        for fname, val in raw.items():
            k = keys.img_to_key(fname)
            if not k or not isinstance(val, dict):
                continue
            ckey = keys.make_key(k[0], k[1], grafi)
            out = ensure(ckey, k[0], k[1], grafi)["outputs"].setdefault(name, {})
            out.setdefault("has_png", ckey in pngmap)

            mpta = _num(val.get(amap.get("mpta"))) if amap else None
            ldfa = _num(val.get(amap.get("ldfa"))) if amap else None
            if mpta is not None and ldfa is not None:
                d = keys.derive(mpta, ldfa)
                out.update({
                    "mpta": d["mpta_raw"], "ldfa": d["ldfa_raw"],
                    "diff": d["diff"], "sum": d["sum"],
                    "dizilim": d["dizilim"], "eklem": d["eklem"], "siniflama": d["siniflama"],
                })
            else:
                out["mpta"], out["ldfa"] = None, None

            extra = {}
            for label, field in emap.items():
                v = _num(val.get(field))
                if v is not None:
                    extra[label] = round(v, 2)
            if extra:
                out["extra"] = extra

        # Model meta + provenance.
        has_test = (paths.TEST_DIR / name / "metrics.json").exists()
        onnx = paths.WEIGHTS_ONNX_DIR / f"{name}.onnx"
        pt_dir = paths.WEIGHTS_PT_DIR / name
        weights = onnx if onnx.exists() else (pt_dir if pt_dir.is_dir() else None)
        models_info.append({
            "name": name, "kind": kind, "grafi": grafi,
            "graphic_label": KIND_LABEL[kind],
            "experimental": cfg["experimental"], "available": True,
            "output_count": len(pngmap),
            "has_test": has_test,
            "weights_present": weights is not None,
            "weights_path": _rel(weights) if weights else None,
            "inference_dir": _rel(inf_dir),
            "metric_files": metric_files,
            "angle_source": cfg["primary_metrics"] if amap else None,
            "test_dir": _rel(paths.TEST_DIR / name) if has_test else None,
        })

    # 3) CSV ground-truth (dataset tanımı). Aynı anahtara birden çok satır
    #    olabileceği için liste tutulur (496 satır, 495 benzersiz).
    if paths.GROUND_TRUTH_CSV.exists():
        with open(paths.GROUND_TRUTH_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        for row in rows[1:]:
            if not row:
                continue
            k = keys.hasta_to_key(_cell(row, 0))
            grafi = _cell(row, 1)
            if not k or grafi not in ("1", "2"):
                continue
            ckey = keys.make_key(k[0], k[1], grafi)
            rec = ensure(ckey, k[0], k[1], grafi)
            rec["display_id"] = _cell(row, 0) or rec["display_id"]
            rec["ground_truth"].append({
                "mpta": _cell_num(row, 2), "ldfa": _cell_num(row, 3),
                "diff": _cell_num(row, 4), "sum": _cell_num(row, 5),
                "dizilim": _cell(row, 6) or None,
                "eklem": _cell(row, 7) or None,
                "siniflama": _cell(row, 8) or None,
            })

    # 4) Sıralı listeler (grid + model-içi gezinme için).
    def _sort_key(ckey):
        pid, side, grafi = keys.parse_key(ckey)
        return (grafi, 0 if keys.is_4xxx(pid) else 1, int(pid), side)

    ordered = sorted(images.keys(), key=_sort_key)
    image_list = [{
        "key": ck,
        "display_id": images[ck]["display_id"],
        "pid": images[ck]["pid"], "side": images[ck]["side"], "grafi": images[ck]["grafi"],
        "graphic_label": images[ck]["graphic_label"],
        "id_class": images[ck]["id_class"],
        "expected_model_kind": images[ck]["expected_model_kind"],
        "has_source": images[ck]["has_source"],
        "has_gt": len(images[ck]["ground_truth"]) > 0,
        "models_with_output": sorted(images[ck]["outputs"].keys()),
    } for ck in ordered]

    model_images = {}
    for cfg in MODELS:
        name = cfg["name"]
        model_images[name] = [{
            "key": ck,
            "display_id": images[ck]["display_id"],
            "has_png": images[ck]["outputs"].get(name, {}).get("has_png", False),
        } for ck in ordered if name in images[ck]["outputs"]]

    # 5) Test paketleri (yalnız metrics.json bulunan modeller — şu an sh14).
    tests = {}
    for cfg in MODELS:
        name = cfg["name"]
        tdir = paths.TEST_DIR / name
        if not (tdir / "metrics.json").exists():
            continue
        bundle = {"has_test": True, "test_dir": _rel(tdir)}
        bundle["metrics"] = json.loads((tdir / "metrics.json").read_text(encoding="utf-8"))
        dist_p = tdir / "distribution.json"
        bundle["distribution"] = json.loads(dist_p.read_text(encoding="utf-8")) if dist_p.exists() else None

        per = {}
        ang = tdir / "angles.csv"
        if ang.exists():
            grafi = KIND_GRAFI[cfg["kind"]]
            with open(ang, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    k = keys.img_to_key(row.get("id", ""))
                    if not k:
                        continue
                    per[keys.make_key(k[0], k[1], grafi)] = {
                        "id": row.get("id"),
                        "f_gt": _f(row.get("f_gt")), "f_pred": _f(row.get("f_pred")), "f_err": _f(row.get("f_err")),
                        "t_gt": _f(row.get("t_gt")), "t_pred": _f(row.get("t_pred")), "t_err": _f(row.get("t_err")),
                        "gt_type": row.get("gt_type"), "pred_type": row.get("pred_type"),
                        "type_correct": str(row.get("type_correct", "")).strip().lower() == "true",
                    }
        bundle["per_image"] = per
        bundle["charts"] = [c for c in ("distribution_plot", "femur_comparison_graphs", "tibia_comparison_graphs")
                            if (tdir / f"{c}.png").exists()]
        tests[name] = bundle

    return {
        "models": models_info,
        "images": images,
        "image_list": image_list,
        "model_images": model_images,
        "source_png": source_png,
        "inference_png": inference_png,
        "tests": tests,
        "stats": {
            "image_count": len(images),
            "source_count": len(source_png),
            "model_count": sum(1 for m in models_info if m["available"]),
            "test_count": len(tests),
        },
    }
