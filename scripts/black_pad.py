#!/usr/bin/env python3
"""
Resimleri hedef en/boya SADECE siyah padding ile genisletir.

Esnetme yok, interpolasyon yok: orijinal pikseller oldugu gibi kalir,
resim hedef tuval icinde ortalanir ve kalan kisim siyahla (0) doldurulur.

Onceden tum klasor taranir; bir resim bile hedeften BUYUK ise
(en VEYA boy) islem hic baslamadan sonlandirilir. Esit olmasi sorun degil.

Kaynak dosyalar degistirilmez; sonuclar yeni bir hedef dizine yazilir.

Kullanim:
    python scripts/black_pad.py --width 640 --height 640 \
        --input <kaynak_klasor> --output <hedef_klasor>
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def collect_images(folder: Path):
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        print(f"Hic resim bulunamadi: {folder}")
        sys.exit(1)
    return files


def precheck(files, tw: int, th: int):
    """Hicbir resim hedeften buyuk olmamali. Buyuk olan(lar) varsa terminate."""
    print(f"{len(files)} resim taraniyor (on kontrol)...")
    oversized = []
    for p in files:
        try:
            with Image.open(p) as img:
                w, h = img.size  # PIL: (width, height)
        except Exception as e:
            print(f"HATA: {p} okunamadi: {e}")
            sys.exit(1)
        if w > tw or h > th:
            oversized.append((p, w, h))

    if oversized:
        print(
            f"\nUYARI: {len(oversized)} resim hedef boyuttan ({tw}x{th}) buyuk. "
            f"Padding ile sadece genisletme yapilabilir, kucultme yapilmaz.\n"
            f"Islem yapilmadi.\n"
        )
        for p, w, h in oversized[:20]:
            print(f"  {w}x{h}  {p}")
        if len(oversized) > 20:
            print(f"  ... ve {len(oversized) - 20} resim daha")
        sys.exit(1)

    print("On kontrol tamam: tum resimler hedefe sigiyor.")


def flatten_alpha(image: np.ndarray) -> np.ndarray:
    """Alpha kanali varsa siyah zemine duzlestirir; opak goruntu dondurur.

    RGBA/LA PNG'lerde seffaf bolgeler, padding sonrasi gridi (dama tahtasi)
    olarak gorunur. Alpha'yi siyah uzerine cakistirip kaldiririz: tam opak
    pikseller (asil goruntu) degismez, seffaf bolgeler siyah olur.
    """
    if image.ndim != 3 or image.shape[2] not in (2, 4):
        return image  # alpha yok (gri, BGR) -> dokunma

    color = image[:, :, :-1].astype(np.float32)
    alpha = image[:, :, -1].astype(np.float32) / 255.0
    flat = (color * alpha[:, :, None]).round().astype(np.uint8)
    # 1 kanal kaldiysa (LA) gri olarak don, degilse BGR
    return flat[:, :, 0] if flat.shape[2] == 1 else flat


def black_pad(image: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Resmi tw x th siyah tuvalde ortalar. Esnetme/interpolasyon yok."""
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else None
    if channels:
        canvas = np.zeros((th, tw, channels), dtype=image.dtype)
    else:
        canvas = np.zeros((th, tw), dtype=image.dtype)

    dy = (th - h) // 2
    dx = (tw - w) // 2
    canvas[dy:dy + h, dx:dx + w] = image
    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Resimleri hedef en/boya sadece siyah padding ile genisletir."
    )
    parser.add_argument("--width", "-W", type=int, required=True, help="Hedef genislik (px)")
    parser.add_argument("--height", "-H", type=int, required=True, help="Hedef yukseklik (px)")
    parser.add_argument("--input", "-i", required=True, help="Kaynak klasor")
    parser.add_argument("--output", "-o", required=True, help="Hedef klasor")
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        print("Genislik ve yukseklik pozitif olmali.")
        sys.exit(1)

    in_dir = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    tw, th = args.width, args.height

    if not in_dir.is_dir():
        print(f"Gecersiz kaynak klasor: {in_dir}")
        sys.exit(1)
    if out_dir == in_dir:
        print("Hedef klasor kaynak ile ayni olamaz (kaynak degistirilmez).")
        sys.exit(1)

    files = collect_images(in_dir)
    precheck(files, tw, th)

    print(f"\n{len(files)} resim {tw}x{th} boyutuna siyah pad ile yaziliyor -> {out_dir}")
    written = 0
    for p in files:
        # cv2.imread BGR okur; gri/renkli kanal yapisi korunur
        image = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"  [atlandi] okunamadi: {p}")
            continue

        image = flatten_alpha(image)  # RGBA/LA -> opak (seffaf padding'i engeller)
        padded = black_pad(image, tw, th)

        rel = p.relative_to(in_dir)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst), padded):
            print(f"  [atlandi] yazilamadi: {dst}")
            continue
        written += 1

    print(f"\nTamam: {written}/{len(files)} resim yazildi -> {out_dir}")


if __name__ == "__main__":
    main()
