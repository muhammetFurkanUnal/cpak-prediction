#!/usr/bin/env python3
"""
Analyze image dimensions in a folder and visualize the distributions.

Usage:
    python scripts/analyze_image_sizes.py <folder>
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def collect_dimensions(folder: Path):
    widths, heights = [], []
    files = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]

    if not files:
        print(f"Hic resim bulunamadi: {folder}")
        sys.exit(1)

    print(f"{len(files)} resim bulundu, boyutlar okunuyor...")
    skipped = 0
    for p in files:
        try:
            with Image.open(p) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception as e:
            skipped += 1
            print(f"  [atlandi] {p.name}: {e}")

    if skipped:
        print(f"{skipped} dosya okunamadi ve atlandi.")
    return np.array(widths), np.array(heights)


def freedman_diaconis_bins(data):
    n = len(data)
    if n < 2:
        return 10
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    if iqr == 0:
        return min(n, 30)
    bw = 2 * iqr / (n ** (1 / 3))
    bins = int(np.ceil((data.max() - data.min()) / bw))
    return max(5, min(bins, 80))


def add_hist_with_kde(ax, data, label, color):
    bins = freedman_diaconis_bins(data)
    ax.hist(data, bins=bins, color=color, alpha=0.6, edgecolor="white",
            linewidth=0.4, density=True, label="histogram")

    xs = np.linspace(data.min(), data.max(), 500)
    kde = gaussian_kde(data, bw_method="scott")
    ax.plot(xs, kde(xs), color=color, linewidth=2.2, label="KDE")

    median = np.median(data)
    mean   = np.mean(data)
    ax.axvline(median, color="red",    linestyle="--", linewidth=1.4,
               label=f"medyan = {median:.0f}")
    ax.axvline(mean,   color="orange", linestyle=":",  linewidth=1.4,
               label=f"ortalama = {mean:.0f}")

    ax.set_xlabel(f"{label} (px)", fontsize=10)
    ax.set_ylabel("yogunluk", fontsize=9)
    ax.set_title(f"{label} dagilimi", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)


def plot(widths, heights, folder: Path):
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        f"Resim boyut analizi  —  {folder}   ({len(widths)} resim)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)
    ax_w   = fig.add_subplot(gs[0, 0])
    ax_h   = fig.add_subplot(gs[0, 1])
    ax_ar  = fig.add_subplot(gs[0, 2])
    ax_sc  = fig.add_subplot(gs[1, 0:2])
    ax_top = fig.add_subplot(gs[1, 2])

    # ----- histogramlar -----
    add_hist_with_kde(ax_w,  widths,  "Genislik (W)", "#4C72B0")
    add_hist_with_kde(ax_h,  heights, "Yukseklik (H)", "#55A868")

    aspect = widths / heights
    add_hist_with_kde(ax_ar, aspect, "En-boy orani W/H", "#C44E52")
    ax_ar.axvline(1.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.6,
                  label="kare (1:1)")
    ax_ar.legend(fontsize=8)

    # ----- scatter -----
    ax_sc.scatter(widths, heights, alpha=0.35, s=8, color="#8172B2", linewidths=0)
    lim = max(widths.max(), heights.max()) * 1.05
    ax_sc.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.4, label="kare (W=H)")
    ax_sc.set_xlabel("Genislik (px)", fontsize=10)
    ax_sc.set_ylabel("Yukseklik (px)", fontsize=10)
    ax_sc.set_title("Genislik vs Yukseklik (scatter)", fontsize=11, fontweight="bold")
    ax_sc.legend(fontsize=8)

    # ----- en cok tekrar eden cozunurlukler -----
    res_counter = Counter(zip(widths.tolist(), heights.tolist()))
    top = res_counter.most_common(10)
    labels = [f"{w}×{h}" for (w, h), _ in top]
    counts = [c for _, c in top]

    bars = ax_top.barh(range(len(labels)), counts, color="#DD8452", alpha=0.85)
    ax_top.set_yticks(range(len(labels)))
    ax_top.set_yticklabels(labels, fontsize=8)
    ax_top.invert_yaxis()
    ax_top.set_xlabel("adet", fontsize=9)
    ax_top.set_title("En sik gorulen cozunurlukler", fontsize=11, fontweight="bold")
    for bar, cnt in zip(bars, counts):
        ax_top.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    str(cnt), va="center", fontsize=7)

    # ----- ozet kutusu -----
    summary = (
        f"Toplam resim : {len(widths)}\n"
        f"Genislik     : {widths.min()} – {widths.max()} px  "
        f"(ort={widths.mean():.0f}, σ={widths.std():.0f})\n"
        f"Yukseklik    : {heights.min()} – {heights.max()} px  "
        f"(ort={heights.mean():.0f}, σ={heights.std():.0f})\n"
        f"En-boy orani : {aspect.min():.2f} – {aspect.max():.2f}  "
        f"(ort={aspect.mean():.2f})"
    )
    fig.text(
        0.01, 0.01, summary, fontsize=8, va="bottom", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFDE7", alpha=0.9),
    )

    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Kullanim: python scripts/analyze_image_sizes.py <klasor>")
        sys.exit(1)

    folder = Path(sys.argv[1]).expanduser().resolve()
    if not folder.is_dir():
        print(f"Gecersiz klasor: {folder}")
        sys.exit(1)

    widths, heights = collect_dimensions(folder)

    print(f"\nGenislik  — min:{widths.min()}  max:{widths.max()}  "
          f"ort:{widths.mean():.1f}  std:{widths.std():.1f}")
    print(f"Yukseklik — min:{heights.min()}  max:{heights.max()}  "
          f"ort:{heights.mean():.1f}  std:{heights.std():.1f}")

    plot(widths, heights, folder)


if __name__ == "__main__":
    main()
