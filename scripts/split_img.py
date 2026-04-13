from PIL import Image
import os
import argparse


def split_image_vertically(input_path, save_dir, split_percent=0.5):
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        img = Image.open(input_path)
        width, height = img.size

        # Yüzdeyi piksele çevir
        split_point = int(width * split_percent)

        if split_point <= 0 or split_point >= width:
            raise ValueError("invalid percentage")

        # Crop alanları
        left_box = (0, 0, split_point, height)
        right_box = (split_point, 0, width, height)

        left_part = img.crop(left_box)
        right_part = img.crop(right_box)

        # Dosya ismi işlemleri
        base_name = os.path.basename(input_path)
        name_only = os.path.splitext(base_name)[0]

        left_path = os.path.join(save_dir, f"{name_only}.l.jpg")
        right_path = os.path.join(save_dir, f"{name_only}.r.jpg")

        left_part.save(left_path)
        right_part.save(right_path)

        print(f"Successfully split: {base_name}")

    except Exception as e:
        print(f"Hata oluştu: {e}")


def main():
    parser = argparse.ArgumentParser(description="Resmi dikey olarak yüzdeye göre ikiye böler.")
    parser.add_argument("input_path", help="Bölünecek resmin dosya yolu")
    parser.add_argument("split_percent", type=float, help="Genişliğin yüzde kaçından bölünecek (0-100 arası)")

    args = parser.parse_args()

    # Hedef klasör sabit: mevcut klasör
    save_dir = "."

    split_image_vertically(args.input_path, save_dir, args.split_percent)


if __name__ == "__main__":
    main()
