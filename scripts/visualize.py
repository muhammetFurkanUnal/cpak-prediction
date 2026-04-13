import pydicom
import matplotlib.pyplot as plt
import os

def display_dicom(file_paths, grid=True):
    if isinstance(file_paths, str):
        file_paths = [file_paths]
        grid = False

    num_images = len(file_paths)
    
    if not grid or num_images == 1:
        for path in file_paths:
            ds = pydicom.dcmread(path)
            plt.figure(figsize=(8, 8))
            plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
            plt.title(f"{os.path.basename(path)}\n{ds.Modality}")
            plt.axis('off')
            plt.show()
    else:
        cols = 2
        rows = (num_images + 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()

        for i, path in enumerate(file_paths):
            try:
                ds = pydicom.dcmread(path)
                axes[i].imshow(ds.pixel_array, cmap=plt.cm.bone)
                axes[i].set_title(f"{os.path.basename(path)}\n{ds.Modality}", fontsize=10)
                axes[i].axis('off')
            except Exception as e:
                axes[i].set_title(f"Error: {os.path.basename(path)}")
                print(f"Error reading {path}: {e}")

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    file_list = [
        "datasets/hospital-data/POSTOP AP/AHMET AYDIN POSTOP AP.Seq1.Ser1001.Img1.dcm",
        "datasets/hospital-data/POSTOP UZ/ALİME KALIPÇI POSTOP UZ.Seq2.Ser1009.Img1.dcm",
        "datasets/hospital-data/PREOP AP/AHMET AYDIN PREOP AP.Seq1.Ser1001.Img1.dcm",
        "datasets/hospital-data/PREOP UZ/AYSUN BAYIN PREOP UZ.Seq1.Ser1001.Img1.dcm"
    ]

    postop_ap_list = [
        "datasets/hospital-data/POSTOP AP/AHMET AYDIN POSTOP AP.Seq1.Ser1001.Img1.dcm",
        "datasets/hospital-data/POSTOP AP/ALİ OSMAN ALTINÇEKİÇ POSTOP AP.Seq2.Ser1002.Img1.dcm",
        "datasets/hospital-data/POSTOP AP/AYŞE TAŞDAN POSTOP AP.Seq1.Ser1001.Img1.dcm",
        "datasets/hospital-data/POSTOP AP/CEMİLE KARATEPE POSTOP AP.Seq2.Ser1002.Img1.dcm"
    ]

    postop_uz_list = [
        "datasets/hospital-data/POSTOP UZ/FAİKA EKİNCİ POSTOP UZ    .Seq1.Ser1001.Img1.dcm",
        "datasets/hospital-data/POSTOP UZ/NEBAHAT ZAZAOĞLU POSTOP UZ.Seq1.Ser1001.Img1.dcm",
        "datasets/hospital-data/POSTOP UZ/SOLMAZ ÜLKER POSTOP UZ.Seq1.Ser1001.Img1.dcm",
        "datasets/hospital-data/POSTOP UZ/HATİCE SEZER POSTOP UZ.Seq1.Ser1001.Img1.dcm"
    ]

    preop_ap_list = [
        "datasets/hospital-data/PREOP AP/AYŞE GÜZİN GÖRMEZ PREOP AP   .Seq1.Ser1001.Img1.dcm",
        "datasets/hospital-data/PREOP AP/BELKİS AKIN PREOP AP.Seq1.Ser1.Img1.dcm",
        "datasets/hospital-data/PREOP AP/CEMİLE KARATEPE PREOP AP.Seq1.Ser1001.Img1.dcm",
        "datasets/hospital-data/PREOP AP/ELMAS ÖNDER PREOP AP.Seq2.Ser1002.Img1.dcm"
    ]

    preop_uz_list = [
        "datasets/hospital-data/PREOP UZ/GÜNGÖR AKIN PREOP UZ.Seq1.Ser1009.Img1.dcm",
        "datasets/hospital-data/PREOP UZ/GÜLER YOLAL PREOP UZ.Seq3.Ser1001.Img1.dcm",
        "datasets/hospital-data/PREOP UZ/HANDAN DEMİRTAŞ PREOP UZ.Seq2.Ser1001.Img1.dcm",
        "datasets/hospital-data/PREOP UZ/KAFİYE ÇAKMAK PREOP UZ.Seq1.Ser1001.Img1.dcm"
    ]
    
    display_dicom(file_list, grid=True)
    display_dicom(postop_ap_list, grid=True)
    display_dicom(postop_uz_list, grid=True)
    display_dicom(preop_ap_list, grid=True)
    display_dicom(preop_uz_list, grid=True)
    