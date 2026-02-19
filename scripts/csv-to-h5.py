import pandas as pd
import os

# Yolları tam (absolute) tanımlayalım
csv_path = "/home/furkan/projects/cpak/deeplabcut/cpak-furkan-2026-02-17/labeled-data/lera/CollectedData_murkanthedestroyer.csv"
h5_path = "/home/furkan/projects/cpak/deeplabcut/cpak-furkan-2026-02-17/labeled-data/lera/CollectedData_murkanthedestroyer.h5"
folder_name="lera"

try:
    # 1. CSV'yi oku
    # DeepLabCut CSV'leri 3 satır header'lıdır
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    
    # 2. Path'leri DLC'nin beklediği formata zorla
    # Indexler: labeled-data/images/resim_adi.jpg şeklinde olmalı
    df.index = [os.path.join('labeled-data', folder_name, os.path.basename(i)) for i in df.index]
    
    # 3. H5 olarak kaydet (DLC'nin beklediği formatta)
    df.to_hdf(h5_path, key='df_with_missing', mode='w')
    
    if os.path.exists(h5_path):
        print(f"BAŞARILI: Dosya oluşturuldu -> {h5_path}")
        print(f"Dosya Boyutu: {os.path.getsize(h5_path)} bytes")
    else:
        print("HATA: to_hdf çalıştı ama dosya diskte görünmüyor!")

except Exception as e:
    print(f"HATA OLUŞTU: {e}")