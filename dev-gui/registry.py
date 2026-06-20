"""
Bildirimsel model kayıt tablosu.

Her model için: türü, hangi metrik dosyasından beslendiği ve MPTA/LDFA gibi
açıların o dosyadaki hangi alandan okunacağı burada açıkça yazılıdır. Uygulama
hiçbir eksik veriyi kendiliğinden üretmez; bir alan ilgili dosyada yoksa o değer
boş (None) kalır ve arayüzde "mevcut değil" diye gösterilir.

YENİ MODEL EKLEMEK
------------------
1. ``notebooks/out/inference/<yeni-model>/`` altına çıktı PNG'lerini ve metrik
   json'unu koy.
2. Buraya bir sözlük ekle:
   - ``kind``: "cpak" (tüm bacak, grafi 1) veya "kneeap" (diz AP, grafi 2).
   - ``primary_metrics``: açıların okunacağı json dosyasının adı.
   - ``angle_map``: {"mpta": "<alan>", "ldfa": "<alan>"} — model MPTA/LDFA
     üretmiyorsa boş bırak ({}), arayüz "mevcut değil" der.
   - ``extra_map``: ek skaler açılar (ör. {"jlca": "jlca"}).
3. Test analizi (``notebooks/out/test/<yeni-model>/``) varsa otomatik bulunur.
Uygulamayı yeniden başlatmak indeksi tazelemeye yeter.
"""

# Modelin türü çıktılarının grafi tipini belirler:
#   cpak  → grafi "1" (tüm bacak / uzunluk filmi)
#   kneeap → grafi "2" (diz AP filmi)
KIND_GRAFI = {"cpak": "1", "kneeap": "2"}
KIND_LABEL = {"cpak": "Tüm bacak (uzunluk filmi)", "kneeap": "Diz (AP filmi)"}
GRAFI_LABEL = {"1": "Tüm bacak (uzunluk filmi)", "2": "Diz (AP filmi)"}
GRAFI_KIND = {"1": "cpak", "2": "kneeap"}


MODELS = [
    {
        # Ortopedik / tam-bacak model. Hem inference hem test analizi var.
        "name": "sh14-ep1000",
        "kind": "cpak",
        "experimental": False,
        "primary_metrics": "orthopedic_metrics.json",
        "angle_map": {"mpta": "tibia_mech_angle_inter", "ldfa": "femur_mech_angle_notch"},
        "extra_map": {},
    },
    {
        # Diz modeli. kneeap_metrics.json yalnızca jlca taşır; MPTA/LDFA üretmez.
        "name": "kneeap-sh1-ep1000",
        "kind": "kneeap",
        "experimental": False,
        "primary_metrics": "kneeap_metrics.json",
        "angle_map": {},
        "extra_map": {"jlca": "jlca"},
    },
    {
        # NOT (aykırılık): bu klasörde ayrıca bir orthopedic_metrics.json var; ancak
        # bir diz (kneeap) modelinde mekanik tam-bacak açısı beklenmez. Kendiliğinden
        # MPTA/LDFA kaynağı yapılmadı — dosyanın varlığı /api/models provenance'ında
        # görünür, karar kullanıcıya bırakılmıştır.
        "name": "kneeap-sh2-ep1000",
        "kind": "kneeap",
        "experimental": False,
        "primary_metrics": "kneeap_metrics.json",
        "angle_map": {},
        "extra_map": {"jlca": "jlca"},
    },
    {
        "name": "kneeap-sh3-ep1000",
        "kind": "kneeap",
        "experimental": False,
        "primary_metrics": "kneeap_metrics.json",
        "angle_map": {},
        "extra_map": {"jlca": "jlca"},
    },
    {
        # Diz modeli; kneeap_angles.json MPTA/LDFA (ampta/aldfa) içerir.
        "name": "kneeap-sh4-ep1000",
        "kind": "kneeap",
        "experimental": False,
        "primary_metrics": "kneeap_angles.json",
        "angle_map": {"mpta": "ampta", "ldfa": "aldfa"},
        "extra_map": {"femorotibial_angle": "femorotibial_angle", "jlca": "jlca"},
    },
    {
        # sh4'ün deneysel yeniden-inference varyantı.
        "name": "new-infer-kneeap-sh4-ep1000",
        "kind": "kneeap",
        "experimental": True,
        "primary_metrics": "kneeap_angles.json",
        "angle_map": {"mpta": "ampta", "ldfa": "aldfa"},
        "extra_map": {"femorotibial_angle": "femorotibial_angle", "jlca": "jlca"},
    },
]
