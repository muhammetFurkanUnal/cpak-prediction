# cpak dev-gui

Geliştirme sürecinde, modellerin daha önce ürettiği çıktıları tek bir hafif
arayüzde gözden geçirmek için yerel bir araç. **Canlı inference yapmaz**;
yalnızca diskte hazır olan dosyaları okuyup gösterir. Mevcut `api/` uygulamasından
(canlı inference, port 8000) tamamen ayrıdır ve **port 8050**'de çalışır; hiçbir
dosyayı değiştirmez.

## Çalıştırma

- **macOS / Linux:** `dev-gui/run.sh`
- **Windows:** `dev-gui\run.bat`

İlk çalıştırmada projenin `.venv`'ine `pywebview` kurulur, sonra bir masaüstü
penceresi açılır. Pencere, işletim sisteminin yerleşik web görüntüleyicisini
kullanır (Chromium paketlemez) → hafif ve düşük bellekli.

Masaüstü penceresi yerine tarayıcıda açmak için:

```
.venv/bin/python dev-gui/app.py --browser
```

## Ne gösterir

- **Modeller:** Mevcut tüm modeller; türü (cpak/kneeap), çıktı sayısı, test
  analizi olup olmadığı, ağırlık dosyası ve her modelin hangi dosyalardan
  beslendiği (veri kaynağı / provenance).
- **Göz at:** Dataset görüntüleri (grid). Bir model seçince o modelin görüntüleri
  arasında ok tuşlarıyla gezilir; tek görüntüde annotasyonlu (çizilmiş) çıktı,
  kaynak görüntü, CSV ground-truth ve modelin tahmini yan yana görülür. Diz
  görüntülerinde birden çok kneeap modeli aynı görüntüde karşılaştırılır.
- **Test analizi:** Yalnızca diskte test klasörü bulunan model için (şu an
  `sh14-ep1000`): başarı metrikleri, hata dağılımı, hazır grafikler ve
  görüntü-başı ground-truth/tahmin tablosu.

## Veriyi nereden okur (provenance)

| Ne | Yol |
|---|---|
| Dataset / ground truth | `data/cpak-grnd-truth.csv` |
| Kaynak görüntüler | `preprocessing/preop-single-prepped/`, `preprocessing/knee-preop-prepped/`, `preprocessing/long-id-split-prepped/` |
| Model çıktıları | `notebooks/out/inference/<model>/` |
| Test analizi | `notebooks/out/test/<model>/` |

Tüm kaynaklar `(pid, side, grafi)` bileşik anahtarında birleşir; normalleştirme
mantığı `scripts/cpak_predict_to_csv.py` ile aynıdır (`keys.py`).

## İlke: yalnızca gerçek veri

Uygulama eksik veriyi kendiliğinden hesaplayıp doldurmaz. Bir şey diskte yoksa
arayüz **"mevcut değil"** der ve nereye baktığını gösterir. Her gösterilen
değerin kaynağı görünür kılınmıştır.

## Yeni model eklemek

1. Çıktıları `notebooks/out/inference/<yeni-model>/` altına koy (annotasyonlu
   PNG'ler + metrik json'u).
2. `registry.py`'ye bir kayıt ekle (`kind`, `primary_metrics`, `angle_map`,
   `extra_map`). Ayrıntı kuralları o dosyanın başında yazılıdır.
3. Test analizi varsa `notebooks/out/test/<yeni-model>/` altına koy; otomatik
   bulunur.
4. Uygulamayı yeniden başlat (indeks açılışta kurulur).

## Mimari (özet)

- `app.py` — uvicorn'u daemon thread'de başlatır + pywebview penceresini açar.
- `server.py` — FastAPI, salt-okuma uç noktaları; görüntüleri diskten akıtır
  (`FileResponse`). `torch`/`onnxruntime`/`cv2` import etmez → hafif kalır.
- `indexer.py` — açılışta CSV + kaynaklar + çıktılar + testi tarayıp bellekte
  birleştirir.
- `registry.py`, `keys.py`, `paths.py` — model tablosu, normalleştirme, yollar.
- `static/` — sade HTML/CSS/vanilla JS ön yüz (derleme adımı yok).
