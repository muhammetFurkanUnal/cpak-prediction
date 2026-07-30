# dev-gui — Mimari ve Tech Stack

Bu döküman, `dev-gui` aracının backend/frontend çalışma mantığını ve kullanılan
tüm teknolojileri açıklar. Aracın ne işe yaradığı ve nasıl çalıştırıldığı için
`README.md`'ye bakın; burada **nasıl çalıştığı** anlatılır.

---

## 1. Tek cümlede

Diskte hazır duran model çıktılarını (annotasyonlu PNG'ler + metrik json'ları +
ground-truth CSV) okuyup tek bir salt-okuma arayüzünde birleştiren, **canlı
inference yapmayan** yerel bir masaüstü aracı. Üç katmandan oluşur: bir **FastAPI**
sunucusu (backend), **vanilla JS** bir tek-sayfa arayüz (frontend) ve ikisini bir
masaüstü penceresinde birleştiren **pywebview** kabuğu.

---

## 2. Tech stack (özet tablo)

| Katman | Teknoloji | Neden |
|---|---|---|
| Masaüstü pencere | **pywebview** | İşletim sisteminin yerleşik WebView'ini kullanır; Chromium paketlemez → hafif. Tek yeni bağımlılık. |
| Web sunucusu | **FastAPI** + **uvicorn** | Projede zaten var; salt-okuma JSON uçları ve `FileResponse` ile görüntü akışı için yeterli. |
| Görüntü akışı | **Starlette `FileResponse`** | Baytı belleğe almadan diskten doğrudan akıtır. |
| İndeks / veri okuma | **Python stdlib** (`csv`, `json`, `pathlib`, `re`) | Ek bağımlılık yok; `torch`/`cv2`/`numpy` **bilinçli olarak import edilmez**. |
| Frontend | **Vanilla JS** (ES, framework yok) + **hash router** | Derleme adımı yok; tek `app.js` + `fetch` + DOM. |
| Stil | Düz **CSS** (tek `style.css`) | Tooling yok. |
| Çalışma ortamı | Projenin mevcut **`.venv`**'i | `run.sh` / `run.bat` ilk açılışta yalnız `pywebview` kurar. |

Bağımlılık olarak eklenen tek paket `pywebview`'dir (`requirements.txt`).

---

## 3. Genel akış

```
run.sh / run.bat
      │  (.venv'i bulur, pywebview yoksa kurar)
      ▼
   app.py  ──────────────────────────────────────────────┐
      │                                                   │
      │ 1) uvicorn'u daemon thread'de başlatır            │
      │    (server.app, 127.0.0.1:8050)                   │
      │ 2) port açılana kadar bekler (_wait_until_ready)  │
      │ 3) ana thread'de pywebview penceresini açar       │
      │    (--browser verilirse normal tarayıcı)          │
      ▼                                                   │
  pywebview penceresi  ──►  http://127.0.0.1:8050/  ──►  FastAPI (server.py)
                                                          │
                                              açılışta bir kez: build_index()
                                              (indexer.py → bellekte INDEX)
```

İki proses değil, **tek proses** vardır: ana thread pencereyi tutar, bir daemon
thread uvicorn'u koşturur. Pencere kapanınca proses biter.

---

## 4. Backend

Dosyalar: `app.py`, `server.py`, `indexer.py`, `registry.py`, `keys.py`, `paths.py`.

### 4.1 Giriş noktası — `app.py`
- `_start_server()`: uvicorn'u `daemon=True` thread'de başlatır. Ana thread
  dışında çalıştığı için sinyal işleyicilerini devre dışı bırakır
  (`server.install_signal_handlers = lambda: None`).
- `_wait_until_ready()`: 8050 portuna TCP bağlanana kadar (en çok 20 sn) bekler;
  pencere erken açılıp boş ekran göstermesin diye.
- `_open_window()`: `webview.create_window(...)` + `webview.start()`. pywebview
  kurulu değilse `ImportError` yakalanır ve `_open_browser()`'a düşülür.

### 4.2 Sunucu — `server.py`
- Saf **FastAPI**; `torch`/`onnxruntime`/`cv2`/`numpy` import etmez → hızlı başlar,
  düşük bellek.
- İndeks açılışta bir kez kurulur: `INDEX = build_index()` ve bellekte tutulur.
  Yeni veri için uygulamayı yeniden başlatmak yeter.
- Uç noktalar:

  | Uç | Döner |
  |---|---|
  | `GET /` | `static/index.html` (no-store) |
  | `GET /health` | indeks istatistikleri |
  | `GET /api/models` | tüm modeller + provenance |
  | `GET /api/images` | grid için hafif liste (bayt/metrik yok) |
  | `GET /api/images/{key}` | tek görüntünün tam kaydı (GT + model çıktıları) |
  | `GET /api/models/{model}/images` | o modelin ürettiği anahtarlar (gezinme) |
  | `GET /api/models/{model}/test` | test analizi paketi; yoksa nereye bakıldığını bildirir |
  | `GET /img/source/{key}` | kaynak (ön-işlenmiş) görüntü — `FileResponse` |
  | `GET /img/inference/{model}/{key}` | modelin annotasyonlu çıktı görüntüsü |
  | `GET /img/test/{model}/{chart}` | test grafiği (sabit beyaz liste ile sınırlı) |

- **Güvenlik notu:** görüntü uçları indekste tutulan yol eşlemesinden okur;
  test grafikleri `_ALLOWED_CHARTS` beyaz listesiyle sınırlıdır → keyfi dosya
  yolu enjeksiyonu (path traversal) yok.

### 4.3 İndeksleyici — `indexer.py` (çekirdek mantık)
Açılışta dört kaynağı tarayıp `(pid, side, grafi)` **bileşik anahtarında**
birleştirir:
1. **Kaynak görüntüler** — `preprocessing/*-prepped/` klasörleri; klasör hangi
   grafi/id-sınıfı olduğunu belirler.
2. **Model çıktıları** — `notebooks/out/inference/<model>/`: annotasyonlu PNG'ler
   + `registry`'de bildirilen `primary_metrics` json'undan görüntü-başı açılar.
   Ham MPTA/LDFA → `keys.derive()` ile DİZİLİM/EKLEM/CPAK-tipi türetilir.
3. **Ground-truth** — `data/cpak-grnd-truth.csv` (aynı anahtara çok satır olabilir,
   liste tutulur).
4. **Test analizi** — `notebooks/out/test/<model>/` (yalnız `metrics.json` bulunan
   model için; şu an `sh14-ep1000`).

Önemli ilke: **görüntü baytları indekste tutulmaz** — yalnız küçük skalerler ve
yol string'leri. Bellek tek haneli MB. Eksik veri **uydurulmaz**; yoksa alan
`None` kalır ve arayüzde "mevcut değil" görünür.

### 4.4 Model kayıt tablosu — `registry.py`
Bildirimsel (declarative) tablo: her modelin `kind`, `primary_metrics`,
`angle_map`, `extra_map` bilgisi. Yeni model eklemek = buraya bir sözlük + diske
çıktı koymak. Aykırılıklar (ör. `kneeap-sh2`'deki beklenmeyen ortopedik metrik
dosyası) kod içinde açıkça not düşülmüş; karar kullanıcıya bırakılmış.

### 4.5 Kimlik normalleştirme — `keys.py`
`scripts/cpak_predict_to_csv.py`'deki mantığın birebir taşınmış hâli; böylece
dev-gui ile üretim script'i aynı anahtar kuralını paylaşır. `4000.r.png`,
`10105873454_r`, ` 4096L` gibi farklı yazımlar tek anahtara indirgenir. Bileşik
anahtar `pid~side~grafi` (URL-güvenli `~` ayraç). CPAK 3×3 ızgarası, dizilim/eklem
eşikleri de burada.

### 4.6 Yollar — `paths.py`
Tüm dataset yollarının tek kaynağı. Yeni kaynak klasör/model eklenince yalnız
burası ve `registry.py` güncellenir.

---

## 5. Frontend

Dosyalar: `static/index.html`, `static/app.js`, `static/style.css`.

- **Framework yok, build yok.** `index.html` tek `app.js`'i yükler; sunucu onu
  `StaticFiles` ile servis eder.
- **Hash router:** gezinme `#/models`, `#/browse` gibi hash rotalarıyla yapılır
  (sayfa yenileme yok, sunucu tarafı routing yok).
- **`h(tag, attrs, ...kids)`** adında mini bir DOM yardımcısıyla element üretilir
  (React'ın `createElement`'inin minik bir karşılığı); `state` nesnesinde basit
  istemci durumu tutulur.
- **`api(path)`** = ince bir `fetch` sarmalayıcı; tüm veri backend'den JSON olarak
  çekilir. Görüntüler doğrudan `<img src="/img/...">` ile yine backend'den gelir.
- Görünümler: **Modeller** (provenance tablosu), **Göz at** (dataset grid + ok
  tuşlarıyla görüntü-görüntü karşılaştırma) ve modelin **Test analizi** (yalnız
  diskte test klasörü olanda).

---

## 6. Tasarım ilkeleri (neden böyle)

- **Salt-okuma, yan etkisiz:** hiçbir dosyayı değiştirmez, hiçbir model
  çalıştırmaz. Canlı API'den (port 8000) tamamen ayrı, port 8050'de.
- **Hafiflik:** ağır ML kütüphaneleri import edilmez; pencere sistemin WebView'ini
  kullanır; frontend'de toolchain yok. Tek ek bağımlılık `pywebview`.
- **Yalnızca gerçek veri:** eksik değer hesaplanıp doldurulmaz; her gösterilen
  değerin diskteki kaynağı (provenance) görünür kılınır.
- **Tek doğruluk kaynağı:** yollar `paths.py`'de, modeller `registry.py`'de,
  anahtar kuralı `keys.py`'de toplanır — biri değişince tek yer güncellenir.
