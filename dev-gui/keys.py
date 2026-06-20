"""
Kimlik normalleştirme + sınıflandırma.

Bu modülün özü ``scripts/cpak_predict_to_csv.py`` içindeki mantıktan birebir
taşınmıştır (``_normalize`` / ``img_to_key`` / ``hasta_to_key`` / ``is_4xxx`` ve
hizalanma/eklem/CPAK-tipi eşikleri). Böylece dev-gui ile üretim script'i aynı
kuralı paylaşır.

TEK BİLİNÇLİ SAPMA: ``_normalize`` regex'i ``[._\\s]``'e genişletildi (alt çizgi
de silinir). Çünkü test dosyası ``angles.csv`` kimlikleri ``10105873454_r`` gibi
alt çizgiyle ayrılır; orijinal ``[.\\s]`` bunu eşleyemezdi. Bu sayede noktayla
ayrılan dosya adları da, alt çizgiyle ayrılan test kimlikleri de aynı anahtara
indirgenir.
"""

import math
import re

# Bir anahtar (pid, side) çiftidir; ör. ("4000", "r") veya ("10105873454", "l").
_KEY_RE = re.compile(r"^(\d+)([lr])$")


def _normalize(text: str) -> str:
    """Nokta/alt-çizgi/boşluk sil ve küçült: '4000.r' / ' 4096L' / '10105873454_r' → '4000r'..."""
    return re.sub(r"[._\s]", "", text).lower()


def img_to_key(filename: str):
    """Görüntü dosya adı → (pid, side). '4000.r.png', '10105873454_r' vb. işler."""
    stem = filename[:-4] if filename.lower().endswith(".png") else filename
    m = _KEY_RE.match(_normalize(stem))
    return (m.group(1), m.group(2)) if m else None


def hasta_to_key(value: str):
    """HASTA hücresi → (pid, side) veya None."""
    m = _KEY_RE.match(_normalize(value))
    return (m.group(1), m.group(2)) if m else None


def is_4xxx(pid: str) -> bool:
    return len(pid) == 4 and pid.startswith("4")


# ── Bileşik anahtar: pid + side + grafi ─────────────────────────────────────
# grafi anahtarın zorunlu parçasıdır: 4000-serisi aynı (pid, side) ile hem tüm
# bacak (grafi 1) hem diz (grafi 2) olarak görünebilir. Ayraç olarak '~' (URL
# güvenli, kodlama gerektirmez) kullanılır; pid yalnız rakam, side l/r, grafi 1/2
# olduğu için ayrışma tek anlamlıdır.
def make_key(pid: str, side: str, grafi: str) -> str:
    return f"{pid}~{side}~{grafi}"


def parse_key(key: str):
    pid, side, grafi = key.split("~")
    return pid, side, grafi


# ── Sınıflandırma (ground-truth tablosuyla aynı eşikler) ────────────────────
def alignment(diff: int) -> str:
    """DİZİLİM (MPTA-LDFA):  <-2 varus / [-2,2] nötr / >2 valgus."""
    if diff < -2:
        return "varus"
    if diff > 2:
        return "valgus"
    return "nötr"


def joint(total: int) -> str:
    """EKLEM (MPTA+LDFA):  <177 apex distal / [177,183] nötr / >183 apex proksimal."""
    if total < 177:
        return "apex distal"
    if total > 183:
        return "apex proksimal"
    return "nötr"


# CPAK 3×3 ızgara → Tip 1..9
_CPAK_GRID = {
    ("varus", "apex distal"): "Tip 1",
    ("nötr", "apex distal"): "Tip 2",
    ("valgus", "apex distal"): "Tip 3",
    ("varus", "nötr"): "Tip 4",
    ("nötr", "nötr"): "Tip 5",
    ("valgus", "nötr"): "Tip 6",
    ("varus", "apex proksimal"): "Tip 7",
    ("nötr", "apex proksimal"): "Tip 8",
    ("valgus", "apex proksimal"): "Tip 9",
}


def cpak_type(align: str, jnt: str) -> str:
    return _CPAK_GRID[(align, jnt)]


def round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def derive(mpta_raw: float, ldfa_raw: float) -> dict:
    """Ham MPTA/LDFA → yuvarlanmış değerler + DİZİLİM/EKLEM/CPAK-tipi.

    ``cpak_predict_to_csv.py`` ile aynı adımlar: önce round-half-up, sonra eşikler.
    """
    mpta = round_half_up(mpta_raw)
    ldfa = round_half_up(ldfa_raw)
    diff = mpta - ldfa
    total = mpta + ldfa
    align = alignment(diff)
    jnt = joint(total)
    return {
        "mpta_raw": round(mpta_raw, 2),
        "ldfa_raw": round(ldfa_raw, 2),
        "mpta": mpta,
        "ldfa": ldfa,
        "diff": diff,
        "sum": total,
        "dizilim": align,
        "eklem": jnt,
        "siniflama": cpak_type(align, jnt),
    }
