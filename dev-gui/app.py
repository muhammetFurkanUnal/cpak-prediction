"""
dev-gui giriş noktası.

uvicorn'u arka planda bir daemon thread'de başlatır, ardından ana thread'de
pywebview ile yerel bir masaüstü penceresi açar. Pencere işletim sisteminin
yerleşik web görüntüleyicisini kullanır (Chromium paketlemez) → hafif kalır.
Pencere kapanınca süreç biter.

Çalıştırma:
    .venv/bin/python dev-gui/app.py            # masaüstü penceresi (varsayılan)
    .venv/bin/python dev-gui/app.py --browser  # bunun yerine tarayıcıda aç
"""

import socket
import sys
import threading
import time

import uvicorn

from server import app

HOST = "127.0.0.1"
PORT = 8050
URL = f"http://{HOST}:{PORT}/"


def _start_server() -> threading.Thread:
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    # Ana thread dışında çalıştığımız için sinyal işleyicilerini devre dışı bırak.
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return thread


def _wait_until_ready(timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect((HOST, PORT))
                return True
            except OSError:
                time.sleep(0.15)
    return False


def _open_browser():
    import webbrowser

    print(f"dev-gui hazır: {URL}  (kapatmak için Ctrl+C)")
    webbrowser.open(URL)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nKapatılıyor.")


def _open_window():
    import webview

    webview.create_window(
        "cpak dev-gui — model çıktı gözden geçirici",
        URL,
        width=1440,
        height=920,
        min_size=(1000, 650),
    )
    webview.start()


def main():
    force_browser = "--browser" in sys.argv[1:]

    _start_server()
    if not _wait_until_ready():
        print("Sunucu zamanında başlamadı.", file=sys.stderr)
        sys.exit(1)

    if force_browser:
        _open_browser()
        return

    try:
        _open_window()
    except ImportError:
        print("pywebview kurulu değil; tarayıcıda açılıyor. "
              "Masaüstü penceresi için: pip install pywebview")
        _open_browser()


if __name__ == "__main__":
    main()
