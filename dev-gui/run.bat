@echo off
rem dev-gui baslatici (windows). Projenin .venv'ini kullanir, gerekirse
rem pywebview'i kurar ve masaustu penceresini acar.
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo Proje sanal ortami bulunamadi: %PY%
  exit /b 1
)

"%PY%" -c "import webview" >nul 2>&1
if errorlevel 1 (
  echo Bagimliliklar kuruluyor ^(pywebview^)...
  "%PY%" -m pip install -r "%ROOT%\dev-gui\requirements.txt"
)

"%PY%" "%ROOT%\dev-gui\app.py" %*
