@echo off
REM ============================================================
REM run_experiment.bat
REM Menjalankan main.py menggunakan Python 3.10 dari .venv
REM Hindari masalah kompatibilitas dengan Python 3.12+ sistem
REM ============================================================

SET SCRIPT_DIR=%~dp0
SET VENV_PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe

IF NOT EXIST "%VENV_PYTHON%" (
    echo [ERROR] .venv\Scripts\python.exe tidak ditemukan.
    echo [INFO]  Buat virtual environment terlebih dahulu:
    echo         uv venv --python 3.10 .venv
    echo         .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo [INFO] Menggunakan Python: %VENV_PYTHON%
"%VENV_PYTHON%" main.py %*
