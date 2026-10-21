@echo off
REM ========================================================
REM RAG Chat PDF - Techtaru Digital
REM First-run setup + launch script for Windows
REM ========================================================

echo.
echo    ____    ____     ___      _    ____   _____   ___
echo   / ___|  |  _ \   / _ \    / \  |  _ \ / ___|  / _ \
echo  ^| |  _   ^| ^|_) ^|^| ^| ^| ^|  / _ \ ^| ^|_) ^| \___ \ ^| ^| ^| ^|
echo  ^| |_| ^|  ^|  _ ^| ^| ^|_^| ^| / ___ \^|  _ ^|  ___) ^|^| ^|_^| ^|
echo   \____^|  ^|_^| \_\  \___/ /_/   \_\_^| \_\ ^|____/  \___/
echo.
echo   RAG Chat PDF ^| Techtaru Digital
echo   ====================================================
echo.

REM ── Virtual environment ──────────────────────────────────
if not exist ".venv" (
    echo [SETUP] No virtual environment found - creating .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create a virtual environment. Is Python installed and on PATH?
        pause
        exit /b 1
    )
)

echo [SETUP] Activating virtual environment ...
call .venv\Scripts\activate.bat

REM ── Dependencies (only install when marker file absent) ──
if not exist ".venv\.deps_installed" (
    echo [SETUP] Installing dependencies - first run takes a while ...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. Check the output above.
        pause
        exit /b 1
    )
    type nul > .venv\.deps_installed
)

REM ── .env ─────────────────────────────────────────────────
if not exist ".env" (
    echo [SETUP] .env not found - copying from .env.example
    copy .env.example .env >nul
    echo [ACTION] Open .env and set your API key, then re-run this script.
    pause
    exit /b 1
)

REM ── Placeholder key check ────────────────────────────────
findstr /C:"your-google-api-key-here" .env >nul
if %ERRORLEVEL%==0 (
    echo [ERROR] Get a FREE Gemini key at https://aistudio.google.com/apikey
    echo         then set GOOGLE_API_KEY in .env and re-run this script.
    pause
    exit /b 1
)

echo [OK] Environment ready
echo [START] Launching Streamlit app ...
echo.

start "" http://localhost:8501
streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
