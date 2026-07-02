@echo off
REM ==============================================================
REM EchoLink — OPEA TTS Model Setup Script
REM ==============================================================
REM This script downloads Intel OPEA SpeechT5 models and speaker
REM embeddings for offline text-to-speech synthesis.
REM
REM What it downloads:
REM   - microsoft/speecht5_tts (Main TTS model, ~150MB)
REM   - microsoft/speecht5_hifigan (Vocoder, ~50MB)
REM   - Speaker embeddings (default + male voice, <1MB)
REM
REM Total download: ~200MB
REM ==============================================================

title EchoLink — OPEA TTS Model Setup

echo.
echo ================================================================
echo  EchoLink — OPEA TTS Model Setup
echo ================================================================
echo.
echo This script will download Intel OPEA SpeechT5 models (~200MB)
echo for offline text-to-speech synthesis.
echo.
echo Models will be stored in:
echo   python-backend\models\tts\opea_speecht5\
echo.
echo Press Ctrl+C to cancel, or
pause

REM ── Check Python availability ──
echo.
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.9+ from: https://python.org
    pause
    exit /b 1
)
echo ✓ Python found

REM ── Check if in correct directory ──
echo.
echo [2/5] Verifying directory structure...
if not exist "python-backend\scripts" (
    echo ERROR: This script must be run from the project root directory
    echo.
    echo Current directory: %CD%
    echo Expected structure: project-root\python-backend\scripts\
    echo.
    echo Please run from: asl-speech-desktop-app v6 complete\
    pause
    exit /b 1
)
echo ✓ Correct directory

REM ── Create model directory ──
echo.
echo [3/5] Creating model directory...
if not exist "python-backend\models\tts\opea_speecht5" (
    mkdir "python-backend\models\tts\opea_speecht5"
    echo ✓ Created: python-backend\models\tts\opea_speecht5\
) else (
    echo ✓ Directory exists: python-backend\models\tts\opea_speecht5\
)

REM ── Check/Install dependencies ──
echo.
echo [4/5] Checking dependencies (transformers, torch, sentencepiece)...
python -c "import transformers, torch, sentencepiece" >nul 2>&1
if errorlevel 1 (
    echo Missing dependencies detected. Installing...
    echo This may take a few minutes...
    echo.
    python -m pip install transformers torch sentencepiece
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        echo.
        echo Try manually: pip install transformers torch sentencepiece
        pause
        exit /b 1
    )
    echo ✓ Dependencies installed
) else (
    echo ✓ Dependencies already installed
)

REM ── Download models ──
echo.
echo [5/5] Downloading OPEA TTS models...
echo.
echo This will download ~200MB from HuggingFace.
echo Progress will be shown below:
echo.
echo ----------------------------------------------------------------

python -c "import sys; sys.path.insert(0, 'python-backend'); from src.speech.opea_tts.model_downloader import download_models_if_needed; success, msg = download_models_if_needed('python-backend/models/tts/opea_speecht5'); print(f'\n{msg}'); sys.exit(0 if success else 1)"

if errorlevel 1 (
    echo.
    echo ================================================================
    echo  ERROR: Model download failed
    echo ================================================================
    echo.
    echo Common issues:
    echo   - No internet connection
    echo   - HuggingFace servers down
    echo   - Insufficient disk space (~500MB required)
    echo.
    echo Try again later or check:
    echo   https://huggingface.co/microsoft/speecht5_tts
    echo.
    pause
    exit /b 1
)

echo.
echo ----------------------------------------------------------------
echo.

REM ── Download speaker embeddings ──
echo Downloading speaker embeddings...
python -c "import os, urllib.request; base_url = 'https://raw.githubusercontent.com/intel/intel-extension-for-transformers/main/intel_extension_for_transformers/neural_chat/assets/speaker_embeddings/'; embeddings = ['spk_embed_default.pt', 'spk_embed_male.pt']; [urllib.request.urlretrieve(base_url + f, os.path.join('python-backend/models/tts/opea_speecht5', f)) if not os.path.exists(os.path.join('python-backend/models/tts/opea_speecht5', f)) else None for f in embeddings]; print('✓ Speaker embeddings downloaded')"

if errorlevel 1 (
    echo Warning: Speaker embedding download failed
    echo The TTS will still work but may have suboptimal voice quality
)

REM ── Verify installation ──
echo.
echo Verifying installation...
python -c "import os; cache_dir = 'python-backend/models/tts/opea_speecht5'; markers = ['models--microsoft--speecht5_tts', 'models--microsoft--speecht5_hifigan']; all_exist = all(os.path.exists(os.path.join(cache_dir, m)) for m in markers); print('✓ All models verified' if all_exist else '✗ Verification failed'); exit(0 if all_exist else 1)"

if errorlevel 1 (
    echo.
    echo ERROR: Model verification failed
    echo Some files may be missing or corrupted
    echo.
    echo Try running this script again
    pause
    exit /b 1
)

REM ── Success ──
echo.
echo ================================================================
echo  SUCCESS — OPEA TTS Models Installed
echo ================================================================
echo.
echo Models are ready at:
echo   python-backend\models\tts\opea_speecht5\
echo.
echo You can now use OPEA TTS in EchoLink!
echo.
echo Optional: Install OpenVINO for 30-40%% faster synthesis:
echo   pip install openvino
echo.
echo ================================================================
echo.
pause
