@echo off
REM ============================================================
REM  Convert sign-recognition ONNX models to OpenVINO IR
REM ============================================================
REM  Task 1: check which models can be converted to IR
REM  Task 2: convert the ones that can (script handles this)
REM  Task 3: place IR files in each model's own folder
REM
REM  Models that cannot be converted (e.g. Model 3 / LSTM with
REM  the unsupported Loop operator) are left in ONNX format and
REM  the app falls back to ONNX Runtime for them automatically.
REM ============================================================

setlocal

REM Resolve paths relative to this .bat file:
REM   this file        -> python-backend\models\sign\convert_to_ir.bat
REM   python-backend   -> ..\..
REM   project root     -> ..\..\..
set "SIGN_DIR=%~dp0"
set "BACKEND_DIR=%SIGN_DIR%..\.."
set "ROOT_DIR=%SIGN_DIR%..\..\.."

REM Prefer the project's .venv (Python 3.11 with OpenVINO). Fall back to system python.
set "VENV_PY=%ROOT_DIR%\.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
) else (
    set "PY=python"
)

REM Force UTF-8 so status symbols print correctly on Windows consoles.
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

echo Using Python: %PY%
echo Script dir  : %SIGN_DIR%
echo.

"%PY%" "%SIGN_DIR%convert_models_to_ir.py" %*
set "EXITCODE=%ERRORLEVEL%"

echo.
if "%EXITCODE%"=="0" (
    echo [DONE] Conversion finished successfully.
) else (
    echo [ERROR] Conversion exited with code %EXITCODE%.
)

endlocal & exit /b %EXITCODE%
