@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

echo ============================================================
echo   EchoLink - Windows Installer Build
echo ============================================================
echo.

where node >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js not found in PATH.
  goto :error
)

where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] npm not found in PATH.
  goto :error
)

set "PYTHON_EXE="
set "PYTHON_ARGS="

call :try_python_path "%cd%\.venv\Scripts\python.exe"
if defined PYTHON_EXE goto :python_found
call :try_python_path "%cd%\venv\Scripts\python.exe"
if defined PYTHON_EXE goto :python_found
call :try_python_path "%cd%\python-backend\.venv\Scripts\python.exe"
if defined PYTHON_EXE goto :python_found
call :try_python_path "%cd%\python-backend\venv\Scripts\python.exe"
if defined PYTHON_EXE goto :python_found

where py >nul 2>nul
if not errorlevel 1 (
  py -3 --version >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3"
    goto :python_found
  )
)

where python >nul 2>nul
if not errorlevel 1 (
  python --version >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_EXE=python"
    set "PYTHON_ARGS="
  )
)

:python_found

if "%PYTHON_EXE%"=="" (
  echo [ERROR] Python not found. Install Python 3.11+ or create a venv.
  goto :error
)

"%PYTHON_EXE%" %PYTHON_ARGS% -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] <= (3,12) else 1)"
if errorlevel 1 (
  echo [ERROR] Python 3.11 or 3.12 is required for backend dependencies ^(MediaPipe/TTS^).
  echo         Detected interpreter: %PYTHON_EXE% %PYTHON_ARGS%
  goto :error
)

echo [1/6] Installing Node dependencies...
npm install --include=dev
if errorlevel 1 goto :error

echo [2/6] Installing Python backend dependencies...
"%PYTHON_EXE%" %PYTHON_ARGS% -m pip install --upgrade pip
if errorlevel 1 goto :error

"%PYTHON_EXE%" %PYTHON_ARGS% -m pip install -r "python-backend\requirements.txt" pyinstaller
if errorlevel 1 goto :error

echo [3/6] Building React frontend (Vite)...
npm run build:frontend
if errorlevel 1 goto :error

echo [4/6] Packaging Python backend with PyInstaller...
if exist "python-backend\dist\echolink-backend" rmdir /s /q "python-backend\dist\echolink-backend"
"%PYTHON_EXE%" %PYTHON_ARGS% -m PyInstaller --noconfirm --clean --distpath "python-backend\dist" --workpath "python-backend\build" "python-backend\echolink-backend.spec"
if errorlevel 1 goto :error

if not exist "python-backend\dist\echolink-backend\echolink-backend.exe" (
  echo [ERROR] PyInstaller output missing: python-backend\dist\echolink-backend\echolink-backend.exe
  goto :error
)

echo [5/6] Building Electron Windows installer...
npm run build:electron
if errorlevel 1 goto :error

echo [6/6] Build complete.
echo.
echo Installer output:
for %%F in ("release\*Setup-*.exe" "release\*.exe") do (
  if exist "%%~fF" echo   %%~fF
)
echo.
echo ============================================================
echo   Success - Windows installer generated in .\release
echo ============================================================
echo.
exit /b 0

:error
echo.
echo ============================================================
echo   Build failed.
echo ============================================================
echo.
exit /b 1

:try_python_path
set "_candidate=%~1"
if exist "%_candidate%" (
  "%_candidate%" --version >nul 2>nul
  if not errorlevel 1 (
    set "PYTHON_EXE=%_candidate%"
    set "PYTHON_ARGS="
  )
)
exit /b 0