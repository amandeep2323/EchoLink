@echo off
setlocal EnableExtensions

title EchoLink — Dev Launch
color 0A

set "ROOT=%~dp0"
set "BACKEND_DIR=%ROOT%python-backend"

echo ============================================================
echo   EchoLink — Dev Launch
echo ============================================================
echo.

:: ── Check Node.js + npm ──
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo   [ERROR] Node.js not found in PATH
    echo   Install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)

where npm >nul 2>nul
if %errorlevel% neq 0 (
    echo   [ERROR] npm not found in PATH
    pause
    exit /b 1
)

:: ── Install npm deps if needed ──
if not exist "%ROOT%node_modules\" (
    echo   [1/3] Installing Node dependencies...
    pushd "%ROOT%"
    npm install
    popd
)

:: ── Locate EchoLink venv python ──
set "VENV_PY="
if exist "%ROOT%.venv\Scripts\python.exe" set "VENV_PY=%ROOT%.venv\Scripts\python.exe"
if not defined VENV_PY if exist "%ROOT%venv\Scripts\python.exe" set "VENV_PY=%ROOT%venv\Scripts\python.exe"
if not defined VENV_PY if exist "%BACKEND_DIR%\.venv\Scripts\python.exe" set "VENV_PY=%BACKEND_DIR%\.venv\Scripts\python.exe"
if not defined VENV_PY if exist "%BACKEND_DIR%\venv\Scripts\python.exe" set "VENV_PY=%BACKEND_DIR%\venv\Scripts\python.exe"
if not defined VENV_PY (
    where python >nul 2>nul
    if %errorlevel% neq 0 (
        echo   [ERROR] Python not found. Install Python 3.11+ or create .venv.
        pause
        exit /b 1
    )
    echo   [WARN] No .venv found — using system python
)

:: ── Start EchoLink Python backend (port 8765) ──
echo   [2/3] Starting EchoLink backend (port 8765)...
if defined VENV_PY (
    start "EchoLink Backend" cmd /k "cd /d "%BACKEND_DIR%" && "%VENV_PY%" main.py"
) else (
    start "EchoLink Backend" cmd /k "cd /d "%BACKEND_DIR%" && python main.py"
)

:: ── Launch Electron dev mode (Vite + Electron together) ──
echo   [3/3] Launching Electron (Vite dev server + app window)...
echo.
echo   The app window will open automatically once Vite is ready.
echo   AvatarLink mode is accessible from the EchoLink header dropdown.
echo.
echo   Close this window to stop the dev servers.
echo ============================================================
echo.

pushd "%ROOT%"
npm run electron:dev
popd
