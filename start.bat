@echo off
setlocal EnableExtensions

title EchoLink — ASL to Speech
color 0A

set "ROOT=%~dp0"
set "BACKEND_DIR=%ROOT%python-backend"

echo ============================================================
echo   EchoLink — ASL to Speech Desktop Application
echo ============================================================
echo.
echo   Starting backend and frontend servers...
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
    echo   Install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)

:: ── Locate venv python ──
set "VENV_PY="
if exist "%ROOT%venv\Scripts\python.exe" set "VENV_PY=%ROOT%venv\Scripts\python.exe"
if not defined VENV_PY if exist "%ROOT%.venv\Scripts\python.exe" set "VENV_PY=%ROOT%.venv\Scripts\python.exe"
if not defined VENV_PY if exist "%BACKEND_DIR%\venv\Scripts\python.exe" set "VENV_PY=%BACKEND_DIR%\venv\Scripts\python.exe"
if not defined VENV_PY if exist "%BACKEND_DIR%\.venv\Scripts\python.exe" set "VENV_PY=%BACKEND_DIR%\.venv\Scripts\python.exe"

:: ── Check Python (only if no venv found) ──
if not defined VENV_PY (
    where python >nul 2>nul
    if %errorlevel% neq 0 (
        echo   [ERROR] Python not found in PATH and no venv detected
        echo   Install Python 3.11+ from https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

:: ── Start Backend ──
echo   [1/2] Starting Python backend...
if not defined VENV_PY echo   [WARN] No venv found, using system python
set "PS_BACKEND=Set-Location -LiteralPath '%BACKEND_DIR%';"
if defined VENV_PY (
    set "PS_BACKEND=%PS_BACKEND% & '%VENV_PY%' main.py"
) else (
    set "PS_BACKEND=%PS_BACKEND% & python main.py"
)
start "EchoLink Backend" powershell -NoExit -Command "%PS_BACKEND%"

:: Wait for backend to start
echo   Waiting for backend to start...
timeout /t 3 /nobreak >nul

:: ── Start Frontend ──
echo   [2/2] Starting frontend dev server...
start "EchoLink Frontend" powershell -NoExit -Command "Set-Location -LiteralPath '%ROOT%'; if (-not (Test-Path 'node_modules')) { Write-Host 'Installing dependencies...'; npm install }; npm run dev"

echo.
echo ============================================================
echo   EchoLink is starting!
echo.
echo   Backend:  ws://127.0.0.1:8765/ws
echo   Frontend: http://localhost:5173
echo.
echo   Close this window or press Ctrl+C to stop.
echo ============================================================
echo.

:: Wait for user (skip in VS Code terminal)
if not defined VSCODE_PID pause
