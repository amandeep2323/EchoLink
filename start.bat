@echo off
title SignSpeak — ASL to Speech
color 0A

echo ============================================================
echo   SignSpeak — ASL to Speech Desktop Application
echo ============================================================
echo.
echo   Starting backend and frontend servers...
echo.

:: ── Check Python ──
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo   [ERROR] Python not found in PATH
    echo   Install Python 3.11+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: ── Check Node.js ──
where node >nul 2>nul
if %errorlevel% neq 0 (
    echo   [ERROR] Node.js not found in PATH
    echo   Install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)

:: ── Start Backend ──
echo   [1/2] Starting Python backend...
start "SignSpeak Backend" powershell -NoExit -Command ^
    "Set-Location '%~dp0python-backend'; ^
    if (Test-Path '..\venv\Scripts\Activate.ps1') { & '..\venv\Scripts\Activate.ps1' } ^
    elseif (Test-Path '.\venv\Scripts\Activate.ps1') { & '.\venv\Scripts\Activate.ps1' } ^
    elseif (Test-Path '../.venv/Scripts/Activate.ps1') { & '../.venv/Scripts/Activate.ps1' } ^
    elseif (Test-Path '.\.venv\Scripts\Activate.ps1') { & '.\.venv\Scripts\Activate.ps1' }; ^
    python main.py"

:: Wait for backend to start
echo   Waiting for backend to start...
timeout /t 3 /nobreak >nul

:: ── Start Frontend ──
echo   [2/2] Starting frontend dev server...
start "SignSpeak Frontend" powershell -NoExit -Command ^
    "Set-Location '%~dp0'; ^
    if (-not (Test-Path 'node_modules')) { echo 'Installing dependencies...'; npm install }; ^
    npm run dev"

echo.
echo ============================================================
echo   SignSpeak is starting!
echo.
echo   Backend:  ws://127.0.0.1:8765/ws
echo   Frontend: http://localhost:5173
echo.
echo   Close this window or press Ctrl+C to stop.
echo ============================================================
echo.

:: Wait for user
pause
