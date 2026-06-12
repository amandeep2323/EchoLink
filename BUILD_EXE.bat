@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo ============================================================
echo   EchoLink - Windows Installer Build
echo ============================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\build-exe.ps1"
set "BUILD_EXIT=%ERRORLEVEL%"

echo.
if "%BUILD_EXIT%"=="0" (
  echo Build completed successfully.
) else (
  echo Build failed with exit code %BUILD_EXIT%.
)
echo.
pause
exit /b %BUILD_EXIT%
