$ErrorActionPreference = "Stop"

Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    Write-Host ""
    Write-Host $Title
    Write-Host ("-" * $Title.Length)
    & $Action
    if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) {
        throw "$Title failed with exit code $LASTEXITCODE"
    }
}

function Test-Command {
    param([Parameter(Mandatory = $true)][string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-PythonVersion {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$Args = @()
    )

    $code = "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    try {
        $output = & $Exe @Args -c $code 2>$null
    } catch {
        $global:LASTEXITCODE = 1
        return $null
    }
    if ($LASTEXITCODE -ne 0 -or -not $output) {
        return $null
    }

    try {
        return [version]($output | Select-Object -First 1)
    } catch {
        return $null
    }
}

function Find-Python {
    $candidates = @()

    if ($env:ECHOLINK_PYTHON) {
        $candidates += @{ Exe = $env:ECHOLINK_PYTHON; Args = @() }
    }

    foreach ($path in @(
        ".venv\Scripts\python.exe",
        "venv\Scripts\python.exe",
        "python-backend\.venv\Scripts\python.exe",
        "python-backend\venv\Scripts\python.exe"
    )) {
        if (Test-Path -LiteralPath $path) {
            $candidates += @{ Exe = (Resolve-Path -LiteralPath $path).Path; Args = @() }
        }
    }

    if (Test-Command "py") {
        $candidates += @{ Exe = "py"; Args = @("-3.12") }
        $candidates += @{ Exe = "py"; Args = @("-3.11") }
    }

    if (Test-Command "python") {
        $candidates += @{ Exe = "python"; Args = @() }
    }

    if (Test-Command "py") {
        $candidates += @{ Exe = "py"; Args = @("-3") }
    }

    foreach ($candidate in $candidates) {
        $version = Get-PythonVersion -Exe $candidate.Exe -Args $candidate.Args
        if ($version) {
            return [pscustomobject]@{
                Exe = $candidate.Exe
                Args = $candidate.Args
                Version = $version
            }
        }
    }

    throw "No working Python interpreter found. Install Python or set ECHOLINK_PYTHON."
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]$Python,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $Python.Exe @($Python.Args + $Arguments)
}

if (-not (Test-Command "node")) {
    throw "Node.js was not found in PATH."
}
if (-not (Test-Command "npm.cmd")) {
    throw "npm.cmd was not found in PATH."
}

$python = Find-Python
Write-Host "Using Python: $($python.Exe) $($python.Args -join ' ') ($($python.Version))"

Invoke-Step "[1/6] Installing Node dependencies" {
    & npm.cmd install --include=dev
}

Invoke-Step "[2/6] Installing Python build dependencies" {
    Invoke-Python $python @("-m", "pip", "install", "--upgrade", "pip")

    if ($python.Version.Major -eq 3 -and $python.Version.Minor -le 12) {
        Invoke-Python $python @("-m", "pip", "install", "-r", "python-backend\requirements.txt", "pyinstaller")
    } else {
        Write-Host "Python $($python.Version) detected; installing compatible backend packages instead of strict pins."
        Invoke-Python $python @(
            "-m", "pip", "install",
            "fastapi",
            "uvicorn[standard]",
            "websockets",
            "opencv-python-headless",
            "numpy",
            "sounddevice",
            "onnxruntime",
            "openvino",
            "pyvirtualcam",
            "piper-tts",
            "pyttsx3",
            "mediapipe",
            "pyinstaller"
        )
    }
}

Invoke-Step "[3/6] Building React frontend (Vite)" {
    & npm.cmd run build:frontend
}

Invoke-Step "[4/6] Packaging Python backend with PyInstaller" {
    $backendDistRoot = "python-backend\dist"
    $backendBuildRoot = "python-backend\build"
    $backendDist = "$backendDistRoot\echolink-backend"
    if (Test-Path -LiteralPath $backendDist) {
        Remove-Item -LiteralPath $backendDist -Recurse -Force
    }

    Invoke-Python $python @(
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        $backendDistRoot,
        "--workpath",
        $backendBuildRoot,
        "python-backend\echolink-backend.spec"
    )

    $backendExe = "$backendDist\echolink-backend.exe"
    if (-not (Test-Path -LiteralPath $backendExe)) {
        throw "PyInstaller output missing: $backendExe"
    }
}

Invoke-Step "[5/6] Building Electron Windows installer" {
    & npm.cmd run build:electron
}

Write-Host ""
Write-Host "[6/6] Build complete"
Write-Host "Installer output:"
Get-ChildItem -Path "release" -Filter "*.exe" -ErrorAction SilentlyContinue |
    ForEach-Object { Write-Host "  $($_.FullName)" }
