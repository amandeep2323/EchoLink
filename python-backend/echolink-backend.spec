# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


project_dir = Path(globals().get("SPECPATH", ".")).resolve()


def safe_collect_data(module_name: str):
    try:
        return collect_data_files(module_name)
    except Exception:
        return []


def safe_collect_submodules(module_name: str):
    try:
        return collect_submodules(module_name)
    except Exception:
        return []


datas = []

# Bundle the models directory but EXCLUDE HuggingFace cache folders that
# contain Windows-incompatible symlinks (models/tts/*/models--microsoft--*).
# These TTS models auto-download on first run — they don't need bundling.
_models_dir = project_dir / "models"
_skip_patterns = [
    "models--microsoft--",         # HuggingFace symlink caches (TTS)
    "model4\\saved_model",         # Conversion intermediate
    "model4/saved_model",          # Conversion intermediate (posix)
    "test_vidoes",                 # Test videos (not needed in prod)
    "best_model_2731.keras",       # Original source file (27 MB, only for reconversion)
]

for item in _models_dir.rglob("*"):
    if not item.is_file():
        continue
    # Skip HuggingFace cache symlink trees
    rel = str(item.relative_to(project_dir))
    if any(pat in rel for pat in _skip_patterns):
        continue
    dest_dir = str(item.parent.relative_to(project_dir))
    datas.append((str(item), dest_dir))

datas += safe_collect_data("mediapipe")
datas += safe_collect_data("onnxruntime")
datas += safe_collect_data("openvino")
datas += safe_collect_data("piper")

hiddenimports = []
hiddenimports += safe_collect_submodules("uvicorn")
hiddenimports += safe_collect_submodules("starlette")
hiddenimports += safe_collect_submodules("mediapipe")
hiddenimports += safe_collect_submodules("onnxruntime")
hiddenimports += safe_collect_submodules("openvino")
hiddenimports += safe_collect_submodules("piper")


a = Analysis(
    [str(project_dir / "main.py")],
    pathex=[str(project_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name="echolink-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    exclude_binaries=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="echolink-backend",
)
