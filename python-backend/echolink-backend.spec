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


datas = [
    (str(project_dir / "models"), "models"),
]

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
