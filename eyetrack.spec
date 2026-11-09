# -*- mode: python ; coding: utf-8 -*-
"""Recette PyInstaller d'Eyetrack.exe. Ne pas lancer directement : python tools/build_exe.py.

Exécutable fenêtré (aucune console), en dossier (démarrage rapide : rien à décompresser à chaque
lancement). Modèles, icône et profils de référence sont dans le paquet.
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / "models" / "face_landmarker.task"), "models"),
    (str(ROOT / "models" / "mobileone_s0_gaze.onnx"), "models"),
    (str(ROOT / "assets" / "eyetrack.ico"), "assets"),
    (str(ROOT / "profiles" / "*.json"), "profiles"),
]
build_info = ROOT / "build" / "build_info.json"
if build_info.exists():
    datas.append((str(build_info), "."))

# MediaPipe charge libmediapipe.dll par ctypes depuis mediapipe/tasks/c : à copier tel quel.
# (matplotlib reste : mediapipe.tasks.python.vision l'importe pour ses outils de dessin.)
datas += collect_data_files("mediapipe", excludes=["**/__pycache__"])
binaries = collect_dynamic_libs("mediapipe")
hiddenimports = ["mediapipe.tasks.c"]

a = Analysis(
    [str(ROOT / "eyetrack_app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # matplotlib sans fenêtre (moteur Agg) : sinon il embarque PyQt5 (~76 Mo) pour rien
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
    excludes=["scipy", "pandas", "IPython", "torch", "tensorflow", "jax", "onnx", "pygame", "pythonnet", "clr",
              "clr_loader", "psutil", "yaml", "setuptools", "pkg_resources", "pyreadline3", "win32com", "pythoncom",
              "PyQt5", "PyQt6", "PySide2", "PySide6", "wx", "gi", "cairo"],
    noarchive=False,
)
# FFmpeg d'OpenCV (~28 Mo chacun) : inutile, la webcam passe par DirectShow / Media Foundation
a.binaries = [b for b in a.binaries if "opencv_videoio_ffmpeg" not in b[0]]
a.datas = [d for d in a.datas if "opencv_videoio_ffmpeg" not in d[0]]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Eyetrack",
    icon=str(ROOT / "assets" / "eyetrack.ico"),
    console=False,  # pas de fenêtre de terminal
    upx=False,      # UPX déclenche souvent les antivirus
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Eyetrack")
