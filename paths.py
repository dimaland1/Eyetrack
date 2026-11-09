"""Chemins du projet, centralisés.

Lancé depuis les sources (python main.py) :
- models/   : modèles téléchargés automatiquement (hors git)
- data/     : calibrations, cache caméra, sorties de diagnostic (hors git, propres à la machine)
- profiles/ : profils de jeux (versionnés)

Lancé depuis l'exécutable (Eyetrack.exe, PyInstaller) :
- modèles, icône et profils de référence sont dans le paquet (lecture seule) ;
- données et profils de l'utilisateur sont dans %LOCALAPPDATA%\\Eyetrack : mettre à jour ou
  réinstaller l'exécutable ne les efface jamais, et ils restent séparés de ceux des sources
  (on peut développer une nouvelle version sans toucher à la calibration du jeu).

Les chemins sont absolus : les scripts fonctionnent quel que soit le dossier de lancement.
"""

from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))

if FROZEN:
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    USER_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Eyetrack"
    ROOT = BUNDLE_DIR
    MODELS_DIR = BUNDLE_DIR / "models"
    DATA_DIR = USER_DIR / "data"
    PROFILES_DIR = USER_DIR / "profiles"
    DEFAULT_PROFILES_DIR = BUNDLE_DIR / "profiles"
    LOG_FILE = USER_DIR / "eyetrack.log"
else:
    ROOT = Path(__file__).resolve().parent
    BUNDLE_DIR = ROOT
    USER_DIR = ROOT
    MODELS_DIR = ROOT / "models"
    DATA_DIR = ROOT / "data"
    PROFILES_DIR = ROOT / "profiles"
    DEFAULT_PROFILES_DIR = PROFILES_DIR
    LOG_FILE = DATA_DIR / "eyetrack.log"

ASSETS_DIR = BUNDLE_DIR / "assets"  # icône de l'application (versionnée)
ICON_FILE = ASSETS_DIR / "eyetrack.ico"


def model_path(filename: str) -> str:
    """Chemin d'un modèle dans models/ (dossier créé si besoin)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return str(MODELS_DIR / filename)


def data_path(filename: str) -> str:
    """Chemin d'un fichier de données dans data/ (dossier créé si besoin)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return str(DATA_DIR / filename)


def prepare_user_dirs() -> None:
    """Crée les dossiers de l'utilisateur et y copie les profils de référence manquants.

    Ne remplace jamais un profil existant : les réglages sauvegardés de l'utilisateur sont gardés.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    if DEFAULT_PROFILES_DIR != PROFILES_DIR and DEFAULT_PROFILES_DIR.is_dir():
        for src in DEFAULT_PROFILES_DIR.glob("*.json"):
            dst = PROFILES_DIR / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
