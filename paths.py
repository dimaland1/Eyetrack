"""Chemins du projet, centralisés.

- models/   : modèles téléchargés automatiquement (hors git)
- data/     : calibrations, cache caméra, sorties de diagnostic (hors git, propres à la machine)
- profiles/ : profils de jeux (versionnés)

Les chemins sont absolus (calculés depuis ce fichier) : les scripts fonctionnent quel que
soit le dossier depuis lequel on les lance.
"""

from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
PROFILES_DIR = ROOT / "profiles"
ASSETS_DIR = ROOT / "assets"  # icône de l'application (versionnée)
ICON_FILE = ASSETS_DIR / "eyetrack.ico"


def model_path(filename: str) -> str:
    """Chemin d'un modèle dans models/ (dossier créé si besoin)."""
    MODELS_DIR.mkdir(exist_ok=True)
    return str(MODELS_DIR / filename)


def data_path(filename: str) -> str:
    """Chemin d'un fichier de données dans data/ (dossier créé si besoin)."""
    DATA_DIR.mkdir(exist_ok=True)
    return str(DATA_DIR / filename)
