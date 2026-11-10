"""Réglages de l'application, indépendants des profils de jeu.

- last_profile     : profil repris au lancement par double-clic sur Eyetrack.exe ;
- launch_opentrack : ouvrir OpenTrack en même temps qu'Eyetrack (s'il n'est pas déjà ouvert) ;
- opentrack_path   : chemin d'opentrack.exe si l'installation n'est pas à l'emplacement habituel ;
- language         : langue de l'interface, "auto" (celle de Windows), "fr" ou "en" (cf. i18n.py).

Stockés dans data/app_settings.json. Un fichier absent ou abîmé redonne les valeurs par défaut.
"""

from __future__ import annotations
import json
import os

from paths import data_path

FILE_NAME = "app_settings.json"
DEFAULTS: dict = {"last_profile": "default", "launch_opentrack": True, "opentrack_path": "", "language": "auto"}


def load() -> dict:
    """Réglages courants (valeurs par défaut pour les clés absentes ou de mauvais type)."""
    settings = dict(DEFAULTS)
    try:
        with open(data_path(FILE_NAME), encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            for key, default in DEFAULTS.items():
                if isinstance(saved.get(key), type(default)):
                    settings[key] = saved[key]
    except (OSError, ValueError):
        pass
    return settings


def update(**changes) -> dict:
    """Modifie et enregistre des réglages ; renvoie les réglages complets."""
    settings = load()
    for key, value in changes.items():
        if key in DEFAULTS and isinstance(value, type(DEFAULTS[key])):
            settings[key] = value
    path = data_path(FILE_NAME)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)  # écriture atomique : jamais de fichier à moitié écrit
    except OSError as e:
        print(f"[réglages] Enregistrement impossible : {e}")
    return settings
