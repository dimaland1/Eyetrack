"""Modules de recherche expérimentale pour le dossier de niveau thèse Eyetrack.

Regroupe :
- l'extraction de patchs oculaires géométriquement alignés (96x48),
- les protocoles d'acquisition solo (Smooth Pursuit et clics passifs avec buffer),
- l'estimation du décalage visuo-moteur,
- les architectures légères personnalisées (Scénario A) et différentielles (Scénario B),
- les bancs d'évaluation, d'ablation et de génération de figures scientifiques.
"""

from __future__ import annotations

from pathlib import Path
import sys


def _preload_stdlib_cprofile() -> None:
    """Évite le masquage de stdlib ``profile`` par ``Eyetrack/profile.py`` lors d'un import PyTorch."""
    if "cProfile" in sys.modules:
        return
    root_dir = str(Path(__file__).resolve().parent.parent).lower()
    saved_path = sys.path[:]
    existing_profile = sys.modules.pop("profile", None)
    try:
        sys.path = [p for p in saved_path if p not in ("", ".") and str(Path(p).resolve()).lower() != root_dir]
        import cProfile  # noqa: F401
    finally:
        sys.path = saved_path
        if existing_profile is not None:
            sys.modules["profile"] = existing_profile
        else:
            sys.modules.pop("profile", None)


_preload_stdlib_cprofile()
