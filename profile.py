"""Module de gestion des profils utilisateur en JSON (Étape 7).

Permet de charger, sauvegarder et basculer entre des configurations
adaptées à différents jeux et simulateurs (ETS2, MSFS, Simracing, Bureau).
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict, field

import paths


@dataclass
class Profile:
    """Profil complet de configuration pour le tracker de tête et de regard."""

    name: str = "default"
    description: str = "Profil polyvalent par défaut pour jeux et simulateurs"

    # Mode d'estimation
    gaze_mode: str = "net"  # 'net' (ONNX MobileOne) ou 'iris' (MVP ratios)
    calib_points: int = 9   # 5 (rapide façon Beam) ou 9 (dense)

    # Paramètres écran
    screen_diag: float = 27.0
    distance_cm: float = 60.0

    # Sortie OpenTrack UDP
    opentrack_host: str = "127.0.0.1"
    opentrack_port: int = 4242

    # Contrôle de caméra par le regard (gaming)
    gaze_yaw_gain: float = 0.50     # Multiplicateur horizontal
    gaze_pitch_gain: float = 0.65   # Multiplicateur vertical
    gaze_deadzone: float = 0.10     # Zone morte douce [0..1] (atténue sans couper)
    gaze_curve_exp: float = 1.20    # Exposant de réponse (1.0 = linéaire)
    gaze_smoothing_s: float = 0.25  # Lissage de la part regard de la caméra (s)

    # Rotation de la tête (1.0 = 1:1 ; >1 amplifie, comme les courbes TrackIR/OpenTrack)
    head_yaw_gain: float = 1.5
    head_pitch_gain: float = 1.2

    # Envoi à OpenTrack (indépendant des 30 images/s de la webcam)
    send_rate_hz: float = 120.0       # 0 = un envoi par image (ancien comportement)
    output_smoothing_s: float = 0.015  # transforme les 30 sauts/s en mouvement continu
    prediction_s: float = 0.010        # avance de la prédiction (rattrape le lissage)

    # Lissage One Euro : réglages séparés car les unités diffèrent
    # Pose tête (degrés et cm)
    pose_min_cutoff: float = 1.0    # Fréquence min en Hz (stabilité au repos)
    pose_beta: float = 0.15         # Ouverture du filtre avec la vitesse (par °/s)
    # Point de regard écran (pixels)
    # Réglé par simulation sur le bruit mesuré du réseau (~2,3° ≈ 100 px par frame) :
    # jitter en fixation 69 px -> ~22 px, latence de saccade ~110 ms.
    gaze_min_cutoff: float = 0.4    # Le regard est bien plus bruité : lissage plus fort au repos
    gaze_beta: float = 0.0005       # Faible : sinon le bruit lui-même ouvre le filtre
    gaze_median: int = 5            # Pré-filtre médian (frames) contre les pics du réseau

    # Cadrage du visage pour le réseau (focale virtuelle, cf. diag_regard.py)
    patch_focal: float = 1900.0

    # Inversion des axes
    invert_yaw: bool = False
    invert_pitch: bool = False
    invert_roll: bool = False
    invert_x: bool = False
    invert_y: bool = False
    invert_z: bool = False

    # Options d'interface
    overlay_enabled: bool = False   # Overlay transparent click-through in-game
    # Recalibration implicite RLS aux clics. Désactivée dans les profils de jeu : la souris y
    # est souvent bloquée au centre et chaque clic apprendrait une fausse cible. Dans tous les
    # cas, les clics faits pendant que le curseur est caché (jeu) sont ignorés.
    recalib_on_click: bool = True
    calibration_file: str = ""      # Chemin du fichier .npz associé (auto si vide)

    # Raccourcis globaux, actifs même quand le jeu a le focus ("" = désactivé).
    # Éviter les lettres : beaucoup de jeux voient quand même la touche (BeamNG : R
    # réinitialise le véhicule). Si Ctrl/Maj passent les vitesses (ETS2, boîte au clavier),
    # prendre une touche seule que le jeu n'utilise pas, ex. "pause" ou "scrolllock".
    hotkey_recenter: str = "ctrl+shift+f9"
    hotkey_gaze: str = "ctrl+shift+f10"
    hotkey_pause: str = "ctrl+shift+f11"

    # Changement de profil automatique : exécutables des jeux qui activent ce profil (sans casse).
    # null (absent) = liste d'Eyetrack (GAME_PROCESSES) ; [] = jamais automatiquement.
    game_processes: list[str] | None = None

    def to_dict(self) -> dict:
        """Sérialise le profil en dictionnaire JSON-compatible."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Profile:
        """Désérialise un profil depuis un dictionnaire en ignorant les champs obsolètes."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


PROFILES_DIR = str(paths.PROFILES_DIR)


def ensure_profiles_dir() -> str:
    """Crée le dossier profiles s'il n'existe pas."""
    os.makedirs(PROFILES_DIR, exist_ok=True)
    return PROFILES_DIR


def list_profiles() -> list[str]:
    """Liste tous les noms de profils JSON disponibles."""
    ensure_profiles_dir()
    profiles = []
    for f in os.listdir(PROFILES_DIR):
        if f.endswith(".json"):
            profiles.append(os.path.splitext(f)[0])
    return sorted(profiles)


def get_profile_path(name_or_path: str) -> str:
    """Résout le chemin d'un profil par son nom ou chemin absolu/relatif."""
    # Un chemin explicite (absolu ou avec un séparateur) est utilisé tel quel ; un nom simple
    # (ex. "ets2") désigne un profil du dossier profiles/.
    if os.path.isabs(name_or_path) or "/" in name_or_path or "\\" in name_or_path:
        if not name_or_path.endswith(".json"):
            name_or_path = f"{name_or_path}.json"
        return name_or_path
    if not name_or_path.endswith(".json"):
        name_or_path = f"{name_or_path}.json"
    return os.path.join(PROFILES_DIR, name_or_path)


def load_profile(name_or_path: str = "default") -> Profile:
    """Charge un profil depuis un fichier JSON. Retourne le profil par défaut si introuvable."""
    ensure_profiles_dir()
    path = get_profile_path(name_or_path)

    if not os.path.isfile(path):
        # Création et sauvegarde du profil par défaut si absent
        profile = create_builtin_profile(name_or_path)
        save_profile(profile, path)
        return profile

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        profile = Profile.from_dict(data)
        return profile
    except Exception as e:
        print(f"[profile] Erreur lors de la lecture de '{path}' : {e}. Utilisation des valeurs par défaut.")
        return Profile()


def save_profile(profile: Profile, name_or_path: str | None = None) -> str:
    """Sauvegarde un profil au format JSON lisible."""
    ensure_profiles_dir()
    if name_or_path is None:
        name_or_path = profile.name
    path = get_profile_path(name_or_path)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=4, ensure_ascii=False)
    return path


def create_builtin_profile(name: str) -> Profile:
    """Instancie un profil calibré pour un jeu ou cas d'usage spécifique."""
    lower = name.lower()
    if "ets2" in lower or "truck" in lower:
        return Profile(
            name="ets2",
            recalib_on_click=False,
            description="Euro Truck Simulator 2 / ATS : grand angle pour rétroviseurs, pitch stable",
            gaze_mode="net",
            gaze_yaw_gain=0.5,
            gaze_pitch_gain=0.3,
            gaze_deadzone=0.1,
            gaze_curve_exp=1.25,
            gaze_smoothing_s=0.3,
            head_yaw_gain=2.5,
            head_pitch_gain=1.5,
            pose_min_cutoff=1.0,
            pose_beta=0.15,
            gaze_min_cutoff=0.4,
            gaze_beta=0.0005,
        )
    elif "flight" in lower or "msfs" in lower or "dcs" in lower:
        return Profile(
            name="flight_sim",
            recalib_on_click=False,
            description="MSFS / DCS / X-Plane : zone morte réduite pour scanner les cadrans du cockpit",
            gaze_mode="net",
            gaze_yaw_gain=0.55,
            gaze_pitch_gain=0.55,
            gaze_deadzone=0.07,
            gaze_curve_exp=1.1,
            gaze_smoothing_s=0.25,
            head_yaw_gain=3.0,
            head_pitch_gain=2.0,
            pose_min_cutoff=0.8,
            pose_beta=0.12,
            gaze_min_cutoff=0.45,
            gaze_beta=0.0005,
        )
    elif "beamng" in lower:
        return Profile(
            name="beamng",
            recalib_on_click=False,
            description="BeamNG.drive : conduite, regard horizontal pour les virages, pas de dérive verticale",
            gaze_mode="net",
            gaze_yaw_gain=0.75,
            gaze_pitch_gain=0.25,
            gaze_deadzone=0.12,
            gaze_curve_exp=1.3,
            gaze_smoothing_s=0.25,
            head_yaw_gain=2.0,
            head_pitch_gain=1.2,
            pose_min_cutoff=1.2,
            pose_beta=0.18,
            gaze_min_cutoff=0.4,
            gaze_beta=0.0005,
        )
    elif "race" in lower or "racing" in lower or "assetto" in lower or "iracing" in lower:
        return Profile(
            name="racing",
            recalib_on_click=False,
            description="Assetto Corsa / iRacing : focalisation sur le point de corde sans dérive verticale",
            gaze_mode="net",
            gaze_yaw_gain=0.75,
            gaze_pitch_gain=0.25,
            gaze_deadzone=0.12,
            gaze_curve_exp=1.3,
            gaze_smoothing_s=0.25,
            head_yaw_gain=2.0,
            head_pitch_gain=1.2,
            pose_min_cutoff=1.2,
            pose_beta=0.18,
            gaze_min_cutoff=0.4,
            gaze_beta=0.0005,
        )
    else:
        return Profile(
            name="default",
            description="Profil standard équilibré pour jeux PC et contrôle de caméra OpenTrack",
            gaze_mode="net",
            gaze_yaw_gain=0.50,
            gaze_pitch_gain=0.65,
            gaze_deadzone=0.1,
            gaze_curve_exp=1.2,
            gaze_smoothing_s=0.25,
            head_yaw_gain=1.5,
            head_pitch_gain=1.2,
            pose_min_cutoff=1.0,
            pose_beta=0.15,
            gaze_min_cutoff=0.4,
            gaze_beta=0.0005,
        )


def initialize_all_default_profiles() -> None:
    """Génère tous les profils de référence si le répertoire est vide ou incomplet."""
    ensure_profiles_dir()
    for name in ["default", "ets2", "beamng", "flight_sim", "racing"]:
        path = os.path.join(PROFILES_DIR, f"{name}.json")
        if not os.path.exists(path):
            p = create_builtin_profile(name)
            save_profile(p, path)


# --------------------------------------------------------------------------- profil selon le jeu
# Exécutables des jeux reconnus, par profil (vérifiés sur une installation Steam : eurotrucks2.exe,
# BeamNG.drive.x64.exe ; les autres d'après leur nom habituel). Un profil peut définir sa propre
# liste dans son JSON (« game_processes »).
GAME_PROCESSES: dict[str, list[str]] = {
    "ets2": ["eurotrucks2.exe", "amtrucks.exe"],
    "beamng": ["BeamNG.drive.x64.exe", "BeamNG.drive.exe"],
    "racing": ["acs.exe", "AC2-Win64-Shipping.exe", "iRacingSim64DX11.exe"],
    "flight_sim": ["FlightSimulator.exe", "FlightSimulator2024.exe", "X-Plane.exe", "DCS.exe"],
}
GAME_NAMES: dict[str, str] = {
    "eurotrucks2.exe": "Euro Truck Simulator 2", "amtrucks.exe": "American Truck Simulator",
    "beamng.drive.x64.exe": "BeamNG.drive", "beamng.drive.exe": "BeamNG.drive",
    "acs.exe": "Assetto Corsa", "ac2-win64-shipping.exe": "Assetto Corsa Competizione",
    "iracingsim64dx11.exe": "iRacing", "flightsimulator.exe": "Microsoft Flight Simulator",
    "flightsimulator2024.exe": "Microsoft Flight Simulator 2024", "x-plane.exe": "X-Plane",
    "dcs.exe": "DCS World",
}


def game_processes_of(profile: Profile) -> list[str]:
    """Exécutables (en minuscules) qui activent ce profil."""
    procs = profile.game_processes if profile.game_processes is not None else GAME_PROCESSES.get(profile.name, [])
    return [p.strip().lower() for p in procs if isinstance(p, str) and p.strip()]


def game_profile_map() -> dict[str, list[str]]:
    """Profil -> exécutables de jeu, pour les profils présents qui en ont."""
    out: dict[str, list[str]] = {}
    for name in list_profiles():
        try:
            procs = game_processes_of(load_profile(name))
        except Exception:
            continue
        if procs:
            out[name] = procs
    return out


def detect_game(running: set[str], mapping: dict[str, list[str]]) -> tuple[str, str] | None:
    """(profil, exécutable) du premier jeu reconnu parmi les processus lancés (noms en minuscules)."""
    for name in sorted(mapping):
        for exe in mapping[name]:
            if exe in running:
                return name, exe
    return None


def game_name(exe: str) -> str:
    return GAME_NAMES.get(exe.lower(), exe)


class AutoProfile:
    """Choisit le profil du jeu lancé, puis revient au profil d'avant quand le jeu se ferme.

    Un changement de profil fait à la main pendant le jeu est respecté : rien n'est imposé, et
    le profil choisi reste en place à la fermeture du jeu.
    """

    def __init__(self) -> None:
        self.game: str | None = None          # exécutable du jeu détecté
        self.game_profile: str | None = None  # profil chargé pour lui
        self.previous: str | None = None      # profil d'avant le jeu
        self.manual = False                   # profil changé à la main pendant le jeu

    def update(self, detected: tuple[str, str] | None, current: str) -> str | None:
        """Profil à charger maintenant (ou None) d'après le jeu détecté et le profil actif."""
        if detected is None:
            if self.game is None:
                return None
            previous, manual = self.previous, self.manual
            self.__init__()
            return None if manual or previous is None or previous == current else previous
        profile, exe = detected
        if exe == self.game:
            if current != self.game_profile:
                self.manual = True
            return None
        if self.game is None:
            self.previous = current
        self.game, self.game_profile, self.manual = exe, profile, False
        return profile if profile != current else None
