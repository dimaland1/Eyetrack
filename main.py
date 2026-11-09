"""Point d'entrée principal du pipeline de suivi de tête et de regard (Eyetrack).

Connecte de façon modulaire :
1. capture      -> Acquisition webcam 60 fps
2. face         -> MediaPipe Face Landmarker (478 pts + pose 6DoF)
3. normalize    -> Préparation géométrique du visage
4. gaze_model   -> Estimation du regard (mode 'iris' ou 'net')
5. geometry     -> Dimensions écran et calculs angulaires
6. calibration  -> Procédure 9 points et modèle Ridge
7. filters      -> Lissage adaptatif One Euro
8. output       -> Envoi OpenTrack UDP et affichage HUD
"""

from __future__ import annotations
import argparse
import sys
import time
import cv2
import numpy as np

from capture import WebcamCapture
from face import FaceDetector
from normalize import FaceNormalizer
from gaze_model import create_gaze_estimator
from geometry import ScreenConfig, get_screen_resolution
from calibration import CalibrationManager
from filters import OneEuroFilter, LossBlender, BlinkDetector
from output import OpenTrackSender, Visualizer, OutputConfig
from overlay import GazeOverlay
from hotkeys import GlobalHotkeys, format_hotkey
from panel import ControlPanel, PANEL_FIELDS, make_preview
from paths import data_path
from profile import (
    Profile,
    load_profile,
    save_profile,
    list_profiles,
    initialize_all_default_profiles,
)


def parse_arguments() -> tuple[argparse.Namespace, Profile]:
    """Analyse les arguments en ligne de commande en les fusionnant avec un profil JSON."""
    initialize_all_default_profiles()

    # Détection préliminaire du profil demandé
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--profile", type=str, default="default", help="Nom du profil JSON")
    pre_args, _ = pre_parser.parse_known_args()

    active_profile = load_profile(pre_args.profile)

    parser = argparse.ArgumentParser(
        description="Eyetrack - Suivi de tête et de regard RGB pour OpenTrack"
    )

    # Profil JSON (Étape 7)
    available_profs = list_profiles()
    parser.add_argument(
        "--profile",
        type=str,
        default=active_profile.name,
        help=f"Profil de configuration ({', '.join(available_profs)})",
    )

    # Paramètres de capture
    parser.add_argument("--camera", type=int, default=0, help="Index de la caméra (défaut: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Largeur demandée (défaut: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Hauteur demandée (défaut: 720)")
    parser.add_argument("--fps", type=int, default=60, help="Framerate cible (défaut: 60)")
    parser.add_argument("--no-flip", action="store_true", help="Désactive l'effet miroir de la caméra")

    # Mode d'estimation du regard
    parser.add_argument(
        "--gaze-mode",
        type=str,
        choices=["iris", "net"],
        default=active_profile.gaze_mode,
        help=f"Mode d'estimation du regard (défaut profil: {active_profile.gaze_mode})",
    )
    parser.add_argument(
        "--calib-points",
        type=int,
        choices=[5, 9],
        default=active_profile.calib_points,
        help=f"Nombre de points pour la calibration (défaut profil: {active_profile.calib_points})",
    )

    # Dimensions écran
    parser.add_argument("--screen-w", type=int, default=None, help="Largeur écran en pixels (auto-détectée si omis)")
    parser.add_argument("--screen-h", type=int, default=None, help="Hauteur écran en pixels (auto-détectée si omis)")
    parser.add_argument("--screen-diag", type=float, default=active_profile.screen_diag, help="Diagonale écran en pouces")
    parser.add_argument("--distance", type=float, default=active_profile.distance_cm, help="Distance utilisateur-écran en cm")

    # Sortie OpenTrack UDP
    parser.add_argument("--host", type=str, default=active_profile.opentrack_host, help="Adresse IP OpenTrack")
    parser.add_argument("--port", type=int, default=active_profile.opentrack_port, help="Port UDP OpenTrack")

    # Inversion des axes
    parser.add_argument("--invert-yaw", action="store_true", default=active_profile.invert_yaw, help="Inverse l'axe Yaw")
    parser.add_argument("--invert-pitch", action="store_true", default=active_profile.invert_pitch, help="Inverse l'axe Pitch")
    parser.add_argument("--invert-roll", action="store_true", default=active_profile.invert_roll, help="Inverse l'axe Roll")
    parser.add_argument("--invert-x", action="store_true", default=active_profile.invert_x, help="Inverse l'axe translation X")
    parser.add_argument("--invert-y", action="store_true", default=active_profile.invert_y, help="Inverse l'axe translation Y")
    parser.add_argument("--invert-z", action="store_true", default=active_profile.invert_z, help="Inverse l'axe translation Z")

    # Contrôle de caméra par le regard (gaming)
    parser.add_argument("--gaze-yaw-gain", type=float, default=active_profile.gaze_yaw_gain, help="Gain regard Yaw caméra")
    parser.add_argument("--gaze-pitch-gain", type=float, default=active_profile.gaze_pitch_gain, help="Gain regard Pitch caméra")
    parser.add_argument("--gaze-deadzone", type=float, default=active_profile.gaze_deadzone, help="Zone morte centrale du regard")
    parser.add_argument("--gaze-curve-exp", type=float, default=active_profile.gaze_curve_exp, help="Exposant gamma de réponse progressive")
    parser.add_argument("--gaze-smoothing", type=float, default=active_profile.gaze_smoothing_s, help="Lissage de la part regard de la caméra (s)")
    parser.add_argument("--head-yaw-gain", type=float, default=active_profile.head_yaw_gain, help="Gain de rotation tête Yaw (1 = 1:1)")
    parser.add_argument("--head-pitch-gain", type=float, default=active_profile.head_pitch_gain, help="Gain de rotation tête Pitch (1 = 1:1)")

    # Envoi à OpenTrack (indépendant du framerate de la webcam)
    parser.add_argument("--send-rate", type=float, default=active_profile.send_rate_hz, help="Fréquence d'envoi à OpenTrack en Hz (0 = une fois par image)")
    parser.add_argument("--prediction", type=float, default=active_profile.prediction_s, help="Avance de la prédiction de la pose (s, 0 = aucune)")
    parser.add_argument("--output-smoothing", type=float, default=active_profile.output_smoothing_s, help="Lissage court de la sortie (s)")

    # Fonctionnalités avancées (Étapes 5 & 6)
    parser.add_argument("--overlay", action="store_true", default=active_profile.overlay_enabled, help="Active l'overlay transparent")
    parser.add_argument("--no-recalib", action="store_true", default=not active_profile.recalib_on_click, help="Désactive recalibration RLS")
    parser.add_argument("--no-panel", action="store_true", help="Sans panneau de contrôle (fenêtre Moniteur seule)")
    parser.add_argument("--moniteur", action="store_true", help="Ouvre aussi la fenêtre Moniteur (vidéo + HUD de diagnostic)")

    # Paramètres de lissage One Euro (séparés : pose en degrés/cm, regard en pixels)
    parser.add_argument("--pose-min-cutoff", type=float, default=active_profile.pose_min_cutoff, help="Fréquence min One Euro de la pose tête (Hz)")
    parser.add_argument("--pose-beta", type=float, default=active_profile.pose_beta, help="Beta One Euro de la pose tête")
    parser.add_argument("--gaze-min-cutoff", type=float, default=active_profile.gaze_min_cutoff, help="Fréquence min One Euro du regard (Hz)")
    parser.add_argument("--gaze-beta", type=float, default=active_profile.gaze_beta, help="Beta One Euro du regard")
    parser.add_argument("--gaze-median", type=int, default=active_profile.gaze_median, help="Pré-filtre médian du regard (frames)")

    # Cadrage du visage pour le réseau (mesurer avec diag_regard.py)
    parser.add_argument("--patch-focal", type=float, default=active_profile.patch_focal, help="Focale virtuelle du patch visage (défaut profil)")

    # Mode debug
    parser.add_argument("--debug", action="store_true", default=True, help="Affiche le HUD de debug et latences")

    args = parser.parse_args()
    return args, active_profile


def main() -> None:
    """Exécute la boucle principale du tracker."""
    args, current_profile = parse_arguments()
    available_profiles = list_profiles()
    current_prof_idx = (
        available_profiles.index(current_profile.name)
        if current_profile.name in available_profiles
        else 0
    )

    # Résolution écran
    auto_w, auto_h = get_screen_resolution()
    screen_w = args.screen_w if args.screen_w is not None else auto_w
    screen_h = args.screen_h if args.screen_h is not None else auto_h

    screen_config = ScreenConfig(
        width_px=screen_w,
        height_px=screen_h,
        diagonal_inches=args.screen_diag,
        distance_cm=args.distance,
    )

    print("=" * 60)
    print(f" Eyetrack - Profil actif : [{current_profile.name.upper()}]")
    print(f" Description     : {current_profile.description}")
    print(f" Caméra demandée : Index {args.camera} ({args.width}x{args.height} @ {args.fps} fps)")
    print(f" Écran cible     : {screen_w}x{screen_h} ({args.screen_diag}\" à {args.distance} cm)")
    print(f" Mode regard     : {args.gaze_mode}")
    print(f" Gaming response : Deadzone {args.gaze_deadzone:.2f} | Exp {args.gaze_curve_exp:.2f} | Gains: Yaw {args.gaze_yaw_gain:.2f}, Pitch {args.gaze_pitch_gain:.2f}")
    print(f" OpenTrack UDP   : {args.host}:{args.port}")
    print("=" * 60)

    # 1. Capture
    try:
        capture = WebcamCapture(
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            target_fps=args.fps,
        )
    except Exception as e:
        print(f"[ERREUR] Impossible d'initialiser la capture : {e}", file=sys.stderr)
        return

    print(f"[capture] Caméra active : {capture.actual_width}x{capture.actual_height} @ {capture.actual_fps:.1f} fps")

    # 2. Détecteur de visage MediaPipe
    face_detector = FaceDetector()

    # 3. Normalisateur faciale (448 pour MobileOne ONNX, 224 pour iris)
    patch_size = 448 if args.gaze_mode == "net" else 224
    normalizer = FaceNormalizer(target_size=patch_size, virtual_focal=args.patch_focal)

    # 4. Modèle de regard
    gaze_estimator = create_gaze_estimator(mode=args.gaze_mode)

    # 5. Gestionnaire de calibration affine
    calib_file = data_path(f"calibration_{args.gaze_mode}.npz")
    calibration_mgr = CalibrationManager(
        screen_config=screen_config,
        mode=args.gaze_mode,
        num_points=args.calib_points,
    )
    import os
    if os.path.exists(calib_file):
        if calibration_mgr.load(calib_file):
            print(f"[main] Calibration existante chargée depuis {calib_file}.")

    # 6. Filtres One Euro (6D pour la pose tête, 2D pour le point de regard écran)
    # Réglages séparés : les unités (degrés/cm vs pixels) et le niveau de bruit diffèrent.
    pose_filter = OneEuroFilter(dim=6, min_cutoff=args.pose_min_cutoff, beta=args.pose_beta)
    gaze_filter = OneEuroFilter(dim=2, min_cutoff=args.gaze_min_cutoff, beta=args.gaze_beta, median_window=args.gaze_median)
    # Maintien pendant les pertes de visage puis retour en douceur à la réacquisition
    pose_blender = LossBlender(dim=6, blend_time=0.35, min_loss=0.1)
    # Clignements : le regard est figé pendant le clignement (sinon il saute)
    blink_detector = BlinkDetector()
    gaze_blender = LossBlender(dim=2, blend_time=0.35, min_loss=0.1)

    # Bornes du point de regard (petite marge hors écran pour l'overlay)
    gaze_margin_x = screen_w * 0.05
    gaze_margin_y = screen_h * 0.05

    def clamp_gaze(px: tuple[float, float]) -> tuple[float, float] | None:
        """Borne le point de regard à l'écran ; None si la valeur est invalide (NaN/inf)."""
        x, y = float(px[0]), float(px[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return (
            float(np.clip(x, -gaze_margin_x, screen_w + gaze_margin_x)),
            float(np.clip(y, -gaze_margin_y, screen_h + gaze_margin_y)),
        )

    # 7. Sortie OpenTrack et interface
    out_cfg = OutputConfig(
        host=args.host,
        port=args.port,
        invert_x=args.invert_x,
        invert_y=args.invert_y,
        invert_z=args.invert_z,
        invert_yaw=args.invert_yaw,
        invert_pitch=args.invert_pitch,
        invert_roll=args.invert_roll,
        gaze_yaw_gain=args.gaze_yaw_gain,
        gaze_pitch_gain=args.gaze_pitch_gain,
        gaze_deadzone=args.gaze_deadzone,
        gaze_curve_exp=args.gaze_curve_exp,
        gaze_smoothing_s=args.gaze_smoothing,
        head_yaw_gain=args.head_yaw_gain,
        head_pitch_gain=args.head_pitch_gain,
        send_rate_hz=args.send_rate,
        max_prediction_s=args.prediction,
        output_smoothing_s=args.output_smoothing,
    )
    sender = OpenTrackSender(config=out_cfg)
    sender.start()
    # Fenêtre Moniteur (vidéo + HUD) : seulement sur demande (--moniteur) ou sans panneau ;
    # l'aperçu caméra du panneau la remplace au quotidien.
    visualizer = Visualizer() if (args.moniteur or args.no_panel) else None

    # Raccourcis globaux (actifs même quand le jeu a le focus)
    hotkey_labels = {"recenter": "Recentrer", "gaze": "Regard on/off", "pause": "Pause (caméra au centre)"}
    hotkey_bindings = {
        "recenter": current_profile.hotkey_recenter,
        "gaze": current_profile.hotkey_gaze,
        "pause": current_profile.hotkey_pause,
    }
    hotkeys = GlobalHotkeys(hotkey_bindings)
    hotkey_status = hotkeys.start()
    short_labels = {"recenter": "recentrer", "gaze": "regard", "pause": "pause"}
    active_hotkeys = [f"{format_hotkey(c)} {short_labels[a]}" for a, c in hotkey_bindings.items()
                      if c and hotkey_status.get(a)]
    hotkey_hint = ("En jeu : " + " | ".join(active_hotkeys)) if active_hotkeys else ""

    # 8. Overlay transparent click-through (Étape 6)
    overlay = GazeOverlay(width=screen_w, height=screen_h)
    overlay.start(visible=args.overlay)

    # Variables d'état
    gaze_enabled = True
    mirror_mode = not args.no_flip
    enable_recalib = not args.no_recalib
    current_gaze_px = (float(screen_w / 2), float(screen_h / 2))
    filtered_xyz = (0.0, 0.0, -60.0)
    filtered_ypr = (0.0, 0.0, 0.0)
    prev_mouse_down = False
    blinking = False
    frame_count = 0

    # Le profil actif reflète ce qui tourne réellement (options de la ligne de commande comprises) :
    # le panneau l'affiche et « Sauvegarder » l'écrit tel quel.
    current_profile.head_yaw_gain = args.head_yaw_gain
    current_profile.head_pitch_gain = args.head_pitch_gain
    current_profile.pose_min_cutoff = args.pose_min_cutoff
    current_profile.gaze_yaw_gain = args.gaze_yaw_gain
    current_profile.gaze_pitch_gain = args.gaze_pitch_gain
    current_profile.gaze_deadzone = args.gaze_deadzone
    current_profile.gaze_smoothing_s = args.gaze_smoothing
    current_profile.recalib_on_click = enable_recalib
    profile_version = 0  # incrémenté à chaque changement de profil (le panneau recharge ses curseurs)
    profile_values = {f: getattr(current_profile, f) for f in PANEL_FIELDS}

    def apply_profile_settings(prof: Profile) -> None:
        """Applique en direct les réglages d'un profil (gains, lissages, recalibration au clic)."""
        nonlocal enable_recalib
        cfg = sender.config
        cfg.gaze_yaw_gain = prof.gaze_yaw_gain
        cfg.gaze_pitch_gain = prof.gaze_pitch_gain
        cfg.gaze_deadzone = prof.gaze_deadzone
        cfg.gaze_curve_exp = prof.gaze_curve_exp
        cfg.gaze_smoothing_s = prof.gaze_smoothing_s
        cfg.head_yaw_gain = prof.head_yaw_gain
        cfg.head_pitch_gain = prof.head_pitch_gain
        cfg.max_prediction_s = prof.prediction_s
        cfg.output_smoothing_s = prof.output_smoothing_s
        enable_recalib = prof.recalib_on_click
        pose_filter.set_params(min_cutoff=prof.pose_min_cutoff, beta=prof.pose_beta)
        gaze_filter.set_params(min_cutoff=prof.gaze_min_cutoff, beta=prof.gaze_beta, median_window=prof.gaze_median)

    def switch_profile(name: str) -> None:
        """Charge et applique un autre profil."""
        nonlocal current_profile, current_prof_idx, available_profiles, profile_version, profile_values
        available_profiles = list_profiles()
        current_profile = load_profile(name)
        if current_profile.name in available_profiles:
            current_prof_idx = available_profiles.index(current_profile.name)
        apply_profile_settings(current_profile)
        profile_version += 1
        profile_values = {f: getattr(current_profile, f) for f in PANEL_FIELDS}
        print(f"\n[profile] Bascule vers [{current_profile.name.upper()}] : {current_profile.description}")
        print(f"          -> Tête : gain {current_profile.head_yaw_gain:.1f} / {current_profile.head_pitch_gain:.1f} | "
              f"Regard : {current_profile.gaze_yaw_gain:.2f} / {current_profile.gaze_pitch_gain:.2f} | "
              f"Recalibration au clic : {'active' if enable_recalib else 'désactivée'} "
              f"(raccourcis globaux et fréquence d'envoi : ceux du lancement)")

    # Panneau de contrôle (réglages en direct, commandes, état)
    panel: ControlPanel | None = None
    if not args.no_panel:
        panel = ControlPanel()
        panel.start()

    def read_for_calibration():
        """Capture pour la calibration avec le même effet miroir que la boucle principale.

        Sans cela, la calibration est apprise sur des images non miroir puis appliquée
        à des images miroir : l'axe horizontal du regard se retrouve inversé.
        """
        ret_c, frame_c, ts_c = capture.read()
        if ret_c and frame_c is not None and mirror_mode:
            frame_c = cv2.flip(frame_c, 1)
        return ret_c, frame_c, ts_c

    import ctypes
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    class CURSORINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("flags", ctypes.c_uint),
                    ("hCursor", ctypes.c_void_p), ("ptScreenPos", POINT)]

    def cursor_visible() -> bool:
        """Le curseur de la souris est-il affiché ? Les jeux le cachent quand ils capturent la
        souris : un clic n'y désigne alors pas l'endroit regardé (recalibration à éviter)."""
        info = CURSORINFO()
        info.cbSize = ctypes.sizeof(CURSORINFO)
        if not ctypes.windll.user32.GetCursorInfo(ctypes.byref(info)):
            return False
        return bool(info.flags & 0x1) and bool(info.hCursor)  # CURSOR_SHOWING

    print("\n[INFO] Tracker démarré.")
    if visualizer is not None:
        print("Raccourcis clavier (fenêtre Moniteur) :")
        print("  [C] : Lancer la calibration affine (5 ou 9 points)")
        print("  [G] : Activer / Désactiver la contribution du regard à la caméra")
        print("  [O] : Activer / Désactiver l'overlay transparent in-game")
        print("  [P] : Basculer vers le profil suivant (default, ets2, flight_sim, racing)")
        print("  [S] : Sauvegarder la configuration dans le profil JSON actif")
        print("  [R] : Recentrer la pose et le regard (zéro)")
        print("  [Espace] : Pause (la caméra revient au centre et y reste)")
        print("  [Q] ou ESC : Quitter")
    print("Raccourcis globaux (marchent aussi en jeu) :")
    for action, combo in hotkey_bindings.items():
        if not combo:
            continue
        if hotkey_status.get(action):
            print(f"  [{format_hotkey(combo)}] : {hotkey_labels[action]}")
        else:
            print(f"  [{format_hotkey(combo)}] : {hotkey_labels[action]} -> INDISPONIBLE "
                  f"({hotkeys.errors.get(action, 'erreur inconnue')}) ; changer « hotkey_{action} » dans le profil")
    print(f"Envoi OpenTrack : {args.send_rate:.0f} Hz" if args.send_rate > 0 else "Envoi OpenTrack : une fois par image")
    print(f"Recalibration au clic : {'active (curseur visible seulement)' if not args.no_recalib else 'désactivée'}")
    print("Panneau de contrôle : " + ("ouvert (le fermer arrête le tracker)" if panel else "désactivé (--no-panel)") + "\n")

    step_latencies: dict[str, float] = {}
    # Mesure de l'agitation sur ~1 s glissante (écart-type des sorties filtrées)
    from collections import deque
    jitter_head: deque = deque(maxlen=60)
    jitter_gaze: deque = deque(maxlen=60)

    try:
        while True:
            t_loop_start = time.perf_counter()

            # 1. Étape Capture
            t0 = time.perf_counter()
            ret, frame, timestamp = capture.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            if mirror_mode:
                frame = cv2.flip(frame, 1)
            t_capture = (time.perf_counter() - t0) * 1000.0

            # 2. Étape Visage (MediaPipe 478 pts + Pose 6DoF)
            t0 = time.perf_counter()
            face_data = face_detector.process(frame, timestamp)
            t_face = (time.perf_counter() - t0) * 1000.0

            # 3. Étape Normalisation (Crop & Espace canonique Zhang 2018)
            t0 = time.perf_counter()
            norm_data = normalizer.process(frame, face_data)

            # 4. Étape Modèle de Regard
            gaze_features = gaze_estimator.estimate(face_data, norm_data=norm_data)
            t_gaze = (time.perf_counter() - t0) * 1000.0

            # 5. Étape Calibration & Filtrage
            t0 = time.perf_counter()
            blinking = False
            if face_data.detected:
                blinking = blink_detector.update(face_data.eye_openness, timestamp)
                # Après une vraie perte de suivi, on repart d'un filtre neuf : sinon le grand dt
                # et la dérivée gonflée laissent passer un saut puis du bruit pendant ~1 s.
                if pose_blender.lost_duration(timestamp) >= pose_blender.min_loss:
                    pose_filter.reset()
                    gaze_filter.reset()

                # Filtrage de la pose tête (x, y, z, yaw, pitch, roll)
                raw_pose_vector = [
                    face_data.pose_xyz[0],
                    face_data.pose_xyz[1],
                    face_data.pose_xyz[2],
                    face_data.pose_ypr[0],
                    face_data.pose_ypr[1],
                    face_data.pose_ypr[2],
                ]
                filtered_pose = pose_filter.filter(raw_pose_vector, timestamp)
                filtered_pose = pose_blender.update(filtered_pose, timestamp)
                filtered_xyz = (float(filtered_pose[0]), float(filtered_pose[1]), float(filtered_pose[2]))
                filtered_ypr = (float(filtered_pose[3]), float(filtered_pose[4]), float(filtered_pose[5]))

                # Prédiction du point écran via le modèle de calibration affine (compensé de la tête)
                if calibration_mgr.is_calibrated:
                    head_ypr = (float(face_data.pose_ypr[0]), float(face_data.pose_ypr[1]))
                    raw_gaze_px = calibration_mgr.predict(gaze_features.raw_features, head_pose=head_ypr)
                else:
                    if args.gaze_mode == "net":
                        # Projection géométrique 3D rayon-plan écran
                        raw_gaze_px = screen_config.ray_screen_intersection(
                            origin_mm=norm_data.origin_3d_mm,
                            direction_cam=gaze_features.gaze_vector_cam,
                        )
                    else:
                        # Approximation non-calibrée basée sur l'écart au ratio moyen 0.5
                        rx, ry = gaze_features.iris_ratios
                        approx_x = screen_w * (0.5 + (rx - 0.5) * 2.5)
                        approx_y = screen_h * (0.5 + (ry - 0.5) * 2.5)
                        raw_gaze_px = (float(approx_x), float(approx_y))

                # Bornage avant filtrage : une prédiction aberrante (regard hors écran,
                # extrapolation de la calibration) ne doit pas entraîner le filtre ni la caméra.
                # Pendant un clignement, le point de regard reste figé (filtre non alimenté).
                clamped_gaze = clamp_gaze(raw_gaze_px)
                if clamped_gaze is not None and not blinking:
                    filtered_gaze = gaze_filter.filter([clamped_gaze[0], clamped_gaze[1]], timestamp)
                    filtered_gaze = gaze_blender.update(filtered_gaze, timestamp)
                    current_gaze_px = (float(filtered_gaze[0]), float(filtered_gaze[1]))
            else:
                # Visage perdu : on garde la dernière sortie lissée (filtered_xyz/ypr et
                # current_gaze_px conservent leur valeur) et on note le début de la perte.
                pose_blender.mark_lost(timestamp)
                gaze_blender.mark_lost(timestamp)

            t_filter = (time.perf_counter() - t0) * 1000.0

            # 6. Étape Sortie (OpenTrack UDP)
            # La pose filtrée est transmise au thread d'envoi, qui envoie à 120 Hz en prédisant
            # entre deux images. Pendant une perte, la valeur maintenue est transmise sans
            # prédiction : le flux reste régulier et la reprise se fait en fondu.
            if face_data.detected or pose_blender.held is not None:
                sender.update(
                    pose_xyz=filtered_xyz,
                    pose_ypr=filtered_ypr,
                    gaze_px=current_gaze_px,
                    screen_config=screen_config,
                    gaze_enabled=gaze_enabled,
                    timestamp=timestamp,
                    tracking=face_data.detected,
                )

            # Mesure du temps total par boucle
            t_total = (time.perf_counter() - t_loop_start) * 1000.0
            step_latencies = {
                "capture": t_capture,
                "face": t_face,
                "gaze": t_gaze,
                "filter": t_filter,
                "total": t_total,
            }

            if face_data.detected:
                jitter_head.append(filtered_ypr[:2])
                jitter_gaze.append(current_gaze_px)
            jitter = None
            if len(jitter_head) >= 30:
                jitter = {
                    "head_deg": float(np.std(np.array(jitter_head), axis=0).max()),
                    "gaze_px": float(np.std(np.array(jitter_gaze), axis=0).max()),
                }

            # Recalibration implicite RLS sur les clics de souris (Étape 5)
            # Seulement si le curseur est visible : en jeu, la souris est capturée (curseur
            # caché, souvent bloqué au centre) et le clic ne désigne pas l'endroit regardé.
            if enable_recalib and ctypes and face_data.detected and calibration_mgr.is_calibrated and not blinking:
                try:
                    is_mouse_down = (ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000) != 0
                    if is_mouse_down and not prev_mouse_down and cursor_visible():
                        pt = POINT()
                        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                        click_x, click_y = float(pt.x), float(pt.y)
                        head_ypr = (float(face_data.pose_ypr[0]), float(face_data.pose_ypr[1]))
                        accepted = calibration_mgr.handle_mouse_click(
                            gaze_raw=gaze_features.raw_features,
                            head_pose=head_ypr,
                            click_px=(click_x, click_y),
                            max_error_px=220.0,
                        )
                        if accepted:
                            calibration_mgr.save(calib_file)
                            print(f"[recalib] RLS : Recalibration adaptative sur clic ({click_x:.0f}, {click_y:.0f}) | Total: +{calibration_mgr.rls_update_count}")
                    prev_mouse_down = is_mouse_down
                except Exception:
                    pass

            # Mise à jour de l'overlay transparent in-game (Étape 6)
            overlay.update_gaze(current_gaze_px[0], current_gaze_px[1])

            # Aperçu caméra du panneau (~10 images/s, seulement s'il est affiché)
            frame_count += 1
            if panel is not None and panel.preview_enabled and frame_count % 3 == 0:
                panel.publish_preview(make_preview(
                    frame, face_data.detected, [face_data.left_iris_px, face_data.right_iris_px], blinking))

            # 7. Étape Rendu visuel & gestion des touches (fenêtre Moniteur, si ouverte)
            key = 255 if visualizer is None else visualizer.render(
                frame=frame,
                face_detected=face_data.detected,
                left_iris=face_data.left_iris_px,
                right_iris=face_data.right_iris_px,
                pose_xyz=filtered_xyz,
                pose_ypr=filtered_ypr,
                gaze_px=current_gaze_px,
                screen_config=screen_config,
                gaze_mode=args.gaze_mode,
                gaze_enabled=gaze_enabled,
                is_calibrated=calibration_mgr.is_calibrated,
                measured_fps=capture.measured_fps,
                step_latencies=step_latencies,
                show_debug=args.debug,
                overlay_active=overlay.is_visible,
                rls_count=calibration_mgr.rls_update_count,
                profile_name=current_profile.name,
                jitter=jitter,
                paused=sender.paused,
                send_rate_hz=sender.measured_rate_hz,
                send_gap_ms=sender.max_gap_ms,
                hotkey_hint=hotkey_hint,
            )

            # État pour le panneau de contrôle
            # Temps de calcul seul (sans l'attente de l'image suivante de la caméra)
            t_face_ms, t_gaze_ms = step_latencies.get("face", 0.0), step_latencies.get("gaze", 0.0)
            t_calc_ms = t_face_ms + t_gaze_ms + step_latencies.get("filter", 0.0)
            diag_text = f"Calcul : {t_calc_ms:.0f} ms par image (visage {t_face_ms:.0f}, regard {t_gaze_ms:.0f})"
            if jitter:
                diag_text += (f"\nAgitation (1 s, si immobile) : tête {jitter['head_deg']:.2f}°, "
                              f"regard {jitter['gaze_px']:.0f} px")

            def panel_status(calibrating: bool = False) -> dict:
                return {
                    "face": face_data.detected, "blinking": blinking, "fps": capture.measured_fps,
                    "send_hz": sender.measured_rate_hz, "gap_ms": sender.max_gap_ms, "paused": sender.paused,
                    "gaze_enabled": gaze_enabled, "calibrated": calibration_mgr.is_calibrated,
                    "rls": calibration_mgr.rls_update_count, "overlay": overlay.is_visible,
                    "profile_name": current_profile.name, "profiles": available_profiles,
                    "profile_version": profile_version, "profile_values": dict(profile_values),
                    "hotkey_hint": hotkey_hint, "calibrating": calibrating,
                    "diag": diag_text,
                }

            if panel is not None:
                panel.publish(panel_status())

            # Commandes : touches du Moniteur, raccourcis globaux (en jeu) et panneau de contrôle
            commands = [("action", a) for a in hotkeys.poll()]
            if panel is not None:
                commands += panel.poll()
            key_actions = {
                ord("r"): "recenter", ord("g"): "gaze", ord(" "): "pause", ord("c"): "calibrate",
                ord("o"): "overlay", ord("p"): "next_profile", ord("s"): "save", ord("q"): "quit", 27: "quit",
            }
            if key in key_actions:
                commands.append(("action", key_actions[key]))

            quit_requested = False
            for command in commands:
                kind = command[0]
                if kind == "set":  # curseur ou case du panneau : réglage appliqué en direct
                    _, field, value = command
                    setattr(current_profile, field, value)
                    profile_values[field] = value
                    apply_profile_settings(current_profile)
                    continue
                if kind == "profile":
                    switch_profile(command[1])
                    continue
                action = command[1]
                if action == "quit":
                    quit_requested = True
                    break
                elif action == "recenter":
                    sender.recenter(
                        current_pose_xyz=filtered_xyz,
                        current_pose_ypr=filtered_ypr,
                        current_gaze_px=current_gaze_px,
                        screen_config=screen_config,
                    )
                elif action == "gaze":
                    gaze_enabled = not gaze_enabled
                    print(f"[main] Contribution du regard à la caméra : {'ACTIVÉ' if gaze_enabled else 'DÉSACTIVÉ'}")
                elif action == "pause":
                    paused = sender.toggle_pause()
                    print(f"[main] {'PAUSE : caméra ramenée au centre' if paused else 'Reprise du suivi'}")
                elif action == "calibrate":
                    print(f"[main] Démarrage de la calibration affine {args.calib_points} points...")
                    if panel is not None:
                        panel.publish(panel_status(calibrating=True))
                    success = calibration_mgr.run_interactive_calibration(
                        capture_fn=read_for_calibration,
                        face_detector=face_detector,
                        gaze_estimator=gaze_estimator,
                        normalizer=normalizer,
                    )
                    if success:
                        calibration_mgr.save(calib_file)
                        print(f"[main] Calibration sauvegardée dans {calib_file}")
                        pose_filter.reset()
                        gaze_filter.reset()
                elif action == "overlay":
                    vis = overlay.toggle()
                    print(f"[main] Overlay transparent in-game : {'ACTIVÉ' if vis else 'DÉSACTIVÉ'}")
                elif action == "next_profile":
                    names = list_profiles()
                    if names:
                        switch_profile(names[(current_prof_idx + 1) % len(names)])
                elif action == "save":
                    save_profile(current_profile)
                    print(f"[profile] Configuration sauvegardée dans 'profiles/{current_profile.name}.json'.")
            if quit_requested:
                print("[main] Arrêt demandé par l'utilisateur.")
                break

    except KeyboardInterrupt:
        print("\n[main] Interruption clavier (Ctrl+C).")
    finally:
        print("[main] Fermeture propre des flux et sockets...")
        hotkeys.stop()
        if panel is not None:
            panel.stop()
        overlay.stop()
        capture.release()
        sender.close()
        if visualizer is not None:
            visualizer.close()
        print("[main] Terminé avec succès.")


if __name__ == "__main__":
    main()
