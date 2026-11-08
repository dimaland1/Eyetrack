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
from filters import OneEuroFilter, LossBlender
from output import OpenTrackSender, Visualizer, OutputConfig
from overlay import GazeOverlay
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

    # Fonctionnalités avancées (Étapes 5 & 6)
    parser.add_argument("--overlay", action="store_true", default=active_profile.overlay_enabled, help="Active l'overlay transparent")
    parser.add_argument("--no-recalib", action="store_true", default=not active_profile.recalib_on_click, help="Désactive recalibration RLS")

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
    )
    sender = OpenTrackSender(config=out_cfg)
    visualizer = Visualizer()

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

    print("\n[INFO] Tracker démarré. Raccourcis clavier :")
    print("  [C] : Lancer la calibration affine (5 ou 9 points)")
    print("  [G] : Activer / Désactiver la contribution du regard à la caméra")
    print("  [O] : Activer / Désactiver l'overlay transparent in-game")
    print("  [P] : Basculer vers le profil suivant (default, ets2, flight_sim, racing)")
    print("  [S] : Sauvegarder la configuration dans le profil JSON actif")
    print("  [R] : Recentrer la pose et le regard (zéro)")
    print("  [Q] ou ESC : Quitter\n")

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
            if face_data.detected:
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
                clamped_gaze = clamp_gaze(raw_gaze_px)
                if clamped_gaze is not None:
                    filtered_gaze = gaze_filter.filter([clamped_gaze[0], clamped_gaze[1]], timestamp)
                    filtered_gaze = gaze_blender.update(filtered_gaze, timestamp)
                    current_gaze_px = (float(filtered_gaze[0]), float(filtered_gaze[1]))
            else:
                # Visage perdu : on garde la dernière sortie lissée (filtered_xyz/ypr et
                # current_gaze_px conservent leur valeur) et on note le début de la perte.
                pose_blender.mark_lost(timestamp)
                gaze_blender.mark_lost(timestamp)

            t_filter = (time.perf_counter() - t0) * 1000.0

            # 6. Étape Sortie (Envoi UDP OpenTrack)
            # Envoi continu, y compris pendant une perte (valeur maintenue) : le flux reste
            # régulier et la reprise se fait en fondu au lieu d'un saut.
            if face_data.detected or pose_blender.held is not None:
                sender.send(
                    pose_xyz=filtered_xyz,
                    pose_ypr=filtered_ypr,
                    gaze_px=current_gaze_px,
                    screen_config=screen_config,
                    gaze_enabled=gaze_enabled,
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
            if enable_recalib and ctypes and face_data.detected and calibration_mgr.is_calibrated:
                try:
                    is_mouse_down = (ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000) != 0
                    if is_mouse_down and not prev_mouse_down:
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

            # 7. Étape Rendu visuel & gestion des touches
            key = visualizer.render(
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
            )

            if key == ord("q") or key == 27:  # 'q' ou ESC
                print("[main] Arrêt demandé par l'utilisateur.")
                break
            elif key == ord("c"):  # Lancer la calibration
                print(f"[main] Démarrage de la calibration affine {args.calib_points} points...")
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
            elif key == ord("g"):  # Toggle regard on/off
                gaze_enabled = not gaze_enabled
                state_str = "ACTIVÉ" if gaze_enabled else "DÉSACTIVÉ"
                print(f"[main] Contribution du regard à la caméra : {state_str}")
            elif key == ord("o"):  # Toggle overlay transparent on/off (Étape 6)
                vis = overlay.toggle()
                print(f"[main] Overlay transparent in-game : {'ACTIVÉ' if vis else 'DÉSACTIVÉ'}")
            elif key == ord("p"):  # Basculer vers le profil JSON suivant (Étape 7)
                available_profiles = list_profiles()
                if available_profiles:
                    current_prof_idx = (current_prof_idx + 1) % len(available_profiles)
                    new_name = available_profiles[current_prof_idx]
                    current_profile = load_profile(new_name)
                    # Mise à jour dynamique des gains et paramètres
                    sender.config.gaze_yaw_gain = current_profile.gaze_yaw_gain
                    sender.config.gaze_pitch_gain = current_profile.gaze_pitch_gain
                    sender.config.gaze_deadzone = current_profile.gaze_deadzone
                    sender.config.gaze_curve_exp = current_profile.gaze_curve_exp
                    pose_filter.set_params(min_cutoff=current_profile.pose_min_cutoff, beta=current_profile.pose_beta)
                    gaze_filter.set_params(min_cutoff=current_profile.gaze_min_cutoff, beta=current_profile.gaze_beta,
                                           median_window=current_profile.gaze_median)
                    print(f"\n[profile] Bascule vers [{current_profile.name.upper()}] : {current_profile.description}")
                    print(f"          -> Yaw gain: {current_profile.gaze_yaw_gain:.2f} | Pitch gain: {current_profile.gaze_pitch_gain:.2f} | Deadzone: {current_profile.gaze_deadzone:.2f} | Exp: {current_profile.gaze_curve_exp:.2f}")
            elif key == ord("s"):  # Sauvegarder la configuration dans le profil
                save_profile(current_profile)
                print(f"[profile] Configuration sauvegardée dans 'profiles/{current_profile.name}.json'.")
            elif key == ord("r"):  # Recentrer
                sender.recenter(
                    current_pose_xyz=filtered_xyz,
                    current_pose_ypr=filtered_ypr,
                    current_gaze_px=current_gaze_px,
                    screen_config=screen_config,
                )

    except KeyboardInterrupt:
        print("\n[main] Interruption clavier (Ctrl+C).")
    finally:
        print("[main] Fermeture propre des flux et sockets...")
        overlay.stop()
        capture.release()
        sender.close()
        visualizer.close()
        print("[main] Terminé avec succès.")


if __name__ == "__main__":
    main()
