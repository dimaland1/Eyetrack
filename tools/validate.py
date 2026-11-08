"""Outil de mesure et de comparaison de précision (validate.py).

Permet d'évaluer et de comparer scientifiquement les modes 'iris' et 'net' :
- Affichage de 9 cibles réparties sur l'écran (avec prise en compte des mouvements de tête).
- Mode individuel (--gaze-mode iris|net) ou mode comparatif simultané (--compare).
- Calcul des erreurs moyennes, médianes et écart-types (° et cm).
- Décomposition spatiale par zone (Centre, Bords, Coins).
- Export CSV détaillé et tableau comparatif synthétique.
"""

from __future__ import annotations
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Les outils vivent dans tools/ : on rend importables les modules du tracker (dossier parent)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture import WebcamCapture
from face import FaceDetector
from normalize import FaceNormalizer
from gaze_model import create_gaze_estimator
from geometry import ScreenConfig, get_screen_resolution
from calibration import CalibrationManager
from filters import OneEuroFilter, BlinkDetector
from paths import data_path
from profile import load_profile


def eye_position_cm(pose_xyz: tuple[float, float, float], screen_config: ScreenConfig) -> tuple[float, float, float]:
    """Position de l'œil dans le repère de l'écran (cm, origine au centre, x droite, y bas).

    MediaPipe donne la tête dans le repère caméra (y vers le HAUT, origine à la webcam). La
    webcam est supposée au centre du bord supérieur de l'écran, d'où le décalage d'une
    demi-hauteur d'écran.
    """
    x, y, z = pose_xyz
    return (float(x), float(-y - screen_config.height_mm / 20.0), float(z))


def clamp_to_screen(px: tuple[float, float], screen_config: ScreenConfig) -> tuple[float, float] | None:
    """Même bornage que le tracker (écran ± 5 %) ; None si la valeur est invalide."""
    x, y = float(px[0]), float(px[1])
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    w, h = screen_config.width_px, screen_config.height_px
    return float(np.clip(x, -0.05 * w, 1.05 * w)), float(np.clip(y, -0.05 * h, 1.05 * h))


def parse_arguments() -> argparse.Namespace:
    """Arguments de la ligne de commande pour validate.py."""
    parser = argparse.ArgumentParser(
        description="Validation et comparaison métrique du tracker de regard"
    )
    parser.add_argument("--camera", type=int, default=0, help="Index de la caméra (défaut: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Largeur demandée (défaut: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Hauteur demandée (défaut: 720)")
    parser.add_argument("--fps", type=int, default=60, help="Framerate cible (défaut: 60)")
    parser.add_argument("--no-flip", action="store_true", help="Désactive le miroir de caméra")
    parser.add_argument(
        "--gaze-mode",
        type=str,
        choices=["iris", "net"],
        default="net",
        help="Mode de regard : 'net' (ONNX) ou 'iris' (MVP)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Mode comparatif : évalue simultanément Iris vs Net sur les mêmes frames",
    )
    parser.add_argument(
        "--calib-file",
        type=str,
        default=None,
        help="Fichier de calibration spécifique (défaut auto: calibration_<mode>.npz)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Fichier CSV de sortie (défaut auto: validation_<mode>.csv ou validation_comparison.csv)",
    )
    parser.add_argument("--screen-w", type=int, default=None, help="Largeur écran pixels")
    parser.add_argument("--screen-h", type=int, default=None, help="Hauteur écran pixels")
    parser.add_argument("--screen-diag", type=float, default=27.0, help="Diagonale écran pouces (défaut: 27.0)")
    parser.add_argument("--distance", type=float, default=60.0, help="Distance écran en cm (défaut: 60.0)")
    parser.add_argument("--samples-per-point", type=int, default=35, help="Échantillons par cible (défaut: 35)")
    parser.add_argument("--settle-frames", type=int, default=15, help="Frames ignorées par cible pour la saccade (défaut: 15)")
    parser.add_argument("--skip-calibration", action="store_true", help="Mesure directement sans proposer de calibration")
    parser.add_argument("--profile", type=str, default="default", help="Profil dont on reprend les réglages de filtre (défaut: default)")

    return parser.parse_args()


# Définition des 9 cibles de validation (grille normalisée à 15%, 50%, 85%)
VALIDATION_TARGETS = [
    {"id": 1, "norm": (0.15, 0.15), "zone": "corner", "name": "Coin Haut-Gauche"},
    {"id": 2, "norm": (0.50, 0.15), "zone": "edge",   "name": "Bord Haut"},
    {"id": 3, "norm": (0.85, 0.15), "zone": "corner", "name": "Coin Haut-Droit"},
    {"id": 4, "norm": (0.15, 0.50), "zone": "edge",   "name": "Bord Gauche"},
    {"id": 5, "norm": (0.50, 0.50), "zone": "center", "name": "Centre"},
    {"id": 6, "norm": (0.85, 0.50), "zone": "edge",   "name": "Bord Droit"},
    {"id": 7, "norm": (0.15, 0.85), "zone": "corner", "name": "Coin Bas-Gauche"},
    {"id": 8, "norm": (0.50, 0.85), "zone": "edge",   "name": "Bord Bas"},
    {"id": 9, "norm": (0.85, 0.85), "zone": "corner", "name": "Coin Bas-Droit"},
]


def run_validation() -> None:
    """Exécute le protocole de validation ou de comparaison."""
    args = parse_arguments()

    auto_w, auto_h = get_screen_resolution()
    screen_w = args.screen_w if args.screen_w is not None else auto_w
    screen_h = args.screen_h if args.screen_h is not None else auto_h

    screen_config = ScreenConfig(
        width_px=screen_w,
        height_px=screen_h,
        diagonal_inches=args.screen_diag,
        distance_cm=args.distance,
    )

    is_compare = args.compare
    out_csv = args.output or data_path("validation_comparison.csv" if is_compare else f"validation_{args.gaze_mode}.csv")

    print("=" * 70)
    print(" Eyetrack - Banc de Validation de Précision")
    print(f" Écran        : {screen_w}x{screen_h} ({args.screen_diag}\", distance {args.distance} cm)")
    print(f" Mode de test : {'COMPARAISON (Iris vs Net)' if is_compare else f'Individuel ({args.gaze_mode.upper()})'}")
    print(f" Sortie CSV   : {out_csv}")
    print("=" * 70)

    # Initialisation capture & vision
    capture = WebcamCapture(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        target_fps=args.fps,
    )
    face_detector = FaceDetector()
    prof = load_profile(args.profile)
    normalizer = FaceNormalizer(target_size=448, virtual_focal=prof.patch_focal)

    def make_gaze_filter() -> OneEuroFilter:
        # Mêmes réglages que le tracker (profil), pour mesurer ce que l'on obtient vraiment en jeu
        return OneEuroFilter(dim=2, min_cutoff=prof.gaze_min_cutoff, beta=prof.gaze_beta,
                             median_window=prof.gaze_median)

    # Clignements : images écartées de la mesure (le tracker fige le regard pendant ce temps)
    blink_detector = BlinkDetector()
    skipped_blinks = 0

    # Estimateurs et calibrations
    if is_compare:
        est_iris = create_gaze_estimator(mode="iris")
        est_net = create_gaze_estimator(mode="net")

        calib_iris = CalibrationManager(screen_config=screen_config, mode="iris")
        calib_net = CalibrationManager(screen_config=screen_config, mode="net")

        # Chargement des fichiers de calibration respectifs
        for mgr, name in ((calib_iris, "iris"), (calib_net, "net")):
            path = data_path(f"calibration_{name}.npz")
            if os.path.exists(path) and mgr.load(path):
                print(f"[validate] Calibration {name} chargée depuis {path}")

        filter_iris = make_gaze_filter()
        filter_net = make_gaze_filter()

    else:
        gaze_estimator = create_gaze_estimator(mode=args.gaze_mode)
        calib_mgr = CalibrationManager(screen_config=screen_config, mode=args.gaze_mode)
        gaze_filter = make_gaze_filter()

        # Fichier de calibration cible
        calib_path = args.calib_file or data_path(f"calibration_{args.gaze_mode}.npz")

        has_calib = False
        if os.path.exists(calib_path):
            if calib_mgr.load(calib_path):
                has_calib = True
                print(f"[validate] Calibration chargée depuis {calib_path}")

        # Proposer calibration si absente
        if not has_calib and not args.skip_calibration:
            print("\n[validate] Aucune calibration trouvée.")
            print("  -> Appuyez sur [C] dans la fenêtre pour calibrer")
            print("  -> Appuyez sur [ESPACE] pour valider sans calibration")

            intro_win = "Eyetrack - Preparation"
            cv2.namedWindow(intro_win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(intro_win, 800, 450)

            while True:
                ret, frame, _ = capture.read()
                if not ret or frame is None:
                    continue
                if not args.no_flip:
                    frame = cv2.flip(frame, 1)

                cv2.putText(frame, f"Mode: {args.gaze_mode.upper()}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Appuyez sur [C] pour calibrer (recommande)", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Appuyez sur [ESPACE] pour tester tel quel", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.imshow(intro_win, frame)

                k = cv2.waitKey(10) & 0xFF
                if k == 27 or k == ord("q"):
                    cv2.destroyAllWindows()
                    capture.release()
                    return
                elif k == ord("c"):
                    cv2.destroyWindow(intro_win)
                    def read_mirrored():
                        # Même effet miroir que la boucle de mesure (sinon l'axe X est inversé)
                        ok_c, fr_c, ts_c = capture.read()
                        if ok_c and fr_c is not None and not args.no_flip:
                            fr_c = cv2.flip(fr_c, 1)
                        return ok_c, fr_c, ts_c

                    success = calib_mgr.run_interactive_calibration(
                        capture_fn=read_mirrored,
                        face_detector=face_detector,
                        gaze_estimator=gaze_estimator,
                        normalizer=normalizer,
                    )
                    if success:
                        calib_mgr.save(data_path(f"calibration_{args.gaze_mode}.npz"))
                    break
                elif k == ord(" ") or k == 13:
                    cv2.destroyWindow(intro_win)
                    break

    # Lancement du test des 9 cibles
    win_name = "Validation du regard (Fixez chaque cible - ESC pour quitter)"
    cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n[validate] Démarrage des 9 cibles de validation...")

    detailed_records: list[dict] = []
    interrupted = False
    mirror_mode = not args.no_flip
    total_targets = len(VALIDATION_TARGETS)

    for idx, target in enumerate(VALIDATION_TARGETS):
        if interrupted:
            break

        nx, ny = target["norm"]
        tx, ty = screen_config.norm_to_px(nx, ny)

        frames_seen = 0
        collected_count = 0

        while collected_count < args.samples_per_point:
            ret, frame, timestamp = capture.read()
            if not ret or frame is None:
                continue
            if mirror_mode:
                frame = cv2.flip(frame, 1)

            face_data = face_detector.process(frame, timestamp)
            norm_data = normalizer.process(frame, face_data)

            # Écran de validation
            val_screen = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            frames_seen += 1

            progress = collected_count / float(args.samples_per_point)
            radius = int(22 + 8 * math.sin(frames_seen * 0.25))

            # Affichage de la consigne
            info_txt = f"Cible {idx + 1}/{total_targets} ({target['name']}) - Fixez la cible"
            cv2.putText(val_screen, info_txt, (screen_w // 2 - 260, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 220, 220), 2)

            # Réticule cyan
            cv2.circle(val_screen, (tx, ty), radius, (255, 200, 0), 2)
            cv2.circle(val_screen, (tx, ty), 5, (255, 255, 0), -1)
            cv2.line(val_screen, (tx - 12, ty), (tx + 12, ty), (255, 255, 0), 1)
            cv2.line(val_screen, (tx, ty - 12), (tx, ty + 12), (255, 255, 0), 1)

            # Jauge de capture
            bar_w = 80
            fill_w = int(bar_w * progress)
            cv2.rectangle(val_screen, (tx - bar_w // 2, ty + 30), (tx - bar_w // 2 + fill_w, ty + 36), (0, 255, 120), -1)

            cv2.imshow(win_name, val_screen)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                print("[validate] Validation interrompue.")
                interrupted = True
                break

            if not face_data.detected:
                continue
            if blink_detector.update(face_data.eye_openness, timestamp):
                if frames_seen > args.settle_frames:
                    skipped_blinks += 1
                continue

            head_ypr = (float(face_data.pose_ypr[0]), float(face_data.pose_ypr[1]))
            eye_cm = eye_position_cm(face_data.pose_xyz, screen_config)

            # Calcul des prédictions
            if is_compare:
                # 1. Mode Iris
                feat_iris = est_iris.estimate(face_data)
                if calib_iris.is_calibrated:
                    px_raw_iris = calib_iris.predict(feat_iris.raw_features, head_pose=head_ypr)
                else:
                    rx, ry = feat_iris.iris_ratios
                    px_raw_iris = (screen_w * (0.5 + (rx - 0.5) * 2.5), screen_h * (0.5 + (ry - 0.5) * 2.5))
                px_raw_iris = clamp_to_screen(px_raw_iris, screen_config)

                # 2. Mode Net
                feat_net = est_net.estimate(face_data, norm_data=norm_data)
                if calib_net.is_calibrated:
                    px_raw_net = calib_net.predict(feat_net.raw_features, head_pose=head_ypr)
                else:
                    px_raw_net = screen_config.ray_screen_intersection(norm_data.origin_3d_mm, feat_net.gaze_vector_cam)
                px_raw_net = clamp_to_screen(px_raw_net, screen_config)
                if px_raw_iris is None or px_raw_net is None:
                    continue
                pt_iris = filter_iris.filter([px_raw_iris[0], px_raw_iris[1]], timestamp)
                pt_net = filter_net.filter([px_raw_net[0], px_raw_net[1]], timestamp)

                if frames_seen > args.settle_frames:
                    collected_count += 1
                    err_deg_iris = screen_config.visual_angle_deg((tx, ty), (pt_iris[0], pt_iris[1]), eye_cm)
                    err_cm_iris = screen_config.distance_cm_between_px((tx, ty), (pt_iris[0], pt_iris[1]))
                    err_deg_net = screen_config.visual_angle_deg((tx, ty), (pt_net[0], pt_net[1]), eye_cm)
                    err_cm_net = screen_config.distance_cm_between_px((tx, ty), (pt_net[0], pt_net[1]))

                    detailed_records.append({
                        "timestamp": f"{timestamp:.4f}",
                        "target_id": target["id"],
                        "target_name": target["name"],
                        "zone": target["zone"],
                        "target_x_px": tx,
                        "target_y_px": ty,
                        "error_deg_iris": round(err_deg_iris, 2),
                        "error_cm_iris": round(err_cm_iris, 2),
                        "error_deg_net": round(err_deg_net, 2),
                        "error_cm_net": round(err_cm_net, 2),
                        "head_yaw": round(head_ypr[0], 2),
                        "head_pitch": round(head_ypr[1], 2),
                    })

            else:
                feat = gaze_estimator.estimate(face_data, norm_data=norm_data)
                if calib_mgr.is_calibrated:
                    px_raw = calib_mgr.predict(feat.raw_features, head_pose=head_ypr)
                else:
                    if args.gaze_mode == "net":
                        px_raw = screen_config.ray_screen_intersection(norm_data.origin_3d_mm, feat.gaze_vector_cam)
                    else:
                        rx, ry = feat.iris_ratios
                        px_raw = (screen_w * (0.5 + (rx - 0.5) * 2.5), screen_h * (0.5 + (ry - 0.5) * 2.5))

                px_raw = clamp_to_screen(px_raw, screen_config)
                if px_raw is None:
                    continue
                pt = gaze_filter.filter([px_raw[0], px_raw[1]], timestamp)

                if frames_seen > args.settle_frames:
                    collected_count += 1
                    err_deg = screen_config.visual_angle_deg((tx, ty), (pt[0], pt[1]), eye_cm)
                    err_cm = screen_config.distance_cm_between_px((tx, ty), (pt[0], pt[1]))

                    detailed_records.append({
                        "timestamp": f"{timestamp:.4f}",
                        "target_id": target["id"],
                        "target_name": target["name"],
                        "zone": target["zone"],
                        "target_x_px": tx,
                        "target_y_px": ty,
                        "pred_x_px": round(pt[0], 1),
                        "pred_y_px": round(pt[1], 1),
                        "error_cm": round(err_cm, 2),
                        "error_deg": round(err_deg, 2),
                        "head_yaw": round(head_ypr[0], 2),
                        "head_pitch": round(head_ypr[1], 2),
                    })

    cv2.destroyAllWindows()
    capture.release()

    print(f"\n[validate] {blink_detector.blink_count} clignements détectés, "
          f"{skipped_blinks} images écartées de la mesure.")

    if not detailed_records:
        print("\n[validate] Aucun échantillon exploitable collecté.")
        return

    # --- Synthèse et Affichage des Résultats ---
    if is_compare:
        iris_deg = [r["error_deg_iris"] for r in detailed_records]
        iris_cm = [r["error_cm_iris"] for r in detailed_records]
        net_deg = [r["error_deg_net"] for r in detailed_records]
        net_cm = [r["error_cm_net"] for r in detailed_records]

        def get_zone_stats(zone_name):
            i_deg = [r["error_deg_iris"] for r in detailed_records if r["zone"] == zone_name]
            n_deg = [r["error_deg_net"] for r in detailed_records if r["zone"] == zone_name]
            i_cm = [r["error_cm_iris"] for r in detailed_records if r["zone"] == zone_name]
            n_cm = [r["error_cm_net"] for r in detailed_records if r["zone"] == zone_name]
            return np.mean(i_deg), np.mean(n_deg), np.mean(i_cm), np.mean(n_cm)

        gain_mean = (np.mean(iris_deg) - np.mean(net_deg)) / np.mean(iris_deg) * 100.0
        gain_med = (np.median(iris_deg) - np.median(net_deg)) / np.median(iris_deg) * 100.0

        print("\n" + "=" * 76)
        print("                 RAPPORT COMPARATIF : IRIS vs NET (ONNX)")
        print("=" * 76)
        print(f"{'Métrique':<22} | {'Mode IRIS (MVP)':<18} | {'Mode NET (ONNX)':<18} | {'Gain':<8}")
        print("-" * 76)
        print(f"{'Moyenne globale':<22} | {np.mean(iris_deg):5.2f}° ({np.mean(iris_cm):4.1f} cm)  | {np.mean(net_deg):5.2f}° ({np.mean(net_cm):4.1f} cm)  | {gain_mean:+5.1f}%")
        print(f"{'Médiane globale':<22} | {np.median(iris_deg):5.2f}° ({np.median(iris_cm):4.1f} cm)  | {np.median(net_deg):5.2f}° ({np.median(net_cm):4.1f} cm)  | {gain_med:+5.1f}%")
        print(f"{'Écart-type (Std)':<22} | {np.std(iris_deg):5.2f}°               | {np.std(net_deg):5.2f}°               |")
        print("-" * 76)

        for z_key, z_label in [("center", "Centre (1 cible)"), ("edge", "Bords (4 cibles)"), ("corner", "Coins (4 cibles)")]:
            i_d, n_d, i_c, n_c = get_zone_stats(z_key)
            g = (i_d - n_d) / max(i_d, 1e-4) * 100.0
            print(f"{z_label:<22} | {i_d:5.2f}° ({i_c:4.1f} cm)  | {n_d:5.2f}° ({n_c:4.1f} cm)  | {g:+5.1f}%")

        print("=" * 76)

    else:
        all_deg = [r["error_deg"] for r in detailed_records]
        all_cm = [r["error_cm"] for r in detailed_records]

        print("\n" + "=" * 70)
        print(f"             RAPPORT DE PRÉCISION : MODE {args.gaze_mode.upper()}")
        print("=" * 70)
        print(f"{'Métrique':<24} | {'Erreur angulaire (°)':<20} | {'Erreur écran (cm)':<18}")
        print("-" * 70)
        print(f"{'Moyenne globale':<24} | {np.mean(all_deg):6.2f}°               | {np.mean(all_cm):6.2f} cm")
        print(f"{'Médiane globale':<24} | {np.median(all_deg):6.2f}°               | {np.median(all_cm):6.2f} cm")
        print(f"{'Écart-type (Std)':<24} | {np.std(all_deg):6.2f}°               | {np.std(all_cm):6.2f} cm")
        print("-" * 70)
        for z_key, z_label in [("center", "Zone Centre"), ("edge", "Zone Bords"), ("corner", "Zone Coins")]:
            z_deg = [r["error_deg"] for r in detailed_records if r["zone"] == z_key]
            z_cm = [r["error_cm"] for r in detailed_records if r["zone"] == z_key]
            if z_deg:
                print(f"{z_label:<24} | {np.mean(z_deg):6.2f}° (méd. {np.median(z_deg):4.2f}°) | {np.mean(z_cm):6.2f} cm")
        print("=" * 70)

    # Sauvegarde CSV
    try:
        keys = detailed_records[0].keys()
        with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in detailed_records:
                writer.writerow(r)
        print(f"\n[validate] Données détaillées enregistrées dans : {out_csv}")
    except Exception as e:
        print(f"[validate] Erreur écriture CSV : {e}", file=sys.stderr)


if __name__ == "__main__":
    run_validation()
