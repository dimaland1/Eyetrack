"""Diagnostic de stabilité : d'où vient l'agitation ?

Pendant 10 secondes, fixez le point au centre de l'écran SANS bouger la tête.
Le script utilise exactement la même chaîne que le tracker (profil, filtres, calibration)
et mesure, pour la tête et pour le regard :
- le « tremblement » image par image (bruit haute fréquence), brut puis filtré ;
- la dérive lente (écart-type total) ;
- le framerate réel et les pertes de visage.

Les mesures image par image sont enregistrées dans diag_stabilite.csv.
Usage : python diag_stabilite.py [--profile default] [--gaze-mode net] [--camera 0]
"""

from __future__ import annotations
import argparse
import csv
import os
import cv2
import numpy as np

from capture import WebcamCapture
from face import FaceDetector
from normalize import FaceNormalizer
from gaze_model import create_gaze_estimator
from geometry import ScreenConfig, get_screen_resolution
from calibration import CalibrationManager
from filters import OneEuroFilter
from profile import load_profile

DURATION_S = 10.0
SETTLE_S = 1.5
POSE_NAMES = ["x_cm", "y_cm", "z_cm", "yaw", "pitch", "roll"]


def jitter(a: np.ndarray) -> float:
    """Bruit image par image : écart-type des différences successives / sqrt(2)."""
    if len(a) < 3:
        return float("nan")
    return float(np.std(np.diff(a)) / np.sqrt(2.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="default")
    ap.add_argument("--gaze-mode", choices=["iris", "net"], default=None)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-flip", action="store_true")
    args = ap.parse_args()

    prof = load_profile(args.profile)
    mode = args.gaze_mode or prof.gaze_mode
    sw, sh = get_screen_resolution()
    sc = ScreenConfig(width_px=sw, height_px=sh, diagonal_inches=prof.screen_diag, distance_cm=prof.distance_cm)

    cap = WebcamCapture(camera_index=args.camera, width=1280, height=720, target_fps=60)
    detector = FaceDetector()
    normalizer = FaceNormalizer(target_size=448 if mode == "net" else 224, virtual_focal=prof.patch_focal)
    estimator = create_gaze_estimator(mode=mode)
    calib = CalibrationManager(screen_config=sc, mode=mode)
    calibrated = calib.load(f"calibration_{mode}.npz")
    pose_f = OneEuroFilter(6, prof.pose_min_cutoff, prof.pose_beta)
    gaze_f = OneEuroFilter(2, prof.gaze_min_cutoff, prof.gaze_beta, median_window=prof.gaze_median)

    print(f"Profil {prof.name} | mode {mode} | calibré : {calibrated} | caméra {cap.actual_width}x{cap.actual_height} @ {cap.actual_fps:.0f}")
    win = "Diagnostic stabilite"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    rows = []
    frames = 0
    lost = 0
    t0 = None
    while True:
        ok, frame, ts = cap.read()
        if not ok or frame is None:
            continue
        if not args.no_flip:
            frame = cv2.flip(frame, 1)
        if t0 is None:
            t0 = ts
        el = ts - t0
        if el > DURATION_S:
            break

        face = detector.process(frame, ts)
        nd = normalizer.process(frame, face)
        g = estimator.estimate(face, norm_data=nd)

        screen = np.zeros((sh, sw, 3), dtype=np.uint8)
        cv2.putText(screen, f"Fixez le point, tete immobile ({max(0, DURATION_S - el):.0f} s)",
                    (sw // 2 - 330, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
        cv2.circle(screen, (sw // 2, sh // 2), 8, (0, 220, 255), -1)
        cv2.imshow(win, screen)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

        if el < SETTLE_S:
            continue
        frames += 1
        if not face.detected:
            lost += 1
            continue
        raw_pose = np.array([*face.pose_xyz, *face.pose_ypr], dtype=np.float64)
        fp = pose_f.filter(raw_pose, ts)
        if calibrated:
            gx, gy = calib.predict(g.raw_features, head_pose=(face.pose_ypr[0], face.pose_ypr[1]))
        else:
            gx, gy = float("nan"), float("nan")
        if np.isfinite(gx):
            gx = float(np.clip(gx, -0.05 * sw, 1.05 * sw))
            gy = float(np.clip(gy, -0.05 * sh, 1.05 * sh))
            fg = gaze_f.filter([gx, gy], ts)
        else:
            fg = np.array([np.nan, np.nan])
        rows.append([ts, *raw_pose, *fp, gx, gy, fg[0], fg[1], *g.raw_features[:2]])

    cv2.destroyAllWindows()
    cap.release()

    if len(rows) < 30:
        print("Pas assez de mesures (visage non détecté ?).")
        return
    A = np.array(rows)
    t = A[:, 0]
    fps = (len(t) - 1) / (t[-1] - t[0])
    dts = np.diff(t)

    with open("diag_stabilite.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["t", *[f"brut_{n}" for n in POSE_NAMES], *[f"filtre_{n}" for n in POSE_NAMES],
                    "regard_brut_x", "regard_brut_y", "regard_filtre_x", "regard_filtre_y", "feat0", "feat1"])
        w.writerows(A.tolist())

    print("\n=== Stabilité (10 s, tête immobile) ===")
    print(f"Framerate réel : {fps:.1f} fps | intervalle max entre images : {dts.max() * 1000:.0f} ms"
          f" | visage perdu : {lost}/{frames} images")
    print(f"\n{'canal':10s} {'tremblement brut':>17s} {'tremblement filtré':>19s} {'dérive totale filtrée':>22s}")
    for i, n in enumerate(POSE_NAMES):
        raw = A[:, 1 + i]
        filt = A[:, 7 + i]
        unit = "cm" if n.endswith("cm") else "°"
        print(f"{n:10s} {jitter(raw):14.2f} {unit:2s} {jitter(filt):16.2f} {unit:2s} {np.std(filt):19.2f} {unit}")
        if np.max(np.abs(np.diff(raw))) > 30 and unit == "°":
            print(f"           /!\\ sauts de plus de 30° sur {n} : angle qui « bascule » (±180°)")
    if calibrated:
        for j, n in enumerate(["regard_x", "regard_y"]):
            raw = A[:, 13 + j]
            filt = A[:, 15 + j]
            print(f"{n:10s} {jitter(raw):14.0f} px {jitter(filt):16.0f} px {np.std(filt):19.0f} px")
        print(f"\nPosition moyenne du regard filtré : ({np.mean(A[:, 15]):.0f}, {np.mean(A[:, 16]):.0f})"
              f"  — centre de l'écran : ({sw // 2}, {sh // 2})")
    else:
        print("\n(Regard non mesuré : pas de calibration pour ce mode.)")
    print(f"\nDétail image par image : {os.path.abspath('diag_stabilite.csv')}")


if __name__ == "__main__":
    main()
