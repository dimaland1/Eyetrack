"""Diagnostic : le réseau de regard suit-il vraiment les yeux, et à quelle résolution ?

Affiche 5 cibles (centre, gauche, droite, haut, bas), une série par résolution de caméra.
Gardez la tête IMMOBILE et ne bougez que les yeux. Pour chaque résolution et chaque façon
de préparer l'image du visage, le script mesure :
- l'amplitude du regard estimé entre les cibles (à comparer à l'amplitude réelle) ;
- le bruit (écart-type quand on fixe une cible) ;
- le rapport signal/bruit (plus il est grand, meilleure est la configuration).

Configurations comparées :
- zhang_f1900 : normalisation de Zhang avec le visage qui remplit le patch (réglage actuel) ;
- crop_serre  : simple recadrage carré serré autour du visage (sans normalisation) ;
- iris        : ratios d'iris (mode MVP), pour comparaison.

Les patchs vus par le réseau sont enregistrés dans data/diag_patches.png.
Usage : python tools/diag_regard.py [--resolutions 640x480,800x600,1280x720] [--camera 0]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Les outils vivent dans tools/ : on rend importables les modules du tracker (dossier parent)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import math
import cv2
import numpy as np

from capture import WebcamCapture
from face import FaceDetector, FaceData
from normalize import FaceNormalizer, NormalizationData
from gaze_model import NetGazeEstimator, IrisGazeEstimator
from geometry import ScreenConfig, get_screen_resolution
from paths import data_path

TARGETS = [("centre", 0.5, 0.5), ("gauche", 0.08, 0.5), ("droite", 0.92, 0.5),
           ("haut", 0.5, 0.08), ("bas", 0.5, 0.92)]
SECONDS_PER_TARGET = 2.5
SETTLE_SECONDS = 0.8
CONFIGS = ["zhang_f1900", "crop_serre", "iris"]


def tight_crop(frame: np.ndarray, face: FaceData, size: int = 448, scale: float = 1.0) -> np.ndarray:
    """Recadrage carré autour de la boîte englobante des landmarks, redimensionné à size."""
    pts = face.landmarks_px
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * scale
    s = size / side
    M = np.array([[s, 0, size / 2 - s * cx], [0, s, size / 2 - s * cy]], dtype=np.float64)
    return cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def show(win: str, sw: int, sh: int, lines: list[str], target: tuple[int, int] | None = None) -> bool:
    """Affiche l'écran de consigne ; renvoie False si ESC."""
    screen = np.zeros((sh, sw, 3), dtype=np.uint8)
    for i, txt in enumerate(lines):
        cv2.putText(screen, txt, (max(20, sw // 2 - 9 * len(txt)), 60 + 45 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
    if target is not None:
        cv2.circle(screen, target, 22, (0, 140, 255), 2)
        cv2.circle(screen, target, 6, (0, 220, 255), -1)
    cv2.imshow(win, screen)
    return (cv2.waitKey(1) & 0xFF) != 27


def run_series(cap: WebcamCapture, detector: FaceDetector, net: NetGazeEstimator, iris: IrisGazeEstimator,
               normalizer: FaceNormalizer, win: str, sw: int, sh: int, res_label: str, flip: bool,
               patches: dict[str, np.ndarray]) -> dict | None:
    """Une série de 5 cibles ; renvoie les échantillons par configuration et par cible."""
    samples = {c: {t[0]: [] for t in TARGETS} for c in CONFIGS}
    n_frames = 0
    t_first = None
    for name, nx, ny in TARGETS:
        tx, ty = int(nx * sw), int(ny * sh)
        t_start = None
        while True:
            ok, frame, ts = cap.read()
            if not ok or frame is None:
                continue
            n_frames += 1
            if t_first is None:
                t_first = ts
            if flip:
                frame = cv2.flip(frame, 1)
            if t_start is None:
                t_start = ts
            elapsed = ts - t_start
            if elapsed > SECONDS_PER_TARGET:
                break

            face = detector.process(frame, ts)
            if face.detected and elapsed > SETTLE_SECONDS:
                nd = normalizer.process(frame, face)
                g = net.estimate(face, norm_data=nd)
                samples["zhang_f1900"][name].append((g.pitch_yaw_deg[1], g.pitch_yaw_deg[0]))
                crop = tight_crop(frame, face)
                gc = net.estimate(face, norm_data=NormalizationData(face_patch=crop, is_valid=True))
                samples["crop_serre"][name].append((gc.pitch_yaw_deg[1], gc.pitch_yaw_deg[0]))
                samples["iris"][name].append(iris.estimate(face).iris_ratios)
                if name == "centre":
                    patches[f"{res_label} f1900"] = nd.face_patch
                    patches[f"{res_label} crop"] = crop

            lines = [f"[{res_label}] Tete IMMOBILE : suivez la cible avec les YEUX seulement"]
            if not face.detected:
                lines.append("Visage non detecte")
            if not show(win, sw, sh, lines, (tx, ty)):
                return None
    fps = n_frames / max(ts - t_first, 1e-6)
    return {"samples": samples, "fps": fps}


def summarize(samples: dict) -> dict[str, tuple[float, float, float, float]]:
    """(amplitude G->D, amplitude H->B, bruit, signal/bruit) par configuration."""
    out = {}
    for c in CONFIGS:
        s = samples[c]
        if any(len(v) < 5 for v in s.values()):
            continue
        m = {k: np.mean(v, axis=0) for k, v in s.items()}
        noise = float(np.mean([np.std(np.array(v), axis=0).mean() for v in s.values()]))
        ax = float(m["droite"][0] - m["gauche"][0])
        ay = float(m["bas"][1] - m["haut"][1])
        out[c] = (ax, ay, noise, (abs(ax) + abs(ay)) / 2.0 / max(noise, 1e-9))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--screen-diag", type=float, default=27.0)
    ap.add_argument("--distance", type=float, default=60.0)
    ap.add_argument("--resolutions", default="640x480,800x600,1280x720",
                    help="Résolutions à comparer, ex. 640x480,800x600,1280x720")
    args = ap.parse_args()

    resolutions = [tuple(int(v) for v in r.lower().split("x")) for r in args.resolutions.split(",") if r]
    sw, sh = get_screen_resolution()
    sc = ScreenConfig(width_px=sw, height_px=sh, diagonal_inches=args.screen_diag, distance_cm=args.distance)
    detector = FaceDetector()
    net = NetGazeEstimator()
    iris = IrisGazeEstimator()

    win = "Diagnostic regard (ESC pour annuler)"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    results: dict[str, dict] = {}
    patches: dict[str, np.ndarray] = {}
    for (rw, rh) in resolutions:
        label = f"{rw}x{rh}"
        show(win, sw, sh, [f"Ouverture de la camera en {label}...", "(quelques secondes)"])
        cv2.waitKey(1)
        cap = WebcamCapture(camera_index=args.camera, width=rw, height=rh, target_fps=60, strict=True)
        real = f"{cap.actual_width}x{cap.actual_height}"
        normalizer = FaceNormalizer(target_size=448)  # focale par défaut (1900)
        res = run_series(cap, detector, net, iris, normalizer, win, sw, sh, real, not args.no_flip, patches)
        cap.release()
        if res is None:
            print("Annulé.")
            break
        results[real] = res

    cv2.destroyAllWindows()

    dist = args.distance * 10.0
    exp_x = 2 * math.degrees(math.atan(0.42 * sc.width_mm / dist))
    exp_y = 2 * math.degrees(math.atan(0.42 * sc.height_mm / dist))
    print("\n=== Diagnostic du regard par résolution ===")
    print(f"Amplitude réelle attendue : gauche->droite ≈ {exp_x:.0f}°, haut->bas ≈ {exp_y:.0f}°")
    for real, res in results.items():
        print(f"\n--- Caméra {real} ({res['fps']:.0f} fps traités) ---")
        print(f"{'configuration':14s} {'G->D':>9s} {'H->B':>9s} {'bruit':>8s} {'signal/bruit':>13s}")
        summ = summarize(res["samples"])
        for c in CONFIGS:
            if c not in summ:
                print(f"{c:14s}  pas assez d'échantillons (visage perdu ?)")
                continue
            ax, ay, noise, snr = summ[c]
            if c == "iris":
                print(f"{c:14s} {ax:+9.3f} {ay:+9.3f} {noise:8.3f} {snr:12.1f}")
            else:
                print(f"{c:14s} {ax:+8.1f}° {ay:+8.1f}° {noise:7.1f}° {snr:12.1f}")
    print("\nLe meilleur choix est celui avec le plus grand 'signal/bruit' (et, pour le réseau,")
    print("une amplitude la plus proche possible de l'amplitude attendue).")

    if patches:
        row = [cv2.resize(p, (240, 240)) for p in patches.values()]
        for img, lab in zip(row, patches.keys()):
            cv2.putText(img, lab, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        out = data_path("diag_patches.png")
        cv2.imwrite(out, np.hstack(row))
        print(f"Patchs vus par le réseau enregistrés dans {out}")


if __name__ == "__main__":
    main()
