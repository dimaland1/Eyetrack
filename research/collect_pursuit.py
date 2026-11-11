"""Collecteur interactif Smooth Pursuit pour jeu de données mono-sujet Eyetrack.

Affiche une cible continue de Lissajous/spirale couvrant tout l'écran (dont les bords
à 6 %) à vitesse angulaire bornée (< 15 deg/s). L'utilisateur suit simplement la cible
du regard tout en variant doucement la pose de sa tête.

Pour chaque frame valide (visage détecté, hors clignement), enregistre :
- les patchs binoculaires alignés 96x48 (uint8 BGR),
- les cibles écran (x, y) et angles visuels 3D exacts (yaw, pitch),
- la pose 6DoF, l'origine 3D, le diamètre de l'iris, l'ouverture oculaire, le reflet,
- les prédictions des baselines actuelles (ratios d'iris + MobileOne-S0 si disponible)
  afin de comparer toutes les méthodes sur exactement les mêmes frames réelles.

Usage :
    python -m research.collect_pursuit --duration 60 --tag "jour_lunettes"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capture import WebcamCapture  # noqa: E402
from face import GLARE_STRONG, FaceDetector, glare_on_iris  # noqa: E402
from filters import BlinkDetector  # noqa: E402
from gaze_model import IrisGazeEstimator, NetGazeEstimator, gaze_model_available  # noqa: E402
from geometry import ScreenConfig, get_screen_resolution  # noqa: E402
from normalize import FaceNormalizer  # noqa: E402
from paths import data_path  # noqa: E402
from research.eye_crop import extract_binocular_sample  # noqa: E402
from research.lag_estimation import PursuitTrajectoryConfig, lissajous_pursuit_target  # noqa: E402
from research.models import screen_px_to_gaze_angles_deg  # noqa: E402


SESSIONS_DIR = Path(data_path("research_sessions"))


def run_pursuit_collection(
    duration_s: float = 60.0,
    tag: str = "session",
    glasses: bool = True,
    lighting: str = "normal",
    camera: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 60,
    screen_diag: float = 27.0,
    distance_cm: float = 60.0,
    flip: bool = True,
) -> Path | None:
    """Lance une session plein écran de poursuite lisse et sauvegarde le .npz."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sw, sh = get_screen_resolution()
    screen = ScreenConfig(width_px=sw, height_px=sh, diagonal_inches=screen_diag, distance_cm=distance_cm)

    cap = WebcamCapture(camera_index=camera, width=width, height=height, target_fps=fps)
    detector = FaceDetector()
    normalizer = FaceNormalizer()
    iris_est = IrisGazeEstimator()
    net_est = NetGazeEstimator() if gaze_model_available() else None
    blink = BlinkDetector()

    win = "Eyetrack Research - Smooth Pursuit (Echap pour annuler)"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Compte à rebours de préparation (3 secondes)
    t_prep = time.perf_counter()
    while time.perf_counter() - t_prep < 3.0:
        ok, frame, _ = cap.read()
        if ok and frame is not None and flip:
            frame = cv2.flip(frame, 1)
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        rem = 3.0 - (time.perf_counter() - t_prep)
        cv2.putText(
            canvas,
            f"Smooth Pursuit ({duration_s:.0f}s) - Demarrage dans {math.ceil(rem)}...",
            (sw // 2 - 420, sh // 2 - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (230, 230, 230),
            2,
        )
        cv2.putText(
            canvas,
            "Suivez la bille des YEUX tout en bougeant doucement la TETE (yaw/pitch/distance)",
            (sw // 2 - 520, sh // 2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 210, 255),
            2,
        )
        cv2.imshow(win, canvas)
        if (cv2.waitKey(1) & 0xFF) == 27:
            cap.release()
            cv2.destroyWindow(win)
            return None

    cfg = PursuitTrajectoryConfig(
        duration_s=duration_s,
        phase_x=float(np.random.uniform(0, 2 * math.pi)),
        phase_y=float(np.random.uniform(0, 2 * math.pi)),
    )

    records: dict[str, list] = {
        "eye_crops": [],
        "geom_vec": [],
        "pose_6dof": [],
        "origin_3d_mm": [],
        "iris_ratios": [],
        "net_pitch_yaw": [],
        "net_hit_mm": [],
        "target_xy": [],
        "target_angles_deg": [],
        "timestamps": [],
        "eye_metrics": [],
        "valid_mask": [],
    }

    t0 = time.perf_counter()
    cancelled = False
    while True:
        now = time.perf_counter()
        elapsed = now - t0
        if elapsed >= duration_s:
            break

        ok, frame, ts = cap.read()
        if not ok or frame is None:
            continue
        if flip:
            frame = cv2.flip(frame, 1)

        tx, ty, speed_deg_s = lissajous_pursuit_target(elapsed, screen, cfg)
        tx_f, ty_f = float(tx), float(ty)

        face = detector.process(frame, ts)
        norm = normalizer.process(frame, face) if face.detected else None
        sample = extract_binocular_sample(frame, face, compute_glare=True)

        iris_feat = iris_est.estimate(face)
        if net_est is not None and norm is not None and norm.is_valid:
            net_feat = net_est.estimate(face, norm)
            net_py = net_feat.pitch_yaw_deg
            net_hit = net_feat.raw_features[:2]
        else:
            net_py = (float("nan"), float("nan"))
            net_hit = np.array([float("nan"), float("nan")], dtype=np.float64)

        is_blink = face.detected and blink.update(face.eye_openness, ts)
        is_valid = bool(face.detected and sample.valid and not is_blink)
        origin = norm.origin_3d_mm if (norm is not None and norm.is_valid) else np.array([0.0, 0.0, distance_cm * 10.0])

        tgt_ang = screen_px_to_gaze_angles_deg(
            np.array([tx_f, ty_f]),
            origin,
            sw,
            sh,
            screen.mm_per_px_x,
            screen.mm_per_px_y,
            screen.camera_offset_y_cm,
        )

        records["eye_crops"].append(sample.eye_crops)
        records["geom_vec"].append(sample.geom_vec)
        records["pose_6dof"].append(sample.pose_6dof)
        records["origin_3d_mm"].append(origin.astype(np.float64))
        records["iris_ratios"].append(np.array(iris_feat.iris_ratios, dtype=np.float64))
        records["net_pitch_yaw"].append(np.array(net_py, dtype=np.float64))
        records["net_hit_mm"].append(np.asarray(net_hit, dtype=np.float64))
        records["target_xy"].append(np.array([tx_f, ty_f], dtype=np.float64))
        records["target_angles_deg"].append(tgt_ang.astype(np.float64))
        records["timestamps"].append(float(elapsed))
        records["eye_metrics"].append(
            np.array([face.iris_diameter_px, sample.eye_openness, sample.glare, float(speed_deg_s)], dtype=np.float64)
        )
        records["valid_mask"].append(is_valid)

        # Affichage stimulus
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        ix, iy = int(round(tx_f)), int(round(ty_f))
        cv2.circle(canvas, (ix, iy), 20, (0, 140, 255), 2)
        cv2.circle(canvas, (ix, iy), 6, (0, 240, 255), -1)

        prog = elapsed / duration_s
        cv2.rectangle(canvas, (20, 20), (320, 34), (40, 40, 40), -1)
        cv2.rectangle(canvas, (20, 20), (20 + int(300 * prog), 34), (60, 210, 90), -1)
        cv2.putText(
            canvas,
            f"{elapsed:4.1f}/{duration_s:.0f}s | {len(records['timestamps'])} frames | v={speed_deg_s:.1f} deg/s",
            (335, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (190, 190, 190),
            1,
        )
        if not face.detected:
            cv2.putText(canvas, "Visage non detecte!", (sw // 2 - 140, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 240), 2)
        elif sample.glare > GLARE_STRONG:
            cv2.putText(canvas, "Reflet fort sur les lunettes", (sw // 2 - 180, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 150, 255), 2)

        # Mini indicateur de couverture de pose de tête (en haut à droite)
        bx, by = sw - 80, 65
        cv2.rectangle(canvas, (bx - 40, by - 40), (bx + 40, by + 40), (35, 35, 35), -1)
        cv2.rectangle(canvas, (bx - 40, by - 40), (bx + 40, by + 40), (95, 95, 95), 1)
        if face.detected:
            dx_h = int(np.clip(face.pose_ypr[0] * 2.0, -35, 35))
            dy_h = int(np.clip(-face.pose_ypr[1] * 2.0, -35, 35))
            cv2.circle(canvas, (bx + dx_h, by + dy_h), 5, (0, 230, 120), -1)

        cv2.imshow(win, canvas)
        if (cv2.waitKey(1) & 0xFF) == 27:
            cancelled = True
            break

    cap.release()
    cv2.destroyWindow(win)
    if cancelled or len(records["timestamps"]) < 30:
        print("[pursuit] Session annulée ou trop courte.")
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
    out_npz = SESSIONS_DIR / f"pursuit_{stamp}_{safe_tag}.npz"
    out_meta = SESSIONS_DIR / f"pursuit_{stamp}_{safe_tag}.json"

    arrays = {k: np.asarray(v) for k, v in records.items()}
    arrays["screen"] = np.array([sw, sh, screen_diag, distance_cm], dtype=np.float64)
    np.savez_compressed(out_npz, **arrays)

    valid_n = int(arrays["valid_mask"].sum())
    meta = {
        "session_id": out_npz.stem,
        "type": "smooth_pursuit",
        "timestamp": stamp,
        "tag": tag,
        "glasses": glasses,
        "lighting": lighting,
        "duration_s": float(arrays["timestamps"][-1]),
        "num_frames": len(arrays["timestamps"]),
        "num_valid": valid_n,
        "fps_effective": float(len(arrays["timestamps"]) / max(arrays["timestamps"][-1], 1e-6)),
        "screen": {"width_px": sw, "height_px": sh, "diagonal_in": screen_diag, "distance_cm": distance_cm},
        "camera": {"width": cap.actual_width, "height": cap.actual_height, "fps_reported": cap.actual_fps},
        "head_yaw_ptp_deg": float(np.ptp(arrays["pose_6dof"][arrays["valid_mask"], 3])) if valid_n > 5 else 0.0,
        "head_pitch_ptp_deg": float(np.ptp(arrays["pose_6dof"][arrays["valid_mask"], 4])) if valid_n > 5 else 0.0,
        "median_iris_px": float(np.median(arrays["eye_metrics"][arrays["valid_mask"], 0])) if valid_n > 5 else 0.0,
    }
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[pursuit] Sauvegardé : {out_npz} ({valid_n}/{meta['num_frames']} frames valides, {meta['fps_effective']:.1f} fps)")
    return out_npz


def main() -> None:
    ap = argparse.ArgumentParser(description="Collecteur Smooth Pursuit pour Eyetrack Research")
    ap.add_argument("--duration", type=float, default=60.0, help="Durée en secondes (défaut: 60)")
    ap.add_argument("--tag", type=str, default="bureau", help="Étiquette de session")
    ap.add_argument("--no-glasses", action="store_true", help="Indiquer une session sans lunettes")
    ap.add_argument("--lighting", type=str, default="normal", help="Condition d'éclairage (jour, lampe, faible...)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--distance", type=float, default=60.0)
    ap.add_argument("--screen-diag", type=float, default=27.0)
    args = ap.parse_args()

    run_pursuit_collection(
        duration_s=args.duration,
        tag=args.tag,
        glasses=not args.no_glasses,
        lighting=args.lighting,
        camera=args.camera,
        distance_cm=args.distance,
        screen_diag=args.screen_diag,
    )


if __name__ == "__main__":
    main()
