"""Enregistreur passif sur clics souris avec buffer circulaire pré-clic (Windows).

Contrairement à une capture prise à l'instant exact du clic (où l'utilisateur amorce déjà
souvent la saccade suivante), ce collecteur conserve un buffer circulaire des dernières
frames traitées (~1,5 s) et prélève automatiquement l'échantillon à ``t_clic - pre_lag_ms``
(défaut : 75 ms avant le clic), uniquement lorsque le curseur Windows est visible et
hors clignement.

Usage :
    python -m research.collect_clicks --max-clicks 200 --tag "navigation_jour"
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capture import WebcamCapture  # noqa: E402
from face import GLARE_STRONG, FaceDetector  # noqa: E402
from filters import BlinkDetector  # noqa: E402
from gaze_model import IrisGazeEstimator, NetGazeEstimator, gaze_model_available  # noqa: E402
from geometry import ScreenConfig, get_screen_resolution  # noqa: E402
from normalize import FaceNormalizer  # noqa: E402
from paths import data_path  # noqa: E402
from research.eye_crop import extract_binocular_sample  # noqa: E402
from research.models import screen_px_to_gaze_angles_deg  # noqa: E402


SESSIONS_DIR = Path(data_path("research_sessions"))


@dataclass
class BufferedFrame:
    """Instantané stocké dans le buffer circulaire pré-clic."""

    ts: float
    eye_crops: np.ndarray
    geom_vec: np.ndarray
    pose_6dof: np.ndarray
    origin_3d_mm: np.ndarray
    iris_ratios: np.ndarray
    net_pitch_yaw: np.ndarray
    net_hit_mm: np.ndarray
    eye_metrics: np.ndarray
    valid: bool


def select_preclick_frame(
    buffer: deque[BufferedFrame],
    click_ts: float,
    pre_lag_ms: float = 75.0,
    max_delta_ms: float = 60.0,
) -> BufferedFrame | None:
    """Sélectionne dans le buffer la frame valide la plus proche de ``click_ts - pre_lag_ms``.

    Rejette si aucune frame valide n'existe dans une fenêtre de ``±max_delta_ms`` autour
    de l'instant cible pré-clic.
    """
    target_ts = click_ts - pre_lag_ms / 1000.0
    best: BufferedFrame | None = None
    best_dist = float("inf")
    for item in buffer:
        if not item.valid:
            continue
        dt_ms = abs(item.ts - target_ts) * 1000.0
        if dt_ms <= max_delta_ms and dt_ms < best_dist:
            best_dist = dt_ms
            best = item
    return best


def _windows_cursor_info():
    import ctypes
    from ctypes import wintypes

    class CURSORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD),
                    ("hCursor", wintypes.HANDLE), ("ptScreenPos", wintypes.POINT)]

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    def is_visible() -> bool:
        ci = CURSORINFO()
        ci.cbSize = ctypes.sizeof(CURSORINFO)
        return bool(ctypes.windll.user32.GetCursorInfo(ctypes.byref(ci)) and (ci.flags & 1))

    def get_pos() -> tuple[float, float] | None:
        pt = POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
            return float(pt.x), float(pt.y)
        return None

    def left_down() -> bool:
        return (ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000) != 0

    return is_visible, get_pos, left_down


def run_click_collection(
    max_clicks: int = 300,
    duration_s: float = 600.0,
    pre_lag_ms: float = 75.0,
    tag: str = "clicks",
    glasses: bool = True,
    camera: int = 0,
    screen_diag: float = 27.0,
    distance_cm: float = 60.0,
    flip: bool = True,
) -> Path | None:
    """Lance l'enregistreur passif sur clics Windows (Ctrl+C pour arrêter proprement)."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sw, sh = get_screen_resolution()
    screen = ScreenConfig(width_px=sw, height_px=sh, diagonal_inches=screen_diag, distance_cm=distance_cm)

    cap = WebcamCapture(camera_index=camera, width=1280, height=720, target_fps=60)
    detector = FaceDetector()
    normalizer = FaceNormalizer()
    iris_est = IrisGazeEstimator()
    net_est = NetGazeEstimator() if gaze_model_available() else None
    blink = BlinkDetector()

    is_visible, get_pos, left_down = _windows_cursor_info()
    ring: deque[BufferedFrame] = deque(maxlen=90)
    prev_down = False
    last_click_ts = -10.0

    saved: dict[str, list] = {
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
    }

    t0 = time.perf_counter()
    print(f"[clicks] Enregistreur passif actif (pre_lag={pre_lag_ms:.0f} ms, max={max_clicks} clics, Ctrl+C pour sauver).")
    try:
        while len(saved["timestamps"]) < max_clicks and (time.perf_counter() - t0) < duration_s:
            ok, frame, ts = cap.read()
            if not ok or frame is None:
                continue
            if flip:
                frame = cv2.flip(frame, 1)

            face = detector.process(frame, ts)
            norm = normalizer.process(frame, face) if face.detected else None
            sample = extract_binocular_sample(frame, face, compute_glare=True)
            iris_feat = iris_est.estimate(face)

            if net_est is not None and norm is not None and norm.is_valid:
                net_feat = net_est.estimate(face, norm)
                net_py = np.array(net_feat.pitch_yaw_deg, dtype=np.float64)
                net_hit = np.asarray(net_feat.raw_features[:2], dtype=np.float64)
            else:
                net_py = np.array([float("nan"), float("nan")], dtype=np.float64)
                net_hit = np.array([float("nan"), float("nan")], dtype=np.float64)

            is_blink = face.detected and blink.update(face.eye_openness, ts)
            origin = norm.origin_3d_mm if (norm is not None and norm.is_valid) else np.array([0.0, 0.0, distance_cm * 10.0])
            valid = bool(face.detected and sample.valid and not is_blink and sample.glare <= GLARE_STRONG)

            ring.append(
                BufferedFrame(
                    ts=ts,
                    eye_crops=sample.eye_crops,
                    geom_vec=sample.geom_vec,
                    pose_6dof=sample.pose_6dof,
                    origin_3d_mm=origin.astype(np.float64),
                    iris_ratios=np.array(iris_feat.iris_ratios, dtype=np.float64),
                    net_pitch_yaw=net_py,
                    net_hit_mm=net_hit,
                    eye_metrics=np.array([face.iris_diameter_px, sample.eye_openness, sample.glare, 0.0], dtype=np.float64),
                    valid=valid,
                )
            )

            down = left_down()
            if down and not prev_down and (ts - last_click_ts) > 0.25 and is_visible():
                pos = get_pos()
                if pos is not None and 0 <= pos[0] < sw and 0 <= pos[1] < sh:
                    chosen = select_preclick_frame(ring, ts, pre_lag_ms=pre_lag_ms)
                    if chosen is not None:
                        tgt_ang = screen_px_to_gaze_angles_deg(
                            np.array(pos),
                            chosen.origin_3d_mm,
                            sw,
                            sh,
                            screen.mm_per_px_x,
                            screen.mm_per_px_y,
                            screen.camera_offset_y_cm,
                        )
                        saved["eye_crops"].append(chosen.eye_crops)
                        saved["geom_vec"].append(chosen.geom_vec)
                        saved["pose_6dof"].append(chosen.pose_6dof)
                        saved["origin_3d_mm"].append(chosen.origin_3d_mm)
                        saved["iris_ratios"].append(chosen.iris_ratios)
                        saved["net_pitch_yaw"].append(chosen.net_pitch_yaw)
                        saved["net_hit_mm"].append(chosen.net_hit_mm)
                        saved["target_xy"].append(np.array(pos, dtype=np.float64))
                        saved["target_angles_deg"].append(tgt_ang.astype(np.float64))
                        saved["timestamps"].append(float(chosen.ts - t0))
                        saved["eye_metrics"].append(chosen.eye_metrics)
                        last_click_ts = ts
                        print(f"[clicks] #{len(saved['timestamps'])} ({pos[0]:.0f}, {pos[1]:.0f})")
            prev_down = down
    except KeyboardInterrupt:
        print("\n[clicks] Interruption clavier : sauvegarde...")
    finally:
        cap.release()

    if len(saved["timestamps"]) == 0:
        print("[clicks] Aucun clic valide capturé.")
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in tag)
    out_npz = SESSIONS_DIR / f"clicks_{stamp}_{safe_tag}.npz"
    out_meta = SESSIONS_DIR / f"clicks_{stamp}_{safe_tag}.json"

    arrays = {k: np.asarray(v) for k, v in saved.items()}
    arrays["valid_mask"] = np.ones(len(saved["timestamps"]), dtype=bool)
    arrays["screen"] = np.array([sw, sh, screen_diag, distance_cm], dtype=np.float64)
    np.savez_compressed(out_npz, **arrays)

    meta = {
        "session_id": out_npz.stem,
        "type": "passive_clicks",
        "timestamp": stamp,
        "tag": tag,
        "glasses": glasses,
        "pre_lag_ms": pre_lag_ms,
        "num_clicks": len(saved["timestamps"]),
    }
    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[clicks] Sauvegardé : {out_npz} ({len(saved['timestamps'])} clics).")
    return out_npz


def main() -> None:
    ap = argparse.ArgumentParser(description="Collecteur passif de clics pré-saccadiques Eyetrack")
    ap.add_argument("--max-clicks", type=int, default=300)
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--pre-lag-ms", type=float, default=75.0)
    ap.add_argument("--tag", type=str, default="bureau")
    ap.add_argument("--no-glasses", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    run_click_collection(
        max_clicks=args.max_clicks,
        duration_s=args.duration,
        pre_lag_ms=args.pre_lag_ms,
        tag=args.tag,
        glasses=not args.no_glasses,
        camera=args.camera,
    )


if __name__ == "__main__":
    main()
