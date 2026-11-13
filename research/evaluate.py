"""Évaluation statistique rigoureuse (métriques angulaires, IC 95 %, latence CPU).

Calcule les métriques attendues d'une évaluation niveau thèse :
- Erreur angulaire 3D exacte (°), moyenne, médiane, écart-type, P90, P95, RMSE,
- Décomposition horizontale (yaw) vs verticale (pitch) et spatiale (centre / bords / coins),
- Erreur en cm à l'écran,
- Intervalle de confiance bootstrap à 95 % et test de Wilcoxon apparié,
- Évaluation multi-ancres (K=1 ou K=5 points) pour ``DifferentialEyeNet`` (Scénario B),
- Benchmark de latence d'inférence CPU (ms/frame).
"""

from __future__ import annotations

import math
import time
from typing import Iterable

import numpy as np
import torch
from scipy.stats import wilcoxon

from geometry import ScreenConfig
from research.augmentations import preprocess_eye_batch_gray_norm, preprocess_eye_batch_rgb
from research.models import DifferentialEyeNet, DualEyePersonalNet, gaze_angles_deg_to_screen_px


def compute_gaze_metrics(
    pred_xy_px: np.ndarray,
    target_xy_px: np.ndarray,
    origin_3d_mm: np.ndarray,
    screen: ScreenConfig,
    seed: int = 0,
) -> dict:
    """Calcule l'ensemble des métriques géométriques et statistiques."""
    pred_xy = np.asarray(pred_xy_px, dtype=np.float64)
    tgt_xy = np.asarray(target_xy_px, dtype=np.float64)
    ori = np.asarray(origin_3d_mm, dtype=np.float64)

    n = len(tgt_xy)
    angles_deg = np.zeros(n, dtype=np.float64)
    yaw_err = np.zeros(n, dtype=np.float64)
    pitch_err = np.zeros(n, dtype=np.float64)
    dist_cm = np.zeros(n, dtype=np.float64)

    for i in range(n):
        eye_cm = (float(ori[i, 0]) / 10.0, float(ori[i, 1]) / 10.0, float(ori[i, 2]) / 10.0)
        angles_deg[i] = screen.visual_angle_deg(
            (float(tgt_xy[i, 0]), float(tgt_xy[i, 1])),
            (float(pred_xy[i, 0]), float(pred_xy[i, 1])),
            eye_offset_xyz_cm=eye_cm,
        )
        dist_cm[i] = screen.distance_cm_between_px(
            (float(tgt_xy[i, 0]), float(tgt_xy[i, 1])),
            (float(pred_xy[i, 0]), float(pred_xy[i, 1])),
        )
        dx_mm = (pred_xy[i, 0] - tgt_xy[i, 0]) * screen.mm_per_px_x
        dy_mm = (pred_xy[i, 1] - tgt_xy[i, 1]) * screen.mm_per_px_y
        z_mm = max(abs(float(ori[i, 2])), 100.0)
        yaw_err[i] = abs(math.degrees(math.atan2(dx_mm, z_mm)))
        pitch_err[i] = abs(math.degrees(math.atan2(dy_mm, z_mm)))

    # Décomposition spatiale (centre <= 25% rayon normalisé, coins hors [0.25..0.75]^2)
    nx = tgt_xy[:, 0] / screen.width_px
    ny = tgt_xy[:, 1] / screen.height_px
    is_center = (np.abs(nx - 0.5) <= 0.22) & (np.abs(ny - 0.5) <= 0.22)
    is_corner = ((nx < 0.25) | (nx > 0.75)) & ((ny < 0.25) | (ny > 0.75))
    is_edge = ~(is_center | is_corner)

    # Bootstrap IC 95 % de la moyenne
    rng = np.random.default_rng(seed)
    boot = np.array([angles_deg[rng.integers(0, n, size=n)].mean() for _ in range(500)])
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    return {
        "n": int(n),
        "mean_deg": float(np.mean(angles_deg)),
        "median_deg": float(np.median(angles_deg)),
        "std_deg": float(np.std(angles_deg)),
        "rmse_deg": float(np.sqrt(np.mean(angles_deg**2))),
        "p90_deg": float(np.percentile(angles_deg, 90)),
        "p95_deg": float(np.percentile(angles_deg, 95)),
        "ci95_low_deg": float(ci_low),
        "ci95_high_deg": float(ci_high),
        "mean_cm": float(np.mean(dist_cm)),
        "yaw_mae_deg": float(np.mean(yaw_err)),
        "pitch_mae_deg": float(np.mean(pitch_err)),
        "center_deg": float(np.mean(angles_deg[is_center])) if is_center.any() else float("nan"),
        "edge_deg": float(np.mean(angles_deg[is_edge])) if is_edge.any() else float("nan"),
        "corner_deg": float(np.mean(angles_deg[is_corner])) if is_corner.any() else float("nan"),
        "errors_deg": angles_deg,
    }


def paired_wilcoxon(errors_a: np.ndarray, errors_b: np.ndarray) -> dict[str, float]:
    """Test non-paramétrique de Wilcoxon apparié entre deux modèles sur les mêmes frames."""
    diff = np.asarray(errors_a) - np.asarray(errors_b)
    if np.allclose(diff, 0.0):
        return {"statistic": 0.0, "p_value": 1.0, "mean_diff_deg": 0.0}
    stat, p_val = wilcoxon(errors_a, errors_b)
    return {"statistic": float(stat), "p_value": float(p_val), "mean_diff_deg": float(np.mean(diff))}


@torch.no_grad()
def predict_personal_session(model: DualEyePersonalNet, session: dict[str, np.ndarray], batch_size: int = 128) -> np.ndarray:
    """Prédit les coordonnées écran (x, y) en pixels sur toute une session."""
    model.eval().cpu()
    rgb = preprocess_eye_batch_rgb(session["eye_crops"])
    geom = session["geom_vec"].astype(np.float32)

    preds = []
    for i in range(0, len(geom), batch_size):
        el = torch.from_numpy(rgb[i : i + batch_size, 0])
        er = torch.from_numpy(rgb[i : i + batch_size, 1])
        g = torch.from_numpy(geom[i : i + batch_size])
        preds.append(model(el, er, g).numpy())
    ang_pred = np.vstack(preds)

    sw, sh, diag, dist = session["screen"]
    sc = ScreenConfig(width_px=int(sw), height_px=int(sh), diagonal_inches=float(diag), distance_cm=float(dist))
    return gaze_angles_deg_to_screen_px(
        ang_pred,
        session["origin_3d_mm"],
        sc.width_px,
        sc.height_px,
        sc.mm_per_px_x,
        sc.mm_per_px_y,
        sc.camera_offset_y_cm,
    )


def select_anchor_indices(target_xy: np.ndarray, k_anchors: int = 5) -> np.ndarray:
    """Sélectionne K points ancres bien répartis sur l'écran (1=centre, 5=croix, etc.)."""
    tx = target_xy[:, 0]
    ty = target_xy[:, 1]
    w, h = tx.max() + tx.min(), ty.max() + ty.min()
    if k_anchors == 1:
        pts = [(0.5 * w, 0.5 * h)]
    else:
        pts = [(0.5 * w, 0.5 * h), (0.15 * w, 0.15 * h), (0.85 * w, 0.15 * h), (0.15 * w, 0.85 * h), (0.85 * w, 0.85 * h)]
        if k_anchors > 5:
            pts += [(0.5 * w, 0.15 * h), (0.5 * w, 0.85 * h), (0.15 * w, 0.5 * h), (0.85 * w, 0.5 * h)]
        pts = pts[:k_anchors]

    chosen = []
    for px, py in pts:
        d2 = (tx - px) ** 2 + (ty - py) ** 2
        idx = int(np.argmin(d2))
        chosen.append(idx)
    return np.array(chosen, dtype=int)


@torch.no_grad()
def predict_differential_session(
    model: DifferentialEyeNet,
    session: dict[str, np.ndarray],
    anchor_indices: Iterable[int],
    batch_size: int = 128,
) -> np.ndarray:
    """Inférence Scénario B par agrégation sur ``K`` ancres calibrées du sujet cible."""
    model.eval().cpu()
    gray = preprocess_eye_batch_gray_norm(session["eye_crops"])
    geom = session["geom_vec"].astype(np.float32)
    angles_gt = session["target_angles_deg"].astype(np.float64)

    anc = np.asarray(list(anchor_indices), dtype=int)
    k = len(anc)
    el_a = torch.from_numpy(gray[anc, 0])
    er_a = torch.from_numpy(gray[anc, 1])
    g_a = torch.from_numpy(geom[anc])
    ang_a = angles_gt[anc]  # (K, 2)

    n = len(geom)
    ang_pred = np.zeros((n, 2), dtype=np.float64)

    for i in range(0, n, batch_size):
        el_q = torch.from_numpy(gray[i : i + batch_size, 0])
        er_q = torch.from_numpy(gray[i : i + batch_size, 1])
        g_q = torch.from_numpy(geom[i : i + batch_size])
        bs = len(g_q)

        # Évalue toutes les paires (bs, K)
        est_per_anchor = np.zeros((bs, k, 2), dtype=np.float64)
        for j in range(k):
            el_r = el_a[j : j + 1].expand(bs, -1, -1, -1)
            er_r = er_a[j : j + 1].expand(bs, -1, -1, -1)
            g_r = g_a[j : j + 1].expand(bs, -1)
            delta = model(el_q, er_q, g_q, el_r, er_r, g_r).numpy()
            est_per_anchor[:, j, :] = ang_a[j] + delta
        ang_pred[i : i + bs] = est_per_anchor.mean(axis=1)

    sw, sh, diag, dist = session["screen"]
    sc = ScreenConfig(width_px=int(sw), height_px=int(sh), diagonal_inches=float(diag), distance_cm=float(dist))
    return gaze_angles_deg_to_screen_px(
        ang_pred,
        session["origin_3d_mm"],
        sc.width_px,
        sc.height_px,
        sc.mm_per_px_x,
        sc.mm_per_px_y,
        sc.camera_offset_y_cm,
    )


@torch.no_grad()
def benchmark_cpu_latency(model: torch.nn.Module, is_differential: bool = False, runs: int = 200) -> dict[str, float]:
    """Mesure la latence d'inférence mono-frame (batch=1) sur CPU en ms."""
    model.eval().cpu()
    if is_differential:
        x = (
            torch.randn(1, 1, 48, 96),
            torch.randn(1, 1, 48, 96),
            torch.randn(1, 9),
            torch.randn(1, 1, 48, 96),
            torch.randn(1, 1, 48, 96),
            torch.randn(1, 9),
        )
    else:
        x = (torch.randn(1, 3, 48, 96), torch.randn(1, 3, 48, 96), torch.randn(1, 9))

    for _ in range(20):
        model(*x)

    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(*x)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times_ms)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }
