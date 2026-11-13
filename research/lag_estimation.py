"""Trajectoires de poursuite lisse (Smooth Pursuit) et estimation du décalage visuo-moteur.

En poursuite oculaire lisse, l'œil humain suit une cible continue de vitesse angulaire
modérée (< 15 deg/s) avec un gain proche de l'unité et un retard visuo-moteur quasi constant
tau (typiquement 50-120 ms). Ce module génère des trajectoires couvrant uniformément
l'écran (y compris les bords) tout en bornant la vitesse angulaire, puis estime tau
empiriquement par minimisation de l'erreur de prédiction sous décalage temporel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from geometry import ScreenConfig


@dataclass
class PursuitTrajectoryConfig:
    """Paramètres d'une trajectoire Smooth Pursuit multi-fréquences."""

    duration_s: float = 60.0
    margin: float = 0.06
    # Deux composantes harmoniques incommensurables par axe pour éviter les cycles fermés
    fx1_hz: float = 0.11
    fx2_hz: float = 0.047
    fy1_hz: float = 0.13
    fy2_hz: float = 0.061
    phase_x: float = 0.0
    phase_y: float = math.pi / 3.0


def lissajous_pursuit_target(
    t_s: float | np.ndarray,
    screen: ScreenConfig,
    cfg: PursuitTrajectoryConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcule la position (x, y en pixels) et la vitesse angulaire (deg/s) à l'instant t_s.

    La trajectoire utilise des sommes sinusoïdales lisses : les bords sont visités grâce à
    une légère saturation tanh (``tanh(1.35 * s) / tanh(1.35)``) qui ralentit doucement
    aux extrémités de l'écran sans discontinuité d'accélération.

    Returns:
        (x_px, y_px, speed_deg_s) sous forme de tableaux/scalaires numpy.
    """
    cfg = cfg or PursuitTrajectoryConfig()
    t = np.asarray(t_s, dtype=np.float64)

    sx = 0.65 * np.sin(2.0 * math.pi * cfg.fx1_hz * t + cfg.phase_x) + 0.35 * np.sin(
        2.0 * math.pi * cfg.fx2_hz * t + 0.7
    )
    sy = 0.65 * np.sin(2.0 * math.pi * cfg.fy1_hz * t + cfg.phase_y) + 0.35 * np.sin(
        2.0 * math.pi * cfg.fy2_hz * t + 1.9
    )

    # Écrêtage doux pour passer plus de temps près des bords et coins tout en restant C^infini
    k = 1.25
    norm_factor = math.tanh(k)
    ux = 0.5 + (0.5 - cfg.margin) * (np.tanh(k * sx) / norm_factor)
    uy = 0.5 + (0.5 - cfg.margin) * (np.tanh(k * sy) / norm_factor)

    x_px = ux * screen.width_px
    y_px = uy * screen.height_px

    # Dérivée analytique pour mesurer la vitesse angulaire instantanée (deg/s)
    dsx = 2.0 * math.pi * (
        0.65 * cfg.fx1_hz * np.cos(2.0 * math.pi * cfg.fx1_hz * t + cfg.phase_x)
        + 0.35 * cfg.fx2_hz * np.cos(2.0 * math.pi * cfg.fx2_hz * t + 0.7)
    )
    dsy = 2.0 * math.pi * (
        0.65 * cfg.fy1_hz * np.cos(2.0 * math.pi * cfg.fy1_hz * t + cfg.phase_y)
        + 0.35 * cfg.fy2_hz * np.cos(2.0 * math.pi * cfg.fy2_hz * t + 1.9)
    )
    dux = (0.5 - cfg.margin) * (k / norm_factor) * (1.0 - np.tanh(k * sx) ** 2) * dsx
    duy = (0.5 - cfg.margin) * (k / norm_factor) * (1.0 - np.tanh(k * sy) ** 2) * dsy

    vx_mm = dux * screen.width_mm
    vy_mm = duy * screen.height_mm
    v_mm_s = np.sqrt(vx_mm**2 + vy_mm**2)
    speed_deg_s = np.degrees(np.arctan2(v_mm_s, screen.distance_cm * 10.0))
    return x_px, y_px, speed_deg_s


def shift_targets_by_lag(
    timestamps_s: np.ndarray,
    target_xy: np.ndarray,
    lag_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Aligne les cibles sur les images en évaluant la cible à ``t - lag_ms/1000``.

    Si l'œil suit la cible avec un retard ``tau > 0``, le regard observé à l'instant ``t``
    correspond à la position de la cible à l'instant ``t - tau``.
    """
    t = np.asarray(timestamps_s, dtype=np.float64)
    tgt = np.asarray(target_xy, dtype=np.float64)
    t_shifted = t - (lag_ms / 1000.0)
    valid = (t_shifted >= t[0]) & (t_shifted <= t[-1])
    interp_x = np.interp(t_shifted, t, tgt[:, 0])
    interp_y = np.interp(t_shifted, t, tgt[:, 1])
    return np.column_stack([interp_x, interp_y]), valid


def estimate_visuomotor_lag(
    timestamps_s: np.ndarray,
    features: np.ndarray,
    target_xy: np.ndarray,
    lag_range_ms: tuple[float, float] = (-50.0, 220.0),
    step_ms: float = 5.0,
    ridge_alpha: float = 1e-2,
) -> dict:
    """Estime le décalage temporel tau (ms) minimisant l'erreur de régression linéaire.

    Utilise une validation croisée temporelle en deux blocs disjoints (première moitié /
    seconde moitié) pour éviter tout surapprentissage du délai.

    Returns:
        Dictionnaire contenant ``best_lag_ms``, ``min_rmse_px``, ``lags_ms``, ``rmses_px``.
    """
    t = np.asarray(timestamps_s, dtype=np.float64)
    X = np.asarray(features, dtype=np.float64)
    tgt = np.asarray(target_xy, dtype=np.float64)

    lags = np.arange(lag_range_ms[0], lag_range_ms[1] + 1e-6, step_ms)
    rmses = np.zeros_like(lags, dtype=np.float64)

    # Masque commun aux extrémités pour comparer tous les délais sur exactement les mêmes frames
    max_abs_lag_s = max(abs(lag_range_ms[0]), abs(lag_range_ms[1])) / 1000.0
    core_mask = (t >= t[0] + max_abs_lag_s) & (t <= t[-1] - max_abs_lag_s)
    if core_mask.sum() < 20:
        core_mask = np.ones(len(t), dtype=bool)

    X_core = X[core_mask]
    n_core = len(X_core)
    split = n_core // 2

    def _cv_rmse(F: np.ndarray, Y: np.ndarray) -> float:
        errs = []
        for tr_slice, te_slice in ((slice(0, split), slice(split, n_core)), (slice(split, n_core), slice(0, split))):
            X_tr, Y_tr = F[tr_slice], Y[tr_slice]
            X_te, Y_te = F[te_slice], Y[te_slice]
            mu = X_tr.mean(axis=0)
            sd = np.maximum(X_tr.std(axis=0), 1e-6)
            Z_tr = np.column_stack([(X_tr - mu) / sd, np.ones(len(X_tr))])
            Z_te = np.column_stack([(X_te - mu) / sd, np.ones(len(X_te))])
            reg = ridge_alpha * len(X_tr) * np.eye(Z_tr.shape[1])
            reg[-1, -1] = 0.0
            W = np.linalg.solve(Z_tr.T @ Z_tr + reg, Z_tr.T @ Y_tr)
            pred = Z_te @ W
            errs.append(np.sum((pred - Y_te) ** 2, axis=1))
        return float(np.sqrt(np.mean(np.concatenate(errs))))

    for i, lag in enumerate(lags):
        Y_shifted, _ = shift_targets_by_lag(t, tgt, float(lag))
        rmses[i] = _cv_rmse(X_core, Y_shifted[core_mask])

    best_idx = int(np.argmin(rmses))
    return {
        "best_lag_ms": float(lags[best_idx]),
        "min_rmse_px": float(rmses[best_idx]),
        "zero_lag_rmse_px": float(rmses[int(np.argmin(np.abs(lags)))]),
        "lags_ms": lags.tolist(),
        "rmses_px": rmses.tolist(),
    }
