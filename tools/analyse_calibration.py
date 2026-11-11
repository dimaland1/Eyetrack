"""Analyse hors ligne d'une calibration : quel modèle prédit le mieux le regard ?

Lit data/calibration_samples_<mode>.npz (enregistré à chaque calibration) et compare
plusieurs combinaisons de features par validation croisée « une cible à la fois » :
le modèle est appris sur 8 cibles et évalué sur la 9e, qu'il n'a jamais vue. C'est
l'erreur à laquelle s'attendre en jeu, contrairement au RMSE d'entraînement.

Usage : python tools/analyse_calibration.py [--mode net] [--file chemin.npz]
"""

from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration import AffineGazeModel, RIDGE_ALPHA  # noqa: E402
from geometry import ScreenConfig  # noqa: E402
from paths import data_path  # noqa: E402

COLS = ["t", "iris_rx", "iris_ry", "pose_x", "pose_y", "pose_z", "pose_yaw", "pose_pitch", "pose_roll",
        "origin_x", "origin_y", "origin_z", "net_yaw", "net_pitch", "iris_px"]

# Écart-type minimal par type de feature (évite d'amplifier ce qui n'a pas varié)
MIN_STD = {"hit": 20.0, "deg": 3.0, "iris": 0.005, "cm": 1.0}


def build(name: str, G: np.ndarray, E: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Features brutes (N, k) et écarts-types minimaux (k,) pour un modèle donné."""
    hx, hy = G[:, 0], G[:, 1]
    head = [E["pose_yaw"], E["pose_pitch"]]
    net = [E["net_yaw"], E["net_pitch"]]
    iris = [E["iris_rx"], E["iris_ry"]]
    feats: list[np.ndarray]
    kinds: list[str]
    if name == "impact seul":
        feats, kinds = [hx, hy], ["hit", "hit"]
    elif name == "impact + tête":
        feats, kinds = [hx, hy, *head], ["hit", "hit", "deg", "deg"]
    elif name == "impact quadratique":
        feats, kinds = [hx, hy, hx * hx / 100, hx * hy / 100, hy * hy / 100], ["hit"] * 5
    elif name == "réseau + tête":
        feats, kinds = [*net, *head], ["deg"] * 4
    elif name == "impact + iris (brut)":
        feats, kinds = [hx, hy, *iris], ["hit", "hit", "iris", "iris"]
    elif name == "impact + tête + iris":
        feats, kinds = [hx, hy, *head, *iris], ["hit", "hit", "deg", "deg", "iris", "iris"]
    elif name == "iris + tête (mode iris)":
        rx, ry = iris
        feats, kinds = [rx, ry, rx * rx, rx * ry, ry * ry, *head], ["iris"] * 5 + ["deg", "deg"]
    else:
        raise ValueError(name)
    return np.column_stack(feats), np.array([MIN_STD[k] if k != "iris" else MIN_STD["iris"] for k in kinds])


MODELS = ["modèle actuel (VOR+centr.)", "impact seul", "impact + tête", "impact quadratique",
          "réseau + tête", "impact + iris (brut)", "impact + tête + iris", "iris + tête (mode iris)"]


def ridge_fit(X: np.ndarray, Y: np.ndarray, floor: np.ndarray, alpha: float):
    mu = X.mean(axis=0)
    sd = np.maximum(X.std(axis=0), floor)
    Z = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    reg = alpha * len(X) * np.eye(Z.shape[1])
    reg[-1, -1] = 0.0
    W = np.linalg.solve(Z.T @ Z + reg, Z.T @ Y)
    return mu, sd, W


def ridge_predict(model, X: np.ndarray) -> np.ndarray:
    mu, sd, W = model
    return np.column_stack([(X - mu) / sd, np.ones(len(X))]) @ W


def px_to_deg(err_px: np.ndarray, sc: ScreenConfig) -> np.ndarray:
    mm = err_px * (sc.mm_per_px_x + sc.mm_per_px_y) / 2.0
    return np.degrees(np.arctan(mm / (sc.distance_cm * 10.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="net")
    ap.add_argument("--file", default=None)
    ap.add_argument("--alpha", type=float, default=1e-3)
    args = ap.parse_args()

    path = args.file or data_path(f"calibration_samples_{args.mode}.npz")
    d = np.load(path)
    G, H, Y, X = d["G"], d["H"], d["Y"], d["extra"]
    cols = [str(c) for c in d["extra_cols"]] if "extra_cols" in d.files else COLS
    E = {c: X[:, i] for i, c in enumerate(cols)}
    sw, sh, diag, dist = d["screen"]
    sc = ScreenConfig(width_px=int(sw), height_px=int(sh), diagonal_inches=float(diag), distance_cm=float(dist))

    targets = np.unique(Y, axis=0)
    tid = np.array([np.where(np.all(targets == y, axis=1))[0][0] for y in Y])
    print(f"Fichier : {path}")
    print(f"{len(Y)} échantillons, {len(targets)} cibles, durée {E['t'][-1] - E['t'][0]:.0f} s | "
          f"tête : yaw {np.ptp(E['pose_yaw']):.1f}°, pitch {np.ptp(E['pose_pitch']):.1f}° d'amplitude | "
          f"iris {np.median(E['iris_px']):.1f} px")

    print("\nErreur sur une cible jamais vue (validation croisée, 1 cible à la fois) :")
    print(f"{'modèle':26s} {'par image':>10s} {'moyenne 0,5 s':>14s} {'biais/cible':>12s} {'centre':>8s}")
    results = {}
    center = int(np.argmin(np.linalg.norm(targets - [sw / 2, sh / 2], axis=1)))
    for name in MODELS:
        err_frame = np.zeros(len(Y))
        err_smooth = np.zeros(len(Y))
        bias = np.zeros(len(targets))
        if name.startswith("modèle actuel"):
            for k in range(len(targets)):
                tr, te = tid != k, tid == k
                m = AffineGazeModel(mode="net", alpha=RIDGE_ALPHA["net"])
                m.fit(G[tr], H[tr], Y[tr])
                P = m._design(G[te], H[te]) @ m.weights
                err_frame[te] = np.linalg.norm(P - Y[te], axis=1)
                w = 15
                Ps = np.array([P[max(0, i - w + 1): i + 1].mean(axis=0) for i in range(len(P))])
                err_smooth[te] = np.linalg.norm(Ps - Y[te], axis=1)
                bias[k] = np.linalg.norm(P.mean(axis=0) - Y[te][0])
        else:
            F, floor = build(name, G, E)
            if not np.all(np.isfinite(F)):
                continue
            for k in range(len(targets)):
                tr, te = tid != k, tid == k
                m = ridge_fit(F[tr], Y[tr], floor, args.alpha)
                P = ridge_predict(m, F[te])
                err_frame[te] = np.linalg.norm(P - Y[te], axis=1)
                # Moyenne glissante ~0,5 s (≈ effet du filtre en fixation)
                w = 15
                Ps = np.array([P[max(0, i - w + 1): i + 1].mean(axis=0) for i in range(len(P))])
                err_smooth[te] = np.linalg.norm(Ps - Y[te], axis=1)
                bias[k] = np.linalg.norm(P.mean(axis=0) - Y[te][0])
        r = (np.mean(px_to_deg(err_frame, sc)), np.mean(px_to_deg(err_smooth, sc)),
             np.mean(px_to_deg(bias, sc)), float(px_to_deg(np.array([bias[center]]), sc)[0]))
        results[name] = r
        print(f"{name:26s} {r[0]:8.1f}°  {r[1]:11.1f}°  {r[2]:10.1f}°  {r[3]:6.1f}°")

    best = min(results, key=lambda n: results[n][1])
    print(f"\nMeilleur modèle : « {best} » -> {results[best][1]:.1f}° en fixation "
          f"(≈ {math.tan(math.radians(results[best][1])) * sc.distance_cm:.1f} cm à l'écran)")


if __name__ == "__main__":
    main()
