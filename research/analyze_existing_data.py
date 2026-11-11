"""Analyse expérimentale approfondie des données réelles existantes et profilage CPU.

Exploite :
1. Le fichier réel ``data/calibration_samples_net.npz`` (405 échantillons, 9 cibles,
   écran 2560x1440 27" à 60 cm, lunettes, webcam Logitech C920 / MediaPipe + MobileOne-S0).
2. Le profilage temps réel sur le CPU local de chaque brique du pipeline (MediaPipe,
   normalisation Zhang 2018, MobileOne-S0 ONNX, ratios d'iris, extracteur 96x48,
   DualEyePersonalNet PyTorch & ONNX, DifferentialEyeNet).

Génère :
- ``data/research_results/real_baseline_report.json``
- ``docs/these/figures/fig1_real_calibration_models.png``
- ``docs/these/figures/fig2_real_alpha_and_spatial.png``
- ``docs/these/figures/fig3_cpu_latency_budget.png``
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face import FaceData, FaceDetector  # noqa: E402
from filters import OneEuroFilter  # noqa: E402
from gaze_model import IrisGazeEstimator, NetGazeEstimator, gaze_model_available  # noqa: E402
from geometry import ScreenConfig  # noqa: E402
from normalize import FaceNormalizer  # noqa: E402
from paths import data_path  # noqa: E402
from research.eye_crop import extract_binocular_sample  # noqa: E402
from research.models import DifferentialEyeNet, DualEyePersonalNet, count_parameters  # noqa: E402
from research.train import export_personal_onnx  # noqa: E402


RESULTS_DIR = Path(data_path("research_results"))
FIGURES_DIR = ROOT / "docs" / "these" / "figures"

MIN_STD = {"hit": 20.0, "deg": 3.0, "iris": 0.005, "cm": 1.0}


def build_feature_matrix(name: str, G: np.ndarray, E: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    hx, hy = G[:, 0], G[:, 1]
    head = [E["pose_yaw"], E["pose_pitch"]]
    net = [E["net_yaw"], E["net_pitch"]]
    iris = [E["iris_rx"], E["iris_ry"]]
    if name == "impact (MobileOne)":
        feats, kinds = [hx, hy], ["hit", "hit"]
    elif name == "impact + iris":
        feats, kinds = [hx, hy, *iris], ["hit", "hit", "iris", "iris"]
    elif name == "impact + tete":
        feats, kinds = [hx, hy, *head], ["hit", "hit", "deg", "deg"]
    elif name == "impact + tete + iris":
        feats, kinds = [hx, hy, *head, *iris], ["hit", "hit", "deg", "deg", "iris", "iris"]
    elif name == "impact quadratique":
        feats, kinds = [hx, hy, hx * hx / 100.0, hx * hy / 100.0, hy * hy / 100.0], ["hit"] * 5
    elif name == "reseau + tete":
        feats, kinds = [*net, *head], ["deg"] * 4
    elif name == "iris + tete (MVP)":
        rx, ry = iris
        feats, kinds = [rx, ry, rx * rx, rx * ry, ry * ry, *head], ["iris"] * 5 + ["deg", "deg"]
    elif name == "iris seul (sans tete)":
        rx, ry = iris
        feats, kinds = [rx, ry, rx * rx, rx * ry, ry * ry], ["iris"] * 5
    else:
        raise ValueError(name)
    return np.column_stack(feats), np.array([MIN_STD[k] for k in kinds], dtype=np.float64)


MODEL_ORDER = [
    "impact (MobileOne)",
    "impact + iris",
    "impact quadratique",
    "impact + tete",
    "reseau + tete",
    "impact + tete + iris",
    "iris seul (sans tete)",
    "iris + tete (MVP)",
]


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


def evaluate_loocv_real(F: np.ndarray, floor: np.ndarray, Y: np.ndarray, tid: np.ndarray,
                        targets: np.ndarray, sc: ScreenConfig, alpha: float, E: dict[str, np.ndarray]) -> dict:
    n = len(Y)
    preds = np.zeros_like(Y)
    for k in range(len(targets)):
        tr, te = tid != k, tid == k
        m = ridge_fit(F[tr], Y[tr], floor, alpha)
        preds[te] = ridge_predict(m, F[te])

    # Erreurs exactes image par image via visual_angle_deg
    frame_deg = np.zeros(n)
    yaw_deg = np.zeros(n)
    pitch_deg = np.zeros(n)
    for i in range(n):
        ori = (float(E["origin_x"][i]) / 10.0, float(E["origin_y"][i]) / 10.0, float(E["origin_z"][i]) / 10.0)
        frame_deg[i] = sc.visual_angle_deg((float(Y[i, 0]), float(Y[i, 1])), (float(preds[i, 0]), float(preds[i, 1])), ori)
        z_mm = max(abs(float(E["origin_z"][i])), 100.0)
        yaw_deg[i] = abs(math.degrees(math.atan2((preds[i, 0] - Y[i, 0]) * sc.mm_per_px_x, z_mm)))
        pitch_deg[i] = abs(math.degrees(math.atan2((preds[i, 1] - Y[i, 1]) * sc.mm_per_px_y, z_mm)))

    # Erreur en fixation (~0.5 s = moyenne des prédictions par bloc de 15 frames) et par cible
    target_mean_deg = np.zeros(len(targets))
    target_std_deg = np.zeros(len(targets))
    fix05_errs = []
    for k in range(len(targets)):
        te = tid == k
        p_k = preds[te]
        t_k = (float(targets[k, 0]), float(targets[k, 1]))
        ori_k = (
            float(np.mean(E["origin_x"][te])) / 10.0,
            float(np.mean(E["origin_y"][te])) / 10.0,
            float(np.mean(E["origin_z"][te])) / 10.0,
        )
        mean_p = p_k.mean(axis=0)
        target_mean_deg[k] = sc.visual_angle_deg(t_k, (float(mean_p[0]), float(mean_p[1])), ori_k)
        target_std_deg[k] = float(np.std(frame_deg[te]))
        # Fenêtres de 0.5 s (15 frames)
        for j in range(0, len(p_k), 15):
            chunk = p_k[j : j + 15].mean(axis=0)
            fix05_errs.append(sc.visual_angle_deg(t_k, (float(chunk[0]), float(chunk[1])), ori_k))

    # Zones (centre = cible 4, coins = {0,2,6,8}, bords = {1,3,5,7})
    center_idx = int(np.argmin(np.linalg.norm(targets - [sc.width_px / 2, sc.height_px / 2], axis=1)))
    nx = targets[:, 0] / sc.width_px
    ny = targets[:, 1] / sc.height_px
    corner_ids = [i for i in range(len(targets)) if (nx[i] < 0.25 or nx[i] > 0.75) and (ny[i] < 0.25 or ny[i] > 0.75)]
    edge_ids = [i for i in range(len(targets)) if i != center_idx and i not in corner_ids]

    return {
        "alpha": float(alpha),
        "frame_mean_deg": float(np.mean(frame_deg)),
        "frame_median_deg": float(np.median(frame_deg)),
        "frame_p95_deg": float(np.percentile(frame_deg, 95)),
        "fix05_mean_deg": float(np.mean(fix05_errs)),
        "target_bias_mean_deg": float(np.mean(target_mean_deg)),
        "yaw_mae_deg": float(np.mean(yaw_deg)),
        "pitch_mae_deg": float(np.mean(pitch_deg)),
        "center_deg": float(target_mean_deg[center_idx]),
        "edge_deg": float(np.mean(target_mean_deg[edge_ids])),
        "corner_deg": float(np.mean(target_mean_deg[corner_ids])),
        "per_target_deg": [float(v) for v in target_mean_deg],
        "per_target_jitter_deg": [float(v) for v in target_std_deg],
    }


def analyze_temporal_filtering(F: np.ndarray, floor: np.ndarray, Y: np.ndarray, tid: np.ndarray,
                               targets: np.ndarray, sc: ScreenConfig, E: dict[str, np.ndarray], alpha: float) -> dict:
    """Compare bruit brut vs median(3) vs OneEuroFilter sur la séquence temporelle réelle."""
    preds = np.zeros_like(Y)
    for k in range(len(targets)):
        tr, te = tid != k, tid == k
        m = ridge_fit(F[tr], Y[tr], floor, alpha)
        preds[te] = ridge_predict(m, F[te])

    t = E["t"]
    # Calcule le jitter intra-cible (écart-type de la dérivée temporelle / résidu autour de la moyenne cible)
    configs = {
        "brut": (1, None),
        "median_3": (3, None),
        "one_euro_1.0_0.007": (1, (1.0, 0.007)),
        "median_3_one_euro": (3, (1.0, 0.007)),
    }
    out = {}
    for cname, (med_w, oe_params) in configs.items():
        filtered = np.zeros_like(preds)
        for k in range(len(targets)):
            idx = np.where(tid == k)[0]
            oe = None if oe_params is None else OneEuroFilter(2, min_cutoff=oe_params[0], beta=oe_params[1], median_window=med_w)
            hist = []
            for ii in idx:
                pt = preds[ii].copy()
                if oe is not None:
                    pt = oe.filter(pt, float(t[ii]))
                elif med_w > 1:
                    hist.append(pt)
                    if len(hist) > med_w:
                        hist.pop(0)
                    pt = np.median(np.array(hist), axis=0)
                filtered[ii] = pt

        # Jitter = écart-type par rapport à la moyenne de chaque cible (en deg)
        jitters = []
        errs = []
        for k in range(len(targets)):
            idx = np.where(tid == k)[0]
            ori_k = (float(np.mean(E["origin_x"][idx])) / 10.0, float(np.mean(E["origin_y"][idx])) / 10.0, float(np.mean(E["origin_z"][idx])) / 10.0)
            mean_k = filtered[idx].mean(axis=0)
            for p in filtered[idx]:
                jitters.append(sc.visual_angle_deg((float(mean_k[0]), float(mean_k[1])), (float(p[0]), float(p[1])), ori_k))
                errs.append(sc.visual_angle_deg((float(targets[k, 0]), float(targets[k, 1])), (float(p[0]), float(p[1])), ori_k))
        out[cname] = {"mean_err_deg": float(np.mean(errs)), "jitter_deg": float(np.mean(jitters))}
    return out


def benchmark_pipeline_latency() -> dict:
    """Profile chaque étage du pipeline réel et des nouveaux modèles sur ce CPU."""
    frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    # Dessine un visage synthétique simple et crée un FaceData valide pour mesurer les temps exacts
    pts = np.zeros((478, 2), dtype=np.float32)
    pts[33], pts[133] = [560, 310], [600, 310]
    pts[159], pts[145] = [580, 302], [580, 318]
    pts[362], pts[263] = [680, 310], [720, 310]
    pts[386], pts[374] = [700, 302], [700, 318]
    pts[61], pts[291] = [605, 410], [675, 410]
    pts[468], pts[473] = [580, 310], [700, 310]
    fake_face = FaceData(
        detected=True,
        landmarks_px=pts,
        pose_xyz=(0.0, 0.0, 60.0),
        pose_ypr=(0.0, 0.0, 0.0),
        iris_diameter_px=20.5,
        left_iris_px=(580.0, 310.0),
        right_iris_px=(700.0, 310.0),
        eye_openness=0.30,
    )

    def _bench(fn, n_warmup=15, n_runs=120):
        for _ in range(n_warmup):
            fn()
        ts = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000.0)
        a = np.asarray(ts)
        return {"mean_ms": float(a.mean()), "std_ms": float(a.std()), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95))}

    detector = FaceDetector()
    normalizer = FaceNormalizer()
    iris_est = IrisGazeEstimator()

    # MediaPipe sur image 1280x720 (testé avec un cercle ovale visage)
    cv2.ellipse(frame, (640, 360), (90, 120), 0, 0, 360, (190, 170, 160), -1)
    t_mp = {"ts": 0.0}

    def _run_mp():
        t_mp["ts"] += 0.033
        return detector.process(frame, t_mp["ts"])

    lat = {}
    lat["mediapipe_1280x720"] = _bench(_run_mp, n_warmup=5, n_runs=60)
    lat["zhang_norm_448x448"] = _bench(lambda: normalizer.process(frame, fake_face))
    lat["iris_ratio"] = _bench(lambda: iris_est.estimate(fake_face))
    lat["eye_crop_96x48x2"] = _bench(lambda: extract_binocular_sample(frame, fake_face, compute_glare=True))

    if gaze_model_available():
        net_est = NetGazeEstimator()
        nd = normalizer.process(frame, fake_face)
        lat["mobileone_s0_onnx"] = _bench(lambda: net_est.estimate(fake_face, nd))

    # Nouveaux modèles : PyTorch et ONNX Runtime
    p_model = DualEyePersonalNet().eval()
    onnx_tmp = RESULTS_DIR / "tmp_dual_eye_bench.onnx"
    export_personal_onnx(p_model, onnx_tmp)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    ort_sess = ort.InferenceSession(str(onnx_tmp), opts, providers=["CPUExecutionProvider"])
    in_el = np.random.randn(1, 3, 48, 96).astype(np.float32)
    in_er = np.random.randn(1, 3, 48, 96).astype(np.float32)
    in_g = np.random.randn(1, 9).astype(np.float32)
    lat["dual_eye_personal_onnx"] = _bench(lambda: ort_sess.run(None, {"eye_left": in_el, "eye_right": in_er, "geom": in_g}))

    t_el = torch.from_numpy(in_el)
    t_er = torch.from_numpy(in_er)
    t_g = torch.from_numpy(in_g)
    with torch.no_grad():
        lat["dual_eye_personal_torch"] = _bench(lambda: p_model(t_el, t_er, t_g))

    d_model = DifferentialEyeNet().eval()
    d_el = torch.randn(1, 1, 48, 96)
    d_er = torch.randn(1, 1, 48, 96)
    # Latence temps réel avec ancres pré-encodées (1 seule passe encodeur + K=5 diff_head)
    with torch.no_grad():
        zl_a, zr_a = d_model.encode(torch.randn(5, 1, 48, 96), torch.randn(5, 1, 48, 96))
        g_a = torch.randn(5, 9)

        def _run_diff_cached():
            zl_q, zr_q = d_model.encode(d_el, d_er)
            zl_qb = zl_q.expand(5, -1)
            zr_qb = zr_q.expand(5, -1)
            g_qb = t_g.expand(5, -1)
            return 0.5 * (d_model._raw_pair(zl_qb, zr_qb, g_qb, zl_a, zr_a, g_a) - d_model._raw_pair(zl_a, zr_a, g_a, zl_qb, zr_qb, g_qb))

        lat["diff_eye_net_k5_torch"] = _bench(_run_diff_cached)

    if onnx_tmp.exists():
        onnx_tmp.unlink()

    return {
        "params": {
            "dual_eye_personal": count_parameters(p_model),
            "differential_eye_net": count_parameters(d_model),
        },
        "latency_ms": lat,
    }


def make_figures(report: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

    # Fig 1 : Comparaison des modèles réels (LOO CV) + décomposition yaw/pitch
    models = report["loocv_models"]
    names = list(models.keys())
    fix_err = [models[m]["fix05_mean_deg"] for m in names]
    frm_err = [models[m]["frame_mean_deg"] for m in names]
    yaw_err = [models[m]["yaw_mae_deg"] for m in names]
    pitch_err = [models[m]["pitch_mae_deg"] for m in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=180)
    y_pos = np.arange(len(names))
    h = 0.36
    ax1.barh(y_pos - h / 2, fix_err, h, label="Fixation 0,5 s (deg)", color="#1f77b4")
    ax1.barh(y_pos + h / 2, frm_err, h, label="Par frame (deg)", color="#ff7f0e", alpha=0.85)
    ax1.axvline(1.5, color="#2ca02c", linestyle="--", linewidth=1.4, label="Cible these (1,5 deg)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names)
    ax1.set_xlim(0, max(frm_err) * 1.15)
    ax1.invert_yaxis()
    ax1.set_xlabel("Erreur angulaire moyenne LOO-CV (deg)")
    ax1.set_title("A. Comparaison des modeles (donnees reelles, N=405)")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8.5)
    for i, v in enumerate(fix_err):
        ax1.text(v + 0.12, i - h / 2, f"{v:.2f}°", va="center", fontsize=8)

    ax2.barh(y_pos - h / 2, yaw_err, h, label="MAE Horizontal (Yaw)", color="#2ca02c")
    ax2.barh(y_pos + h / 2, pitch_err, h, label="MAE Vertical (Pitch)", color="#d62728", alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([""] * len(names))
    ax2.set_xlim(0, max(max(yaw_err), max(pitch_err)) * 1.12)
    ax2.invert_yaxis()
    ax2.set_xlabel("Erreur par axe (deg)")
    ax2.set_title("B. Asymetrie horizontale vs verticale")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_real_calibration_models.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 2 : Balayage alpha Ridge + carte spatiale 3x3 du meilleur modèle réel
    sweep = report["alpha_sweep"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180)
    for mname, pts in sweep.items():
        alphas = [p["alpha"] for p in pts]
        errs = [p["fix05_mean_deg"] for p in pts]
        ax1.semilogx(alphas, errs, marker="o", linewidth=1.8, label=mname)
    ax1.set_xlabel("Regularisation Ridge alpha (echelle log)")
    ax1.set_ylabel("Erreur LOO-CV fixation 0,5 s (deg)")
    ax1.set_title("A. Sensibilite a la regularisation Ridge")
    ax1.legend(fontsize=8.5)

    best_grid = np.array(models["impact (MobileOne)"]["per_target_deg"]).reshape(3, 3)
    im = ax2.imshow(best_grid, cmap="YlOrRd", vmin=0.0, vmax=max(8.5, best_grid.max()))
    for r in range(3):
        for c in range(3):
            ax2.text(c, r, f"{best_grid[r, c]:.1f}°", ha="center", va="center", color="black", fontweight="bold", fontsize=11)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["10% G", "50% C", "90% D"])
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["10% H", "50% M", "90% B"])
    ax2.set_title("B. Carte spatiale LOO-CV : impact (MobileOne-S0)")
    fig.colorbar(im, ax=ax2, label="Erreur (deg)")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_real_alpha_and_spatial.png")
    plt.close(fig)

    # Fig 3 : Budget de latence CPU
    lat = report["profiling"]["latency_ms"]
    stages = [
        ("MediaPipe 720p", lat["mediapipe_1280x720"]["mean_ms"]),
        ("Zhang Norm 448", lat["zhang_norm_448x448"]["mean_ms"]),
        ("MobileOne-S0 ONNX", lat.get("mobileone_s0_onnx", {}).get("mean_ms", 0.0)),
        ("Crop 96x48 (2 yeux)", lat["eye_crop_96x48x2"]["mean_ms"]),
        ("DualEyePersonal ONNX", lat["dual_eye_personal_onnx"]["mean_ms"]),
        ("DualEyePersonal Torch", lat["dual_eye_personal_torch"]["mean_ms"]),
        ("DiffEyeNet K=5 Torch", lat["diff_eye_net_k5_torch"]["mean_ms"]),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=180)
    lbls = [s[0] for s in stages]
    vals = [s[1] for s in stages]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#64b5cd", "#ccb974", "#dd8452"]
    bars = ax.barh(np.arange(len(lbls)), vals, color=colors)
    ax.axvline(16.67, color="#d62728", linestyle="--", linewidth=1.3, label="Budget 60 Hz (16,7 ms)")
    ax.set_yticks(np.arange(len(lbls)))
    ax.set_yticklabels(lbls)
    ax.invert_yaxis()
    ax.set_xlabel("Latence moyenne CPU mono-frame (ms)")
    ax.set_title("Budget de latence CPU mesure sur le poste cible")
    for b, v in zip(bars, vals):
        ax.text(v + 0.15, b.get_y() + b.get_height() / 2, f"{v:.2f} ms", va="center", fontsize=9)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_cpu_latency_budget.png")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(data_path("calibration_samples_net.npz"))
    d = np.load(path)
    G, H, Y, X = d["G"], d["H"], d["Y"], d["extra"]
    cols = [str(c) for c in d["extra_cols"]]
    E = {c: X[:, i] for i, c in enumerate(cols)}
    sw, sh, diag, dist = d["screen"]
    sc = ScreenConfig(width_px=int(sw), height_px=int(sh), diagonal_inches=float(diag), distance_cm=float(dist))

    targets = np.unique(Y, axis=0)
    tid = np.array([np.where(np.all(targets == y, axis=1))[0][0] for y in Y])

    loocv_models = {}
    for mname in MODEL_ORDER:
        F, floor = build_feature_matrix(mname, G, E)
        alpha = 1e-3 if "iris" in mname and "impact" not in mname else 1e-2
        loocv_models[mname] = evaluate_loocv_real(F, floor, Y, tid, targets, sc, alpha, E)

    alpha_grid = [1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.3, 1.0]
    alpha_sweep = {}
    for mname in ["impact (MobileOne)", "impact + iris", "impact + tete", "iris + tete (MVP)"]:
        F, floor = build_feature_matrix(mname, G, E)
        alpha_sweep[mname] = [
            {"alpha": a, "fix05_mean_deg": evaluate_loocv_real(F, floor, Y, tid, targets, sc, a, E)["fix05_mean_deg"]}
            for a in alpha_grid
        ]

    F_best, floor_best = build_feature_matrix("impact (MobileOne)", G, E)
    temporal_filtering = analyze_temporal_filtering(F_best, floor_best, Y, tid, targets, sc, E, alpha=1e-2)

    profiling = benchmark_pipeline_latency()

    report = {
        "dataset_summary": {
            "file": str(path),
            "n_samples": int(len(Y)),
            "n_targets": int(len(targets)),
            "duration_s": float(E["t"][-1] - E["t"][0]),
            "screen": {"width_px": int(sw), "height_px": int(sh), "diag_in": float(diag), "distance_cm": float(dist)},
            "head_yaw_ptp_deg": float(np.ptp(E["pose_yaw"])),
            "head_pitch_ptp_deg": float(np.ptp(E["pose_pitch"])),
            "head_roll_ptp_deg": float(np.ptp(E["pose_roll"])),
            "median_iris_px": float(np.median(E["iris_px"])),
            "mean_glare": float(np.mean(E["glare"])),
            "max_glare": float(np.max(E["glare"])),
        },
        "loocv_models": loocv_models,
        "alpha_sweep": alpha_sweep,
        "temporal_filtering": temporal_filtering,
        "profiling": profiling,
    }

    out_json = RESULTS_DIR / "real_baseline_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    make_figures(report)
    print(f"[analyze_existing_data] Rapport généré : {out_json}")
    print(json.dumps(
        {k: {"fix05": round(v["fix05_mean_deg"], 2), "frame": round(v["frame_mean_deg"], 2),
             "yaw": round(v["yaw_mae_deg"], 2), "pitch": round(v["pitch_mae_deg"], 2)}
         for k, v in loocv_models.items()},
        indent=2,
        ensure_ascii=False,
    ))
    print("[profiling]", json.dumps(profiling, indent=2))


if __name__ == "__main__":
    main()
