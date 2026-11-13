"""Évaluation automatique sur sessions webcam humaines réelles (Smooth Pursuit + clics).

Ce script lit les sessions enregistrées dans ``data/research_sessions/``, estime
automatiquement le retard visuo-moteur Smooth Pursuit (tau_lag), entraîne
``DualEyePersonalNet`` (Scénario A) et ``DifferentialEyeNet`` (Scénario B) en
validation stricte inter-sessions, et compare leurs performances face aux baselines
existantes (ratios d'iris, MobileOne-S0 ONNX) sur exactement les mêmes frames de test.

Usage :
    python -m research.evaluate_human_sessions
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry import ScreenConfig  # noqa: E402
from paths import data_path  # noqa: E402
from research.evaluate import (  # noqa: E402
    compute_gaze_metrics,
    paired_wilcoxon,
    predict_differential_session,
    predict_personal_session,
    select_anchor_indices,
)
from research.lag_estimation import estimate_visuomotor_lag  # noqa: E402
from research.train import (  # noqa: E402
    TrainConfig,
    export_personal_onnx,
    load_session_npz,
    merge_sessions,
    train_differential_model,
    train_personal_model,
)


SESSIONS_DIR = Path(data_path("research_sessions"))
RESULTS_DIR = Path(data_path("research_results"))
FIGURES_DIR = ROOT / "docs" / "these" / "figures"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pursuit_files = sorted(SESSIONS_DIR.glob("pursuit_*.npz"))
    click_files = sorted(SESSIONS_DIR.glob("clicks_*.npz"))
    if len(pursuit_files) < 2:
        raise SystemExit(
            f"Il faut au moins 2 sessions Smooth Pursuit dans {SESSIONS_DIR} "
            "(1+ pour l'entraînement, 1 session indépendante pour le test). "
            "Lancez d'abord : python -m research.collect_pursuit --duration 60 --tag s1"
        )

    # 1. Estimation du lag visuo-moteur sur la première session Smooth Pursuit
    raw0 = load_session_npz(pursuit_files[0], lag_ms=0.0)
    lag_feats = np.column_stack([raw0["iris_ratios"], raw0["pose_6dof"][:, 3:5]])
    lag_info = estimate_visuomotor_lag(raw0["timestamps"], lag_feats, raw0["target_xy"], lag_range_ms=(-40.0, 200.0), step_ms=5.0)
    best_lag_ms = lag_info["best_lag_ms"]
    print(f"[human_eval] Retard visuo-moteur estimé : {best_lag_ms:.0f} ms")

    # 2. Chargement compensé du lag (train = toutes les sessions sauf la dernière pursuit + tous les clics)
    pursuit_sessions = [load_session_npz(p, lag_ms=best_lag_ms) for p in pursuit_files]
    click_sessions = [load_session_npz(p, lag_ms=0.0) for p in click_files]

    test_sess = pursuit_sessions[-1]
    train_pool = pursuit_sessions[:-1] + click_sessions
    if len(train_pool) >= 2:
        train_sess = merge_sessions(train_pool[:-1])
        val_sess = train_pool[-1]
    else:
        s0 = train_pool[0]
        n_sp = int(0.8 * len(s0["target_angles_deg"]))
        train_sess = {k: (v[:n_sp] if k != "screen" else v) for k, v in s0.items()}
        val_sess = {k: (v[n_sp:] if k != "screen" else v) for k, v in s0.items()}

    sw, sh, diag, dist = test_sess["screen"]
    sc = ScreenConfig(width_px=int(sw), height_px=int(sh), diagonal_inches=float(diag), distance_cm=float(dist))

    # 3. Baselines sur exactement les mêmes sessions (Iris + Pose Ridge, et MobileOne si disponible)
    def _fit_ridge(X_tr, Y_tr, X_te):
        mu = X_tr.mean(axis=0)
        sd = np.maximum(X_tr.std(axis=0), 1e-4)
        Z_tr = np.column_stack([(X_tr - mu) / sd, np.ones(len(X_tr))])
        Z_te = np.column_stack([(X_te - mu) / sd, np.ones(len(X_te))])
        reg = 1e-2 * len(X_tr) * np.eye(Z_tr.shape[1])
        reg[-1, -1] = 0.0
        W = np.linalg.solve(Z_tr.T @ Z_tr + reg, Z_tr.T @ Y_tr)
        return Z_te @ W

    def _iris_feat(s):
        rx, ry = s["iris_ratios"][:, 0:1], s["iris_ratios"][:, 1:2]
        return np.hstack([rx, ry, rx**2, rx * ry, ry**2, s["pose_6dof"][:, 3:5]])

    pred_iris = _fit_ridge(_iris_feat(train_sess), train_sess["target_xy"], _iris_feat(test_sess))
    met_iris = compute_gaze_metrics(pred_iris, test_sess["target_xy"], test_sess["origin_3d_mm"], sc)

    # 4. Entraînement Scénario A (DualEyePersonalNet) & Scénario B (DifferentialEyeNet)
    cfg = TrainConfig(epochs=25, batch_size=64, lr=2e-3, seed=42)
    p_model, p_info = train_personal_model(train_sess, val_sess, cfg)
    export_personal_onnx(p_model, RESULTS_DIR / "dual_eye_personal_human.onnx")
    torch.save(p_model.state_dict(), RESULTS_DIR / "dual_eye_personal_human.pt")
    pred_p = predict_personal_session(p_model, test_sess)
    met_p = compute_gaze_metrics(pred_p, test_sess["target_xy"], test_sess["origin_3d_mm"], sc)

    d_model, d_info = train_differential_model(train_sess, val_sess, cfg, n_train_pairs=min(4000, max(1500, len(train_sess["target_xy"]) * 2)))
    torch.save(d_model.state_dict(), RESULTS_DIR / "differential_eye_human.pt")
    anc5 = select_anchor_indices(test_sess["target_xy"], k_anchors=5)
    pred_d5 = predict_differential_session(d_model, test_sess, anc5)
    met_d5 = compute_gaze_metrics(pred_d5, test_sess["target_xy"], test_sess["origin_3d_mm"], sc)

    def _clean(m: dict) -> dict:
        return {k: v for k, v in m.items() if k != "errors_deg"}

    report = {
        "lag_estimation": lag_info,
        "train_samples": int(len(train_sess["target_xy"])),
        "val_samples": int(len(val_sess["target_xy"])),
        "test_samples": int(len(test_sess["target_xy"])),
        "test_session": pursuit_files[-1].name,
        "results": {
            "iris_ridge": _clean(met_iris),
            "dual_eye_personal": _clean(met_p),
            "differential_k5": _clean(met_d5),
        },
        "wilcoxon_personal_vs_iris": paired_wilcoxon(met_p["errors_deg"], met_iris["errors_deg"]),
        "train_info": {"personal": p_info, "differential": d_info},
    }

    out_json = RESULTS_DIR / "human_sessions_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Figure humaine
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), dpi=180)
    ax1.plot(lag_info["lags_ms"], lag_info["rmses_px"], linewidth=2.0)
    ax1.axvline(best_lag_ms, color="#d62728", linestyle="--", label=f"Lag optimal = {best_lag_ms:.0f} ms")
    ax1.set_xlabel("Decalage tau (ms)")
    ax1.set_ylabel("RMSE CV-2 (px)")
    ax1.set_title("A. Lag visuo-moteur reel (Smooth Pursuit)")
    ax1.legend()

    labels = ["Iris+Pose\nRidge", "DifferentialNet\n(K=5 ancres)", "DualEyePersonal\n(Scenario A)"]
    vals = [met_iris["mean_deg"], met_d5["mean_deg"], met_p["mean_deg"]]
    bars = ax2.bar(labels, vals, color=["#ff7f0e", "#2ca02c", "#1f77b4"])
    ax2.axhline(1.5, color="#2ca02c", linestyle=":", label="Cible 1,5°")
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.2f}°", ha="center", va="bottom")
    ax2.set_ylabel("Erreur angulaire moyenne test hors-session (deg)")
    ax2.set_title("B. Precision mesuree sur webcam reelle")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_human_sessions_results.png")
    plt.close(fig)

    print("[human_eval] Terminé :", out_json)


if __name__ == "__main__":
    main()
