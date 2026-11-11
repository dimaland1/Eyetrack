"""Ablations contrôlées sur banc synthétique (Scénario A & Scénario B) + validation ONNX.

Mesure de façon reproductible sur CPU :
1. Scénario A : courbe d'apprentissage de ``DualEyePersonalNet`` en fonction du nombre
   d'échantillons mono-sujet (S0), comparée à la baseline géométrique ``Iris + Pose Ridge``,
   et récupération du retard visuo-moteur Smooth Pursuit (tau = 80 ms).
2. Scénario B : transfert inter-sujets depuis un unique sujet S0 vers 5 sujets inédits S1..S5
   (angle kappa, ouverture palpébrale, carnation, lunettes différents), comparant :
   - DualEyePersonalNet en zero-shot,
   - DualEyePersonalNet + correction de biais sur 5 ancres,
   - DifferentialEyeNet sans augmentation morphologique (K=1 et K=5 ancres),
   - DifferentialEyeNet avec augmentations morphologiques et photométriques (K=1 et K=5 ancres).
3. Vérification de l'équivalence numérique PyTorch <-> ONNX Runtime.

Génère :
- ``data/research_results/synthetic_ablation_report.json``
- ``docs/these/figures/fig4_scenarioA_sample_efficiency_and_lag.png``
- ``docs/these/figures/fig5_scenarioB_cross_subject_ablation.png``
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry import ScreenConfig  # noqa: E402
from paths import data_path  # noqa: E402
from research.augmentations import preprocess_eye_batch_rgb  # noqa: E402
from research.evaluate import (  # noqa: E402
    compute_gaze_metrics,
    paired_wilcoxon,
    predict_differential_session,
    predict_personal_session,
    select_anchor_indices,
)
from research.lag_estimation import (  # noqa: E402
    estimate_visuomotor_lag,
    lissajous_pursuit_target,
    shift_targets_by_lag,
)
from research.models import gaze_angles_deg_to_screen_px  # noqa: E402
from research.synthetic_bench import generate_synthetic_session  # noqa: E402
from research.train import (  # noqa: E402
    TrainConfig,
    export_personal_onnx,
    train_differential_model,
    train_personal_model,
)


RESULTS_DIR = Path(data_path("research_results"))
FIGURES_DIR = ROOT / "docs" / "these" / "figures"


def subset_session(sess: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    return {k: (v[:n] if k != "screen" else v) for k, v in sess.items()}


def fit_eval_iris_baseline(train_sess: dict[str, np.ndarray], test_sess: dict[str, np.ndarray], sc: ScreenConfig) -> dict:
    def _feat(s):
        rx = s["iris_ratios"][:, 0:1]
        ry = s["iris_ratios"][:, 1:2]
        hy = s["pose_6dof"][:, 3:4]
        hp = s["pose_6dof"][:, 4:5]
        return np.hstack([rx, ry, rx**2, rx * ry, ry**2, hy, hp])

    X_tr = _feat(train_sess)
    Y_tr = train_sess["target_xy"]
    mu = X_tr.mean(axis=0)
    sd = np.maximum(X_tr.std(axis=0), 1e-4)
    Z_tr = np.column_stack([(X_tr - mu) / sd, np.ones(len(X_tr))])
    reg = 1e-3 * len(X_tr) * np.eye(Z_tr.shape[1])
    reg[-1, -1] = 0.0
    W = np.linalg.solve(Z_tr.T @ Z_tr + reg, Z_tr.T @ Y_tr)

    X_te = _feat(test_sess)
    Z_te = np.column_stack([(X_te - mu) / sd, np.ones(len(X_te))])
    pred = Z_te @ W
    return compute_gaze_metrics(pred, test_sess["target_xy"], test_sess["origin_3d_mm"], sc)


def run_lag_recovery_experiment(sc: ScreenConfig) -> dict:
    t = np.linspace(0.0, 30.0, 900)
    x, y, spd = lissajous_pursuit_target(t, sc)
    tgt = np.column_stack([x, y])
    rng = np.random.default_rng(7)
    shifted, _ = shift_targets_by_lag(t, tgt, lag_ms=80.0)
    feats = np.column_stack([
        (shifted[:, 0] - sc.width_px / 2.0) / 1000.0 + rng.normal(0.0, 0.015, len(t)),
        (shifted[:, 1] - sc.height_px / 2.0) / 1000.0 + rng.normal(0.0, 0.015, len(t)),
    ])
    res = estimate_visuomotor_lag(t, feats, tgt, lag_range_ms=(-40.0, 200.0), step_ms=5.0)
    res["max_target_speed_deg_s"] = float(np.max(spd))
    res["mean_target_speed_deg_s"] = float(np.mean(spd))
    return res


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sc = ScreenConfig(width_px=2560, height_px=1440, diagonal_inches=27.0, distance_cm=60.0)

    # Données S0 (train / val / test par sessions distinctes)
    s0_train_full = generate_synthetic_session(subject_id=0, n_samples=1600, seed=101, head_pose_std_deg=9.0, glare_prob=0.15)
    s0_val = generate_synthetic_session(subject_id=0, n_samples=350, seed=202, head_pose_std_deg=9.0, glare_prob=0.15)
    s0_test = generate_synthetic_session(subject_id=0, n_samples=500, seed=303, head_pose_std_deg=10.0, lighting_gain=0.92, glare_prob=0.20)

    # 1. Scénario A : courbe d'apprentissage selon N
    sample_sizes = [100, 250, 500, 1000, 1600]
    iris_base = fit_eval_iris_baseline(s0_train_full, s0_test, sc)
    learning_curve = []
    best_personal_model = None

    for n_samp in sample_sizes:
        print(f"[ablation] Scénario A N={n_samp}...", flush=True)
        sub_tr = subset_session(s0_train_full, n_samp)
        cfg = TrainConfig(epochs=20 if n_samp >= 500 else 24, batch_size=64, lr=2.5e-3, seed=42)
        m, tr_info = train_personal_model(sub_tr, s0_val, cfg)
        preds = predict_personal_session(m, s0_test)
        met = compute_gaze_metrics(preds, s0_test["target_xy"], s0_test["origin_3d_mm"], sc)
        learning_curve.append(
            {
                "n_samples": n_samp,
                "mean_deg": met["mean_deg"],
                "median_deg": met["median_deg"],
                "p95_deg": met["p95_deg"],
                "ci95": [met["ci95_low_deg"], met["ci95_high_deg"]],
                "yaw_mae_deg": met["yaw_mae_deg"],
                "pitch_mae_deg": met["pitch_mae_deg"],
                "train_time_s": tr_info["train_time_s"],
            }
        )
        if n_samp == sample_sizes[-1]:
            best_personal_model = m

    # Vérification export ONNX
    onnx_path = RESULTS_DIR / "dual_eye_personal_synth.onnx"
    export_personal_onnx(best_personal_model, onnx_path)
    ort_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rgb_test = preprocess_eye_batch_rgb(s0_test["eye_crops"][:16])
    geom_test = s0_test["geom_vec"][:16].astype(np.float32)
    with torch.no_grad():
        torch_out = best_personal_model(
            torch.from_numpy(rgb_test[:, 0]), torch.from_numpy(rgb_test[:, 1]), torch.from_numpy(geom_test)
        ).numpy()
    ort_out = ort_sess.run(None, {"eye_left": rgb_test[:, 0], "eye_right": rgb_test[:, 1], "geom": geom_test})[0]
    onnx_max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))

    # 2. Lag Smooth Pursuit
    lag_res = run_lag_recovery_experiment(sc)

    # 3. Scénario B : entraînement sur S0 uniquement, transfert sur S1..S5
    print("[ablation] Scénario B DiffNet sans augmentation...", flush=True)
    diff_no_aug, diff_no_aug_info = train_differential_model(
        subset_session(s0_train_full, 1200),
        s0_val,
        TrainConfig(epochs=18, batch_size=64, lr=2.5e-3, seed=42, morph_strength=0.0, photo_strength=0.0),
        n_train_pairs=2400,
    )
    print("[ablation] Scénario B DiffNet + Morph/Photo Aug...", flush=True)
    diff_aug, diff_aug_info = train_differential_model(
        subset_session(s0_train_full, 1200),
        s0_val,
        TrainConfig(epochs=18, batch_size=64, lr=2.5e-3, seed=42, morph_strength=1.0, photo_strength=1.0),
        n_train_pairs=2400,
    )

    unseen_subjects = [
        generate_synthetic_session(subject_id=sid, n_samples=400, seed=500 + sid, head_pose_std_deg=9.0, glare_prob=0.15)
        for sid in range(1, 6)
    ]

    # Évaluation sur S0 (intra-sujet) et sur S1..S5 (inter-sujets)
    method_keys = [
        "PersonalNet_ZeroShot",
        "PersonalNet_Bias5pt",
        "DiffNet_NoAug_K1",
        "DiffNet_NoAug_K5",
        "DiffNet_MorphAug_K1",
        "DiffNet_MorphAug_K5",
    ]
    per_subject_scores: dict[str, list[float]] = {k: [] for k in method_keys}
    all_errors: dict[str, list[np.ndarray]] = {k: [] for k in method_keys}

    for sess_k in unseen_subjects:
        anc1 = select_anchor_indices(sess_k["target_xy"], k_anchors=1)
        anc5 = select_anchor_indices(sess_k["target_xy"], k_anchors=5)

        # Direct zero-shot
        p_direct = predict_personal_session(best_personal_model, sess_k)
        m_direct = compute_gaze_metrics(p_direct, sess_k["target_xy"], sess_k["origin_3d_mm"], sc)
        per_subject_scores["PersonalNet_ZeroShot"].append(m_direct["mean_deg"])
        all_errors["PersonalNet_ZeroShot"].append(m_direct["errors_deg"])

        # Direct + correction biais angulaire moyen sur 5 ancres (estime delta kappa)
        with torch.no_grad():
            rgb_a = preprocess_eye_batch_rgb(sess_k["eye_crops"][anc5])
            ang_a_pred = best_personal_model(
                torch.from_numpy(rgb_a[:, 0]), torch.from_numpy(rgb_a[:, 1]), torch.from_numpy(sess_k["geom_vec"][anc5])
            ).numpy()
            bias_deg = np.mean(sess_k["target_angles_deg"][anc5] - ang_a_pred, axis=0)
            rgb_all = preprocess_eye_batch_rgb(sess_k["eye_crops"])
            ang_all_pred = best_personal_model(
                torch.from_numpy(rgb_all[:, 0]), torch.from_numpy(rgb_all[:, 1]), torch.from_numpy(sess_k["geom_vec"])
            ).numpy() + bias_deg
        p_bias = gaze_angles_deg_to_screen_px(
            ang_all_pred, sess_k["origin_3d_mm"], sc.width_px, sc.height_px, sc.mm_per_px_x, sc.mm_per_px_y
        )
        m_bias = compute_gaze_metrics(p_bias, sess_k["target_xy"], sess_k["origin_3d_mm"], sc)
        per_subject_scores["PersonalNet_Bias5pt"].append(m_bias["mean_deg"])
        all_errors["PersonalNet_Bias5pt"].append(m_bias["errors_deg"])

        # Differential sans augmentation
        for k_name, anc in (("K1", anc1), ("K5", anc5)):
            p_noaug = predict_differential_session(diff_no_aug, sess_k, anc)
            m_noaug = compute_gaze_metrics(p_noaug, sess_k["target_xy"], sess_k["origin_3d_mm"], sc)
            per_subject_scores[f"DiffNet_NoAug_{k_name}"].append(m_noaug["mean_deg"])
            all_errors[f"DiffNet_NoAug_{k_name}"].append(m_noaug["errors_deg"])

            p_aug = predict_differential_session(diff_aug, sess_k, anc)
            m_aug = compute_gaze_metrics(p_aug, sess_k["target_xy"], sess_k["origin_3d_mm"], sc)
            per_subject_scores[f"DiffNet_MorphAug_{k_name}"].append(m_aug["mean_deg"])
            all_errors[f"DiffNet_MorphAug_{k_name}"].append(m_aug["errors_deg"])

    # Évaluation intra-sujet de DiffNet_MorphAug_K5 sur S0_test pour comparaison directe
    anc5_s0 = select_anchor_indices(s0_test["target_xy"], k_anchors=5)
    m_diff_s0 = compute_gaze_metrics(
        predict_differential_session(diff_aug, s0_test, anc5_s0), s0_test["target_xy"], s0_test["origin_3d_mm"], sc
    )

    cross_summary = {}
    for k in method_keys:
        vals = np.array(per_subject_scores[k])
        pooled = np.concatenate(all_errors[k])
        cross_summary[k] = {
            "subject_means_deg": [float(v) for v in vals],
            "mean_across_subjects_deg": float(vals.mean()),
            "std_across_subjects_deg": float(vals.std()),
            "pooled_median_deg": float(np.median(pooled)),
            "pooled_p95_deg": float(np.percentile(pooled, 95)),
        }

    # Test statistique apparié : DiffNet_MorphAug_K5 vs DiffNet_NoAug_K5 et vs PersonalNet_Bias5pt
    wilcoxon_aug_vs_noaug = paired_wilcoxon(
        np.concatenate(all_errors["DiffNet_MorphAug_K5"]), np.concatenate(all_errors["DiffNet_NoAug_K5"])
    )
    wilcoxon_diff_vs_bias = paired_wilcoxon(
        np.concatenate(all_errors["DiffNet_MorphAug_K5"]), np.concatenate(all_errors["PersonalNet_Bias5pt"])
    )

    report = {
        "scenario_a": {
            "iris_baseline_mean_deg": iris_base["mean_deg"],
            "learning_curve": learning_curve,
            "onnx_max_abs_diff": onnx_max_abs_diff,
            "lag_recovery": {
                "true_lag_ms": 80.0,
                "estimated_lag_ms": lag_res["best_lag_ms"],
                "min_rmse_px": lag_res["min_rmse_px"],
                "zero_lag_rmse_px": lag_res["zero_lag_rmse_px"],
                "max_speed_deg_s": lag_res["max_target_speed_deg_s"],
                "mean_speed_deg_s": lag_res["mean_target_speed_deg_s"],
            },
        },
        "scenario_b": {
            "train_info_no_aug": {"time_s": diff_no_aug_info["train_time_s"], "params": diff_no_aug_info["params"]},
            "train_info_aug": {"time_s": diff_aug_info["train_time_s"], "params": diff_aug_info["params"]},
            "s0_within_subject_diff_k5_deg": m_diff_s0["mean_deg"],
            "cross_subject_results": cross_summary,
            "wilcoxon_morph_aug_vs_no_aug_k5": wilcoxon_aug_vs_noaug,
            "wilcoxon_diff_aug_vs_personal_bias5_k5": wilcoxon_diff_vs_bias,
        },
    }

    out_json = RESULTS_DIR / "synthetic_ablation_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Figures 4 et 5
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.7), dpi=180)
    ns = [r["n_samples"] for r in learning_curve]
    means = [r["mean_deg"] for r in learning_curve]
    ci_lo = [r["mean_deg"] - r["ci95"][0] for r in learning_curve]
    ci_hi = [r["ci95"][1] - r["mean_deg"] for r in learning_curve]
    ax1.errorbar(ns, means, yerr=[ci_lo, ci_hi], marker="o", linewidth=2.0, capsize=4, label="DualEyePersonalNet (S0)")
    ax1.axhline(iris_base["mean_deg"], color="#ff7f0e", linestyle="--", label=f"Baseline Iris+Pose ({iris_base['mean_deg']:.2f}°)")
    ax1.axhline(1.5, color="#2ca02c", linestyle=":", linewidth=1.6, label="Objectif < 1,5°")
    ax1.set_xlabel("Nombre d'echantillons d'entrainement (S0)")
    ax1.set_ylabel("Erreur moyenne hors-session (deg)")
    ax1.set_title("A. Efficacite d'echantillonnage (Scenario A, banc synthetique)")
    ax1.legend(fontsize=8.5)

    ax2.plot(lag_res["lags_ms"], lag_res["rmses_px"], linewidth=2.0, color="#1f77b4")
    ax2.axvline(lag_res["best_lag_ms"], color="#d62728", linestyle="--", label=f"Lag estime = {lag_res['best_lag_ms']:.0f} ms (vrai = 80 ms)")
    ax2.set_xlabel("Decalage temporel applique tau (ms)")
    ax2.set_ylabel("RMSE en validation croisee (px)")
    ax2.set_title("B. Calibration du retard visuo-moteur Smooth Pursuit")
    ax2.legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_scenarioA_sample_efficiency_and_lag.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=180)
    labels_pretty = {
        "PersonalNet_ZeroShot": "PersonalNet\nZero-Shot",
        "PersonalNet_Bias5pt": "PersonalNet\n+ Biais 5 pts",
        "DiffNet_NoAug_K1": "DiffNet (sans aug)\nK=1 ancre",
        "DiffNet_NoAug_K5": "DiffNet (sans aug)\nK=5 ancres",
        "DiffNet_MorphAug_K1": "DiffNet (+MorphAug)\nK=1 ancre",
        "DiffNet_MorphAug_K5": "DiffNet (+MorphAug)\nK=5 ancres",
    }
    x_pos = np.arange(len(method_keys))
    m_vals = [cross_summary[k]["mean_across_subjects_deg"] for k in method_keys]
    s_vals = [cross_summary[k]["std_across_subjects_deg"] for k in method_keys]
    cols = ["#c44e52", "#dd8452", "#8172b3", "#64b5cd", "#55a868", "#2ca02c"]
    bars = ax.bar(x_pos, m_vals, yerr=s_vals, capsize=5, color=cols, alpha=0.92)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([labels_pretty[k] for k in method_keys], fontsize=9)
    ax.set_ylabel("Erreur angulaire inter-sujets S1..S5 (deg)")
    ax.set_title("Generalisation inter-sujets depuis un unique sujet S0 (banc synthetique)")
    for b, v, s in zip(bars, m_vals, s_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + s + 0.12, f"{v:.2f}°", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(v + s for v, s in zip(m_vals, s_vals)) * 1.16)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_scenarioB_cross_subject_ablation.png")
    plt.close(fig)

    print("[run_synthetic_ablations] Terminé :", out_json)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
