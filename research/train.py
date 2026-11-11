"""Entraînement CPU/GPU des modèles Scénario A (perso) et Scénario B (différentiel).

Respecte une séparation stricte par *session* (jamais de découpage aléatoire frame par frame)
afin d'éviter toute fuite temporelle entre l'entraînement et la validation.
Supporte aussi l'export ONNX et la mesure de latence CPU via ONNX Runtime.

Usage :
    python -m research.train --arch personal --sessions data/research_sessions/*.npz
    python -m research.train --arch differential --sessions data/research_sessions/*.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.augmentations import (  # noqa: E402
    augment_eye_crop,
    preprocess_eye_batch_gray_norm,
    preprocess_eye_batch_rgb,
)
from research.lag_estimation import shift_targets_by_lag  # noqa: E402
from research.models import (  # noqa: E402
    DifferentialEyeNet,
    DualEyePersonalNet,
    count_parameters,
    screen_px_to_gaze_angles_deg,
)
from geometry import ScreenConfig  # noqa: E402


def load_session_npz(path: str | Path, lag_ms: float = 0.0) -> dict[str, np.ndarray]:
    """Charge une session .npz, filtre ``valid_mask`` et applique si demandé le décalage lag_ms."""
    d = np.load(path, allow_pickle=False)
    data = {k: d[k] for k in d.files}

    if lag_ms != 0.0 and "timestamps" in data and len(data["timestamps"]) > 5:
        shifted_xy, lag_valid = shift_targets_by_lag(data["timestamps"], data["target_xy"], lag_ms)
        data["target_xy"] = shifted_xy
        sw, sh, diag, dist = data["screen"]
        sc = ScreenConfig(width_px=int(sw), height_px=int(sh), diagonal_inches=float(diag), distance_cm=float(dist))
        data["target_angles_deg"] = screen_px_to_gaze_angles_deg(
            shifted_xy,
            data["origin_3d_mm"],
            sc.width_px,
            sc.height_px,
            sc.mm_per_px_x,
            sc.mm_per_px_y,
            sc.camera_offset_y_cm,
        )
        data["valid_mask"] = data["valid_mask"].astype(bool) & lag_valid

    mask = data["valid_mask"].astype(bool)
    finite_y = np.all(np.isfinite(data["target_angles_deg"]), axis=1)
    mask = mask & finite_y

    out = {}
    for k in ("eye_crops", "geom_vec", "pose_6dof", "origin_3d_mm", "iris_ratios", "target_xy", "target_angles_deg", "timestamps"):
        if k in data:
            out[k] = data[k][mask]
    out["screen"] = data["screen"]
    return out


class PersonalEyeDataset(Dataset):
    """Dataset Scénario A : (eye_l, eye_r, geom) -> angles (yaw, pitch).

    Pré-calcule ``aug_copies`` vues augmentées en RAM (ou 1 vue brute) pour éliminer la
    surcharge OpenCV à chaque époque tout en conservant la variabilité géométrique.
    """

    def __init__(
        self,
        session: dict[str, np.ndarray],
        augment: bool = False,
        morph_strength: float = 0.25,
        photo_strength: float = 0.6,
        aug_copies: int = 3,
        seed: int = 42,
    ) -> None:
        crops = session["eye_crops"]
        geom = session["geom_vec"].astype(np.float32)
        angles = session["target_angles_deg"].astype(np.float32)
        rng = np.random.default_rng(seed)

        if augment and aug_copies > 1:
            all_crops = [crops]
            all_geom = [geom]
            all_angles = [angles]
            for _ in range(aug_copies - 1):
                aug_c = np.empty_like(crops)
                for i in range(len(crops)):
                    aug_c[i, 0] = augment_eye_crop(crops[i, 0], rng, morph_strength, photo_strength, add_glare_prob=0.15)
                    aug_c[i, 1] = augment_eye_crop(crops[i, 1], rng, morph_strength, photo_strength, add_glare_prob=0.15)
                all_crops.append(aug_c)
                all_geom.append(geom)
                all_angles.append(angles)
            crops_full = np.concatenate(all_crops, axis=0)
            self.geom = np.concatenate(all_geom, axis=0)
            self.angles = np.concatenate(all_angles, axis=0)
        else:
            crops_full = crops
            self.geom = geom
            self.angles = angles

        self.rgb = torch.from_numpy(preprocess_eye_batch_rgb(crops_full))
        self.geom_t = torch.from_numpy(self.geom)
        self.angles_t = torch.from_numpy(self.angles)

    def __len__(self) -> int:
        return len(self.angles_t)

    def __getitem__(self, idx: int):
        return self.rgb[idx, 0], self.rgb[idx, 1], self.geom_t[idx], self.angles_t[idx]


class DifferentialPairDataset(Dataset):
    """Dataset Scénario B : paires (requête, référence) -> delta (yaw, pitch).

    Si ``morph_strength > 0``, pré-génère ``aug_copies`` versions morphologiquement et
    photométriquement perturbées des patchs avant l'échantillonnage des paires, afin de
    briser la mémorisation d'identité avec un coût CPU nul pendant les époques.
    """

    def __init__(
        self,
        session: dict[str, np.ndarray],
        n_pairs: int = 2400,
        augment: bool = True,
        morph_strength: float = 1.0,
        photo_strength: float = 1.0,
        aug_copies: int = 4,
        seed: int = 42,
    ) -> None:
        crops = session["eye_crops"]
        geom = session["geom_vec"].astype(np.float32)
        angles = session["target_angles_deg"].astype(np.float32)
        rng = np.random.default_rng(seed)

        if augment and (morph_strength > 0.0 or photo_strength > 0.0):
            views = []
            for _ in range(max(1, aug_copies)):
                aug_c = np.empty_like(crops)
                for i in range(len(crops)):
                    aug_c[i, 0] = augment_eye_crop(crops[i, 0], rng, morph_strength, photo_strength, add_glare_prob=0.20)
                    aug_c[i, 1] = augment_eye_crop(crops[i, 1], rng, morph_strength, photo_strength, add_glare_prob=0.20)
                views.append(preprocess_eye_batch_gray_norm(aug_c))
            # (V, N, 2, 1, H, W) -> (V*N, 2, 1, H, W)
            gray_bank = np.concatenate(views, axis=0)
            geom_bank = np.tile(geom, (len(views), 1))
            angles_bank = np.tile(angles, (len(views), 1))
        else:
            gray_bank = preprocess_eye_batch_gray_norm(crops)
            geom_bank = geom
            angles_bank = angles

        self.gray_t = torch.from_numpy(gray_bank)
        self.geom_t = torch.from_numpy(geom_bank)
        self.angles_t = torch.from_numpy(angles_bank)
        n_bank = len(self.angles_t)
        self.idx_q = rng.integers(0, n_bank, size=n_pairs)
        self.idx_r = rng.integers(0, n_bank, size=n_pairs)
        self.n_pairs = n_pairs

    def __len__(self) -> int:
        return self.n_pairs

    def __getitem__(self, idx: int):
        iq = int(self.idx_q[idx])
        ir = int(self.idx_r[idx])
        return (
            self.gray_t[iq, 0],
            self.gray_t[iq, 1],
            self.geom_t[iq],
            self.gray_t[ir, 0],
            self.gray_t[ir, 1],
            self.geom_t[ir],
            self.angles_t[iq] - self.angles_t[ir],
        )


def merge_sessions(sessions: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatène plusieurs sessions compatibles dans un unique dictionnaire."""
    merged = {}
    keys = [k for k in sessions[0].keys() if k != "screen"]
    for k in keys:
        merged[k] = np.concatenate([s[k] for s in sessions], axis=0)
    merged["screen"] = sessions[0]["screen"]
    return merged


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 42
    morph_strength: float = 1.0
    photo_strength: float = 1.0


def train_personal_model(
    train_session: dict[str, np.ndarray],
    val_session: dict[str, np.ndarray] | None = None,
    cfg: TrainConfig | None = None,
) -> tuple[DualEyePersonalNet, dict]:
    """Entraîne ``DualEyePersonalNet`` (Scénario A) sur CPU/GPU."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PersonalEyeDataset(train_session, augment=True, morph_strength=0.15, photo_strength=0.45, seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    val_loader = None
    if val_session is not None and len(val_session["angles" if "angles" in val_session else "target_angles_deg"]) > 0:
        val_ds = PersonalEyeDataset(val_session, augment=False, seed=cfg.seed + 1)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = DualEyePersonalNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    best_val = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    history = {"train_loss": [], "val_mae_deg": []}

    t_start = time.perf_counter()
    for _ in range(cfg.epochs):
        model.train()
        tot_loss, n_seen = 0.0, 0
        for el, er, g, y in train_loader:
            el, er, g, y = el.to(device), er.to(device), g.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(el, er, g)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            tot_loss += float(loss.item()) * len(y)
            n_seen += len(y)
        sched.step()
        history["train_loss"].append(tot_loss / max(n_seen, 1))

        if val_loader is not None:
            model.eval()
            errs = []
            with torch.no_grad():
                for el, er, g, y in val_loader:
                    pred = model(el.to(device), er.to(device), g.to(device)).cpu().numpy()
                    ang_err = np.linalg.norm(pred - y.numpy(), axis=1)
                    errs.append(ang_err)
            val_mae = float(np.mean(np.concatenate(errs)))
            history["val_mae_deg"].append(val_mae)
            if val_mae < best_val:
                best_val = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.cpu().eval()
    return model, {
        "train_time_s": float(time.perf_counter() - t_start),
        "params": count_parameters(model),
        "best_val_mae_deg": best_val if val_loader is not None else float("nan"),
        "history": history,
    }


def train_differential_model(
    train_session: dict[str, np.ndarray],
    val_session: dict[str, np.ndarray] | None = None,
    cfg: TrainConfig | None = None,
    n_train_pairs: int = 4000,
) -> tuple[DifferentialEyeNet, dict]:
    """Entraîne ``DifferentialEyeNet`` (Scénario B) par paires différentielles."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DifferentialPairDataset(
        train_session,
        n_pairs=n_train_pairs,
        augment=True,
        morph_strength=cfg.morph_strength,
        photo_strength=cfg.photo_strength,
        seed=cfg.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    val_loader = None
    if val_session is not None and len(val_session["target_angles_deg"]) > 5:
        val_ds = DifferentialPairDataset(val_session, n_pairs=800, augment=False, seed=cfg.seed + 1)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = DifferentialEyeNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    best_val = float("inf")
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    history = {"train_loss": [], "val_mae_deg": []}

    t_start = time.perf_counter()
    for _ in range(cfg.epochs):
        model.train()
        tot_loss, n_seen = 0.0, 0
        for elq, erq, gq, elr, err, gr, dy in train_loader:
            elq, erq, gq = elq.to(device), erq.to(device), gq.to(device)
            elr, err, gr, dy = elr.to(device), err.to(device), gr.to(device), dy.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(elq, erq, gq, elr, err, gr)
            loss = loss_fn(pred, dy)
            loss.backward()
            opt.step()
            tot_loss += float(loss.item()) * len(dy)
            n_seen += len(dy)
        sched.step()
        history["train_loss"].append(tot_loss / max(n_seen, 1))

        if val_loader is not None:
            model.eval()
            errs = []
            with torch.no_grad():
                for elq, erq, gq, elr, err, gr, dy in val_loader:
                    pred = model(
                        elq.to(device), erq.to(device), gq.to(device),
                        elr.to(device), err.to(device), gr.to(device),
                    ).cpu().numpy()
                    errs.append(np.linalg.norm(pred - dy.numpy(), axis=1))
            val_mae = float(np.mean(np.concatenate(errs)))
            history["val_mae_deg"].append(val_mae)
            if val_mae < best_val:
                best_val = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.cpu().eval()
    return model, {
        "train_time_s": float(time.perf_counter() - t_start),
        "params": count_parameters(model),
        "best_val_mae_deg": best_val if val_loader is not None else float("nan"),
        "history": history,
    }


def export_personal_onnx(model: DualEyePersonalNet, out_path: str | Path) -> Path:
    """Exporte ``DualEyePersonalNet`` vers ONNX (batch dynamique ou 1)."""
    import inspect

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    el = torch.randn(1, 3, 48, 96)
    er = torch.randn(1, 3, 48, 96)
    geom = torch.randn(1, 9)
    kwargs = dict(
        input_names=["eye_left", "eye_right", "geom"],
        output_names=["gaze_yaw_pitch_deg"],
        dynamic_axes={"eye_left": {0: "batch"}, "eye_right": {0: "batch"}, "geom": {0: "batch"}},
        opset_version=17,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(model, (el, er, geom), str(out), **kwargs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Entraînement Eyetrack Research")
    ap.add_argument("--arch", choices=["personal", "differential"], required=True)
    ap.add_argument("--sessions", nargs="+", required=True, help="Fichiers .npz (ou motif glob)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lag-ms", type=float, default=0.0)
    ap.add_argument("--out", type=str, default="data/research_results/model.pt")
    ap.add_argument("--export-onnx", type=str, default=None)
    args = ap.parse_args()

    paths = []
    for p in args.sessions:
        paths.extend(sorted(glob.glob(p)))
    if len(paths) < 1:
        raise SystemExit("Aucune session trouvée.")

    loaded = [load_session_npz(p, lag_ms=args.lag_ms) for p in paths]
    if len(loaded) >= 2:
        train_s = merge_sessions(loaded[:-1])
        val_s = loaded[-1]
    else:
        # Si une seule session est fournie, split temporel contigu 80/20 (pas aléatoire)
        s0 = loaded[0]
        n_split = int(0.8 * len(s0["target_angles_deg"]))
        train_s = {k: (v[:n_split] if k != "screen" else v) for k, v in s0.items()}
        val_s = {k: (v[n_split:] if k != "screen" else v) for k, v in s0.items()}

    cfg = TrainConfig(epochs=args.epochs)
    if args.arch == "personal":
        model, info = train_personal_model(train_s, val_s, cfg)
        if args.export_onnx:
            export_personal_onnx(model, args.export_onnx)
    else:
        model, info = train_differential_model(train_s, val_s, cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    (out_path.with_suffix(".json")).write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[train] {args.arch} terminé en {info['train_time_s']:.1f}s | params={info['params']} | val_mae={info['best_val_mae_deg']:.2f}°")


if __name__ == "__main__":
    main()
