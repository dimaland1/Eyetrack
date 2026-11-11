"""Banc synthétique paramétrique binoculaire pour ablations contrôlées (Scénarios A & B).

Permet de tester de manière strictement reproductible :
1. Scénario A (mono-sujet) : précision en fonction du nombre de sessions et d'échantillons,
   avec pose de tête variable et reflets.
2. Scénario B (généralisation inter-sujets depuis un seul sujet) : entraînement sur un
   unique sujet S0 (avec ou sans augmentations morphologiques) et évaluation sur des sujets
   inédits S1..Sk dotés d'angles kappa différents, d'ouvertures palpébrales distinctes,
   de carnations variées et de bruits capteur réalistes.

Important : ce banc ne remplace pas les mesures webcam réelles ; il sert à isoler
mathématiquement l'effet de chaque choix d'architecture et d'augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import cv2
import numpy as np

from geometry import ScreenConfig
from research.models import screen_px_to_gaze_angles_deg


@dataclass
class SubjectProfile:
    """Paramètres anatomiques et photométriques d'un sujet synthétique."""

    subject_id: int
    kappa_yaw_deg: float
    kappa_pitch_deg: float
    skin_bgr: tuple[int, int, int]
    iris_bgr: tuple[int, int, int]
    eyelid_open_px: float
    iris_radius_px: float
    sclera_level: float
    glasses: bool
    noise_std: float


def sample_subject_profile(subject_id: int, rng: np.random.Generator) -> SubjectProfile:
    """Génère un profil anatomique réaliste et reproductible."""
    if subject_id == 0:
        # Sujet de référence (S0)
        return SubjectProfile(
            subject_id=0,
            kappa_yaw_deg=1.8,
            kappa_pitch_deg=-1.2,
            skin_bgr=(140, 165, 205),
            iris_bgr=(45, 55, 75),
            eyelid_open_px=15.0,
            iris_radius_px=13.0,
            sclera_level=222.0,
            glasses=True,
            noise_std=4.5,
        )
    # Sujets inédits avec variabilité marquée d'angle kappa, carnation, ouverture et iris
    skin_base = rng.uniform(0.55, 1.15)
    skin_bgr = (
        int(np.clip(125 * skin_base + rng.uniform(-18, 18), 45, 235)),
        int(np.clip(152 * skin_base + rng.uniform(-18, 18), 55, 240)),
        int(np.clip(190 * skin_base + rng.uniform(-18, 18), 70, 248)),
    )
    iris_palette = [(42, 48, 60), (95, 70, 45), (55, 90, 70), (35, 35, 42)]
    iris_bgr = iris_palette[int(rng.integers(0, len(iris_palette)))]
    return SubjectProfile(
        subject_id=subject_id,
        kappa_yaw_deg=float(rng.uniform(-4.5, 4.5)),
        kappa_pitch_deg=float(rng.uniform(-3.5, 3.5)),
        skin_bgr=skin_bgr,
        iris_bgr=iris_bgr,
        eyelid_open_px=float(rng.uniform(11.5, 18.5)),
        iris_radius_px=float(rng.uniform(11.0, 14.5)),
        sclera_level=float(rng.uniform(195.0, 238.0)),
        glasses=bool(rng.random() < 0.5),
        noise_std=float(rng.uniform(3.0, 7.0)),
    )


def _render_single_eye_patch(
    iris_shift_x: float,
    iris_shift_y: float,
    prof: SubjectProfile,
    rng: np.random.Generator,
    lighting_gain: float,
    glare_strength: float,
    w: int = 96,
    h: int = 48,
) -> np.ndarray:
    """Rend un patch oculaire 48x96 réaliste (sclère, iris, pupille, paupières, reflet)."""
    patch = np.empty((h, w, 3), dtype=np.float32)
    patch[..., 0] = prof.skin_bgr[0]
    patch[..., 1] = prof.skin_bgr[1]
    patch[..., 2] = prof.skin_bgr[2]

    # Léger gradient vertical de paupière supérieure (ombre arcade sourcilière)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * 0.5, h * 0.5
    patch *= (0.86 + 0.24 * (yy / h))[..., None]

    # Ouverture oculaire en amande : demi-largeur ~30 px, demi-hauteur ~eyelid_open_px
    rx = 30.0
    ry = prof.eyelid_open_px
    # Paupière supérieure légèrement abaissée quand le regard descend
    lid_drop = 0.22 * iris_shift_y
    eye_mask = (((xx - cx) / rx) ** 2 + ((yy - (cy + lid_drop)) / ry) ** 2) <= 1.0

    # Sclère ombrée près des commissures
    sclera_shading = 1.0 - 0.22 * (((xx - cx) / rx) ** 2) - 0.12 * (((yy - cy) / ry) ** 2)
    for c in range(3):
        ch = patch[..., c]
        ch[eye_mask] = np.clip(prof.sclera_level * sclera_shading[eye_mask], 80, 250)

    # Iris + limbe + pupille
    ix = cx + iris_shift_x
    iy = cy + iris_shift_y
    dist_iris = np.sqrt((xx - ix) ** 2 + (yy - iy) ** 2)
    r_iris = prof.iris_radius_px
    iris_mask = eye_mask & (dist_iris <= r_iris)
    pupil_mask = eye_mask & (dist_iris <= 0.42 * r_iris)
    limbus = np.clip(1.0 - 0.35 * (dist_iris / max(r_iris, 1.0)) ** 3, 0.55, 1.0)
    for c in range(3):
        ch = patch[..., c]
        ch[iris_mask] = prof.iris_bgr[c] * limbus[iris_mask]
        ch[pupil_mask] = 16.0

    # Reflet cornéen + reflet éventuel de lunettes
    cornea_glint = ((xx - (ix + 2.0)) ** 2 + (yy - (iy - 2.0)) ** 2) <= 2.2
    patch[cornea_glint & eye_mask] = 245.0

    # Repli palpébral supérieur
    crease = np.exp(-0.5 * ((yy - (cy - ry - 2.5 + 0.12 * ((xx - cx) ** 2) / 30.0)) / 1.5) ** 2)
    patch *= (1.0 - 0.28 * crease)[..., None]

    if prof.glasses and glare_strength > 0.05:
        gx = float(rng.uniform(w * 0.25, w * 0.75))
        gy = float(rng.uniform(h * 0.25, h * 0.75))
        g_mask = np.exp(-(((xx - gx) / 10.0) ** 2 + ((yy - gy) / 6.0) ** 2))
        patch = patch * (1.0 - glare_strength * g_mask[..., None]) + 250.0 * (glare_strength * g_mask[..., None])

    patch = cv2.GaussianBlur(patch, (3, 3), sigmaX=0.55)
    patch = patch * lighting_gain + rng.normal(0.0, prof.noise_std, size=patch.shape).astype(np.float32)
    return np.clip(patch, 0, 255).astype(np.uint8)


def generate_synthetic_session(
    subject_id: int,
    n_samples: int = 600,
    seed: int = 0,
    head_pose_std_deg: float = 8.0,
    lighting_gain: float = 1.0,
    glare_prob: float = 0.15,
) -> dict[str, np.ndarray]:
    """Génère une session synthétique complète au même schéma que ``collect_pursuit.py``."""
    rng = np.random.default_rng(seed + 1009 * subject_id)
    prof = sample_subject_profile(subject_id, rng)
    screen = ScreenConfig(width_px=2560, height_px=1440, diagonal_inches=27.0, distance_cm=60.0)

    # Cibles sur tout l'écran (avec marge 8%)
    tx = rng.uniform(0.08 * screen.width_px, 0.92 * screen.width_px, size=n_samples)
    ty = rng.uniform(0.08 * screen.height_px, 0.92 * screen.height_px, size=n_samples)
    target_xy = np.column_stack([tx, ty])

    # Pose de tête 6DoF réaliste
    pose_x_cm = rng.normal(0.0, 2.5, size=n_samples)
    pose_y_cm = rng.normal(0.0, 2.0, size=n_samples)
    pose_z_cm = rng.normal(60.0, 4.0, size=n_samples)
    pose_yaw = rng.normal(0.0, head_pose_std_deg, size=n_samples)
    pose_pitch = rng.normal(0.0, 0.7 * head_pose_std_deg, size=n_samples)
    pose_roll = rng.normal(0.0, 3.0, size=n_samples)

    origin_3d_mm = np.column_stack([pose_x_cm * 10.0, pose_y_cm * 10.0, pose_z_cm * 10.0])
    target_angles_deg = screen_px_to_gaze_angles_deg(
        target_xy,
        origin_3d_mm,
        screen.width_px,
        screen.height_px,
        screen.mm_per_px_x,
        screen.mm_per_px_y,
        screen.camera_offset_y_cm,
    )

    # Rotation oculaire dans l'orbite = direction de regard caméra - rotation de tête + angle kappa
    # (+ légère non-linéarité réaliste et décalage binoculaire de parallaxe)
    eye_in_head_yaw = target_angles_deg[:, 0] - pose_yaw + prof.kappa_yaw_deg
    eye_in_head_pitch = target_angles_deg[:, 1] - pose_pitch + prof.kappa_pitch_deg

    # Déplacement de l'iris dans le patch 96x48 (~0.68 px / deg horizontal, ~0.52 px / deg vertical)
    px_per_deg_x = 0.68
    px_per_deg_y = 0.52

    crops = np.zeros((n_samples, 2, 48, 96, 3), dtype=np.uint8)
    geom_vec = np.zeros((n_samples, 9), dtype=np.float32)
    pose_6dof = np.column_stack([pose_x_cm, pose_y_cm, pose_z_cm, pose_yaw, pose_pitch, pose_roll])
    iris_ratios = np.zeros((n_samples, 2), dtype=np.float64)

    for i in range(n_samples):
        g_str = float(rng.uniform(0.25, 0.55)) if (prof.glasses and rng.random() < glare_prob) else 0.0
        l_gain = lighting_gain * float(rng.uniform(0.92, 1.08))
        # Légère dissymétrie vergence gauche/droite (~0.02 * pose_z)
        dx_l = float(eye_in_head_yaw[i] * px_per_deg_x + 0.4)
        dx_r = float(eye_in_head_yaw[i] * px_per_deg_x - 0.4)
        dy_eye = float(eye_in_head_pitch[i] * px_per_deg_y)

        crops[i, 0] = _render_single_eye_patch(dx_l, dy_eye, prof, rng, l_gain, g_str)
        crops[i, 1] = _render_single_eye_patch(dx_r, dy_eye, prof, rng, l_gain, g_str)

        geom_vec[i] = np.array(
            [
                pose_x_cm[i] / 100.0,
                pose_y_cm[i] / 100.0,
                pose_z_cm[i] / 100.0,
                math.radians(pose_yaw[i]),
                math.radians(pose_pitch[i]),
                math.radians(pose_roll[i]),
                pose_x_cm[i] / 60.0,
                pose_y_cm[i] / 60.0,
                (prof.iris_radius_px * 2.0) / 1280.0,
            ],
            dtype=np.float32,
        )
        iris_ratios[i] = np.array(
            [
                0.5 + eye_in_head_yaw[i] * 0.011 + rng.normal(0.0, 0.008),
                0.5 + eye_in_head_pitch[i] * 0.013 + rng.normal(0.0, 0.011),
            ]
        )

    return {
        "eye_crops": crops,
        "geom_vec": geom_vec,
        "pose_6dof": pose_6dof,
        "origin_3d_mm": origin_3d_mm,
        "iris_ratios": iris_ratios,
        "target_xy": target_xy,
        "target_angles_deg": target_angles_deg,
        "timestamps": np.arange(n_samples, dtype=np.float64) / 30.0,
        "valid_mask": np.ones(n_samples, dtype=bool),
        "screen": np.array([2560.0, 1440.0, 27.0, 60.0], dtype=np.float64),
    }
