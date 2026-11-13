"""Extraction géométriquement alignée de patchs binoculaires (96x48) et pose 6DoF.

Alignement canonique par œil :
- L'axe horizontal du patch est aligné sur le segment joignant les deux commissures
  (coins interne et externe) de l'œil, supprimant le roulis (roll) de la tête.
- La largeur du patch est normalisée proportionnellement à la distance inter-commissurale
  (facteur ``margin_scale``, défaut 1.6) pour absorber les variations d'échelle/distance.
- Un vecteur géométrique compact (pose 6DoF, origine 3D de l'œil, ouverture, reflets)
  complète l'apparence locale afin de dissocier la pose de tête et le mouvement oculaire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import cv2
import numpy as np

from face import FaceData, LEFT_IMAGE_EYE, RIGHT_IMAGE_EYE, glare_on_iris


EYE_CROP_WIDTH = 96
EYE_CROP_HEIGHT = 48
DEFAULT_MARGIN_SCALE = 1.6


@dataclass
class BinocularSample:
    """Représentation compacte d'une frame pour les modèles basés patchs oculaires."""

    # Patchs BGR (2, H, W, 3) : [0] = œil droit du sujet (gauche image), [1] = œil gauche du sujet
    eye_crops: np.ndarray
    # Vecteur géométrique (9,) : [pose_x_m, pose_y_m, pose_z_m, yaw_rad, pitch_rad, roll_rad,
    #                             eye_cx_norm, eye_cy_norm, iris_diam_norm]
    geom_vec: np.ndarray
    # Pose 6DoF brute (x_cm, y_cm, z_cm, yaw_deg, pitch_deg, roll_deg)
    pose_6dof: np.ndarray
    # Ouverture oculaire moyenne et taux de reflet
    eye_openness: float
    glare: float
    valid: bool = True
    affine_mats: tuple[np.ndarray, np.ndarray] = field(
        default_factory=lambda: (np.eye(2, 3), np.eye(2, 3))
    )


def _ordered_corners(pts: np.ndarray, eye_spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie (coin_gauche_image, coin_droit_image) garantis ordonnés selon X."""
    p_a = pts[eye_spec["outer"]].astype(np.float64)
    p_b = pts[eye_spec["inner"]].astype(np.float64)
    if p_a[0] <= p_b[0]:
        return p_a, p_b
    return p_b, p_a


def extract_aligned_eye_crop(
    frame_bgr: np.ndarray,
    landmarks_px: np.ndarray,
    eye_spec: dict,
    out_size: tuple[int, int] = (EYE_CROP_WIDTH, EYE_CROP_HEIGHT),
    margin_scale: float = DEFAULT_MARGIN_SCALE,
    grayscale: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extrait un patch oculaire aligné horizontalement sur les commissures.

    Args:
        frame_bgr: Image source BGR (H, W, 3).
        landmarks_px: Landmarks faciaux en pixels (478, 2).
        eye_spec: Dictionnaire ``LEFT_IMAGE_EYE`` ou ``RIGHT_IMAGE_EYE``.
        out_size: Dimensions (largeur, hauteur) du patch de sortie (défaut 96x48).
        margin_scale: Ratio largeur_patch / largeur_œil (défaut 1.6 pour inclure paupières).
        grayscale: Convertit le patch en niveaux de gris 2D si True.

    Returns:
        (patch, matrice_affine_2x3, largeur_oeil_px).
    """
    p_left, p_right = _ordered_corners(landmarks_px, eye_spec)
    vec = p_right - p_left
    eye_w = float(np.linalg.norm(vec))
    if eye_w < 4.0:
        out_w, out_h = out_size
        empty = np.zeros((out_h, out_w) if grayscale else (out_h, out_w, 3), dtype=np.uint8)
        return empty, np.eye(2, 3, dtype=np.float64), eye_w

    center = 0.5 * (p_left + p_right)
    # Légère compensation verticale vers le milieu paupière haut/bas si disponible
    p_top = landmarks_px[eye_spec["top"]].astype(np.float64)
    p_bot = landmarks_px[eye_spec["bottom"]].astype(np.float64)
    center = 0.5 * center + 0.5 * (0.5 * (p_top + p_bot))

    angle_rad = math.atan2(float(vec[1]), float(vec[0]))
    out_w, out_h = out_size
    scale = float(out_w) / (margin_scale * eye_w)

    cos_a = math.cos(-angle_rad) * scale
    sin_a = math.sin(-angle_rad) * scale
    cx, cy = float(center[0]), float(center[1])
    tx = out_w * 0.5 - (cos_a * cx - sin_a * cy)
    ty = out_h * 0.5 - (sin_a * cx + cos_a * cy)
    M = np.array([[cos_a, -sin_a, tx], [sin_a, cos_a, ty]], dtype=np.float64)

    patch = cv2.warpAffine(
        frame_bgr,
        M,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if grayscale:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return patch, M, eye_w


def build_geom_vector(face_data: FaceData, frame_shape: tuple[int, ...]) -> np.ndarray:
    """Construit le vecteur géométrique normalisé (9,) décrivant la pose et la géométrie faciale."""
    h, w = float(frame_shape[0]), float(frame_shape[1])
    px, py, pz = face_data.pose_xyz
    yaw, pitch, roll = face_data.pose_ypr

    if face_data.landmarks_px is not None:
        pts = face_data.landmarks_px
        mid_eye = 0.25 * (
            pts[LEFT_IMAGE_EYE["outer"]]
            + pts[LEFT_IMAGE_EYE["inner"]]
            + pts[RIGHT_IMAGE_EYE["inner"]]
            + pts[RIGHT_IMAGE_EYE["outer"]]
        )
        eye_cx_norm = float(mid_eye[0]) / max(w, 1.0) - 0.5
        eye_cy_norm = float(mid_eye[1]) / max(h, 1.0) - 0.5
    else:
        eye_cx_norm, eye_cy_norm = 0.0, 0.0

    iris_norm = float(face_data.iris_diameter_px) / max(w, 1.0)
    return np.array(
        [
            px / 100.0,
            py / 100.0,
            pz / 100.0,
            math.radians(yaw),
            math.radians(pitch),
            math.radians(roll),
            eye_cx_norm,
            eye_cy_norm,
            iris_norm,
        ],
        dtype=np.float32,
    )


def extract_binocular_sample(
    frame_bgr: np.ndarray,
    face_data: FaceData,
    out_size: tuple[int, int] = (EYE_CROP_WIDTH, EYE_CROP_HEIGHT),
    margin_scale: float = DEFAULT_MARGIN_SCALE,
    compute_glare: bool = True,
) -> BinocularSample:
    """Extrait les deux patchs oculaires alignés et le vecteur de pose 6DoF."""
    out_w, out_h = out_size
    if not face_data.detected or face_data.landmarks_px is None:
        return BinocularSample(
            eye_crops=np.zeros((2, out_h, out_w, 3), dtype=np.uint8),
            geom_vec=np.zeros(9, dtype=np.float32),
            pose_6dof=np.zeros(6, dtype=np.float64),
            eye_openness=0.0,
            glare=0.0,
            valid=False,
        )

    crop_l, mat_l, ew_l = extract_aligned_eye_crop(
        frame_bgr, face_data.landmarks_px, LEFT_IMAGE_EYE, out_size=out_size, margin_scale=margin_scale
    )
    crop_r, mat_r, ew_r = extract_aligned_eye_crop(
        frame_bgr, face_data.landmarks_px, RIGHT_IMAGE_EYE, out_size=out_size, margin_scale=margin_scale
    )
    crops = np.stack([crop_l, crop_r], axis=0)
    pose_6dof = np.array([*face_data.pose_xyz, *face_data.pose_ypr], dtype=np.float64)
    geom_vec = build_geom_vector(face_data, frame_bgr.shape)
    glare = glare_on_iris(frame_bgr, face_data) if compute_glare else 0.0
    valid = ew_l >= 6.0 and ew_r >= 6.0

    return BinocularSample(
        eye_crops=crops,
        geom_vec=geom_vec,
        pose_6dof=pose_6dof,
        eye_openness=float(face_data.eye_openness),
        glare=float(glare),
        valid=valid,
        affine_mats=(mat_l, mat_r),
    )
