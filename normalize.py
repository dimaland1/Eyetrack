"""Module de normalisation de données (Zhang et al., ETRA 2018).

Référence : "Revisiting Data Normalization for Appearance-Based Gaze Estimation"
(Zhang, Sugano, Bulling, ETRA 2018).

Construit une caméra virtuelle canonique pointée sur le visage à distance fixe (600 mm),
applique la transformation projective de l'image (warp perspective 448x448) et conserve
la matrice de rotation R_norm pour dé-normaliser ensuite le vecteur de regard.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import cv2
import numpy as np
from face import FaceData

# Diamètre moyen de l'iris humain (mm), très peu variable d'une personne à l'autre
IRIS_DIAMETER_MM = 11.7
# Landmarks MediaPipe du centre du visage : coins des yeux (33, 133, 362, 263) + bouche (61, 291)
FACE_CENTER_IDX = [33, 133, 362, 263, 61, 291]
# Focale virtuelle par défaut (patch 448 px, visage à 600 mm). Mesuré avec diag_regard.py :
# à 960 (Zhang standard) le visage ne remplit que la moitié du patch et le réseau MobileOne
# (entraîné sur des visages recadrés serré) ne suit les yeux qu'à ~30 % ; à 1900, signal/bruit x4.
DEFAULT_VIRTUAL_FOCAL = 1900.0


@dataclass
class NormalizationData:
    """Résultat de la normalisation faciale géométrique."""

    # Image normalisée du visage (H, W, 3) pour l'inférence du réseau
    face_patch: np.ndarray | None = None
    # Matrice de rotation 3x3 de l'espace caméra réel vers l'espace virtuel canonique
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    # Matrice de projection projective 3x3 appliquée à l'image
    warp_matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    # Distance virtuelle fixée en mm
    distance_mm: float = 600.0
    # Origine 3D du visage/yeux dans le repère caméra en mm
    origin_3d_mm: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -600.0], dtype=np.float64))
    # Indicateur de validité
    is_valid: bool = False


class FaceNormalizer:
    """Normalisateur d'image faciale basé sur la formulation de Zhang et al. (2018)."""

    def __init__(
        self,
        target_size: int = 448,
        virtual_distance_mm: float = 600.0,
        camera_fov_deg: float = 60.0,
        virtual_focal: float = DEFAULT_VIRTUAL_FOCAL,
    ) -> None:
        """Initialise le normalisateur.

        Args:
            target_size: Taille carrée du patch de sortie (448 pour MobileOne/ResNet).
            virtual_distance_mm: Distance canonique fixe de la caméra virtuelle (600 mm).
            camera_fov_deg: Champ de vision horizontal estimé de la webcam (défaut: 60°).
            virtual_focal: Focale de la caméra virtuelle pour un patch 448 px. Plus elle est
                grande, plus le visage remplit le patch.
        """
        self.target_size = target_size
        self.virtual_distance_mm = virtual_distance_mm
        self.camera_fov_deg = camera_fov_deg
        self.virtual_focal = float(virtual_focal)
        # Distance lissée (mm) estimée par la taille de l'iris
        self._dist_smooth: float | None = None

    def _estimate_camera_matrix(self, width: int, height: int) -> np.ndarray:
        """Estime la matrice intrinsèque K de la webcam à partir de son FOV."""
        fov_rad = math.radians(self.camera_fov_deg)
        fx = width / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def process(self, frame_bgr: np.ndarray, face_data: FaceData) -> NormalizationData:
        """Normalise l'image du visage selon le protocole de Zhang 2018.

        La caméra virtuelle est placée à partir des landmarks 2D (centre du visage, ligne
        des yeux) et la distance est estimée par la taille de l'iris (~11,7 mm). On ne
        dépend donc pas des conventions de la matrice 3D MediaPipe (repère OpenGL, Y vers
        le haut, FOV interne différent de celui supposé ici), qui faisaient viser la
        caméra virtuelle à côté du visage et tournaient le patch avec le roll de la tête.

        Args:
            frame_bgr: Image source de la webcam.
            face_data: Données faciales MediaPipe.

        Returns:
            NormalizationData avec le patch normalisé et la matrice R_norm.
        """
        if not face_data.detected or face_data.landmarks_px is None:
            return NormalizationData(is_valid=False)

        h, w = frame_bgr.shape[:2]
        K = self._estimate_camera_matrix(w, h)
        K_inv = np.linalg.inv(K)
        f = float(K[0, 0])
        pts = face_data.landmarks_px.astype(np.float64)

        # 1. Distance œil-caméra estimée par le diamètre apparent de l'iris (lissée)
        iris_px = float(face_data.iris_diameter_px)
        if iris_px > 2.0:
            dist_mm = f * IRIS_DIAMETER_MM / iris_px
            dist_mm = float(np.clip(dist_mm, 250.0, 1500.0))
            if self._dist_smooth is None:
                self._dist_smooth = dist_mm
            else:
                self._dist_smooth += 0.15 * (dist_mm - self._dist_smooth)
        elif self._dist_smooth is None:
            self._dist_smooth = self.virtual_distance_mm
        dist_mm = self._dist_smooth

        # 2. Centre du visage (Zhang : coins des yeux + coins de la bouche), rétro-projeté
        center_px = pts[FACE_CENTER_IDX].mean(axis=0)
        ray = K_inv @ np.array([center_px[0], center_px[1], 1.0])
        ray = ray / np.linalg.norm(ray)
        origin_3d = ray * dist_mm  # repère OpenCV : x droite, y bas, z devant la caméra
        norm_e = float(np.linalg.norm(origin_3d))

        # 3. Axe optique de la caméra virtuelle pointant vers le centre du visage (Z_v)
        z_v = origin_3d / norm_e

        # 4. Axe horizontal de la tête : ligne des coins externes des yeux, à la profondeur du visage
        p_l = K_inv @ np.array([pts[33, 0], pts[33, 1], 1.0])
        p_r = K_inv @ np.array([pts[263, 0], pts[263, 1], 1.0])
        x_head = (p_r - p_l)
        nx = np.linalg.norm(x_head)
        x_head = x_head / nx if nx > 1e-9 else np.array([1.0, 0.0, 0.0])

        # Y_v orthogonal à Z_v et X_head, puis X_v
        y_v = np.cross(z_v, x_head)
        norm_y = np.linalg.norm(y_v)
        y_v = y_v / norm_y if norm_y > 1e-6 else np.array([0.0, 1.0, 0.0])
        x_v = np.cross(y_v, z_v)
        x_v = x_v / np.linalg.norm(x_v)

        # Matrice de rotation canonique R_norm (lignes = axes de la caméra virtuelle)
        R_norm = np.vstack([x_v, y_v, z_v])

        # 5. Matrice d'échelle S (ramène le visage à la distance virtuelle fixe)
        S = np.diag([1.0, 1.0, self.virtual_distance_mm / norm_e])

        # 6. Intrinsèques de la caméra virtuelle (focale proportionnelle à la taille du patch)
        c_virtual = self.target_size / 2.0
        f_virtual = self.virtual_focal * (self.target_size / 448.0)
        C_v = np.array([
            [f_virtual, 0.0, c_virtual],
            [0.0, f_virtual, c_virtual],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # 7. Warp perspective M = C_v * S * R_norm * K^-1
        M = C_v @ S @ R_norm @ K_inv
        warped = cv2.warpPerspective(
            frame_bgr,
            M,
            (self.target_size, self.target_size),
            flags=cv2.INTER_LINEAR,
        )

        return NormalizationData(
            face_patch=warped,
            rotation_matrix=R_norm,
            warp_matrix=M,
            distance_mm=self.virtual_distance_mm,
            origin_3d_mm=origin_3d,
            is_valid=True,
        )
