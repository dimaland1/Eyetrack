"""Module de géométrie écran et d'espace 3D.

Gère les dimensions physiques de l'écran, les conversions pixels <-> millimètres,
et prépare l'intersection rayon 3D / plan écran pour les étapes futures.
"""

from __future__ import annotations
from dataclasses import dataclass
import platform
import math
import numpy as np


def get_screen_resolution() -> tuple[int, int]:
    """Détecte la résolution native de l'écran principal."""
    if platform.system() == "Windows":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            # Prise en compte du DPI scaling Windows
            user32.SetProcessDPIAware()
            w = int(user32.GetSystemMetrics(0))
            h = int(user32.GetSystemMetrics(1))
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
    # Fallback standard Full HD
    return 1920, 1080


@dataclass
class ScreenConfig:
    """Configuration géométrique de l'écran et de la webcam."""

    width_px: int
    height_px: int
    diagonal_inches: float = 27.0
    aspect_ratio: float = 16.0 / 9.0
    distance_cm: float = 60.0
    # Décalage de la webcam par rapport au centre supérieur de l'écran (cm)
    camera_offset_y_cm: float = 0.0

    @property
    def width_mm(self) -> float:
        """Calcule la largeur physique de l'écran en mm."""
        diag_mm = self.diagonal_inches * 25.4
        theta = math.atan(1.0 / self.aspect_ratio)
        return diag_mm * math.cos(theta)

    @property
    def height_mm(self) -> float:
        """Calcule la hauteur physique de l'écran en mm."""
        diag_mm = self.diagonal_inches * 25.4
        theta = math.atan(1.0 / self.aspect_ratio)
        return diag_mm * math.sin(theta)

    @property
    def mm_per_px_x(self) -> float:
        """Facteur d'échelle horizontal mm / pixel."""
        return self.width_mm / max(self.width_px, 1)

    @property
    def mm_per_px_y(self) -> float:
        """Facteur d'échelle vertical mm / pixel."""
        return self.height_mm / max(self.height_px, 1)

    def px_to_norm(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Convertit des coordonnées pixels en coordonnées normalisées [0..1]."""
        return x_px / self.width_px, y_px / self.height_px

    def norm_to_px(self, norm_x: float, norm_y: float) -> tuple[int, int]:
        """Convertit des coordonnées normalisées [0..1] en coordonnées pixels."""
        x = int(round(norm_x * self.width_px))
        y = int(round(norm_y * self.height_px))
        return x, y

    def px_to_angle_deg(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Convertit une position pixel à l'écran en angle visuel par rapport au centre (degrés)."""
        center_x = self.width_px / 2.0
        center_y = self.height_px / 2.0

        dx_mm = (x_px - center_x) * self.mm_per_px_x
        dy_mm = (y_px - center_y) * self.mm_per_px_y
        dist_mm = self.distance_cm * 10.0

        yaw_deg = math.degrees(math.atan2(dx_mm, dist_mm))
        pitch_deg = math.degrees(math.atan2(dy_mm, dist_mm))
        return yaw_deg, pitch_deg

    def distance_cm_between_px(self, p1_px: tuple[float, float], p2_px: tuple[float, float]) -> float:
        """Calcule la distance physique euclidienne en cm entre deux points pixels de l'écran."""
        dx_mm = (p1_px[0] - p2_px[0]) * self.mm_per_px_x
        dy_mm = (p1_px[1] - p2_px[1]) * self.mm_per_px_y
        return float(math.sqrt(dx_mm**2 + dy_mm**2) / 10.0)

    def visual_angle_deg(
        self,
        p_target_px: tuple[float, float],
        p_pred_px: tuple[float, float],
        eye_offset_xyz_cm: tuple[float, float, float] | None = None,
    ) -> float:
        """Calcule l'erreur angulaire visuelle 3D en degrés entre le point cible et le point prédit."""
        center_x = self.width_px / 2.0
        center_y = self.height_px / 2.0

        # Position de l'œil dans l'espace (origine au centre de l'écran)
        if eye_offset_xyz_cm is not None:
            eye_x = eye_offset_xyz_cm[0] * 10.0
            eye_y = eye_offset_xyz_cm[1] * 10.0
            eye_z = -abs(eye_offset_xyz_cm[2]) * 10.0
        else:
            eye_x = 0.0
            eye_y = 0.0
            eye_z = -self.distance_cm * 10.0

        # Vecteurs 3D depuis l'œil vers les points écran (en mm)
        t_x = (p_target_px[0] - center_x) * self.mm_per_px_x
        t_y = (p_target_px[1] - center_y) * self.mm_per_px_y
        v_target = np.array([t_x - eye_x, t_y - eye_y, 0.0 - eye_z], dtype=np.float64)

        p_x = (p_pred_px[0] - center_x) * self.mm_per_px_x
        p_y = (p_pred_px[1] - center_y) * self.mm_per_px_y
        v_pred = np.array([p_x - eye_x, p_y - eye_y, 0.0 - eye_z], dtype=np.float64)

        norm_t = np.linalg.norm(v_target)
        norm_p = np.linalg.norm(v_pred)
        if norm_t < 1e-6 or norm_p < 1e-6:
            return 0.0

        cos_angle = float(np.clip(np.dot(v_target, v_pred) / (norm_t * norm_p), -1.0, 1.0))
        return float(math.degrees(math.acos(cos_angle)))

    def ray_screen_intersection(
        self,
        origin_mm: np.ndarray,
        direction_cam: np.ndarray,
        camera_offset_y_cm: float = 0.0,
    ) -> tuple[float, float]:
        """Calcule l'intersection géométrique du rayon 3D de regard avec le plan écran.

        Args:
            origin_mm: Position 3D de l'œil dans le repère caméra (mm) [Ox, Oy, Oz].
            direction_cam: Vecteur de direction 3D unitaire dé-normalisé [dx, dy, dz].
            camera_offset_y_cm: Décalage vertical webcam -> haut de l'écran en cm (défaut: 0).

        Returns:
            Coordonnées du point de regard écran en pixels (x_px, y_px).
        """
        ox, oy, oz = float(origin_mm[0]), float(origin_mm[1]), float(origin_mm[2])
        dx, dy, dz = float(direction_cam[0]), float(direction_cam[1]), float(direction_cam[2])

        # Sécurité : le rayon doit pointer vers l'écran (dz < 0 en convention OpenCV)
        if dz > -0.05:
            dz = -0.05

        # Plan écran à Z = 0 dans le repère caméra
        # P(t) = O + t * d  avec  Oz + t * dz = 0  =>  t = -Oz / dz
        t = -oz / dz
        if t <= 0:
            t = self.distance_cm * 10.0

        # Point d'impact en mm sur le plan écran
        ix_mm = ox + t * dx
        iy_mm = oy + t * dy

        # Conversion mm -> pixels
        # Repère écran : origine (0,0) en haut à gauche
        # Webcam située au centre horizontal (X = 0) et au-dessus de l'écran
        cam_offset_mm = camera_offset_y_cm * 10.0

        px_x = (self.width_px / 2.0) + (ix_mm / self.mm_per_px_x)
        px_y = ((iy_mm - cam_offset_mm) / self.mm_per_px_y)

        # Clamping doux avec marge d'affichage de 15% hors écran
        margin_x = self.width_px * 0.15
        margin_y = self.height_px * 0.15
        clamped_x = float(np.clip(px_x, -margin_x, self.width_px + margin_x))
        clamped_y = float(np.clip(px_y, -margin_y, self.height_px + margin_y))

        return clamped_x, clamped_y
