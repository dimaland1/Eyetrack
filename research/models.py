"""Architectures neuronales légères pour le Scénario A (perso) et le Scénario B (différentiel).

Principes architecturaux communs :
1. Entrée bi-oculaire alignée (48x96 par œil) plutôt que visage complet, afin d'empêcher
   toute mémorisation du décor ou des vêtements et d'obtenir une latence < 2 ms sur CPU.
2. Ajout de canaux de coordonnées normalisés (CoordConv) et préservation d'une grille
   spatiale (3x6) avant projection, indispensable pour préserver la position sub-pixellique
   de l'iris par rapport aux coins de l'œil.
3. Scénario A (``DualEyePersonalNet``) : prédit directement les angles (yaw, pitch) de la
   direction de regard 3D à partir des deux yeux RGB + vecteur géométrique 6DoF.
4. Scénario B (``DifferentialEyeNet``) : architecture siamoise sur niveaux de gris normalisés
   qui prédit la variation angulaire (d_yaw, d_pitch) entre une image requête et une ou
   plusieurs images ancres calibrées, avec antisymétrie structurelle f(q, r) = -f(r, q).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordConv2d(nn.Module):
    """Concatène deux canaux de coordonnées spatiales (x, y) in [-1, 1] avant la convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        hw: tuple[int, int] = (48, 96),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        h, w = hw
        yy = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(1, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(1, 1, h, w)
        self.register_buffer("coord_grid", torch.cat([xx, yy], dim=1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == self.coord_grid.shape[-2:]:
            grid = self.coord_grid.to(dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1, -1)
        else:
            b, _, h, w = x.shape
            yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
            xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
            grid = torch.cat([xx, yy], dim=1)
        return self.act(self.bn(self.conv(torch.cat([x, grid], dim=1))))


class DSConvBlock(nn.Module):
    """Bloc convolutionnel séparable en profondeur (style MobileNet / MobileOne)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
        self.use_res = stride == 1 and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.dw(x)))
        out = self.bn2(self.pw(out))
        if self.use_res:
            out = out + x
        return self.act(out)


class SpatialEyeEncoder(nn.Module):
    """Encodeur léger préservant la topologie spatiale de l'œil (48x96 -> vecteur D)."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 64, width_mult: float = 1.0) -> None:
        super().__init__()
        c1 = max(12, int(round(16 * width_mult)))
        c2 = max(24, int(round(32 * width_mult)))
        c3 = max(32, int(round(48 * width_mult)))
        c4 = max(48, int(round(64 * width_mult)))

        self.stem = CoordConv2d(in_channels, c1, kernel_size=3, stride=2, padding=1)  # 24x48
        self.stage1 = nn.Sequential(DSConvBlock(c1, c2, stride=2), DSConvBlock(c2, c2, stride=1))  # 12x24
        self.stage2 = nn.Sequential(DSConvBlock(c2, c3, stride=2), DSConvBlock(c3, c3, stride=1))  # 6x12
        self.stage3 = DSConvBlock(c3, c4, stride=2)  # 3x6
        self.pool = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))  # -> grille spatiale 3x3 compatible ONNX
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c4 * 3 * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.proj(x)


class DualEyePersonalNet(nn.Module):
    """Modèle personnalisé (Scénario A) : patchs binoculaires RGB + pose 6DoF -> (yaw, pitch) en deg."""

    def __init__(self, geom_dim: int = 9, eye_embed_dim: int = 64, hidden_dim: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        # Encodeur partagé (poids siamois) ; un biais latéral d'asymétrie (gauche vs droite)
        # est fourni au niveau de la fusion pour respecter la dissymétrie des deux yeux.
        self.eye_encoder = SpatialEyeEncoder(in_channels=3, embed_dim=eye_embed_dim)
        self.pose_mlp = nn.Sequential(
            nn.Linear(geom_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 32),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * eye_embed_dim + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, eye_l: torch.Tensor, eye_r: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        zl = self.eye_encoder(eye_l)
        zr = self.eye_encoder(eye_r)
        zp = self.pose_mlp(geom)
        return self.head(torch.cat([zl, zr, zp], dim=-1))


class DifferentialEyeNet(nn.Module):
    """Modèle différentiel siamois (Scénario B) : (I_q, I_r, pose_q, pose_r) -> delta (yaw, pitch).

    Structurellement antisymétrique : ``f(q, r) = 0.5 * (g(q, r) - g(r, q))``, garantissant
    exactement ``f(r, r) = 0`` (cohérence d'ancre) et ``f(q, r) = -f(r, q)``.
    Entrée en niveaux de gris normalisés pour découpler la couleur de peau et l'éclairage.
    """

    def __init__(self, geom_dim: int = 9, eye_embed_dim: int = 64, hidden_dim: int = 96, dropout: float = 0.15) -> None:
        super().__init__()
        self.eye_encoder = SpatialEyeEncoder(in_channels=1, embed_dim=eye_embed_dim)
        # On passe uniquement la variation géométrique d_geom et un faible contexte pose (yaw, pitch, roll)
        # pour corriger le couplage 3D sans fournir de signature de position statique du sujet.
        self.diff_head = nn.Sequential(
            nn.Linear(2 * eye_embed_dim + geom_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),
        )

    def encode(self, eye_l: torch.Tensor, eye_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.eye_encoder(eye_l), self.eye_encoder(eye_r)

    def _raw_pair(
        self,
        zl_q: torch.Tensor,
        zr_q: torch.Tensor,
        geom_q: torch.Tensor,
        zl_r: torch.Tensor,
        zr_r: torch.Tensor,
        geom_r: torch.Tensor,
    ) -> torch.Tensor:
        dz_l = zl_q - zl_r
        dz_r = zr_q - zr_r
        d_geom = geom_q - geom_r
        pose_mean = 0.5 * (geom_q[:, 3:6] + geom_r[:, 3:6])
        feat = torch.cat([dz_l, dz_r, d_geom, pose_mean], dim=-1)
        return self.diff_head(feat)

    def forward(
        self,
        eye_l_q: torch.Tensor,
        eye_r_q: torch.Tensor,
        geom_q: torch.Tensor,
        eye_l_r: torch.Tensor,
        eye_r_r: torch.Tensor,
        geom_r: torch.Tensor,
    ) -> torch.Tensor:
        zl_q, zr_q = self.encode(eye_l_q, eye_r_q)
        zl_r, zr_r = self.encode(eye_l_r, eye_r_r)
        return 0.5 * (
            self._raw_pair(zl_q, zr_q, geom_q, zl_r, zr_r, geom_r)
            - self._raw_pair(zl_r, zr_r, geom_r, zl_q, zr_q, geom_q)
        )


def count_parameters(model: nn.Module) -> int:
    """Nombre total de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def screen_px_to_gaze_angles_deg(
    target_xy: torch.Tensor | "np.ndarray",
    origin_3d_mm: torch.Tensor | "np.ndarray",
    width_px: int,
    height_px: int,
    mm_per_px_x: float,
    mm_per_px_y: float,
    camera_offset_y_cm: float = 0.0,
):
    """Convertit une cible écran (px) et une origine oculaire 3D (mm) en angles (yaw, pitch) deg.

    Suit exactement les conventions de ``geometry.ScreenConfig.ray_screen_intersection`` :
    repère caméra OpenCV (x droite, y bas, z vers l'utilisateur < 0 à la position de l'œil,
    écran dans le plan z = 0 avec centre horizontal x = 0 et bord supérieur à y = cam_offset_mm).
    """
    import numpy as np

    tgt = np.asarray(target_xy, dtype=np.float64)
    ori = np.asarray(origin_3d_mm, dtype=np.float64)
    cam_off_mm = camera_offset_y_cm * 10.0

    ix_mm = (tgt[..., 0] - width_px / 2.0) * mm_per_px_x
    iy_mm = tgt[..., 1] * mm_per_px_y + cam_off_mm
    iz_mm = np.zeros_like(ix_mm)

    dx = ix_mm - ori[..., 0]
    dy = iy_mm - ori[..., 1]
    dz = iz_mm - ori[..., 2]  # > 0 car origin_z < 0 (attention : gaze_model utilise Z négatif vers écran)

    # Distance le long de l'axe optique vers l'écran : positive
    dist = np.maximum(np.abs(dz), 1e-3)
    yaw_deg = np.degrees(np.arctan2(dx, dist))
    pitch_deg = np.degrees(np.arctan2(dy, np.sqrt(dx**2 + dist**2)))
    return np.stack([yaw_deg, pitch_deg], axis=-1)


def gaze_angles_deg_to_screen_px(
    angles_deg,
    origin_3d_mm,
    width_px: int,
    height_px: int,
    mm_per_px_x: float,
    mm_per_px_y: float,
    camera_offset_y_cm: float = 0.0,
):
    """Inverse exact de ``screen_px_to_gaze_angles_deg`` : angles (yaw, pitch) -> (x_px, y_px)."""
    import numpy as np

    ang = np.asarray(angles_deg, dtype=np.float64)
    ori = np.asarray(origin_3d_mm, dtype=np.float64)
    yaw = np.radians(ang[..., 0])
    pitch = np.radians(ang[..., 1])

    # dz positif (écran à z=0 depuis origin_z < 0), ou t = -ori_z / dir_z
    dir_x = np.cos(pitch) * np.sin(yaw)
    dir_y = np.sin(pitch)
    dir_z = np.cos(pitch) * np.cos(yaw)
    dir_z = np.maximum(dir_z, 0.05)

    t = np.abs(ori[..., 2]) / dir_z
    ix_mm = ori[..., 0] + t * dir_x
    iy_mm = ori[..., 1] + t * dir_y

    cam_off_mm = camera_offset_y_cm * 10.0
    px_x = (width_px / 2.0) + ix_mm / mm_per_px_x
    px_y = (iy_mm - cam_off_mm) / mm_per_px_y
    return np.stack([px_x, px_y], axis=-1)
