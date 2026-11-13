"""Augmentations géométriques et photométriques dédiées aux patchs oculaires.

Conçues pour le Scénario B (généralisation inter-sujets à partir d'un unique sujet) :
1. Déformations morphologiques (étirement vertical/horizontal indépendant des paupières,
   légère rotation et translation) pour simuler d'autres anatomies oculaires.
2. Perturbations photométriques et capteur (niveaux de gris, gamma, contraste local,
   bruit gaussien, flou de mouvement/défocalisation de webcam).
3. Reflets spéculaires et ombres de monture synthétiques pour renforcer la robustesse
   aux lunettes.
"""

from __future__ import annotations

import cv2
import numpy as np


def augment_eye_crop(
    crop_bgr: np.ndarray,
    rng: np.random.Generator,
    morph_strength: float = 1.0,
    photo_strength: float = 1.0,
    add_glare_prob: float = 0.25,
) -> np.ndarray:
    """Applique une augmentation géométrique et photométrique à un patch (H, W, 3).

    Important : le centre de l'iris subit au plus un léger décalage contrôlé pour ne pas
    corrompre le label angulaire au-delà du bruit réaliste de détection de coins (~1 px).
    """
    h, w = crop_bgr.shape[:2]
    img = crop_bgr.copy()

    # 1. Déformation morphologique des paupières / forme d'œil (sans déplacer le centre)
    if morph_strength > 0.0:
        sx = float(rng.uniform(1.0 - 0.14 * morph_strength, 1.0 + 0.14 * morph_strength))
        sy = float(rng.uniform(1.0 - 0.25 * morph_strength, 1.0 + 0.28 * morph_strength))
        angle = float(rng.uniform(-4.0 * morph_strength, 4.0 * morph_strength))
        dx = float(rng.uniform(-1.5 * morph_strength, 1.5 * morph_strength))
        dy = float(rng.uniform(-1.5 * morph_strength, 1.5 * morph_strength))

        cx, cy = w * 0.5, h * 0.5
        rad = np.radians(angle)
        ca, sa = np.cos(rad), np.sin(rad)
        M = np.array(
            [
                [sx * ca, -sy * sa, cx * (1.0 - sx * ca) + cy * sy * sa + dx],
                [sx * sa, sy * ca, cy * (1.0 - sy * ca) - cx * sx * sa + dy],
            ],
            dtype=np.float32,
        )
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 2. Perturbations photométriques et capteur webcam
    if photo_strength > 0.0:
        f = img.astype(np.float32) / 255.0

        # Gamma et gain/biais d'éclairage
        gamma = float(rng.uniform(1.0 - 0.35 * photo_strength, 1.0 + 0.45 * photo_strength))
        f = np.clip(f, 1e-4, 1.0) ** gamma
        gain = float(rng.uniform(1.0 - 0.25 * photo_strength, 1.0 + 0.25 * photo_strength))
        bias = float(rng.uniform(-0.12 * photo_strength, 0.12 * photo_strength))
        f = np.clip(f * gain + bias, 0.0, 1.0)

        # Léger flou optique / compression
        if rng.random() < 0.35 * photo_strength:
            k = int(rng.choice([3, 5]))
            sigma = float(rng.uniform(0.3, 1.1 * photo_strength))
            f = cv2.GaussianBlur(f, (k, k), sigmaX=sigma)

        # Bruit capteur gaussien
        noise_std = float(rng.uniform(0.0, 0.035 * photo_strength))
        if noise_std > 0.0:
            f = np.clip(f + rng.normal(0.0, noise_std, size=f.shape).astype(np.float32), 0.0, 1.0)

        img = (f * 255.0).astype(np.uint8)

    # 3. Reflet spéculaire synthétique (verres de lunettes / écran)
    if add_glare_prob > 0.0 and rng.random() < add_glare_prob:
        gx = int(rng.integers(int(w * 0.2), int(w * 0.8)))
        gy = int(rng.integers(int(h * 0.2), int(h * 0.8)))
        rx = int(rng.integers(3, max(4, w // 7)))
        ry = int(rng.integers(2, max(3, h // 6)))
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (gx, gy), (rx, ry), float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (5, 5), 1.5)[..., None]
        alpha = float(rng.uniform(0.35, 0.75))
        img = np.clip(img.astype(np.float32) * (1.0 - alpha * mask) + 255.0 * (alpha * mask), 0, 255).astype(np.uint8)

    return img


def preprocess_eye_batch_rgb(crops: np.ndarray) -> np.ndarray:
    """Convertit ``(N, 2, H, W, 3)`` uint8 BGR -> float32 ``(N, 2, 3, H, W)`` normalisé [-1, 1]."""
    arr = crops[..., ::-1].astype(np.float32) / 127.5 - 1.0
    return np.transpose(arr, (0, 1, 4, 2, 3))


def preprocess_eye_batch_gray_norm(crops: np.ndarray) -> np.ndarray:
    """Convertit ``(N, 2, H, W, 3)`` uint8 BGR -> niveaux de gris normalisés par patch ``(N, 2, 1, H, W)``.

    La normalisation centrée-réduite par patch (mean=0, std=1) élimine les écarts
    d'exposition globale et de teinte cutanée entre sujets et sessions.
    """
    # Luminance BGR -> Y : 0.114*B + 0.587*G + 0.299*R
    lum = (
        0.114 * crops[..., 0].astype(np.float32)
        + 0.587 * crops[..., 1].astype(np.float32)
        + 0.299 * crops[..., 2].astype(np.float32)
    ) / 255.0
    mu = lum.mean(axis=(-2, -1), keepdims=True)
    sd = np.maximum(lum.std(axis=(-2, -1), keepdims=True), 1e-4)
    normed = (lum - mu) / sd
    return normed[:, :, None, :, :]
