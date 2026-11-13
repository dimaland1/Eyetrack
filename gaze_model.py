"""Module d'estimation du regard.

Implémente deux approches interchangeables :
1. Mode 'iris' (MVP) : calcul des ratios de position de l'iris par rapport aux coins/paupières.
2. Mode 'net' (SOTA léger) : inférence d'un modèle d'apparence ONNX (MobileOne-S0 / ResNet-18)
   sur l'image normalisée selon Zhang 2018, produisant une direction de regard 3D canonique.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import os
import urllib.request
import cv2
import numpy as np
import onnxruntime as ort

from face import FaceData, LEFT_IMAGE_EYE, RIGHT_IMAGE_EYE
from normalize import NormalizationData
from paths import USER_MODELS_DIR

# Poids pré-entraînés (yakhyo/gaze-estimation, entraînés sur Gaze360 : usage non commercial).
# Jamais fournis avec Eyetrack : l'utilisateur les télécharge lui-même depuis la source officielle,
# après avoir accepté leur licence (installeur ou bouton du panneau). Sans eux : mode 'iris'.
MODEL_FILE = "mobilenetv2_gaze.onnx"
FALLBACK_MODEL_FILE = "mobileone_s0_gaze.onnx"
MODEL_URL = "https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobilenetv2_gaze.onnx"
MODEL_SOURCE = "https://github.com/yakhyo/gaze-estimation"
# Empreinte du fichier attendu : un fichier modifié ou corrompu est refusé
MODEL_SHA256 = "b81312df85c7ac1c1b5f78c573620d22c2719cb839650e15f12dc7eecb7744a4"


def gaze_model_path() -> str:
    """Emplacement du modèle de précision (dossier de l'utilisateur, pas le paquet)."""
    env_model = os.environ.get("EYETRACK_GAZE_MODEL", "").strip()
    if env_model:
        candidate = USER_MODELS_DIR / env_model
        if os.path.isfile(candidate):
            return str(candidate)
    primary = USER_MODELS_DIR / MODEL_FILE
    if os.path.isfile(primary):
        return str(primary)
    fallback = USER_MODELS_DIR / FALLBACK_MODEL_FILE
    if os.path.isfile(fallback):
        return str(fallback)
    return str(primary)


def gaze_model_available() -> bool:
    return os.path.isfile(gaze_model_path())


def download_gaze_model(dest: str | None = None, url: str = MODEL_URL, sha256: str = MODEL_SHA256,
                        timeout: float = 30.0) -> str:
    """Télécharge le modèle depuis sa source officielle (à appeler seulement après l'accord de
    l'utilisateur sur sa licence). Vérifie l'empreinte SHA-256 avant de le mettre en place."""
    dest = dest or str(USER_MODELS_DIR / MODEL_FILE)  # jamais le nom du modèle de repli
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(part, "wb") as f:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                digest.update(chunk)
                f.write(chunk)
        if digest.hexdigest().lower() != sha256.lower():
            raise ValueError("fichier reçu différent du modèle attendu (empreinte SHA-256)")
        os.replace(part, dest)
    finally:
        if os.path.exists(part):
            os.remove(part)
    return dest


@dataclass
class GazeFeatures:
    """Caractéristiques du regard extraites pour la calibration et le pointage."""

    # Ratios moyens normalisés (x, y) pour le mode iris
    iris_ratios: tuple[float, float] = (0.5, 0.5)
    # Angles de regard (pitch, yaw en degrés)
    pitch_yaw_deg: tuple[float, float] = (0.0, 0.0)
    # Vecteur de direction 3D unitaire dans l'espace caméra réel (dx, dy, dz)
    gaze_vector_cam: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float64))
    # Vecteur de direction 3D unitaire dans l'espace normalisé canonique
    gaze_vector_norm: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=np.float64))
    # Vecteur de features numpy brut pour la régression polynomiale / calibration
    raw_features: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5], dtype=np.float64))
    # Confiance de la mesure (0.0 à 1.0)
    confidence: float = 1.0


class IrisGazeEstimator:
    """Estimateur basé sur la géométrie et les ratios de position de l'iris dans l'œil."""

    def __init__(self) -> None:
        pass

    def estimate(self, face_data: FaceData, norm_data: NormalizationData | None = None) -> GazeFeatures:
        """Calcule les ratios d'iris moyens des deux yeux.

        Args:
            face_data: Données de visage détecté avec les 478 landmarks.
            norm_data: Données de normalisation (non utilisées en mode iris).

        Returns:
            GazeFeatures contenant les ratios (rx, ry) et le vecteur de features.
        """
        if not face_data.detected or face_data.landmarks_px is None:
            return GazeFeatures(confidence=0.0)

        pts = face_data.landmarks_px

        # --- Œil gauche de l'image (œil droit du sujet) ---
        p_outer1 = pts[LEFT_IMAGE_EYE["outer"]]
        p_inner1 = pts[LEFT_IMAGE_EYE["inner"]]
        p_top1 = pts[LEFT_IMAGE_EYE["top"]]
        p_bot1 = pts[LEFT_IMAGE_EYE["bottom"]]
        p_iris1 = pts[LEFT_IMAGE_EYE["iris_center"]]

        dx1 = float(p_inner1[0] - p_outer1[0])
        dy1 = float(p_bot1[1] - p_top1[1])
        h_ratio1 = (p_iris1[0] - p_outer1[0]) / max(abs(dx1), 1.0)
        v_ratio1 = (p_iris1[1] - p_top1[1]) / max(abs(dy1), 1.0)

        # --- Œil droit de l'image (œil gauche du sujet) ---
        p_inner2 = pts[RIGHT_IMAGE_EYE["inner"]]
        p_outer2 = pts[RIGHT_IMAGE_EYE["outer"]]
        p_top2 = pts[RIGHT_IMAGE_EYE["top"]]
        p_bot2 = pts[RIGHT_IMAGE_EYE["bottom"]]
        p_iris2 = pts[RIGHT_IMAGE_EYE["iris_center"]]

        dx2 = float(p_outer2[0] - p_inner2[0])
        dy2 = float(p_bot2[1] - p_top2[1])
        h_ratio2 = (p_iris2[0] - p_inner2[0]) / max(abs(dx2), 1.0)
        v_ratio2 = (p_iris2[1] - p_top2[1]) / max(abs(dy2), 1.0)

        # Moyenne des deux yeux pour réduire le bruit
        ratio_x = float(0.5 * (h_ratio1 + h_ratio2))
        ratio_y = float(0.5 * (v_ratio1 + v_ratio2))

        # Vecteur de features : ratios combinés
        raw_feat = np.array([ratio_x, ratio_y], dtype=np.float64)

        return GazeFeatures(
            iris_ratios=(ratio_x, ratio_y),
            raw_features=raw_feat,
            confidence=1.0,
        )


class NetGazeEstimator:
    """Estimateur basé sur réseau d'apparence ONNX (MobileOne-S0 / ResNet-18)."""

    def __init__(self, model_path: str | None = None) -> None:
        """Initialise la session ONNX Runtime.

        Args:
            model_path: Chemin du fichier ONNX (défaut : gaze_model_path()). Jamais téléchargé
                automatiquement : FileNotFoundError s'il est absent.
        """
        self.model_path = model_path or gaze_model_path()
        # Nom du modèle chargé : enregistré avec la calibration (une calibration faite avec un
        # autre modèle de regard ne vaut rien et doit être refaite)
        self.model_name = os.path.basename(self.model_path)
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Modèle de regard absent : {self.model_path}")
        self._iris = IrisGazeEstimator()

        # Initialisation session ONNX Runtime sur CPU (ultra-rapide avec AVX2)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4  # mesuré : 2 threads ≈ 12 ms, 4 threads ≈ 5 ms
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(self.model_path, opts, providers=["CPUExecutionProvider"])

        self.input_name = self.session.get_inputs()[0].name
        # Format attendu : [1, 3, 448, 448]
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Constantes de décodage classification-régression L2CS
        self._bins = 90
        self._binwidth = 4.0
        self._angle_offset = 180.0
        self.idx_tensor = np.arange(self._bins, dtype=np.float32)

        # Normalisation ImageNet standard
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        # Pré-filtre temporel anti-pics sur les angles du réseau (médiane 3 images + lissage adaptatif)
        self._angle_hist: list[tuple[float, float]] = []
        self._angle_smooth: tuple[float, float] | None = None

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Calcule le softmax numériquement stable sur l'axe 1."""
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def estimate(self, face_data: FaceData, norm_data: NormalizationData | None = None) -> GazeFeatures:
        """Estime la direction du regard 3D à partir de l'image normalisée.

        Args:
            face_data: Données de visage détecté.
            norm_data: Données de normalisation de Zhang 2018.

        Returns:
            GazeFeatures contenant la direction 3D dans le repère caméra réel.
        """
        if not face_data.detected or norm_data is None or not norm_data.is_valid or norm_data.face_patch is None:
            return GazeFeatures(confidence=0.0)

        patch = norm_data.face_patch
        # Prétraitement : BGR -> RGB, normalisation float32, CHW, batch dim
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm_img = (rgb - self.mean) / self.std
        tensor = np.transpose(norm_img, (2, 0, 1))[None, ...].astype(np.float32)

        # Inférence ONNX (~5 ms sur CPU)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        yaw_logits = outputs[0]
        pitch_logits = outputs[1]

        # Décodage Softmax expectation
        yaw_probs = self._softmax(yaw_logits)
        pitch_probs = self._softmax(pitch_logits)

        yaw_deg = float(np.sum(yaw_probs * self.idx_tensor) * self._binwidth - self._angle_offset)
        pitch_deg = float(np.sum(pitch_probs * self.idx_tensor) * self._binwidth - self._angle_offset)

        # Pré-filtrage temporel : rejette les sauts isolés (±8° sur 1 image dus au bruit MJPEG)
        # tout en laissant passer immédiatement les vraies saccades (> 6° maintenus).
        self._angle_hist.append((yaw_deg, pitch_deg))
        if len(self._angle_hist) > 3:
            self._angle_hist.pop(0)
        med_yaw = float(np.median([a[0] for a in self._angle_hist]))
        med_pitch = float(np.median([a[1] for a in self._angle_hist]))
        if self._angle_smooth is None:
            self._angle_smooth = (med_yaw, med_pitch)
        else:
            dyaw = med_yaw - self._angle_smooth[0]
            dpitch = med_pitch - self._angle_smooth[1]
            step = float(np.hypot(dyaw, dpitch))
            # Gain adaptatif : 0.28 au repos (divise le bruit par ~3), monte à 0.85 sur saccade
            alpha_net = min(0.85, 0.28 + 0.08 * max(0.0, step - 2.0))
            self._angle_smooth = (
                self._angle_smooth[0] + alpha_net * dyaw,
                self._angle_smooth[1] + alpha_net * dpitch,
            )
        yaw_deg, pitch_deg = self._angle_smooth

        # Construction du vecteur unitaire 3D dans l'espace normalisé canonique
        # Convention OpenCV / MPIIGaze : regard vers l'écran = Z négatif (-1)
        pitch_rad = np.radians(pitch_deg)
        yaw_rad = np.radians(yaw_deg)

        g_norm = np.array([
            -np.cos(pitch_rad) * np.sin(yaw_rad),
            -np.sin(pitch_rad),
            -np.cos(pitch_rad) * np.cos(yaw_rad),
        ], dtype=np.float64)
        g_norm = g_norm / np.linalg.norm(g_norm)

        # Dé-normalisation de Zhang 2018 :
        # Le vecteur regard caméra = R_norm^-1 * g_norm = R_norm^T * g_norm
        R_norm = norm_data.rotation_matrix
        g_cam = R_norm.T @ g_norm
        g_cam = g_cam / np.linalg.norm(g_cam)

        # Features de calibration : point d'impact du rayon de regard sur le plan de l'écran
        # (z = 0 dans le repère caméra OpenCV : x droite, y bas), en mm, NON borné.
        # La géométrie gère seule les translations de la tête (1 cm de déplacement = 1 cm à
        # l'écran) ; la calibration n'a plus qu'à corriger gains et décalages du réseau.
        # (Apprendre l'effet de la position de la tête depuis la calibration, où elle ne bouge
        # que de quelques mm, donnait des gains absurdes : ~1000 px par cm.)
        origin = np.asarray(norm_data.origin_3d_mm, dtype=np.float64)
        dz = min(float(g_cam[2]), -0.05)
        t = -origin[2] / dz
        hit_x = float(origin[0] + t * g_cam[0])
        hit_y = float(origin[1] + t * g_cam[1])

        # + ratios d'iris : le réseau donne la direction globale, l'iris affine l'horizontale.
        # Mesuré sur une vraie calibration (avec lunettes) : 3,7° -> 3,1° dans la zone calibrée.
        iris_rx, iris_ry = self._iris.estimate(face_data).iris_ratios

        return GazeFeatures(
            iris_ratios=(iris_rx, iris_ry),
            pitch_yaw_deg=(pitch_deg, yaw_deg),
            gaze_vector_cam=g_cam,
            gaze_vector_norm=g_norm,
            raw_features=np.array([hit_x, hit_y, iris_rx, iris_ry], dtype=np.float64),
            confidence=1.0,
        )


def create_gaze_estimator(mode: str = "iris", model_path: str | None = None) -> IrisGazeEstimator | NetGazeEstimator:
    """Factory pour créer l'estimateur de regard selon le mode demandé."""
    if mode == "iris":
        return IrisGazeEstimator()
    elif mode == "net":
        return NetGazeEstimator(model_path=model_path)
    else:
        raise ValueError(f"Mode de regard inconnu : {mode}. Choix possibles : 'iris', 'net'")
