"""Module de détection faciale et d'estimation de pose 6DoF via MediaPipe Tasks.

Utilise le modèle FaceLandmarker (478 points, incluant les iris) et la matrice
de transformation faciale 4x4 pour extraire la pose de tête métrique (cm)
et les rotations d'Euler (degrés).
"""

from __future__ import annotations
from dataclasses import dataclass
import os
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

from paths import model_path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Indices canoniques MediaPipe 478 points
# Œil droit du sujet (côté gauche de l'image caméra standard)
LEFT_IMAGE_EYE = {
    "outer": 33,
    "inner": 133,
    "top": 159,
    "bottom": 145,
    "iris_center": 468,
    "iris_rim": (469, 470, 471, 472),
}

# Œil gauche du sujet (côté droit de l'image caméra standard)
RIGHT_IMAGE_EYE = {
    "inner": 362,
    "outer": 263,
    "top": 386,
    "bottom": 374,
    "iris_center": 473,
    "iris_rim": (474, 475, 476, 477),
}

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class FaceData:
    """Structure contenant toutes les données extraites du visage par frame."""

    detected: bool
    # Landmarks normalisés [0..1] de dimension (478, 3)
    landmarks_norm: np.ndarray | None = None
    # Landmarks en pixels de l'image courante (478, 2)
    landmarks_px: np.ndarray | None = None
    # Matrice de transformation 4x4 brute issue de MediaPipe
    transform_matrix: np.ndarray | None = None
    # Pose 6DoF : translation (x, y, z en cm) et rotation (yaw, pitch, roll en degrés)
    pose_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pose_ypr: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Diamètre estimé de l'iris en pixels
    iris_diameter_px: float = 0.0
    # Centres des iris en coordonnées pixels (x, y)
    left_iris_px: tuple[float, float] = (0.0, 0.0)
    right_iris_px: tuple[float, float] = (0.0, 0.0)
    # Ouverture des yeux : distance paupière haute-basse / largeur de l'œil (moyenne des 2 yeux).
    # ~0,25-0,35 œil ouvert, chute brutale pendant un clignement.
    eye_openness: float = 0.0


def pose_from_matrix(mat: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Matrice de transformation faciale MediaPipe (4x4) -> (x, y, z en cm), (yaw, pitch, roll en °).

    Repère MediaPipe : x vers la droite de l'image, y vers le haut, z vers la caméra.
    Angles d'Euler dans l'ordre d'OpenTrack : yaw (autour de Y), puis pitch (autour de l'axe X
    de la tête), puis roll (autour de l'axe avant de la tête), soit R = Ry(yaw)·Rx(pitch)·Rz(roll).
    L'ancien RQDecomp3x3 (ordre X-Y-Z) mélangeait les axes dès que la tête était tournée : tête
    tournée de 40° puis penchée de 15° donnait un faux pitch de 12° (cf. tests/test_axes.py).
    """
    mat = np.asarray(mat, dtype=np.float64)
    pose_xyz = (float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3]))
    # Rotation pure (au cas où la matrice contient une échelle)
    U, _, Vt = np.linalg.svd(mat[:3, :3])
    R = U @ Vt
    yaw = float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))
    pitch = float(np.degrees(np.arcsin(np.clip(-R[1, 2], -1.0, 1.0))))
    roll = float(np.degrees(np.arctan2(R[1, 0], R[1, 1])))
    return pose_xyz, (yaw, pitch, roll)


class FaceDetector:
    """Détecteur de visage et de pose basé sur MediaPipe FaceLandmarker."""

    def __init__(self, model_file: str | None = None) -> None:
        """Initialise le modèle FaceLandmarker.

        Args:
            model_file: Chemin vers le fichier modèle .task (défaut : models/face_landmarker.task).
                Téléchargé automatiquement si absent.
        """
        self.model_path = model_file or model_path("face_landmarker.task")
        self._ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        # Mode VIDEO : MediaPipe suit le visage d'une frame à l'autre au lieu de le
        # redétecter à chaque image -> landmarks plus stables, moins de pertes, plus rapide.
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        # Le mode VIDEO exige des timestamps (ms) strictement croissants
        self._last_ts_ms: int = -1

    def _ensure_model_exists(self) -> None:
        """Vérifie la présence du modèle et le télécharge si nécessaire."""
        if not os.path.exists(self.model_path):
            print(f"[face] Téléchargement du modèle FaceLandmarker depuis {MODEL_URL}...")
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print("[face] Modèle téléchargé avec succès.")

    def process(self, frame_bgr: np.ndarray, timestamp_s: float | None = None) -> FaceData:
        """Traite une image BGR et extrait les landmarks et la pose 6DoF.

        Args:
            frame_bgr: Image BGR OpenCV (H, W, 3).
            timestamp_s: Horodatage de la frame en secondes (time.perf_counter()).
                Si omis, l'heure courante est utilisée.

        Returns:
            FaceData contenant les informations faciales ou detected=False.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if timestamp_s is None:
            timestamp_s = time.perf_counter()
        ts_ms = max(int(timestamp_s * 1000.0), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms

        result = self.detector.detect_for_video(mp_image, ts_ms)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return FaceData(detected=False)

        raw_landmarks = result.face_landmarks[0]
        # Conversion en tableau numpy normalisé (478, 3)
        lm_norm = np.array(
            [[lm.x, lm.y, lm.z] for lm in raw_landmarks], dtype=np.float32
        )
        # Coordonnées en pixels
        lm_px = np.zeros((len(raw_landmarks), 2), dtype=np.float32)
        lm_px[:, 0] = lm_norm[:, 0] * w
        lm_px[:, 1] = lm_norm[:, 1] * h

        # Pose 6DoF depuis la matrice 4x4 MediaPipe
        pose_xyz = (0.0, 0.0, -60.0)
        pose_ypr = (0.0, 0.0, 0.0)
        transform_mat = None

        if result.facial_transformation_matrixes and len(result.facial_transformation_matrixes) > 0:
            mat = np.array(result.facial_transformation_matrixes[0], dtype=np.float64)
            transform_mat = mat
            pose_xyz, pose_ypr = pose_from_matrix(mat)

        # Extraction des iris et calcul du diamètre métrique
        left_iris_center = (float(lm_px[LEFT_IMAGE_EYE["iris_center"], 0]), float(lm_px[LEFT_IMAGE_EYE["iris_center"], 1]))
        right_iris_center = (float(lm_px[RIGHT_IMAGE_EYE["iris_center"], 0]), float(lm_px[RIGHT_IMAGE_EYE["iris_center"], 1]))

        # Calcul du diamètre de l'iris en pixels (distance entre les points du contour)
        p_left_1 = lm_px[LEFT_IMAGE_EYE["iris_rim"][0]]
        p_left_2 = lm_px[LEFT_IMAGE_EYE["iris_rim"][2]]
        d1 = float(np.linalg.norm(p_left_1 - p_left_2))

        p_right_1 = lm_px[RIGHT_IMAGE_EYE["iris_rim"][0]]
        p_right_2 = lm_px[RIGHT_IMAGE_EYE["iris_rim"][2]]
        d2 = float(np.linalg.norm(p_right_1 - p_right_2))
        iris_diam_px = (d1 + d2) / 2.0

        # Ouverture des yeux (détection des clignements)
        def _openness(eye: dict) -> float:
            width = float(np.linalg.norm(lm_px[eye["outer"]] - lm_px[eye["inner"]]))
            height = float(np.linalg.norm(lm_px[eye["top"]] - lm_px[eye["bottom"]]))
            return height / max(width, 1.0)

        eye_open = 0.5 * (_openness(LEFT_IMAGE_EYE) + _openness(RIGHT_IMAGE_EYE))

        return FaceData(
            detected=True,
            landmarks_norm=lm_norm,
            landmarks_px=lm_px,
            transform_matrix=transform_mat,
            pose_xyz=pose_xyz,
            pose_ypr=pose_ypr,
            iris_diameter_px=iris_diam_px,
            left_iris_px=left_iris_center,
            right_iris_px=right_iris_center,
            eye_openness=eye_open,
        )
