"""Tests du choix du mode caméra : ordre des réglages DirectShow, mémorisation, repli.

Sans webcam : cv2.VideoCapture est remplacé par une fausse caméra. Ces tests verrouillent un
défaut constaté sur une Logitech C920 : le format MJPG réglé avant la taille était perdu, la
caméra retombait en non compressé (720p à 10 images/s) et le 640x480 était mémorisé à tort.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402

import capture  # noqa: E402
from capture import CAMERA_CACHE_VERSION, WebcamCapture  # noqa: E402

W, H = cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT
FPS, FOURCC = cv2.CAP_PROP_FPS, cv2.CAP_PROP_FOURCC
MJPG = cv2.VideoWriter_fourcc(*"MJPG")
YUY2 = cv2.VideoWriter_fourcc(*"YUY2")


class FakeCap:
    """Fausse caméra : mémorise les réglages reçus et leur ordre."""

    has_mjpg = True          # False : la webcam n'a pas de mode compressé
    opened: list["FakeCap"] = []

    def __init__(self, *_args) -> None:
        self.calls: list[int] = []
        self.props = {W: 640, H: 480, FOURCC: YUY2}
        FakeCap.opened.append(self)

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: float) -> bool:
        self.calls.append(prop)
        if prop == FOURCC and not self.has_mjpg:
            return False
        self.props[prop] = value
        return True

    def get(self, prop: int) -> float:
        return self.props.get(prop, 0)

    def read(self):
        return True, None

    def release(self) -> None:
        pass


def bare_capture(fps: int = 60) -> WebcamCapture:
    """Un WebcamCapture sans ouverture de caméra, pour tester ses méthodes une par une."""
    cap = WebcamCapture.__new__(WebcamCapture)
    cap.camera_index, cap.requested_width, cap.requested_height = 0, 1280, 720
    cap.requested_fps, cap.strict, cap.cap = fps, False, None
    return cap


class TestOrdreDesReglages(unittest.TestCase):
    def test_format_regle_en_dernier_avec_directshow(self):
        fake = FakeCap()
        bare_capture()._configure(fake, cv2.CAP_DSHOW, 1280, 720, True)
        self.assertEqual(fake.calls, [FPS, W, H, FOURCC])
        self.assertEqual(fake.props[FOURCC], MJPG)

    def test_pas_de_format_si_mjpg_non_demande(self):
        fake = FakeCap()
        bare_capture()._configure(fake, cv2.CAP_DSHOW, 640, 480, False)
        self.assertEqual(fake.calls, [FPS, W, H])

    def test_autres_systemes_format_en_premier(self):
        fake = FakeCap()
        bare_capture()._configure(fake, cv2.CAP_ANY, 1280, 720, True)
        self.assertEqual(fake.calls, [FOURCC, W, H, FPS])


class AvecFichierTemporaire(unittest.TestCase):
    """Redirige camera_mode.json vers un dossier temporaire."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self._tmp.name, "camera_mode.json")
        patcher = mock.patch.object(capture, "CAMERA_CACHE_FILE", self.cache)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def write_cache(self, entry: dict) -> None:
        with open(self.cache, "w", encoding="utf-8") as f:
            json.dump({"0:1280x720@60": entry}, f)

    def read_cache(self) -> dict:
        with open(self.cache, "r", encoding="utf-8") as f:
            return json.load(f)["0:1280x720@60"]


class TestModeMemorise(AvecFichierTemporaire):
    def test_ancien_mode_memorise_ignore(self):
        # Entrée écrite par l'ancien code (sans version) : 640x480 retenu à tort
        self.write_cache({"label": "DirectShow 640x480", "backend": "dshow", "w": 640, "h": 480, "mjpg": False})
        self.assertIsNone(bare_capture()._load_cached_mode())

    def test_mode_memorise_relu(self):
        cap = bare_capture()
        cap._save_cached_mode(("DirectShow MJPG", cv2.CAP_DSHOW, 1280, 720, True))
        self.assertEqual(self.read_cache()["v"], CAMERA_CACHE_VERSION)
        self.assertEqual(cap._load_cached_mode(), ("DirectShow MJPG", cv2.CAP_DSHOW, 1280, 720, True))

    def test_fichier_illisible_ignore(self):
        with open(self.cache, "w", encoding="utf-8") as f:
            f.write("{pas du json")
        self.assertIsNone(bare_capture()._load_cached_mode())


class TestChoixDuMode(AvecFichierTemporaire):
    """Recherche complète du mode, avec une fausse caméra et des images/s imposées."""

    def open_with(self, fps_of, has_mjpg: bool = True, strict: bool = False) -> WebcamCapture:
        """Ouvre un WebcamCapture ; `fps_of(largeur, compressé)` donne les images/s de chaque mode."""
        FakeCap.opened = []

        def measure(cap: FakeCap) -> float:
            return float(fps_of(int(cap.get(W)), cap.get(FOURCC) == MJPG))

        with mock.patch.object(FakeCap, "has_mjpg", has_mjpg), \
                mock.patch.object(capture.cv2, "VideoCapture", FakeCap), \
                mock.patch.object(capture.platform, "system", return_value="Windows"), \
                mock.patch.object(WebcamCapture, "_measure_fps", staticmethod(measure)), \
                contextlib.redirect_stdout(io.StringIO()):
            return WebcamCapture(camera_index=0, width=1280, height=720, target_fps=60, strict=strict)

    def test_mjpg_720p_retenu_du_premier_coup(self):
        cam = self.open_with(lambda width, mjpg: 30 if mjpg else (10 if width > 640 else 30))
        self.assertEqual((cam.actual_width, cam.actual_height), (1280, 720))
        self.assertEqual(len(FakeCap.opened), 1)  # pas d'essai de repli inutile
        self.assertEqual(self.read_cache()["w"], 1280)
        self.assertTrue(self.read_cache()["mjpg"])

    def test_piece_sombre_garde_la_resolution_demandee(self):
        # Pose longue : tous les modes plafonnent vers 20 images/s, le 640x480 à peine mieux
        cam = self.open_with(lambda width, mjpg: 20 if width > 640 else 22)
        self.assertEqual(cam.actual_width, 1280)
        self.assertEqual(self.read_cache()["w"], 1280)

    def test_mode_unique_lent_ouvert_une_seule_fois(self):
        # Diagnostics (strict) dans une pièce sombre : un seul mode, gardé sans le rouvrir
        cam = self.open_with(lambda width, mjpg: 20, strict=True)
        self.assertEqual(cam.actual_width, 1280)
        self.assertEqual(len(FakeCap.opened), 1)
        self.assertFalse(os.path.exists(self.cache))  # strict : rien n'est mémorisé

    def test_webcam_sans_mjpg_replie_en_640(self):
        # Non compressé seulement : 720p limité par l'USB à 10 images/s, 640x480 à 30
        cam = self.open_with(lambda width, mjpg: 10 if width > 640 else 30, has_mjpg=False)
        self.assertEqual(cam.actual_width, 640)
        self.assertEqual(self.read_cache()["w"], 640)
        self.assertFalse(self.read_cache()["mjpg"])

    def test_ancien_640_memorise_ne_bloque_plus_le_720p(self):
        self.write_cache({"label": "DirectShow 640x480", "backend": "dshow", "w": 640, "h": 480, "mjpg": False})
        cam = self.open_with(lambda width, mjpg: 30 if mjpg else (10 if width > 640 else 30))
        self.assertEqual(cam.actual_width, 1280)
        self.assertEqual(self.read_cache()["v"], CAMERA_CACHE_VERSION)


if __name__ == "__main__":
    unittest.main()
