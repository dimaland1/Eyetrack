"""Modèle de regard de précision : jamais fourni, téléchargé seulement après accord, vérifié."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import gaze_model  # noqa: E402


class TestModele(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "source.onnx"
        self.src.write_bytes(b"faux modele" * 1000)
        self.sha = hashlib.sha256(self.src.read_bytes()).hexdigest()
        self.url = self.src.as_uri()

    def tearDown(self):
        self.tmp.cleanup()

    def test_telechargement_verifie(self):
        dest = self.dir / "models" / gaze_model.MODEL_FILE
        out = gaze_model.download_gaze_model(str(dest), url=self.url, sha256=self.sha)
        self.assertEqual(out, str(dest))
        self.assertEqual(dest.read_bytes(), self.src.read_bytes())
        self.assertFalse(Path(str(dest) + ".part").exists())

    def test_fichier_different_refuse(self):
        dest = self.dir / "models" / gaze_model.MODEL_FILE
        with self.assertRaises(ValueError):
            gaze_model.download_gaze_model(str(dest), url=self.url, sha256="0" * 64)
        self.assertFalse(dest.exists())
        self.assertFalse(Path(str(dest) + ".part").exists())

    def test_jamais_telecharge_automatiquement(self):
        with self.assertRaises(FileNotFoundError):
            gaze_model.NetGazeEstimator(model_path=str(self.dir / "absent.onnx"))

    def test_hors_du_paquet(self):
        # Ni dans le paquet PyInstaller, ni dans l'installeur (licence non commerciale d'un tiers)
        with open(os.path.join(ROOT, "eyetrack.spec"), encoding="utf-8") as f:
            self.assertNotIn(gaze_model.MODEL_FILE, f.read())
        iss = os.path.join(ROOT, "installer", "eyetrack.iss")
        if os.path.exists(iss):
            with open(iss, encoding="utf-8") as f:
                text = f.read()
            self.assertNotIn("Source: \"" + gaze_model.MODEL_FILE, text)
            self.assertIn(gaze_model.MODEL_SHA256, text)  # même empreinte que l'application
            self.assertIn(gaze_model.MODEL_URL, text)

    def test_empreinte_du_modele_local(self):
        local = os.path.join(ROOT, "models", gaze_model.MODEL_FILE)
        if not os.path.exists(local):
            self.skipTest("modèle non téléchargé sur ce PC")
        with open(local, "rb") as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), gaze_model.MODEL_SHA256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
