"""Outil de comparaison des modèles de regard : features concaténées et analyse hors ligne."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("compare_gaze_models", ROOT / "tools" / "compare_gaze_models.py")
cgm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cgm)

from gaze_model import GazeFeatures  # noqa: E402
from geometry import ScreenConfig  # noqa: E402


class FakeNet:
    """Réseau factice à mémoire (comme le pré-filtre temporel) : renvoie le nombre d'images vues."""

    def __init__(self, name, offset, ok=True):
        self.model_name = name
        self.offset = offset
        self.ok = ok
        self.seen = []

    def estimate(self, face_data, norm_data=None):
        self.seen.append(int(norm_data.face_patch[0, 0, 0]))
        if not self.ok:
            return GazeFeatures(confidence=0.0)
        n = float(len(self.seen))
        return GazeFeatures(pitch_yaw_deg=(n, self.offset),
                            raw_features=np.array([10.0, 20.0, 0.5, 0.4]) + self.offset, confidence=1.0)


class FaceStub:
    def __init__(self, detected=True):
        self.detected = detected
        self.landmarks_px = np.zeros((478, 2))


class NormStub:
    def __init__(self, k, valid=True):
        self.is_valid = valid
        self.face_patch = np.full((4, 4, 3), k, dtype=np.uint8)
        self.rotation_matrix = np.eye(3)
        self.origin_3d_mm = np.array([0.0, 0.0, -600.0])


class SequenceTest(unittest.TestCase):
    def test_enregistrement_puis_reseaux_dans_l_ordre(self):
        rec = cgm.RecordingEstimator()
        indices = []
        for k in range(6):
            face_ok = k != 2  # image 2 : visage perdu
            g = rec.estimate(FaceStub(face_ok), NormStub(k))
            indices.append(int(g.raw_features[0]))
            self.assertEqual(g.confidence, 1.0 if face_ok else 0.0)
        self.assertEqual(indices, list(range(6)))
        self.assertIsNone(rec.frames[2])

        nets = [FakeNet("a.onnx", 0.0), FakeNet("b.onnx", 1.0), FakeNet("c.onnx", 0.0, ok=False)]
        feats, ms = cgm.run_networks(rec.frames, nets)
        self.assertEqual(feats.shape, (6, 3 * cgm.FEATS_PER_MODEL))
        # Chaque réseau voit toute la séquence dans l'ordre, sans l'image sans visage (filtre inchangé)
        self.assertEqual(nets[0].seen, [0, 1, 3, 4, 5])
        self.assertEqual(nets[1].seen, [0, 1, 3, 4, 5])
        np.testing.assert_allclose(feats[0, :6], [10, 20, 0.5, 0.4, 1, 0])
        np.testing.assert_allclose(feats[5, 6:12], [11, 21, 1.5, 1.4, 5, 1])
        self.assertTrue(np.isnan(feats[2]).all())         # image sans visage
        self.assertTrue(np.isnan(feats[:, 12:]).all())    # réseau sans résultat : NaN, pas de faux zéro
        self.assertEqual(len(ms), 3)


def fake_session(path, noise_mm=(15.0, 60.0), passes=2, n=45, seed=1):
    """Séance simulée : le second modèle est 4x plus bruité que le premier."""
    rng = np.random.default_rng(seed)
    screen = ScreenConfig(width_px=2560, height_px=1440, diagonal_inches=27.0, distance_cm=60.0)
    G, H, Y, point, pass_ids = [], [], [], [], []
    for p in range(passes):
        for idx in (range(9) if p == 0 else cgm.PASS2_ORDER):
            nx, ny = [(x, y) for y in (0.1, 0.5, 0.9) for x in (0.1, 0.5, 0.9)][idx]
            tx, ty = screen.norm_to_px(nx, ny)
            hx = (tx - 1280) * screen.mm_per_px_x
            hy = ty * screen.mm_per_px_y + 20.0
            rows = []
            for sigma in noise_mm:
                block = np.column_stack([
                    hx + rng.normal(0, sigma, n), hy + rng.normal(0, sigma, n),
                    0.5 + 0.0004 * hx + rng.normal(0, 0.004, n), 0.45 + 0.0003 * hy + rng.normal(0, 0.004, n),
                    rng.normal(0, 1, n), rng.normal(0, 1, n)])
                rows.append(block)
            G.append(np.hstack(rows))
            H.append(rng.normal(0, 0.3, (n, 2)))
            Y.append(np.tile([tx, ty], (n, 1)))
            point.append(np.full(n, idx))
            pass_ids.append(np.full(n, p))
    np.savez(path, models=np.array(["calme.onnx", "bruite.onnx"]), G=np.vstack(G), H=np.vstack(H),
             Y=np.vstack(Y).astype(float), point=np.concatenate(point), pass_id=np.concatenate(pass_ids),
             screen=np.array([2560, 1440, 27.0, 60.0]), camera=np.array([1280, 720, 30.0]),
             infer_ms=np.array([5.0, 6.0]), session_fps=np.array(29.0), date=np.array("2026-11-15 12:00"))


class AnalyseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "s.npz"

    def tearDown(self):
        self.tmp.cleanup()

    def test_le_modele_bruite_est_moins_bon(self):
        fake_session(self.path)
        report = cgm.analyse(self.path)
        self.assertIn("calme.onnx", report)
        bruite = next(line for line in report.splitlines() if line.strip().startswith("bruite.onnx") and "IC 95" in line)
        self.assertIn("moins bon", bruite)

    def test_une_seule_serie(self):
        fake_session(self.path, passes=1)
        report = cgm.analyse(self.path)
        self.assertIn("1 série(s)", report)
        self.assertNotIn("IC 95", report)  # pas de série croisée sans seconde série


if __name__ == "__main__":
    unittest.main()
