"""Tests de la calibration guidée : note de qualité, points à refaire, reflets, sauvegarde."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from calibration import (  # noqa: E402
    AffineGazeModel, CalibrationManager, classify_points, draw_review, evaluate_points,
    points_to_redo, quality_label,
)
from face import GLARE_STRONG, FaceData, glare_label, glare_on_iris  # noqa: E402
from geometry import ScreenConfig  # noqa: E402

SC = ScreenConfig(width_px=2560, height_px=1440, diagonal_inches=27.0, distance_cm=60.0)


def synthetic_points(bad: int | None = None, noise_mm: float = 8.0, seed: int = 0):
    """9 points : point d'impact (mm) ≈ cible + bruit ; `bad` = regard posé sur une autre cible."""
    rng = np.random.default_rng(seed)
    mgr = CalibrationManager(SC, mode="net", num_points=9)
    targets = [np.array(SC.norm_to_px(nx, ny), dtype=float) for nx, ny in mgr.points_norm]
    pts = []
    for i, t in enumerate(targets):
        looked = targets[(i + 4) % 9] if i == bad else t
        hit = np.array([(looked[0] - SC.width_px / 2) * SC.mm_per_px_x, looked[1] * SC.mm_per_px_y])
        G = np.column_stack([hit[0] + rng.normal(0, noise_mm, 40), hit[1] + rng.normal(0, noise_mm, 40),
                             0.5 + hit[0] / 4000 + rng.normal(0, 0.01, 40), 0.5 + hit[1] / 4000 + rng.normal(0, 0.01, 40)])
        pts.append((G, np.zeros((40, 2)), t))
    return pts


class TestNote(unittest.TestCase):
    def test_libelles(self):
        self.assertEqual(quality_label(3.9), "bonne")
        self.assertEqual(quality_label(6.5), "moyenne")
        self.assertEqual(quality_label(9.0), "à refaire")
        self.assertEqual(quality_label(float("nan")), "inconnue")
        self.assertEqual(quality_label(8.2, num_points=5), "bonne")  # 5 points : seuils élargis

    def test_points_a_refaire(self):
        errs = [1.4, 3.8, 8.0, 5.5, 6.0, 2.0, 5.2, 4.6, 6.6]  # vraie calibration avec lunettes
        self.assertNotIn("bad", classify_points(errs))
        self.assertEqual(points_to_redo(errs), [2])           # sinon : le moins bon
        errs_bad = [4.0, 4.5, 21.9, 5.0, 4.2, 3.9, 4.8, 9.5, 4.1]
        self.assertEqual(points_to_redo(errs_bad), [2])       # voisin gonflé à 9,5° : épargné
        errs_two = [4.0, 4.5, 21.9, 5.0, 4.2, 3.9, 4.8, 13.0, 4.1]
        self.assertEqual(points_to_redo(errs_two), [2, 7])    # au-delà de 10° : refait aussi

    def test_calibration_propre_bien_notee(self):
        errs, _, mean = evaluate_points("net", synthetic_points(), SC)
        self.assertEqual(quality_label(mean), "bonne", f"{mean:.1f}° {errs}")
        self.assertNotIn("bad", classify_points(errs))

    def test_point_rate_repere(self):
        for bad in (0, 4, 8):
            errs, _, mean = evaluate_points("net", synthetic_points(bad=bad), SC)
            self.assertIn(bad, points_to_redo(errs), f"point {bad + 1} : {[round(e, 1) for e in errs]}")
            self.assertEqual(int(np.argmax(errs)), bad)

    def test_ecran_de_resultats(self):
        pts = synthetic_points(bad=2)
        errs, preds, mean = evaluate_points("net", pts, SC)
        points = [{"target": p[2], "glare": 0.3 if i == 5 else 0.0} for i, p in enumerate(pts)]
        img = draw_review(SC, points, errs, preds, mean)
        self.assertEqual(img.shape, (SC.height_px, SC.width_px, 3))
        self.assertGreater(int(img.max()), 0)


class TestGrille(unittest.TestCase):
    def test_cibles_a_10_pourcent_des_bords(self):
        for n in (5, 9):
            pts = CalibrationManager(SC, mode="net", num_points=n).points_norm
            self.assertEqual(len(pts), n)
            self.assertAlmostEqual(min(min(p) for p in pts), 0.10)
            self.assertAlmostEqual(max(max(p) for p in pts), 0.90)


class TestSauvegarde(unittest.TestCase):
    def test_note_sauvegardee_et_rechargee(self):
        pts = synthetic_points()
        m = AffineGazeModel(mode="net", alpha=1e-2)
        m.fit(np.vstack([p[0] for p in pts]), np.vstack([p[1] for p in pts]),
              np.vstack([np.tile(p[2], (40, 1)) for p in pts]))
        m.quality_deg = 3.7
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "c.npz")
            m.save(f)
            mgr = CalibrationManager(SC, mode="net")
            self.assertTrue(mgr.load(f))
            self.assertAlmostEqual(mgr.quality_deg, 3.7)
            self.assertEqual(mgr.quality_label, "bonne")
            # Ancienne calibration sans note : chargée, note « inconnue »
            data = dict(np.load(f))
            data.pop("quality_deg")
            np.savez(f, **data)
            mgr2 = CalibrationManager(SC, mode="net")
            self.assertTrue(mgr2.load(f))
            self.assertEqual(mgr2.quality_label, "inconnue")


class TestReflets(unittest.TestCase):
    def frame_and_face(self, glare_radius: int = 0):
        frame = np.full((480, 640, 3), 150, np.uint8)
        for cx in (300, 340):
            cv2.circle(frame, (cx, 200), 6, (40, 30, 20), -1)       # iris sombre
            cv2.circle(frame, (cx + 1, 199), 1, (255, 255, 255), -1)  # reflet normal de la cornée
            if glare_radius:
                cv2.circle(frame, (cx + 2, 202), glare_radius, (255, 255, 255), -1)  # reflet de verre
        face = FaceData(detected=True, left_iris_px=(300.0, 200.0), right_iris_px=(340.0, 200.0),
                        iris_diameter_px=12.0)
        return frame, face

    def test_sans_reflet(self):
        frame, face = self.frame_and_face()
        g = glare_on_iris(frame, face)
        self.assertLess(g, 0.05)
        self.assertEqual(glare_label(g), "aucun")

    def test_reflet_de_lunettes(self):
        frame, face = self.frame_and_face(glare_radius=6)
        g = glare_on_iris(frame, face)
        self.assertGreater(g, GLARE_STRONG)
        self.assertEqual(glare_label(g), "gênant")

    def test_sans_visage(self):
        frame, _ = self.frame_and_face(glare_radius=6)
        self.assertEqual(glare_on_iris(frame, FaceData(detected=False)), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
