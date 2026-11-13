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
    HEAD_PITCH_LEAKAGE_MM, HEAD_YAW_LEAKAGE_MM, IRIS_PITCH_LEAKAGE_RAD, IRIS_YAW_LEAKAGE_RAD,
    AffineGazeModel, CalibrationManager, classify_points, draw_intro, draw_review,
    evaluate_points, points_to_redo, quality_label,
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

    def test_calibration_liee_au_modele_de_regard(self):
        # Une calibration faite avec un autre modèle de regard (ou l'ancien calcul) est refusée :
        # l'écran d'accueil propose alors d'en refaire une
        pts = synthetic_points()
        m = AffineGazeModel(mode="net", alpha=1e-2)
        m.fit(np.vstack([p[0] for p in pts]), np.vstack([p[1] for p in pts]),
              np.vstack([np.tile(p[2], (40, 1)) for p in pts]))
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "c.npz")
            m.save(f, model_name="mobilenetv2_gaze.onnx")
            self.assertTrue(CalibrationManager(SC, mode="net", model_name="mobilenetv2_gaze.onnx").load(f))
            self.assertFalse(CalibrationManager(SC, mode="net", model_name="mobileone_s0_gaze.onnx").load(f))
            self.assertTrue(CalibrationManager(SC, mode="net").load(f))  # outils : pas de vérification
            data = dict(np.load(f))
            data["version"] = np.array(5)  # calcul d'avant la compensation de la fuite de tête
            np.savez(f, **data)
            self.assertFalse(CalibrationManager(SC, mode="net", model_name="mobilenetv2_gaze.onnx").load(f))


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


class TestAccueil(unittest.TestCase):
    """Écran d'accueil du premier lancement (calibration proposée)."""

    def test_tient_dans_l_ecran(self):
        frame = np.full((480, 640, 3), 77, np.uint8)
        for w, h in [(1280, 720), (1366, 768), (1920, 1080), (2560, 1440), (3840, 2160)]:
            img = draw_intro(w, h, frame, True, 0.0, 9)
            self.assertEqual(img.shape, (h, w, 3))
            rows = np.where(img.max(axis=(1, 2)) > 0)[0]
            cols = np.where(img.max(axis=(0, 2)) > 0)[0]
            self.assertLess(rows.max(), h - 10, f"{w}x{h} : texte coupé en bas")
            self.assertLess(cols.max(), w - 10, f"{w}x{h} : texte coupé à droite")
            self.assertTrue((img == 77).all(axis=2).any(), f"{w}x{h} : image caméra absente")

    def test_cadre_vert_ou_rouge(self):
        frame = np.full((480, 640, 3), 77, np.uint8)
        ok = draw_intro(1920, 1080, frame, True, 0.0)
        lost = draw_intro(1920, 1080, frame, False, 0.0)
        self.assertTrue((ok == (60, 200, 60)).all(axis=2).any())
        self.assertFalse((ok == (60, 60, 230)).all(axis=2).any())
        self.assertTrue((lost == (60, 60, 230)).all(axis=2).any())
        self.assertEqual(draw_intro(800, 600, None, False, 0.0).shape, (600, 800, 3))

    def test_echap_remet_a_plus_tard(self):
        from unittest import mock
        import calibration
        frames = []

        def capture():
            frames.append(1)
            return True, np.full((480, 640, 3), 77, np.uint8), 0.0

        detector = mock.Mock()
        detector.process.return_value = FaceData(detected=False)
        mgr = CalibrationManager(SC, mode="net", num_points=9)
        keys = iter([255, 255, 27])  # rien, rien, puis Échap
        with mock.patch.multiple(calibration.cv2, namedWindow=mock.DEFAULT, setWindowProperty=mock.DEFAULT,
                                 imshow=mock.DEFAULT, destroyWindow=mock.DEFAULT,
                                 waitKey=mock.Mock(side_effect=lambda *_: next(keys, 27))),                 mock.patch.object(calibration.winapp, "bring_to_front", return_value=True):
            done = mgr.run_interactive_calibration(capture, detector, gaze_estimator=None, intro=True)
        self.assertFalse(done)
        self.assertFalse(mgr.is_calibrated)
        self.assertGreaterEqual(len(frames), 2)  # la caméra tourne pendant l'accueil


class TestPrecisionPiste1(unittest.TestCase):
    """Vérifie la compensation VOR de la tête et le rejet des micro-saccades (Piste 1)."""

    def test_compensation_fuite_rotation_tete(self):
        pts = synthetic_points(noise_mm=5.0)
        # Sur le point 6 (bas-gauche), l'utilisateur tourne la tête de +7° en yaw et +10° en pitch :
        # le réseau subit la fuite VOR (+500*tan(yaw) en X, -320*tan(pitch) en Y).
        G6, _, t6 = pts[6]
        H6 = np.column_stack([np.full(len(G6), 7.0), np.full(len(G6), 10.0)])
        G6_leaked = G6.copy()
        yaw_tan = np.tan(np.radians(H6[:, 0]))
        pitch_tan = np.tan(np.radians(H6[:, 1]))
        G6_leaked[:, 0] += HEAD_YAW_LEAKAGE_MM * yaw_tan
        G6_leaked[:, 1] -= HEAD_PITCH_LEAKAGE_MM * pitch_tan
        G6_leaked[:, 2] -= IRIS_YAW_LEAKAGE_RAD * yaw_tan
        G6_leaked[:, 3] -= IRIS_PITCH_LEAKAGE_RAD * pitch_tan
        pts[6] = (G6_leaked, H6, t6)

        errs, _, mean = evaluate_points("net", pts, SC)
        self.assertLess(errs[6], 2.5, f"Fuite de tête non compensée au point 7 : {errs[6]:.2f}°")
        self.assertLess(mean, 2.0, f"Moyenne globale dégradée : {mean:.2f}°")

    def test_rejet_outliers_saccades(self):
        pts = synthetic_points(noise_mm=5.0)
        # 15 % des images de chaque cible ont une saccade parasite de +250 mm
        corrupted = []
        for G, H, t in pts:
            Gc = G.copy()
            Gc[:6, 0] += 250.0
            Gc[:6, 1] -= 200.0
            corrupted.append((Gc, H, t))
        _, _, mean = evaluate_points("net", corrupted, SC)
        self.assertLess(mean, 2.5, f"Les saccades parasites ont faussé le fit : {mean:.2f}°")


if __name__ == "__main__":
    unittest.main(verbosity=2)
