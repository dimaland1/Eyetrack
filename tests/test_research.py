"""Tests unitaires du package de recherche (eye_crop, lag, modèles, buffer clics, banc)."""

from __future__ import annotations

from collections import deque
import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from face import FaceData, LEFT_IMAGE_EYE, RIGHT_IMAGE_EYE  # noqa: E402
from geometry import ScreenConfig  # noqa: E402
from research.augmentations import augment_eye_crop, preprocess_eye_batch_gray_norm, preprocess_eye_batch_rgb  # noqa: E402
from research.collect_clicks import BufferedFrame, select_preclick_frame  # noqa: E402
from research.eye_crop import extract_aligned_eye_crop, extract_binocular_sample  # noqa: E402
from research.lag_estimation import (  # noqa: E402
    estimate_visuomotor_lag,
    lissajous_pursuit_target,
    shift_targets_by_lag,
)
from research.models import (  # noqa: E402
    DifferentialEyeNet,
    DualEyePersonalNet,
    gaze_angles_deg_to_screen_px,
    screen_px_to_gaze_angles_deg,
)
from research.synthetic_bench import generate_synthetic_session  # noqa: E402
from research.train import DifferentialPairDataset, PersonalEyeDataset, export_personal_onnx  # noqa: E402


class TestEyeCrop(unittest.TestCase):
    def test_rotation_alignment_keeps_eye_horizontal_and_upright(self):
        # Crée une image avec un repère coloré au-dessus de l'œil pour vérifier l'absence d'inversion
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        # Demi-disque supérieur vert autour de l'œil incliné à ~20°
        theta = np.radians(20.0)
        cx, cy = 120.0, 100.0
        p_outer = np.array([cx - 25 * np.cos(theta), cy - 25 * np.sin(theta)])
        p_inner = np.array([cx + 25 * np.cos(theta), cy + 25 * np.sin(theta)])
        # Point haut (paupière supérieure) décalé vers le haut perpendiculairement
        normal_up = np.array([np.sin(theta), -np.cos(theta)])
        p_top = np.array([cx, cy]) + 8.0 * normal_up
        p_bot = np.array([cx, cy]) - 8.0 * normal_up

        # Dessine bande verte côté paupière supérieure, bleue côté paupière inférieure
        for d in range(-25, 26):
            pt_u = (np.array([cx, cy]) + d * np.array([np.cos(theta), np.sin(theta)]) + 6 * normal_up).astype(int)
            pt_l = (np.array([cx, cy]) + d * np.array([np.cos(theta), np.sin(theta)]) - 6 * normal_up).astype(int)
            img[pt_u[1] - 2 : pt_u[1] + 3, pt_u[0] - 2 : pt_u[0] + 3] = (0, 255, 0)
            img[pt_l[1] - 2 : pt_l[1] + 3, pt_l[0] - 2 : pt_l[0] + 3] = (255, 0, 0)

        pts = np.zeros((478, 2), dtype=np.float32)
        pts[LEFT_IMAGE_EYE["outer"]] = p_outer
        pts[LEFT_IMAGE_EYE["inner"]] = p_inner
        pts[LEFT_IMAGE_EYE["top"]] = p_top
        pts[LEFT_IMAGE_EYE["bottom"]] = p_bot

        patch, _, ew = extract_aligned_eye_crop(img, pts, LEFT_IMAGE_EYE)
        self.assertEqual(patch.shape, (48, 96, 3))
        self.assertGreater(ew, 45.0)
        # La moitié haute du patch doit être verte (canal 1 > canal 0), la moitié basse bleue
        top_half = patch[:20, 20:76].mean(axis=(0, 1))
        bot_half = patch[28:, 20:76].mean(axis=(0, 1))
        self.assertGreater(top_half[1], top_half[0])
        self.assertGreater(bot_half[0], bot_half[1])

    def test_binocular_sample_dimensions(self):
        img = np.full((360, 640, 3), 120, dtype=np.uint8)
        pts = np.zeros((478, 2), dtype=np.float32)
        pts[33], pts[133] = [200, 150], [250, 150]
        pts[159], pts[145] = [225, 142], [225, 158]
        pts[362], pts[263] = [390, 150], [440, 150]
        pts[386], pts[374] = [415, 142], [415, 158]
        face = FaceData(
            detected=True,
            landmarks_px=pts,
            pose_xyz=(1.0, -2.0, 60.0),
            pose_ypr=(5.0, -3.0, 1.0),
            iris_diameter_px=20.0,
            left_iris_px=(225.0, 150.0),
            right_iris_px=(415.0, 150.0),
            eye_openness=0.32,
        )
        sample = extract_binocular_sample(img, face)
        self.assertTrue(sample.valid)
        self.assertEqual(sample.eye_crops.shape, (2, 48, 96, 3))
        self.assertEqual(sample.geom_vec.shape, (9,))


class TestLagAndPursuit(unittest.TestCase):
    def test_pursuit_speed_bounded_and_lag_recovered(self):
        sc = ScreenConfig(width_px=2560, height_px=1440, diagonal_inches=27.0, distance_cm=60.0)
        t = np.linspace(0.0, 30.0, 900)
        x, y, spd = lissajous_pursuit_target(t, sc)
        self.assertTrue(np.all((x >= 0) & (x <= sc.width_px)))
        self.assertTrue(np.all((y >= 0) & (y <= sc.height_px)))
        self.assertLess(float(np.max(spd)), 25.0)  # adapté à la poursuite lisse

        # Simule des features qui suivent la cible avec un retard exact de 80 ms
        tgt = np.column_stack([x, y])
        true_lag_ms = 80.0
        shifted, _ = shift_targets_by_lag(t, tgt, true_lag_ms)
        feats = np.column_stack([shifted[:, 0] * 0.5 + 10.0, shifted[:, 1] * 0.5 - 5.0])
        res = estimate_visuomotor_lag(t, feats, tgt, lag_range_ms=(-20.0, 160.0), step_ms=5.0)
        self.assertAlmostEqual(res["best_lag_ms"], true_lag_ms, delta=10.0)
        self.assertLess(res["min_rmse_px"], res["zero_lag_rmse_px"])


class TestModelsAndGeometry(unittest.TestCase):
    def test_screen_angle_round_trip(self):
        ori = np.array([[15.0, -25.0, 610.0], [-30.0, 10.0, 580.0]])
        tgt = np.array([[400.0, 300.0], [2100.0, 1250.0]])
        sc = ScreenConfig(width_px=2560, height_px=1440, diagonal_inches=27.0, distance_cm=60.0)
        ang = screen_px_to_gaze_angles_deg(tgt, ori, sc.width_px, sc.height_px, sc.mm_per_px_x, sc.mm_per_px_y)
        rec = gaze_angles_deg_to_screen_px(ang, ori, sc.width_px, sc.height_px, sc.mm_per_px_x, sc.mm_per_px_y)
        np.testing.assert_allclose(rec, tgt, atol=1e-6)

    def test_differential_antisymmetry_and_zero_self_delta(self):
        model = DifferentialEyeNet()
        model.eval()
        el_q = torch.randn(4, 1, 48, 96)
        er_q = torch.randn(4, 1, 48, 96)
        g_q = torch.randn(4, 9)
        el_r = torch.randn(4, 1, 48, 96)
        er_r = torch.randn(4, 1, 48, 96)
        g_r = torch.randn(4, 9)

        with torch.no_grad():
            self_delta = model(el_q, er_q, g_q, el_q, er_q, g_q)
            d_qr = model(el_q, er_q, g_q, el_r, er_r, g_r)
            d_rq = model(el_r, er_r, g_r, el_q, er_q, g_q)

        np.testing.assert_allclose(self_delta.numpy(), 0.0, atol=1e-5)
        np.testing.assert_allclose(d_qr.numpy(), -d_rq.numpy(), atol=1e-5)

    def test_personal_net_forward_and_preprocessors(self):
        sess = generate_synthetic_session(subject_id=0, n_samples=8, seed=1)
        rng = np.random.default_rng(0)
        aug = augment_eye_crop(sess["eye_crops"][0, 0], rng)
        self.assertEqual(aug.shape, (48, 96, 3))
        rgb = preprocess_eye_batch_rgb(sess["eye_crops"])
        gray = preprocess_eye_batch_gray_norm(sess["eye_crops"])
        self.assertEqual(rgb.shape, (8, 2, 3, 48, 96))
        self.assertEqual(gray.shape, (8, 2, 1, 48, 96))

        net = DualEyePersonalNet()
        net.eval()
        out = net(torch.from_numpy(rgb[:, 0]), torch.from_numpy(rgb[:, 1]), torch.from_numpy(sess["geom_vec"]))
        self.assertEqual(out.shape, (8, 2))

        # Vérifie l'export ONNX et l'équivalence numérique sous ONNX Runtime
        import tempfile
        import onnxruntime as ort
        with tempfile.TemporaryDirectory() as td:
            onnx_p = os.path.join(td, "personal.onnx")
            export_personal_onnx(net, onnx_p)
            ort_sess = ort.InferenceSession(onnx_p, providers=["CPUExecutionProvider"])
            ort_out = ort_sess.run(
                None,
                {"eye_left": rgb[:, 0], "eye_right": rgb[:, 1], "geom": sess["geom_vec"]},
            )[0]
            np.testing.assert_allclose(out.detach().numpy(), ort_out, atol=1e-4)

        p_ds = PersonalEyeDataset(sess, augment=True, aug_copies=2, seed=0)
        self.assertEqual(len(p_ds), 16)
        d_ds = DifferentialPairDataset(sess, n_pairs=12, augment=True, aug_copies=2, seed=0)
        self.assertEqual(len(d_ds), 12)


class TestPreclickBuffer(unittest.TestCase):
    def test_select_preclick_picks_75ms_prior_valid_frame(self):
        buf = deque()
        for k in range(20):
            ts = 10.0 + k * 0.030  # ~33 fps
            valid = k != 15  # 10.450 invalide (clignement)
            buf.append(
                BufferedFrame(
                    ts=ts,
                    eye_crops=np.zeros((2, 48, 96, 3), dtype=np.uint8),
                    geom_vec=np.zeros(9, dtype=np.float32),
                    pose_6dof=np.zeros(6),
                    origin_3d_mm=np.zeros(3),
                    iris_ratios=np.zeros(2),
                    net_pitch_yaw=np.zeros(2),
                    net_hit_mm=np.zeros(2),
                    eye_metrics=np.zeros(4),
                    valid=valid,
                )
            )
        # Clic à 10.525 s -> cible pré-clic = 10.450 s (invalide), doit prendre voisin valide (10.420 ou 10.480)
        chosen = select_preclick_frame(buf, click_ts=10.525, pre_lag_ms=75.0, max_delta_ms=60.0)
        self.assertIsNotNone(chosen)
        self.assertTrue(chosen.valid)
        self.assertLessEqual(abs(chosen.ts - 10.450), 0.035)


if __name__ == "__main__":
    unittest.main()
