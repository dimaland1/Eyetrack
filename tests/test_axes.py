"""Tests du sens des axes envoyés à OpenTrack (et de l'envoi lui-même).

Les sens de référence sont ceux vérifiés en jeu (ETS2, BeamNG) en novembre 2026, sans aucune
inversion cochée dans OpenTrack. Si un test échoue après une modification, c'est qu'un axe a
changé de sens : la caméra du jeu partirait du mauvais côté.

Chaque test part d'un mouvement physique de la tête, traduit en matrice de transformation
MediaPipe, et lit les 6 valeurs réellement envoyées en UDP. L'image est en mode miroir (comme
dans le tracker) : la droite de l'image est la droite de l'utilisateur.

Lancer :  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import socket
import struct
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from face import pose_from_matrix  # noqa: E402
from geometry import ScreenConfig  # noqa: E402
from hotkeys import parse_hotkey, format_hotkey, MOD_CONTROL, MOD_SHIFT  # noqa: E402
from output import OpenTrackSender, OutputConfig  # noqa: E402
import profile as profiles  # noqa: E402

# Repère MediaPipe : x vers la droite de l'image, y vers le haut, z vers la caméra
RIGHT = np.array([1.0, 0.0, 0.0])
UP = np.array([0.0, 1.0, 0.0])
FORWARD = np.array([0.0, 0.0, 1.0])  # direction du nez quand on regarde la caméra
NEUTRAL_Z = -60.0                    # visage à 60 cm de la caméra

SCREEN = ScreenConfig(width_px=1920, height_px=1080)
CENTER = (960.0, 540.0)


def rot(axis: np.ndarray, deg: float) -> np.ndarray:
    """Rotation de `deg` degrés autour de `axis` (règle de la main droite, Rodrigues)."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    th = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def face_matrix(R: np.ndarray | None = None, t=(0.0, 0.0, NEUTRAL_Z)) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = np.eye(3) if R is None else R
    M[:3, 3] = t
    return M


# Mouvements physiques (vérifiés par check_physics ci-dessous)
def turn_right(deg: float) -> np.ndarray:
    """Tourner la tête vers la droite (le nez part vers la droite)."""
    return rot(UP, deg)


def look_down(deg: float) -> np.ndarray:
    """Baisser la tête (le nez descend)."""
    return rot(RIGHT, deg)


def tilt_right(deg: float) -> np.ndarray:
    """Pencher la tête à droite, oreille droite vers l'épaule droite (le haut du crâne part à droite)."""
    return rot(FORWARD, -deg)


class Harness:
    """Émetteur relié à une vraie socket UDP locale : on lit les paquets envoyés."""

    def __init__(self, **cfg):
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("127.0.0.1", 0))
        self.rx.settimeout(1.0)
        base = dict(host="127.0.0.1", port=self.rx.getsockname()[1], send_rate_hz=0.0,
                    max_prediction_s=0.0, output_smoothing_s=0.0, head_yaw_gain=1.0,
                    head_pitch_gain=1.0, gaze_yaw_gain=0.0, gaze_pitch_gain=0.0)
        base.update(cfg)
        self.sender = OpenTrackSender(OutputConfig(**base))

    def recenter_neutral(self) -> None:
        xyz, ypr = pose_from_matrix(face_matrix())
        self.sender.recenter(xyz, ypr, CENTER, SCREEN)

    def send(self, M: np.ndarray, gaze_px=CENTER, gaze_enabled=False) -> dict[str, float]:
        xyz, ypr = pose_from_matrix(M)
        self.sender.update(xyz, ypr, gaze_px, SCREEN, gaze_enabled=gaze_enabled)
        return self.receive()

    def receive(self) -> dict[str, float]:
        data = self.rx.recv(128)
        assert len(data) == 48, f"paquet OpenTrack de {len(data)} octets (48 attendus)"
        return dict(zip(("x", "y", "z", "yaw", "pitch", "roll"), struct.unpack("<6d", data)))

    def close(self) -> None:
        self.sender.close()
        self.rx.close()


class TestPhysics(unittest.TestCase):
    """Garde-fou : les mouvements de test font bien ce que leur nom dit."""

    def test_check_physics(self):
        self.assertGreater((turn_right(30) @ FORWARD)[0], 0.4)   # nez vers la droite
        self.assertLess((look_down(30) @ FORWARD)[1], -0.4)      # nez vers le bas
        self.assertGreater((tilt_right(30) @ UP)[0], 0.4)        # crâne vers la droite


class TestAxes(unittest.TestCase):
    """Sens validés en jeu : chaque mouvement doit produire ce signe dans OpenTrack."""

    def setUp(self):
        self.h = Harness()
        self.h.recenter_neutral()

    def tearDown(self):
        self.h.close()

    def assertOnly(self, out: dict[str, float], axis: str, sign: int, value: float | None = None):
        """`axis` a le signe attendu, les autres axes restent à ~0."""
        self.assertEqual(int(np.sign(out[axis])), sign, f"{axis} = {out[axis]:+.2f} : mauvais sens ({out})")
        if value is not None:
            self.assertAlmostEqual(abs(out[axis]), value, delta=0.05 * value + 1e-3)
        for k, v in out.items():
            if k != axis:
                self.assertAlmostEqual(v, 0.0, delta=0.05, msg=f"{k} parasite = {v:+.3f} ({out})")

    def test_pose_neutre_donne_zero(self):
        out = self.h.send(face_matrix())
        for k, v in out.items():
            self.assertAlmostEqual(v, 0.0, delta=1e-6, msg=k)

    def test_tourner_a_droite_yaw_positif(self):
        self.assertOnly(self.h.send(face_matrix(turn_right(20))), "yaw", +1, 20.0)

    def test_tourner_a_gauche_yaw_negatif(self):
        self.assertOnly(self.h.send(face_matrix(turn_right(-20))), "yaw", -1, 20.0)

    def test_baisser_la_tete_pitch_negatif(self):
        # Tête baissée -> la caméra du jeu descend (avant la correction, elle montait)
        self.assertOnly(self.h.send(face_matrix(look_down(15))), "pitch", -1, 15.0)

    def test_lever_la_tete_pitch_positif(self):
        self.assertOnly(self.h.send(face_matrix(look_down(-15))), "pitch", +1, 15.0)

    def test_pencher_a_droite_roll_negatif(self):
        self.assertOnly(self.h.send(face_matrix(tilt_right(15))), "roll", -1, 15.0)

    def test_se_decaler_a_droite_x_negatif(self):
        # OpenTrack : X négatif = la caméra glisse à droite (pré-inversion de X faite dans le code)
        self.assertOnly(self.h.send(face_matrix(t=(5.0, 0.0, NEUTRAL_Z))), "x", -1, 5.0)

    def test_monter_y_positif(self):
        self.assertOnly(self.h.send(face_matrix(t=(0.0, 4.0, NEUTRAL_Z))), "y", +1, 4.0)

    def test_avancer_vers_l_ecran_z_negatif(self):
        # Avancer le visage faisait reculer la caméra avant la correction
        self.assertOnly(self.h.send(face_matrix(t=(0.0, 0.0, NEUTRAL_Z + 10.0))), "z", -1, 10.0)

    def test_tourner_puis_pencher_sans_faux_pitch(self):
        # Ancien bug (RQDecomp3x3) : tête tournée de 40° puis penchée de 15° -> faux pitch de 12°
        out = self.h.send(face_matrix(turn_right(40) @ tilt_right(15)))
        self.assertAlmostEqual(out["yaw"], 40.0, delta=0.5)
        self.assertAlmostEqual(out["pitch"], 0.0, delta=0.5)
        self.assertAlmostEqual(out["roll"], -15.0, delta=0.5)

    def test_tourner_puis_baisser(self):
        out = self.h.send(face_matrix(turn_right(30) @ look_down(20)))
        self.assertAlmostEqual(out["yaw"], 30.0, delta=0.5)
        self.assertAlmostEqual(out["pitch"], -20.0, delta=0.5)
        self.assertAlmostEqual(out["roll"], 0.0, delta=0.5)


class TestGains(unittest.TestCase):
    def test_gain_de_tete_sur_yaw_et_pitch_seulement(self):
        h = Harness(head_yaw_gain=2.5, head_pitch_gain=1.5)
        try:
            h.recenter_neutral()
            out = h.send(face_matrix(turn_right(10) @ look_down(10) @ tilt_right(10)))
            self.assertAlmostEqual(out["yaw"], 25.0, delta=1.0)
            self.assertAlmostEqual(out["pitch"], -15.0, delta=1.0)
            self.assertAlmostEqual(out["roll"], -10.0, delta=1.0)  # roll non amplifié
        finally:
            h.close()

    def test_inversions_du_profil(self):
        h = Harness(invert_yaw=True, invert_x=True)
        try:
            h.recenter_neutral()
            self.assertLess(h.send(face_matrix(turn_right(20)))["yaw"], 0)
            self.assertGreater(h.send(face_matrix(t=(5.0, 0.0, NEUTRAL_Z)))["x"], 0)
        finally:
            h.close()


class ManualClock(unittest.TestCase):
    """Émetteur piloté avec une horloge choisie (thread non démarré, _tick appelé à la main)."""

    def make(self, **cfg) -> OpenTrackSender:
        base = dict(send_rate_hz=120.0, max_prediction_s=0.0, output_smoothing_s=0.0,
                    head_yaw_gain=1.0, head_pitch_gain=1.0, port=9)  # port 9 = « discard »
        base.update(cfg)
        s = OpenTrackSender(OutputConfig(**base))
        self.addCleanup(s.close)
        xyz, ypr = pose_from_matrix(face_matrix())
        s.recenter(xyz, ypr, CENTER, SCREEN)
        return s

    def run_for(self, s: OpenTrackSender, t0: float, seconds: float, step: float = 1 / 120):
        out = None
        for k in range(int(seconds / step)):
            out = s._tick(t0 + k * step)
        return out


class TestGaze(ManualClock):
    """Le regard tourne la caméra dans le même sens que la tête."""

    def gaze_out(self, gaze_px):
        s = self.make(gaze_yaw_gain=0.5, gaze_pitch_gain=0.5, gaze_smoothing_s=0.1)
        xyz, ypr = pose_from_matrix(face_matrix())
        t0 = time.perf_counter()
        s.update(xyz, ypr, gaze_px, SCREEN, gaze_enabled=True, timestamp=t0)
        return dict(zip(("x", "y", "z", "yaw", "pitch", "roll"), self.run_for(s, t0, 1.5)))

    def test_regard_a_droite_comme_tete_a_droite(self):
        self.assertGreater(self.gaze_out((1900.0, 540.0))["yaw"], 3.0)

    def test_regard_en_bas_comme_tete_baissee(self):
        self.assertLess(self.gaze_out((960.0, 1070.0))["pitch"], -2.0)

    def test_regard_au_centre_sans_effet(self):
        out = self.gaze_out(CENTER)
        self.assertAlmostEqual(out["yaw"], 0.0, delta=1e-6)
        self.assertAlmostEqual(out["pitch"], 0.0, delta=1e-6)


class TestPauseEtPrediction(ManualClock):
    def test_pause_ramene_au_centre_puis_reprise(self):
        s = self.make(pause_fade_s=0.4)
        xyz, ypr = pose_from_matrix(face_matrix(turn_right(20)))
        t0 = time.perf_counter()
        s.update(xyz, ypr, CENTER, SCREEN, gaze_enabled=False, timestamp=t0)
        self.assertAlmostEqual(self.run_for(s, t0, 0.1)[3], 20.0, delta=0.01)
        s.set_paused(True)
        mid = s._tick(t0 + 0.3)  # en plein fondu : ni 20 ni 0
        self.assertTrue(0.5 < mid[3] < 19.5, mid)
        self.assertAlmostEqual(self.run_for(s, t0 + 0.3, 0.6)[3], 0.0, delta=1e-6)
        s.set_paused(False)
        self.assertAlmostEqual(self.run_for(s, t0 + 0.9, 0.6)[3], 20.0, delta=0.01)

    def test_recentrage_progressif(self):
        s = self.make()
        xyz, ypr = pose_from_matrix(face_matrix(turn_right(20)))
        t0 = time.perf_counter()
        s.update(xyz, ypr, CENTER, SCREEN, gaze_enabled=False, timestamp=t0)
        self.assertAlmostEqual(self.run_for(s, t0, 0.1)[3], 20.0, delta=0.01)
        s.recenter(xyz, ypr, CENTER, SCREEN)  # recentrer tête tournée de 20°
        self.assertGreater(s._tick(t0 + 0.1 + 1 / 120)[3], 15.0)  # pas de saut immédiat à 0
        self.assertAlmostEqual(self.run_for(s, t0 + 0.11, 1.0)[3], 0.0, delta=0.05)

    def test_prediction_en_mouvement(self):
        s = self.make(max_prediction_s=0.010)
        t0 = time.perf_counter()
        for k in range(6):  # tête qui tourne à 60°/s, 30 images/s
            t = t0 + k / 30
            xyz, ypr = pose_from_matrix(face_matrix(turn_right(60.0 * k / 30)))
            s.update(xyz, ypr, CENTER, SCREEN, gaze_enabled=False, timestamp=t)
        last = 60.0 * 5 / 30
        yaw = s._tick(t + 0.020)[3]
        self.assertAlmostEqual(yaw, last + 60.0 * 0.010, delta=0.1)  # avance bornée à 10 ms

    def test_pas_de_prediction_au_repos_ni_visage_perdu(self):
        s = self.make(max_prediction_s=0.05)
        t0 = time.perf_counter()
        xyz, ypr = pose_from_matrix(face_matrix(turn_right(10)))
        s.update(xyz, ypr, CENTER, SCREEN, gaze_enabled=False, timestamp=t0)
        s.update(xyz, ypr, CENTER, SCREEN, gaze_enabled=False, timestamp=t0 + 1 / 30)
        self.assertAlmostEqual(s._tick(t0 + 0.08)[3], 10.0, delta=1e-6)
        xyz2, ypr2 = pose_from_matrix(face_matrix(turn_right(20)))
        s.update(xyz2, ypr2, CENTER, SCREEN, gaze_enabled=False, timestamp=t0 + 2 / 30, tracking=False)
        self.assertAlmostEqual(s._tick(t0 + 0.2)[3], 20.0, delta=1e-6)


class TestEnvoiRegulier(unittest.TestCase):
    def test_thread_envoie_a_120_hz(self):
        h = Harness(send_rate_hz=120.0)
        try:
            h.rx.settimeout(0.2)
            xyz, ypr = pose_from_matrix(face_matrix())
            h.sender.update(xyz, ypr, CENTER, SCREEN, gaze_enabled=False)
            h.sender.start()
            t_end = time.perf_counter() + 1.0
            n = 0
            while time.perf_counter() < t_end:
                try:
                    h.rx.recv(128)
                    n += 1
                except socket.timeout:
                    pass
            self.assertGreater(n, 95, f"{n} paquets en 1 s")
            self.assertLess(n, 135, f"{n} paquets en 1 s")
        finally:
            h.close()


class TestRaccourcisEtProfils(unittest.TestCase):
    def test_analyse_des_raccourcis(self):
        self.assertEqual(parse_hotkey("ctrl+shift+f9"), (MOD_CONTROL | MOD_SHIFT, 0x78))
        self.assertEqual(parse_hotkey("Ctrl + Maj + F11"), (MOD_CONTROL | MOD_SHIFT, 0x7A))
        self.assertEqual(parse_hotkey("pause"), (0, 0x13))
        self.assertEqual(parse_hotkey("numpad5"), (0, 0x65))
        self.assertEqual(format_hotkey("ctrl+shift+f9"), "Ctrl+Maj+F9")
        for bad in ("", "ctrl+", "ctrl+shift", "hyper+f9", "ctrl+toto"):
            with self.assertRaises(ValueError, msg=bad):
                parse_hotkey(bad)

    def test_profils_de_jeu_sans_recalibration_au_clic(self):
        for name in ("ets2", "flight_sim", "racing"):
            self.assertFalse(profiles.create_builtin_profile(name).recalib_on_click, name)
            self.assertFalse(profiles.load_profile(name).recalib_on_click, f"profiles/{name}.json")

    def test_raccourcis_des_profils_valides(self):
        for name in profiles.list_profiles():
            p = profiles.load_profile(name)
            for combo in (p.hotkey_recenter, p.hotkey_gaze, p.hotkey_pause):
                if combo:
                    parse_hotkey(combo)


if __name__ == "__main__":
    unittest.main(verbosity=2)
