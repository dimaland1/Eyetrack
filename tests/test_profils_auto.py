"""Changement de profil automatique selon le jeu lancé."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import profile as profiles  # noqa: E402
import winapp  # noqa: E402
from profile import AutoProfile, Profile, detect_game, game_name, game_processes_of  # noqa: E402

MAPPING = {"ets2": ["eurotrucks2.exe", "amtrucks.exe"], "beamng": ["beamng.drive.x64.exe"]}
ETS2 = ("ets2", "eurotrucks2.exe")
BEAMNG = ("beamng", "beamng.drive.x64.exe")


class TestDetection(unittest.TestCase):
    def test_jeu_reconnu(self):
        self.assertEqual(detect_game({"explorer.exe", "eurotrucks2.exe"}, MAPPING), ETS2)
        self.assertEqual(detect_game({"beamng.drive.x64.exe"}, MAPPING), BEAMNG)
        self.assertIsNone(detect_game({"explorer.exe", "chrome.exe"}, MAPPING))

    def test_liste_du_profil(self):
        self.assertIn("eurotrucks2.exe", game_processes_of(Profile(name="ets2")))   # liste d'Eyetrack
        self.assertEqual(game_processes_of(Profile(name="ets2", game_processes=[])), [])  # désactivé
        self.assertEqual(game_processes_of(Profile(name="x", game_processes=["Jeu.EXE"])), ["jeu.exe"])
        self.assertEqual(game_processes_of(Profile(name="inconnu")), [])

    def test_profils_fournis_pour_chaque_jeu(self):
        # Chaque profil de la liste des jeux existe dans les profils fournis
        bundled = {f[:-5] for f in os.listdir(os.path.join(ROOT, "profiles")) if f.endswith(".json")}
        self.assertLessEqual(set(profiles.GAME_PROCESSES), bundled)
        self.assertEqual(game_name("EuroTrucks2.exe"), "Euro Truck Simulator 2")

    def test_programmes_lances(self):
        self.assertIn(os.path.basename(sys.executable).lower(), winapp.running_processes())


class TestAutoProfile(unittest.TestCase):
    def test_jeu_lance_puis_ferme(self):
        a = AutoProfile()
        self.assertIsNone(a.update(None, "default"))
        self.assertEqual(a.update(ETS2, "default"), "ets2")     # jeu lancé : son profil
        self.assertIsNone(a.update(ETS2, "ets2"))               # toujours lancé : rien
        self.assertEqual(a.update(None, "ets2"), "default")     # fermé : profil d'avant

    def test_deja_sur_le_bon_profil(self):
        a = AutoProfile()
        self.assertIsNone(a.update(ETS2, "ets2"))
        self.assertIsNone(a.update(None, "ets2"))

    def test_changement_a_la_main_respecte(self):
        a = AutoProfile()
        self.assertEqual(a.update(ETS2, "default"), "ets2")
        self.assertIsNone(a.update(ETS2, "racing"))   # changé à la main pendant le jeu : gardé
        self.assertIsNone(a.update(ETS2, "racing"))
        self.assertIsNone(a.update(None, "racing"))   # à la fermeture : pas de retour imposé

    def test_un_jeu_puis_un_autre(self):
        a = AutoProfile()
        self.assertEqual(a.update(ETS2, "default"), "ets2")
        self.assertEqual(a.update(BEAMNG, "ets2"), "beamng")
        self.assertEqual(a.update(None, "beamng"), "default")  # retour au profil d'avant les jeux


if __name__ == "__main__":
    unittest.main(verbosity=2)
