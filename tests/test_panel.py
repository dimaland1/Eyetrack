"""Tests du panneau de contrôle : les curseurs correspondent bien aux réglages des profils."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import profile as profiles  # noqa: E402
from panel import ADVANCED_SLIDERS, CHECKS, ESSENTIAL_SLIDERS, PANEL_FIELDS, PREVIEW_WIDTH, make_preview  # noqa: E402


class TestPanel(unittest.TestCase):
    def test_vignette_camera(self):
        frame = np.full((480, 640, 3), 128, np.uint8)
        rgb = make_preview(frame, True, [(320.0, 240.0)])
        self.assertEqual(rgb.shape, (240, PREVIEW_WIDTH, 3))  # proportions conservées
        self.assertEqual(tuple(rgb[120, 160]), (255, 255, 0))   # point jaune sur l'iris (RGB)
        self.assertEqual(tuple(rgb[0, 0]), (40, 180, 40))       # cadre vert : visage suivi
        lost = make_preview(frame, False, [(320.0, 240.0)])
        self.assertEqual(tuple(lost[0, 0]), (220, 40, 40))      # cadre rouge : visage perdu
        self.assertEqual(tuple(lost[120, 160]), (128, 128, 128))  # pas de point sans visage
        blink = make_preview(frame, True, [], blinking=True)
        self.assertEqual(tuple(blink[0, 0]), (255, 140, 0))     # cadre orange : clignement

    def test_champs_du_panneau_existent_dans_le_profil(self):
        fields = set(profiles.Profile.__dataclass_fields__)
        for f in PANEL_FIELDS:
            self.assertIn(f, fields)

    def test_valeurs_des_profils_dans_les_bornes_des_curseurs(self):
        # Sinon le curseur tronquerait la valeur du profil sans prévenir
        for name in profiles.list_profiles():
            p = profiles.load_profile(name)
            for field, _label, lo, hi, _step in ESSENTIAL_SLIDERS + ADVANCED_SLIDERS:
                v = getattr(p, field)
                self.assertTrue(lo <= v <= hi, f"{name}.{field} = {v} hors du curseur [{lo}, {hi}]")
            for field, _label in CHECKS:
                self.assertIsInstance(getattr(p, field), bool)

    def test_aucun_objet_tk_garde_apres_fermeture(self):
        # Un objet Tk (case à cocher, image…) encore référencé après la fermeture est détruit plus
        # tard par le thread principal : « main thread is not in main loop » puis plantage de Tcl
        # à la sortie (code 0x80000003, constaté avec Eyetrack.exe).
        import tkinter as tk
        from panel import ControlPanel

        def tk_objects(v):
            if isinstance(v, (tk.Variable, tk.Misc, tk.Image)):
                return [v]
            if isinstance(v, dict):
                return [o for x in v.values() for o in tk_objects(x)]
            if isinstance(v, (list, tuple, set)):
                return [o for x in v for o in tk_objects(x)]
            return []

        panel = ControlPanel()
        panel.start()
        self.assertTrue(panel.is_running)
        panel.stop()
        kept = {name: tk_objects(v) for name, v in vars(panel).items() if tk_objects(v)}
        self.assertEqual(kept, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
