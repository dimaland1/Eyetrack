"""Tests du panneau de contrôle : les curseurs correspondent bien aux réglages des profils."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import profile as profiles  # noqa: E402
from panel import CHECKS, GAZE_SLIDERS, HEAD_SLIDERS, PANEL_FIELDS  # noqa: E402


class TestPanel(unittest.TestCase):
    def test_champs_du_panneau_existent_dans_le_profil(self):
        fields = set(profiles.Profile.__dataclass_fields__)
        for f in PANEL_FIELDS:
            self.assertIn(f, fields)

    def test_valeurs_des_profils_dans_les_bornes_des_curseurs(self):
        # Sinon le curseur tronquerait la valeur du profil sans prévenir
        for name in profiles.list_profiles():
            p = profiles.load_profile(name)
            for field, _label, lo, hi, _step in HEAD_SLIDERS + GAZE_SLIDERS:
                v = getattr(p, field)
                self.assertTrue(lo <= v <= hi, f"{name}.{field} = {v} hors du curseur [{lo}, {hi}]")
            for field, _label in CHECKS:
                self.assertIsInstance(getattr(p, field), bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
