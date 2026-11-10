"""Tests de la langue de l'interface : chaque texte affiché a sa traduction anglaise."""

from __future__ import annotations

import ast
import os
import re
import string
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import i18n  # noqa: E402
from i18n import EN, tr  # noqa: E402

SOURCES = ["panel.py", "tray.py", "main.py", "calibration.py", "eyetrack_app.py", "hotkeys.py", "i18n.py"]


def tr_literals() -> list[tuple[str, int, str]]:
    """Textes passés en dur à tr("…") dans les sources (fichier, ligne, texte)."""
    found = []
    for name in SOURCES:
        with open(os.path.join(ROOT, name), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "tr"
                    and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                found.append((name, node.lineno, node.args[0].value))
    return found


def fields(text: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


class TestTraductions(unittest.TestCase):
    def tearDown(self):
        i18n.set_language("fr")

    def test_chaque_texte_a_sa_traduction(self):
        literals = tr_literals()
        self.assertGreater(len(literals), 80)  # le scan trouve bien les textes
        missing = [f"{n}:{line} {t!r}" for n, line, t in literals if t not in EN]
        self.assertEqual(missing, [])

    def test_textes_passes_par_variable(self):
        # Libellés gardés en français dans des listes, traduits à l'affichage
        from panel import ADVANCED_SLIDERS, APP_CHECKS, CHECKS, ESSENTIAL_SLIDERS
        texts = [t for _, t, *_ in ESSENTIAL_SLIDERS + ADVANCED_SLIDERS] + [t for _, t in CHECKS + APP_CHECKS]
        texts += ["Visage", "Reflets lunettes", "Caméra", "Regard", "Calibration", "Caméra du jeu",
                  "Recentrer", "Pause", "Couper le regard", "Calibrer le regard", "Afficher l'overlay",
                  "Voir la caméra", "aucun", "léger", "gênant", "bonne", "moyenne", "à refaire", "inconnue",
                  "BONNE", "MOYENNE", "A REFAIRE", "Maj"]
        self.assertEqual([t for t in texts if t not in EN], [])

    def test_memes_valeurs_inserees(self):
        bad = [k for k, v in EN.items() if fields(k) != fields(v)]
        self.assertEqual(bad, [])

    def test_ecrans_de_calibration_sans_accents(self):
        # cv2.putText (police Hershey) n'affiche que l'ASCII : un texte français sans accents
        # doit le rester en anglais (sinon « ? » à l'écran)
        bad = [v for k, v in EN.items() if k.isascii() and not v.isascii()]
        self.assertEqual(bad, [])

    def test_tr(self):
        i18n.set_language("fr")
        self.assertEqual(tr("Recentrer"), "Recentrer")
        self.assertEqual(tr("{fps} images/s", fps=30), "30 images/s")
        i18n.set_language("en")
        self.assertEqual(tr("Recentrer"), "Recenter")
        self.assertEqual(tr("{fps} images/s", fps=30), "30 fps")
        self.assertEqual(tr("texte sans traduction"), "texte sans traduction")

    def test_choix_de_langue(self):
        self.assertEqual(i18n.set_language("en"), "en")
        self.assertEqual(i18n.set_language("fr"), "fr")
        self.assertIn(i18n.set_language("auto"), ("fr", "en"))
        self.assertEqual(i18n.resolve("xx"), i18n.system_language())

    def test_raccourci_en_anglais(self):
        from hotkeys import format_hotkey
        self.assertEqual(format_hotkey("ctrl+shift+f9"), "Ctrl+Maj+F9")
        i18n.set_language("en")
        self.assertEqual(format_hotkey("ctrl+shift+f9"), "Ctrl+Shift+F9")

    def test_ecrans_en_anglais(self):
        import numpy as np
        from calibration import draw_intro, draw_review
        from geometry import ScreenConfig
        i18n.set_language("en")
        img = draw_intro(1920, 1080, np.full((480, 640, 3), 77, np.uint8), True, 0.5)
        self.assertEqual(img.shape, (1080, 1920, 3))
        sc = ScreenConfig(width_px=1920, height_px=1080, diagonal_inches=24.0, distance_cm=60.0)
        pts = [{"target": np.array([100.0 + 200 * i, 300.0]), "glare": 0.5 if i == 0 else 0.0} for i in range(9)]
        errs = [3.0] * 8 + [20.0]
        img = draw_review(sc, pts, errs, [p["target"] for p in pts], 5.0)
        self.assertEqual(img.shape, (1080, 1920, 3))

    def test_panneau_en_anglais_et_reconstruit(self):
        # Changer de langue reconstruit le panneau : arrêt puis nouveau panneau, sans erreur Tk
        from panel import ControlPanel
        i18n.set_language("en")
        for _ in range(2):
            panel = ControlPanel()
            self.assertEqual(panel.title, "Eyetrack - Control panel")
            panel.start()
            self.assertTrue(panel.is_running)
            panel.publish({"face": True, "fps": 30.0, "glare": 0.2, "glare_label": "gênant", "calibrated": True,
                           "calib_quality": "bonne", "calib_deg": 3.2, "rls": 2, "opentrack_running": False,
                           "app_settings": {"launch_opentrack": True, "language": "en"}})
            import time
            time.sleep(0.4)  # quelques rafraîchissements avec cet état
            panel.stop()
            self.assertFalse(panel._thread.is_alive())


if __name__ == "__main__":
    unittest.main(verbosity=2)
