"""Licences livrées avec l'exécutable : bons composants, textes copiés, pas de faux positifs."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_exe", ROOT / "tools" / "build_exe.py")
build_exe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_exe)


class LicencesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.build_dir = base / "build"
        self.dist_dir = base / "dist"
        self.build_dir.mkdir()
        self.dist_dir.mkdir()
        # Listes de PyInstaller réduites : numpy en module pur, cv2 en extension, un fichier de données
        (self.build_dir / "PYZ-00.toc").write_text(
            repr(("PYZ-00.pyz", [("numpy.core", "numpy/core/__init__.py", "PYMODULE")])), encoding="utf-8")
        (self.build_dir / "Analysis-00.toc").write_text(repr((
            ["eyetrack_app.py"],
            [("cv2" + os.sep + "cv2.pyd", "cv2.pyd", "EXTENSION")],
            [("typing" + os.sep + "py.typed", "py.typed", "DATA")],  # « py » ne doit pas donner pytest
        )), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_composants_et_textes(self):
        count = build_exe.write_licenses(self.dist_dir, self.build_dir, "abc1234")
        notices = (self.dist_dir / "THIRD_PARTY_NOTICES.txt").read_text(encoding="utf-8")
        self.assertIn("GNU General Public License v3", notices)
        self.assertIn("/tree/abc1234", notices)
        self.assertIn("numpy", notices)
        self.assertIn("opencv", notices)
        self.assertNotIn("pytest", notices)
        self.assertEqual((self.dist_dir / "LICENSE.txt").read_bytes(), (ROOT / "LICENSE").read_bytes())
        numpy_dirs = list((self.dist_dir / "licenses").glob("numpy-*"))
        self.assertEqual(len(numpy_dirs), 1)
        self.assertTrue((numpy_dirs[0] / "LICENSE.txt").is_file())
        self.assertGreaterEqual(count, 3)  # Python, numpy, OpenCV

    def test_une_seule_distribution_opencv(self):
        # opencv-python et opencv-contrib-python peuvent coexister : seule celle qui possède cv2 compte
        found = build_exe.bundled_distributions(self.build_dir)
        self.assertEqual(len([n for n in found if n.lower().startswith("opencv")]), 1)


if __name__ == "__main__":
    unittest.main()
