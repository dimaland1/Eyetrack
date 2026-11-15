"""Numéro de version des constructions : la version publiée seulement sur le commit étiqueté et propre."""

import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_exe", ROOT / "tools" / "build_exe.py")
build_exe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_exe)

from version import VERSION  # noqa: E402


class VersionTest(unittest.TestCase):
    def setUp(self):
        self.real_git = build_exe.git

    def tearDown(self):
        build_exe.git = self.real_git

    def fake_git(self, tags: str, status: str):
        answers = {("rev-parse", "--short", "HEAD"): "abc1234",
                   ("tag", "--points-at", "HEAD"): tags,
                   ("status", "--porcelain", "--untracked-files=no"): status}
        build_exe.git = lambda *args: answers[args]

    def test_format(self):
        self.assertRegex(VERSION, r"^\d+\.\d+\.\d+$")

    def test_commit_etiquete_et_propre(self):
        self.fake_git(f"v{VERSION}", "")
        self.assertEqual(build_exe.release_version(), VERSION)

    def test_commit_non_etiquete(self):
        self.fake_git("", "")
        self.assertEqual(build_exe.release_version(), f"{VERSION}-dev+abc1234")

    def test_fichiers_modifies(self):
        self.fake_git(f"v{VERSION}", " M main.py")
        self.assertEqual(build_exe.release_version(), f"{VERSION}-dev+abc1234")

    def test_changelog_a_jour(self):
        # Chaque version publiée a son entrée dans CHANGELOG.md
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertTrue(re.search(rf"^## \[?{re.escape(VERSION)}\]?", text, re.MULTILINE))


if __name__ == "__main__":
    unittest.main()
