"""Tests de l'application sans console : réglages, dossiers utilisateur, instance unique, OpenTrack."""

import json
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app_settings  # noqa: E402
import paths  # noqa: E402
import winapp  # noqa: E402


class TempDataDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._saved = (paths.DATA_DIR, paths.PROFILES_DIR, paths.DEFAULT_PROFILES_DIR)
        paths.DATA_DIR = self.tmp / "data"

    def tearDown(self):
        paths.DATA_DIR, paths.PROFILES_DIR, paths.DEFAULT_PROFILES_DIR = self._saved
        self._tmp.cleanup()


class TestAppSettings(TempDataDir):
    def test_defaults_when_missing(self):
        self.assertEqual(app_settings.load(), app_settings.DEFAULTS)

    def test_update_persists(self):
        app_settings.update(last_profile="ets2", launch_opentrack=False)
        s = app_settings.load()
        self.assertEqual(s["last_profile"], "ets2")
        self.assertFalse(s["launch_opentrack"])

    def test_bad_values_ignored(self):
        app_settings.update(last_profile=42, unknown="x")  # mauvais type, clé inconnue
        self.assertEqual(app_settings.load(), app_settings.DEFAULTS)

    def test_corrupt_file_gives_defaults(self):
        paths.DATA_DIR.mkdir(parents=True)
        (paths.DATA_DIR / app_settings.FILE_NAME).write_text("{pas du json", encoding="utf-8")
        self.assertEqual(app_settings.load(), app_settings.DEFAULTS)
        (paths.DATA_DIR / app_settings.FILE_NAME).write_text(json.dumps({"launch_opentrack": "oui"}), encoding="utf-8")
        self.assertTrue(app_settings.load()["launch_opentrack"])  # mauvais type : défaut


class TestUserDirs(TempDataDir):
    def test_profiles_seeded_without_overwrite(self):
        paths.DEFAULT_PROFILES_DIR = self.tmp / "bundle_profiles"
        paths.PROFILES_DIR = self.tmp / "user_profiles"
        paths.DEFAULT_PROFILES_DIR.mkdir()
        (paths.DEFAULT_PROFILES_DIR / "default.json").write_text('{"v": "paquet"}', encoding="utf-8")
        (paths.DEFAULT_PROFILES_DIR / "ets2.json").write_text('{"v": "paquet"}', encoding="utf-8")
        paths.PROFILES_DIR.mkdir()
        (paths.PROFILES_DIR / "ets2.json").write_text('{"v": "utilisateur"}', encoding="utf-8")
        paths.prepare_user_dirs()
        self.assertEqual((paths.PROFILES_DIR / "default.json").read_text(encoding="utf-8"), '{"v": "paquet"}')
        self.assertEqual((paths.PROFILES_DIR / "ets2.json").read_text(encoding="utf-8"), '{"v": "utilisateur"}')
        self.assertTrue(paths.DATA_DIR.is_dir())


@unittest.skipUnless(sys.platform == "win32", "API Windows")
class TestWindows(unittest.TestCase):
    def test_single_instance(self):
        name = f"Local\\Eyetrack.Test.{uuid.uuid4().hex}"
        first = winapp.SingleInstance(name)
        second = winapp.SingleInstance(name)
        try:
            self.assertFalse(first.already_running)
            self.assertTrue(second.already_running)
        finally:
            second.release()
            first.release()
        third = winapp.SingleInstance(name)
        self.assertFalse(third.already_running)  # verrou libéré
        third.release()

    def test_process_detection(self):
        me = os.path.basename(sys.executable)
        self.assertTrue(winapp.is_process_running(me))
        self.assertTrue(winapp.is_process_running(me.upper()))
        self.assertFalse(winapp.is_process_running(f"pas_un_programme_{uuid.uuid4().hex[:8]}.exe"))

    def test_find_opentrack_custom_path(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "opentrack.exe"
            exe.write_bytes(b"")
            self.assertEqual(winapp.find_opentrack(str(exe)), str(exe))
        self.assertNotEqual(winapp.find_opentrack(r"C:\nulle\part\opentrack.exe"), r"C:\nulle\part\opentrack.exe")

    def test_console_detected_in_tests(self):
        self.assertIsNone(winapp.setup_log_file(Path(tempfile.gettempdir()) / "ne_doit_pas_servir.log"))

    @unittest.skipUnless(winapp.IS_WINDOWS, "Windows uniquement")
    def test_premier_plan_jamais_pour_un_autre_programme(self):
        # La mise au premier plan (fenêtre de calibration) ne doit jamais toucher la fenêtre
        # d'un autre programme portant le même titre (ici le bureau, tenu par l'Explorateur).
        self.assertFalse(winapp.bring_to_front(f"Eyetrack fenetre absente {uuid.uuid4()}"))
        self.assertFalse(winapp.bring_to_front("Program Manager"))


if __name__ == "__main__":
    unittest.main()
