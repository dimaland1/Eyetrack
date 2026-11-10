"""Point d'entrée de l'exécutable Eyetrack.exe (application fenêtrée, sans console).

Double-clic sur Eyetrack.exe :
- reprend le dernier profil utilisé (ou celui passé en argument, ex. --profile ets2) ;
- ouvre OpenTrack s'il n'est pas déjà ouvert (case dans « Réglages avancés ») ;
- lance le tracker avec son panneau et son icône près de l'horloge.

Relancer Eyetrack.exe pendant qu'il tourne rouvre simplement son panneau (une seule instance :
la caméra et le port UDP ne se partagent pas). Sans console, les messages vont dans le journal
%LOCALAPPDATA%\\Eyetrack\\eyetrack.log et les erreurs bloquantes s'affichent en fenêtre.

Depuis les sources, `python eyetrack_app.py` fait la même chose (messages dans la console).
"""

from __future__ import annotations
import json
import sys
import time
import traceback

import paths
import winapp


def build_info() -> str:
    """Version de l'exécutable (commit et date de construction), écrite par tools/build_exe.py."""
    try:
        info = json.loads((paths.BUNDLE_DIR / "build_info.json").read_text(encoding="utf-8"))
        return f"version {info.get('commit', '?')} du {info.get('date', '?')}"
    except (OSError, ValueError):
        return "version des sources"


def self_test() -> int:
    """Eyetrack.exe --self-test : vérifie le paquet sans caméra ni fenêtre visible.

    Charge tout ce dont le tracker a besoin (MediaPipe et sa DLL, modèle du visage, modèle du
    regard ONNX, OpenCV, Tkinter) et analyse une image noire. Résultat dans
    eyetrack.selftest.log ; code de sortie 0 si tout va bien. Lancé par tools/build_exe.py
    avant d'installer : une construction cassée ne remplace jamais la version qui marche.
    """
    winapp.setup_log_file(paths.LOG_FILE.with_name("eyetrack.selftest.log"))
    print(f"Auto-test Eyetrack - {build_info()}")
    t0 = time.perf_counter()
    try:
        import numpy as np
        import tkinter as tk
        import main  # noqa: F401  (tous les modules du tracker)
        from face import FaceDetector
        from gaze_model import create_gaze_estimator

        face = FaceDetector().process(np.zeros((480, 640, 3), np.uint8))
        print(f"MediaPipe : OK (visage sur image noire : {face.detected})")
        create_gaze_estimator(mode="net")
        print("Modèle du regard (ONNX) : OK")
        root = tk.Tk()
        root.withdraw()
        root.destroy()
        print("Tkinter : OK")
    except Exception:
        print(traceback.format_exc())
        print("SELFTEST ÉCHEC")
        return 1
    print(f"SELFTEST OK en {time.perf_counter() - t0:.1f} s")
    return 0


def run() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    # Instance unique vérifiée avant d'ouvrir le journal : celui de l'instance lancée reste intact
    import app_settings
    from i18n import set_language, tr
    set_language(app_settings.load()["language"])
    instance = winapp.SingleInstance()
    if instance.already_running:
        if not winapp.show_running_instance():
            winapp.message_box("Eyetrack", tr("Eyetrack est déjà lancé : son icône est près de l'horloge."),
                               error=False)
        return 0

    paths.prepare_user_dirs()
    winapp.setup_log_file(paths.LOG_FILE)
    print(f"Eyetrack - {build_info()} - démarré le {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Données : {paths.DATA_DIR}")

    try:
        settings = app_settings.load()
        if len(sys.argv) == 1:  # double-clic : dernier profil utilisé
            sys.argv += ["--profile", settings["last_profile"]]
        if settings["launch_opentrack"]:
            print(f"[opentrack] {winapp.launch_opentrack(settings['opentrack_path'])}")
        import main  # import lourd (MediaPipe, ONNX) : une erreur ici doit aussi s'afficher
        main.main()
    except SystemExit as e:  # arguments invalides (argparse)
        if e.code not in (0, None):
            winapp.report_error(tr("Arguments invalides : {args}\n\nDétails : {log}",
                                   args=" ".join(sys.argv[1:]), log=paths.LOG_FILE))
            return int(e.code) if isinstance(e.code, int) else 2
    except Exception as e:
        print(traceback.format_exc())
        winapp.report_error(
            tr("Eyetrack s'est arrêté à cause d'une erreur :\n\n{kind} : {err}\n\nDétails dans le journal :\n{log}",
               kind=type(e).__name__, err=e, log=paths.LOG_FILE),
            title=tr("Eyetrack - erreur"))
        return 1
    finally:
        instance.release()
    print(f"Arrêt le {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
