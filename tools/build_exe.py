"""Construit Eyetrack.exe et l'installe pour l'utilisateur courant.

Usage (depuis le dossier du projet, ou option E du menu Eyetrack.bat) :
    python tools/build_exe.py               construire puis installer / mettre à jour
    python tools/build_exe.py --build-only  construire seulement (dist/Eyetrack/Eyetrack.exe)

Installation (sans droits administrateur) :
- programme : %LOCALAPPDATA%\\Programs\\Eyetrack, remplacé à chaque mise à jour ;
- données   : %LOCALAPPDATA%\\Eyetrack (calibration, profils, journal), jamais effacées. À la
  première installation, la calibration, le cache caméra et les profils des sources y sont
  copiés, sans rien écraser : pas besoin de se recalibrer ;
- raccourcis « Eyetrack » sur le Bureau et dans le menu Démarrer.

Le programme installé est indépendant du dossier des sources : on peut continuer à développer
(nouvelle branche, essais) sans toucher à l'exécutable utilisé pour jouer.
"""

from __future__ import annotations
import argparse
import ast
import base64
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import winapp  # noqa: E402

LOCAL = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
APP_DIR = LOCAL / "Programs" / "Eyetrack"
USER_DIR = LOCAL / "Eyetrack"
DIST_DIR = ROOT / "dist" / "Eyetrack"
BUILD_DIR = ROOT / "build" / "eyetrack"  # listes de PyInstaller (PYZ-00.toc, Analysis-00.toc)
REPO_URL = "https://github.com/dimaland1/Eyetrack"
MODELS = ["face_landmarker.task"]  # fournis dans le paquet (Apache 2.0)
# Modèle de regard de précision : licence non commerciale d'un tiers, jamais dans le paquet ni
# l'installeur. Sert seulement à l'auto-test, et copié pour l'utilisateur de CE PC (sa copie).
GAZE_MODEL = ROOT / "models" / "mobilenetv2_gaze.onnx"  # = gaze_model.MODEL_FILE (vérifié par les tests)
DATA_TO_COPY = ["calibration_*.npz", "calibration_samples_*.npz", "camera_mode.json", "app_settings.json"]
NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def write_build_info() -> str:
    commit = git("rev-parse", "--short", "HEAD") or "inconnu"
    if git("status", "--porcelain", "--untracked-files=no"):
        commit += " (modifié)"
    branch = git("branch", "--show-current")
    info = {"commit": commit, "branch": branch, "date": time.strftime("%Y-%m-%d %H:%M")}
    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build" / "build_info.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return f"{commit} ({branch}) du {info['date']}"


def build() -> None:
    missing = [m for m in MODELS if not (ROOT / "models" / m).is_file()]
    if missing:
        sys.exit(f"Modèles absents de models/ : {', '.join(missing)}.\n"
                 "Lancez une fois le tracker depuis les sources (option 1) : ils sont téléchargés.")
    print(f"Construction de la version {write_build_info()}…")
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--log-level", "WARN", str(ROOT / "eyetrack.spec")]
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        sys.exit("Échec de PyInstaller (voir les messages ci-dessus).")
    if not (DIST_DIR / "Eyetrack.exe").is_file():
        sys.exit(f"Eyetrack.exe introuvable dans {DIST_DIR}.")
    count = write_licenses(DIST_DIR, BUILD_DIR, git("rev-parse", "HEAD") or "main")
    print(f"Licences : LICENSE.txt, THIRD_PARTY_NOTICES.txt et {count} composants dans licenses/")
    size = sum(f.stat().st_size for f in DIST_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"Construit : {DIST_DIR / 'Eyetrack.exe'} ({size:.0f} Mo au total)")


# --- Licences livrées avec l'exécutable (GPL-3.0 + textes des bibliothèques embarquées) ---

_LICENSE_NAME = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|AUTHORS|ThirdPartyNotices)", re.IGNORECASE)
# Mentions de copyright rangées ailleurs que dans un fichier LICENSE
_EXTRA_NOTICES = {"sounddevice": ["_sounddevice_data/portaudio-binaries/README.md"]}  # PortAudio (MIT)


def _owns_files(dist: metadata.Distribution, top: str) -> bool:
    """Vrai si les fichiers .py de `top` sur le disque sont bien ceux de cette distribution.

    Deux distributions peuvent fournir le même module (opencv-python et opencv-contrib-python
    pour cv2) : la dernière installée a écrasé l'autre, c'est elle qui est embarquée.
    """
    checked = 0
    for f in dist.files or []:
        if f.parts[0] != top or f.suffix != ".py" or f.hash is None:
            continue
        path = Path(dist.locate_file(f))
        if not path.is_file():
            return False
        digest = base64.urlsafe_b64encode(hashlib.new(f.hash.mode, path.read_bytes()).digest()).rstrip(b"=")
        if digest.decode() != f.hash.value:
            return False
        checked += 1
    return checked > 0


def bundled_distributions(build_dir: Path) -> dict[str, metadata.Distribution]:
    """Distributions Python réellement embarquées, d'après les listes de PyInstaller."""
    names: set[str] = set()
    pyz = ast.literal_eval((build_dir / "PYZ-00.toc").read_text(encoding="utf-8"))
    names.update(entry[0].split(".")[0] for entry in pyz[1])
    for part in ast.literal_eval((build_dir / "Analysis-00.toc").read_text(encoding="utf-8")):
        if not isinstance(part, list):
            continue
        for entry in part:  # binaires et données : (destination, source, type)
            if isinstance(entry, tuple) and len(entry) == 3 and isinstance(entry[0], str):
                segments = re.split(r"[\\/]", entry[0])
                names.add(segments[0].split(".")[0])  # mediapipe/..., cv2/..., _cffi_backend.pyd
                if entry[2] == "EXTENSION":  # win32/win32api.pyd -> win32api (pywin32)
                    names.add(segments[-1].split(".")[0])
    owners = metadata.packages_distributions()
    found: dict[str, metadata.Distribution] = {}
    for name in names:
        candidates = [metadata.distribution(d) for d in dict.fromkeys(owners.get(name, []))]
        if len(candidates) > 1:
            candidates = [d for d in candidates if _owns_files(d, name)] or candidates
        for dist in candidates:
            found[dist.metadata["Name"]] = dist
    return dict(sorted(found.items(), key=lambda item: item[0].lower()))


def _license_label(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    if meta.get("License-Expression"):
        return meta["License-Expression"]
    text = (meta.get("License") or "").strip()
    if text and "\n" not in text and len(text) <= 40:
        return text
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    return ", ".join(classifiers) or "voir les fichiers"


def write_licenses(dist_dir: Path, build_dir: Path, commit: str) -> int:
    """Copie LICENSE (GPL-3.0) et les licences des composants embarqués à côté d'Eyetrack.exe.

    Écrit THIRD_PARTY_NOTICES.txt (liste exacte, versions comprises) et licenses/<composant>/.
    Renvoie le nombre de composants.
    """
    shutil.copy2(ROOT / "LICENSE", dist_dir / "LICENSE.txt")
    out = dist_dir / "licenses"
    shutil.rmtree(out, ignore_errors=True)
    rows = []
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if python_license.is_file():
        folder = out / f"Python-{sys.version.split()[0]}"
        folder.mkdir(parents=True)
        shutil.copy2(python_license, folder / "LICENSE.txt")
        rows.append((f"Python {sys.version.split()[0]}", "PSF-2.0 (with its bundled libraries)", folder.name))
    for name, dist in bundled_distributions(build_dir).items():
        folder = out / f"{name}-{dist.version}"
        copied = 0
        extra = _EXTRA_NOTICES.get(name, [])
        for f in dist.files or []:
            if f.as_posix() not in extra and (not _LICENSE_NAME.search(f.name) or f.suffix in (".py", ".pyc", ".pyi")):
                continue
            src = Path(dist.locate_file(f))
            if not src.is_file():
                continue
            # Sans le dossier .dist-info : numpy-1.26.4.dist-info/LICENSE.txt -> LICENSE.txt
            parts = [p for p in f.parts if not p.endswith(".dist-info") and p != "licenses"]
            dst = folder.joinpath(*parts)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        rows.append((f"{name} {dist.version}", _license_label(dist), folder.name if copied else ""))
    width = max(len(r[0]) for r in rows)
    lines = [
        "Eyetrack - third-party components included in this build",
        "",
        "Eyetrack is free software under the GNU General Public License v3 (LICENSE.txt).",
        f"Source code of this build: {REPO_URL}/tree/{commit}",
        "",
        "Eyetrack.exe includes the components below. Their license texts are in the licenses folder.",
        "The MediaPipe face landmark model (models/face_landmarker.task) is under Apache-2.0, like MediaPipe.",
        "The optional precision gaze model is NOT included: users download it themselves from its",
        "official source (yakhyo/gaze-estimation), under its own non-commercial license.",
        "",
    ]
    lines += [f"{label.ljust(width)}  {lic}" + (f"  ->  licenses/{where}" if where else "  (no license file in the package)")
              for label, lic, where in rows]
    (dist_dir / "THIRD_PARTY_NOTICES.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows)


def _close_windows_of(pid: int) -> list[str]:
    """Ferme les fenêtres visibles d'un processus (fenêtre d'erreur) et renvoie leurs textes."""
    import ctypes
    from ctypes import wintypes
    u = ctypes.windll.user32
    texts: list[str] = []
    found: list[int] = []
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def on_window(hwnd, _):
        owner = wintypes.DWORD()
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and u.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    def on_child(hwnd, _):
        buf = ctypes.create_unicode_buffer(4096)
        u.SendMessageW(hwnd, 0x000D, 4096, buf)  # WM_GETTEXT (marche aussi d'un processus à l'autre)
        if buf.value.strip():
            texts.append(buf.value.strip())
        return True

    u.EnumWindows(enum_proc(on_window), 0)
    for hwnd in found:
        u.EnumChildWindows(hwnd, enum_proc(on_child), 0)
        u.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
    return texts


def self_test(exe: Path) -> None:
    """Lance « Eyetrack.exe --self-test » : arrêt (sans rien installer) si le paquet est cassé."""
    log = USER_DIR / "eyetrack.selftest.log"
    print("Auto-test de l'exécutable (modèles, MediaPipe, interface)…")
    # Modèle de précision passé à part (pas dans le paquet) : ONNX Runtime vérifié avec le vrai modèle
    extra = ["--gaze-model", str(GAZE_MODEL)] if GAZE_MODEL.is_file() else []
    p = subprocess.Popen([str(exe), "--self-test", *extra], cwd=exe.parent)
    # L'auto-test n'ouvre aucune fenêtre : une fenêtre visible = message d'erreur du lanceur
    deadline = time.monotonic() + 120
    while p.poll() is None:
        texts = _close_windows_of(p.pid) if time.monotonic() > deadline else []
        if not texts and time.monotonic() <= deadline:
            time.sleep(0.5)
            texts = _close_windows_of(p.pid)
        if texts or time.monotonic() > deadline:
            time.sleep(1)
            p.kill()
            detail = "\n".join(texts) or "aucune réponse en 2 minutes"
            sys.exit(f"ÉCHEC de l'auto-test : rien n'a été installé.\n{detail}")
    code = p.returncode
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
    if code != 0 or not any(line.startswith("SELFTEST OK") for line in lines):
        sys.exit("ÉCHEC de l'auto-test : rien n'a été installé.\n" + "\n".join(lines[-15:]))
    print(f"Auto-test : {next(line for line in lines if line.startswith('SELFTEST OK'))}")


def install_program() -> None:
    if winapp.is_process_running("Eyetrack.exe"):
        sys.exit("Eyetrack.exe est en cours d'exécution : quittez-le (clic droit sur l'icône près de "
                 "l'horloge > Quitter), puis relancez la mise à jour.")
    if APP_DIR.exists():
        # Garde-fou : ne supprimer que l'ancienne installation d'Eyetrack, jamais un autre dossier
        if not (APP_DIR / "Eyetrack.exe").is_file() and any(APP_DIR.iterdir()):
            sys.exit(f"{APP_DIR} existe mais ne contient pas Eyetrack.exe : rien n'est remplacé.")
        shutil.rmtree(APP_DIR)
    shutil.copytree(DIST_DIR, APP_DIR)
    print(f"Installé : {APP_DIR / 'Eyetrack.exe'}")


def copy_user_data() -> None:
    """Première installation : reprend la calibration et les profils des sources (sans écraser)."""
    copied = []
    for pattern, src_dir, dst_dir in [(p, ROOT / "data", USER_DIR / "data") for p in DATA_TO_COPY] + \
                                     [("*.json", ROOT / "profiles", USER_DIR / "profiles"),
                                      (GAZE_MODEL.name, GAZE_MODEL.parent, USER_DIR / "models")]:
        for src in sorted(src_dir.glob(pattern)):
            dst = dst_dir / src.name
            if not dst.exists():
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied.append(f"{dst_dir.name}/{src.name}")
    if copied:
        print(f"Repris des sources dans {USER_DIR} : {', '.join(copied)}")
    else:
        print(f"Données conservées dans {USER_DIR} (rien à reprendre des sources)")


def create_shortcuts() -> None:
    """Raccourcis Bureau et menu Démarrer (dossiers réels, même si le Bureau est sur OneDrive)."""
    script = (
        "$ErrorActionPreference = 'Stop'; $sh = New-Object -ComObject WScript.Shell; "
        "foreach ($dir in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) { "
        "  $lnk = $sh.CreateShortcut((Join-Path $dir 'Eyetrack.lnk')); "
        "  $lnk.TargetPath = $env:EYETRACK_EXE; $lnk.WorkingDirectory = $env:EYETRACK_DIR; "
        "  $lnk.IconLocation = $env:EYETRACK_EXE + ',0'; "
        "  $lnk.Description = 'Suivi de la tete et du regard pour OpenTrack'; $lnk.Save(); "
        "  Write-Output (Join-Path $dir 'Eyetrack.lnk') }"
    )
    env = dict(os.environ, EYETRACK_EXE=str(APP_DIR / "Eyetrack.exe"), EYETRACK_DIR=str(APP_DIR))
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                       env=env, capture_output=True, text=True, creationflags=NO_WINDOW)
    if r.returncode != 0:
        print(f"Raccourcis non créés : {r.stderr.strip()}")
        return
    for line in r.stdout.splitlines():
        print(f"Raccourci : {line.strip()}")


ISCC_CANDIDATES = [LOCAL / "Programs" / "Inno Setup 6" / "ISCC.exe",
                   Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Inno Setup 6" / "ISCC.exe",
                   Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Inno Setup 6" / "ISCC.exe"]
INSTALLER = ROOT / "dist" / "installer" / "Eyetrack-Setup.exe"


def build_installer() -> None:
    """Installeur Eyetrack-Setup.exe (Inno Setup) à partir de la construction testée de dist/."""
    iscc = next((p for p in ISCC_CANDIDATES if p.is_file()), None)
    if iscc is None:
        sys.exit("Inno Setup 6 introuvable. Installation (gratuite, sans droits administrateur) : "
                 "winget install --id JRSoftware.InnoSetup -e --source winget --scope user")
    version = time.strftime("%Y.%m.%d") + "-" + (git("rev-parse", "--short", "HEAD") or "dev")
    print(f"Compilation de l'installeur (version {version})…")
    cmd = [str(iscc), "/Qp", f"/DAppVersion={version}", f"/DSourceDir={DIST_DIR}",
           str(ROOT / "installer" / "eyetrack.iss")]
    if subprocess.run(cmd, cwd=ROOT / "installer").returncode != 0 or not INSTALLER.is_file():
        sys.exit("Échec de la compilation de l'installeur (voir les messages ci-dessus).")
    print(f"Installeur : {INSTALLER} ({INSTALLER.stat().st_size / 1e6:.0f} Mo)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construit et installe Eyetrack.exe")
    parser.add_argument("--build-only", action="store_true", help="Construire sans installer")
    parser.add_argument("--install-only", action="store_true", help="Installer la dernière construction (dist/)")
    parser.add_argument("--installer", action="store_true",
                        help="Construire l'installeur Eyetrack-Setup.exe (à donner), sans installer sur ce PC")
    args = parser.parse_args()
    if not args.install_only:
        build()
    elif not (DIST_DIR / "Eyetrack.exe").is_file():
        sys.exit(f"Aucune construction dans {DIST_DIR} : lancez d'abord sans --install-only.")
    self_test(DIST_DIR / "Eyetrack.exe")
    if args.installer:
        build_installer()
        return
    if args.build_only:
        return
    install_program()
    copy_user_data()
    create_shortcuts()
    print("\nC'est prêt : double-cliquez sur « Eyetrack » sur le Bureau.")


if __name__ == "__main__":
    main()
