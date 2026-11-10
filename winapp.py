"""Outils Windows pour l'application sans console (Eyetrack.exe).

- journal : sans console, les messages vont dans un fichier (eyetrack.log) ;
- messages d'erreur en fenêtre (une erreur au démarrage ne doit pas fermer l'appli en silence) ;
- une seule instance : relancer Eyetrack rouvre le panneau de celle qui tourne déjà ;
- OpenTrack : détection de l'installation, du processus, et lancement sans fenêtre de console.

Tout passe par ctypes (aucune dépendance). Hors Windows, les fonctions ne font rien.
"""

from __future__ import annotations
import ctypes
import os
import sys
from ctypes import wintypes
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"

MB_OK = 0x0
MB_ICONERROR = 0x10
MB_ICONINFORMATION = 0x40
MB_SETFOREGROUND = 0x10000
MB_TOPMOST = 0x40000
ERROR_ALREADY_EXISTS = 183
TH32CS_SNAPPROCESS = 0x2
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

WM_APP = 0x8000
WM_SHOW_PANEL = WM_APP + 3  # demandé à l'instance déjà lancée (cf. tray.py)
TRAY_CLASS_PREFIX = "EyetrackTray"
MUTEX_NAME = "Local\\Eyetrack.Tracker"

OPENTRACK_EXE = "opentrack.exe"


# --------------------------------------------------------------------------- console et journal

def windowed() -> bool:
    """Vrai sans sortie texte (Eyetrack.exe, pythonw.exe) : les messages doivent aller dans un
    journal et les erreurs dans une fenêtre. Faux depuis un terminal, même si la sortie est
    redirigée vers un fichier ou un tube."""
    return sys.__stdout__ is None


def setup_log_file(log_file: Path) -> object | None:
    """Sans sortie texte : redirige print() et les erreurs vers log_file (l'ancien est gardé en .old).

    Renvoie le fichier ouvert, ou None si une sortie texte existe (rien n'est changé).
    """
    if not windowed():
        return None
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if log_file.exists():
            os.replace(log_file, log_file.with_suffix(".old.log"))
        f = open(log_file, "w", encoding="utf-8", buffering=1, errors="replace")
    except OSError:
        return None
    sys.stdout = sys.stderr = f
    try:
        import faulthandler
        faulthandler.enable(f)  # plantage natif (DLL) : la pile Python est quand même écrite
    except Exception:
        pass
    return f


# --------------------------------------------------------------------------- fenêtres de message

def message_box(title: str, text: str, error: bool = True) -> None:
    """Fenêtre de message Windows, au premier plan (même par-dessus un jeu en fenêtre)."""
    if not IS_WINDOWS:
        print(f"{title} : {text}")
        return
    flags = MB_OK | MB_SETFOREGROUND | MB_TOPMOST | (MB_ICONERROR if error else MB_ICONINFORMATION)
    ctypes.windll.user32.MessageBoxW(None, text, title, flags)


def report_error(text: str, title: str = "Eyetrack") -> None:
    """Erreur visible dans tous les cas : texte dans le terminal ou, sans terminal, fenêtre de message."""
    print(f"[ERREUR] {text}", file=sys.stderr)
    if windowed():
        message_box(title, text, error=True)


def bring_to_front(title: str) -> bool:
    """Met la fenêtre `title` de ce programme au-dessus de tout et lui donne le clavier.

    Au-dessus de tout : « toujours visible », devant le jeu en fenêtre, le navigateur, le panneau.
    Le clavier : Windows refuse le premier plan à un programme qui n'est pas au premier plan
    (raccourci reçu pendant le jeu…) ; on se rattache donc, le temps de l'appel, à la file
    d'entrée de la fenêtre active. À appeler depuis le thread qui a créé la fenêtre.
    Renvoie vrai si la fenêtre est bien devenue la fenêtre active.
    """
    if not IS_WINDOWS:
        return False
    u = ctypes.WinDLL("user32")
    k = ctypes.WinDLL("kernel32")
    u.FindWindowW.restype = wintypes.HWND
    u.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    u.GetForegroundWindow.restype = wintypes.HWND
    u.GetWindowThreadProcessId.restype = wintypes.DWORD
    u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, wintypes.UINT]
    u.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
    for fn in (u.BringWindowToTop, u.SetForegroundWindow, u.SetFocus):
        fn.argtypes = [wintypes.HWND]

    hwnd = u.FindWindowW(None, title)
    if not hwnd:
        return False
    pid = wintypes.DWORD()
    u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if pid.value != os.getpid():  # jamais la fenêtre d'un autre programme
        return False

    HWND_TOPMOST = wintypes.HWND(-1)
    SWP_NOSIZE, SWP_NOMOVE, SWP_SHOWWINDOW = 0x1, 0x2, 0x40
    u.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW)

    me = k.GetCurrentThreadId()
    fg = u.GetForegroundWindow()
    other = u.GetWindowThreadProcessId(fg, None) if fg else 0
    attached = bool(other and other != me and u.AttachThreadInput(me, other, True))
    try:
        u.BringWindowToTop(hwnd)
        u.SetForegroundWindow(hwnd)
        u.SetFocus(hwnd)
    finally:
        if attached:
            u.AttachThreadInput(me, other, False)
    return u.GetForegroundWindow() == hwnd


# --------------------------------------------------------------------------- instance unique

class SingleInstance:
    """Verrou nommé : une seule instance d'Eyetrack à la fois (caméra et port UDP partagés)."""

    def __init__(self, name: str = MUTEX_NAME) -> None:
        self._handle = None
        self.already_running = False
        if not IS_WINDOWS:
            return
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateMutexW.restype = wintypes.HANDLE
        k.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        self._handle = k.CreateMutexW(None, False, name)
        self.already_running = ctypes.get_last_error() == ERROR_ALREADY_EXISTS

    def release(self) -> None:
        if self._handle:
            ctypes.windll.kernel32.CloseHandle(wintypes.HANDLE(self._handle))
            self._handle = None


def _find_tray_windows() -> list[int]:
    """Fenêtres cachées des icônes Eyetrack (classe « EyetrackTray… ») des autres instances."""
    found: list[int] = []
    if not IS_WINDOWS:
        return found
    u = ctypes.windll.user32
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _lparam):
        buf = ctypes.create_unicode_buffer(64)
        u.GetClassNameW(hwnd, buf, 64)
        if buf.value.startswith(TRAY_CLASS_PREFIX):
            found.append(hwnd)
        return True

    u.EnumWindows(proc(callback), 0)
    return found


def show_running_instance() -> bool:
    """Demande à l'instance déjà lancée d'afficher son panneau. Faux si elle n'a pas d'icône."""
    windows = _find_tray_windows()
    if not windows:
        return False
    # Autorise l'autre processus à passer au premier plan (sinon Windows le laisse derrière)
    ASFW_ANY = -1
    ctypes.windll.user32.AllowSetForegroundWindow(ASFW_ANY)
    for hwnd in windows:
        ctypes.windll.user32.PostMessageW(hwnd, WM_SHOW_PANEL, 0, 0)
    return True


# --------------------------------------------------------------------------- OpenTrack

class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


def is_process_running(exe_name: str) -> bool:
    """Vrai si un processus de ce nom (ex. « opentrack.exe ») tourne, sans ouvrir de console."""
    if not IS_WINDOWS:
        return False
    k = ctypes.windll.kernel32
    k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snap or snap == INVALID_HANDLE_VALUE:
        return False
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = k.Process32FirstW(wintypes.HANDLE(snap), ctypes.byref(entry))
        target = exe_name.lower()
        while ok:
            if entry.szExeFile.lower() == target:
                return True
            ok = k.Process32NextW(wintypes.HANDLE(snap), ctypes.byref(entry))
    finally:
        k.CloseHandle(wintypes.HANDLE(snap))
    return False


def find_opentrack(custom_path: str = "") -> str | None:
    """Chemin d'opentrack.exe : chemin personnalisé, sinon emplacements d'installation habituels."""
    candidates = [custom_path] if custom_path else []
    for var in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
        base = os.environ.get(var)
        if base:
            candidates.append(os.path.join(base, "opentrack", OPENTRACK_EXE))
            candidates.append(os.path.join(base, "Programs", "opentrack", OPENTRACK_EXE))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def launch_opentrack(custom_path: str = "") -> str:
    """Ouvre OpenTrack s'il ne tourne pas déjà. Renvoie « déjà ouvert », « lancé » ou « introuvable »."""
    if is_process_running(OPENTRACK_EXE):
        return "déjà ouvert"
    path = find_opentrack(custom_path)
    if path is None:
        return "introuvable"
    try:
        os.startfile(path, cwd=os.path.dirname(path))  # comme un double-clic : pas de console
    except OSError as e:
        print(f"[opentrack] Lancement impossible : {e}")
        return "introuvable"
    return "lancé"
