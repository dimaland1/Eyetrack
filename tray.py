"""Icône Eyetrack dans la zone de notification de Windows (à côté de l'horloge).

API Windows directe (Shell_NotifyIcon via ctypes) : aucune dépendance, licence propre.
L'icône vit dans son propre thread (fenêtre cachée + boucle de messages) et ne communique
avec la boucle principale que par une file d'actions (`poll()`), comme les raccourcis globaux.

- clic gauche : afficher / masquer le panneau de contrôle ;
- clic droit : menu (panneau, recentrer, pause, regard, quitter).
"""

from __future__ import annotations
import ctypes
import platform
import queue
import threading
from ctypes import wintypes

WM_NULL = 0x0000
WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
WM_LBUTTONUP = 0x0202
WM_RBUTTONUP = 0x0205
WM_APP = 0x8000
WM_TRAY = WM_APP + 1     # événements souris sur l'icône
WM_REFRESH = WM_APP + 2  # mise à jour de l'infobulle / bulle demandée par un autre thread
WM_SHOW_PANEL = WM_APP + 3  # une 2e instance d'Eyetrack demande d'afficher le panneau (winapp.py)

NIM_ADD, NIM_MODIFY, NIM_DELETE = 0, 1, 2
NIF_MESSAGE, NIF_ICON, NIF_TIP, NIF_INFO = 0x1, 0x2, 0x4, 0x10
NIIF_INFO = 0x1
MF_STRING, MF_SEPARATOR = 0x0, 0x800
TPM_RIGHTBUTTON, TPM_NONOTIFY, TPM_RETURNCMD = 0x2, 0x80, 0x100
IMAGE_ICON, LR_LOADFROMFILE, LR_DEFAULTSIZE = 1, 0x10, 0x40
IDI_APPLICATION = 32512

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)


class WNDCLASSW(ctypes.Structure):
    _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]


class GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort), ("Data3", ctypes.c_ushort),
                ("Data4", ctypes.c_ubyte * 8)]


class NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("hWnd", wintypes.HWND), ("uID", wintypes.UINT),
                ("uFlags", wintypes.UINT), ("uCallbackMessage", wintypes.UINT), ("hIcon", wintypes.HICON),
                ("szTip", wintypes.WCHAR * 128), ("dwState", wintypes.DWORD), ("dwStateMask", wintypes.DWORD),
                ("szInfo", wintypes.WCHAR * 256), ("uVersion", wintypes.UINT),
                ("szInfoTitle", wintypes.WCHAR * 64), ("dwInfoFlags", wintypes.DWORD),
                ("guidItem", GUID), ("hBalloonIcon", wintypes.HICON)]


def _api():
    """Fonctions Win32 avec leurs types (indispensable en 64 bits pour les handles)."""
    u = ctypes.WinDLL("user32", use_last_error=True)
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    s = ctypes.WinDLL("shell32", use_last_error=True)
    u.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    u.DefWindowProcW.restype = LRESULT
    u.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
    u.RegisterClassW.restype = wintypes.ATOM
    u.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
    u.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
    u.CreateWindowExW.restype = wintypes.HWND
    u.DestroyWindow.argtypes = [wintypes.HWND]
    u.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    u.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
    u.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    u.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    u.DispatchMessageW.restype = LRESULT
    u.RegisterWindowMessageW.argtypes = [wintypes.LPCWSTR]
    u.RegisterWindowMessageW.restype = wintypes.UINT
    u.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT, ctypes.c_int, ctypes.c_int, wintypes.UINT]
    u.LoadImageW.restype = wintypes.HANDLE
    u.LoadIconW.argtypes = [wintypes.HINSTANCE, wintypes.LPVOID]
    u.LoadIconW.restype = wintypes.HICON
    u.DestroyIcon.argtypes = [wintypes.HICON]
    u.CreatePopupMenu.restype = wintypes.HMENU
    u.AppendMenuW.argtypes = [wintypes.HMENU, wintypes.UINT, ctypes.c_size_t, wintypes.LPCWSTR]
    u.TrackPopupMenu.argtypes = [wintypes.HMENU, wintypes.UINT, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 wintypes.HWND, wintypes.LPVOID]
    u.TrackPopupMenu.restype = ctypes.c_int
    u.DestroyMenu.argtypes = [wintypes.HMENU]
    u.SetForegroundWindow.argtypes = [wintypes.HWND]
    u.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    k.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    k.GetModuleHandleW.restype = wintypes.HMODULE
    s.Shell_NotifyIconW.argtypes = [wintypes.DWORD, ctypes.POINTER(NOTIFYICONDATAW)]
    s.Shell_NotifyIconW.restype = wintypes.BOOL
    return u, k, s


class TrayIcon:
    """Icône de la zone de notification, dans un thread dédié."""

    def __init__(self, icon_path: str | None = None, tooltip: str = "Eyetrack") -> None:
        self.icon_path = icon_path
        self._tooltip = tooltip
        self._balloon: tuple[str, str] | None = None
        self._actions: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._hwnd = 0
        self.available = False
        # État affiché dans le menu (mis à jour par la boucle principale)
        self.paused = False
        self.gaze_enabled = True
        self.panel_visible = True

    # ------------------------------------------------------------------ interface (autres threads)
    def start(self) -> bool:
        """Crée l'icône ; False si indisponible (autre système, erreur Windows)."""
        if platform.system() != "Windows":
            return False
        self._thread = threading.Thread(target=self._run, name="zone-notification", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=3.0)
        return self.available

    def poll(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self._actions.get_nowait())
            except queue.Empty:
                return out

    def set_tooltip(self, text: str) -> None:
        """Texte affiché au survol de l'icône (128 caractères max)."""
        if text != self._tooltip:
            self._tooltip = text
            self._post(WM_REFRESH)

    def notify(self, title: str, text: str) -> None:
        """Petite notification (bulle) près de l'horloge."""
        self._balloon = (title, text)
        self._post(WM_REFRESH)

    def stop(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._post(WM_CLOSE)
            self._thread.join(timeout=2.0)
        self._thread = None

    def _post(self, msg: int) -> None:
        if self._hwnd:
            ctypes.windll.user32.PostMessageW(wintypes.HWND(self._hwnd), msg, 0, 0)

    # ------------------------------------------------------------------ thread de l'icône
    def _menu_items(self) -> list[tuple[str, str] | None]:
        return [
            ("toggle_panel", "Masquer le panneau" if self.panel_visible else "Afficher le panneau"),
            ("recenter", "Recentrer"),
            ("pause", "Reprendre le suivi" if self.paused else "Pause (caméra au centre)"),
            ("gaze", "Couper le regard" if self.gaze_enabled else "Réactiver le regard"),
            None,
            ("quit", "Quitter Eyetrack"),
        ]

    def _nid(self, flags: int) -> NOTIFYICONDATAW:
        nid = NOTIFYICONDATAW()
        nid.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
        nid.hWnd = self._hwnd
        nid.uID = 1
        nid.uFlags = flags
        nid.uCallbackMessage = WM_TRAY
        nid.hIcon = self._hicon
        nid.szTip = self._tooltip[:127]
        return nid

    def _add_icon(self) -> bool:
        nid = self._nid(NIF_MESSAGE | NIF_ICON | NIF_TIP)
        return bool(self._s.Shell_NotifyIconW(NIM_ADD, ctypes.byref(nid)))

    def _refresh(self) -> None:
        flags = NIF_TIP
        nid = self._nid(flags)
        if self._balloon is not None:
            title, text = self._balloon
            self._balloon = None
            nid.uFlags |= NIF_INFO
            nid.szInfoTitle = title[:63]
            nid.szInfo = text[:255]
            nid.dwInfoFlags = NIIF_INFO
        self._s.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(nid))

    def _show_menu(self) -> None:
        u = self._u
        menu = u.CreatePopupMenu()
        ids: dict[int, str] = {}
        for i, item in enumerate(self._menu_items(), start=1):
            if item is None:
                u.AppendMenuW(menu, MF_SEPARATOR, 0, None)
            else:
                u.AppendMenuW(menu, MF_STRING, i, item[1])
                ids[i] = item[0]
        pt = wintypes.POINT()
        u.GetCursorPos(ctypes.byref(pt))
        u.SetForegroundWindow(self._hwnd)  # sinon le menu ne se ferme pas en cliquant ailleurs
        cmd = u.TrackPopupMenu(menu, TPM_RIGHTBUTTON | TPM_NONOTIFY | TPM_RETURNCMD, pt.x, pt.y, 0, self._hwnd, None)
        u.PostMessageW(self._hwnd, WM_NULL, 0, 0)
        u.DestroyMenu(menu)
        if cmd in ids:
            self._actions.put(ids[cmd])

    def _proc(self, hwnd, msg, wparam, lparam):
        try:
            if msg == WM_TRAY:
                event = lparam & 0xFFFF
                if event == WM_LBUTTONUP:
                    self._actions.put("toggle_panel")
                elif event == WM_RBUTTONUP:
                    self._show_menu()
                return 0
            if msg == WM_REFRESH:
                self._refresh()
                return 0
            if msg == WM_SHOW_PANEL:
                self._actions.put("show_panel")
                return 0
            if msg == self._taskbar_created:  # explorateur relancé : l'icône a disparu
                self._add_icon()
                return 0
            if msg == WM_CLOSE:
                self._u.DestroyWindow(hwnd)
                return 0
            if msg == WM_DESTROY:
                nid = self._nid(0)
                self._s.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(nid))
                ctypes.windll.user32.PostQuitMessage(0)
                return 0
        except Exception as e:  # une exception ici ne doit pas faire tomber Windows... ni le tracker
            print(f"[zone de notification] Erreur : {e}")
        return self._u.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _run(self) -> None:
        self._hicon = None
        loaded_icon = False
        cls_name = f"EyetrackTray{id(self)}"
        hinst = None
        try:
            self._u, k, self._s = _api()
            u = self._u
            hinst = k.GetModuleHandleW(None)
            self._wndproc = WNDPROC(self._proc)  # garder la référence (sinon le rappel est libéré)
            wc = WNDCLASSW()
            wc.lpfnWndProc = self._wndproc
            wc.hInstance = hinst
            wc.lpszClassName = cls_name
            if not u.RegisterClassW(ctypes.byref(wc)):
                raise OSError(f"RegisterClassW : erreur {ctypes.get_last_error()}")
            self._taskbar_created = u.RegisterWindowMessageW("TaskbarCreated")
            self._hwnd = u.CreateWindowExW(0, cls_name, "Eyetrack", 0, 0, 0, 0, 0, None, None, hinst, None)
            if not self._hwnd:
                raise OSError(f"CreateWindowExW : erreur {ctypes.get_last_error()}")
            if self.icon_path:
                self._hicon = u.LoadImageW(None, str(self.icon_path), IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE)
                loaded_icon = bool(self._hicon)
            if not self._hicon:
                self._hicon = u.LoadIconW(None, ctypes.c_void_p(IDI_APPLICATION))
            self.available = self._add_icon()
            self._ready.set()
            if not self.available:
                u.DestroyWindow(self._hwnd)
                return
            msg = wintypes.MSG()
            while u.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                u.TranslateMessage(ctypes.byref(msg))
                u.DispatchMessageW(ctypes.byref(msg))
        except Exception as e:
            print(f"[zone de notification] Indisponible : {e}")
        finally:
            self._ready.set()
            self._hwnd = 0
            try:
                if loaded_icon and self._hicon:
                    self._u.DestroyIcon(self._hicon)
                if hinst is not None:
                    self._u.UnregisterClassW(cls_name, hinst)
            except Exception:
                pass
