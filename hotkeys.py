"""Raccourcis clavier globaux (Windows) : actifs même quand le jeu a le focus.

Les touches de la fenêtre « Moniteur » ne marchent que si elle a le focus : en jeu, `R` part
au jeu. Ici, les combinaisons sont enregistrées auprès de Windows (RegisterHotKey) et reçues
dans un thread dédié, quelle que soit la fenêtre active.

Attention : beaucoup de jeux lisent le clavier en direct (Raw Input / DirectInput) et voient
quand même les touches d'une combinaison. Il faut donc une touche que le jeu n'utilise pas :
les lettres sont à éviter (BeamNG : R = réinitialiser le véhicule), d'où Ctrl+Maj+F9/F10/F11
par défaut. Les combinaisons se règlent dans le profil (`hotkey_*`).
"""

from __future__ import annotations
import platform
import queue
import threading

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000  # une seule notification par appui, même si la touche reste enfoncée

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

MODIFIERS = {
    "ctrl": MOD_CONTROL, "control": MOD_CONTROL,
    "shift": MOD_SHIFT, "maj": MOD_SHIFT,
    "alt": MOD_ALT,
    "win": MOD_WIN,
}

NAMED_KEYS: dict[str, int] = {
    "space": 0x20, "espace": 0x20,
    "pageup": 0x21, "pagedown": 0x22, "end": 0x23, "fin": 0x23, "home": 0x24,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "insert": 0x2D, "inser": 0x2D, "delete": 0x2E, "suppr": 0x2E,
    "pause": 0x13, "scrolllock": 0x91,
    "numpad*": 0x6A, "numpad+": 0x6B, "numpad-": 0x6D, "numpad.": 0x6E, "numpad/": 0x6F,
}
NAMED_KEYS.update({f"f{i}": 0x70 + i - 1 for i in range(1, 25)})      # F1..F24
NAMED_KEYS.update({f"numpad{i}": 0x60 + i for i in range(10)})        # pavé numérique 0..9
NAMED_KEYS.update({chr(c): c for c in range(ord("0"), ord("9") + 1)})  # chiffres
NAMED_KEYS.update({chr(c).lower(): c for c in range(ord("A"), ord("Z") + 1)})  # lettres


def parse_hotkey(text: str) -> tuple[int, int]:
    """Convertit « ctrl+shift+f9 » en (modificateurs, code de touche virtuelle Windows).

    Lève ValueError si la combinaison est invalide (touche inconnue, pas de touche, etc.).
    """
    parts = [p.strip().lower() for p in str(text).replace(" ", "").split("+") if p.strip()]
    if not parts:
        raise ValueError(f"raccourci vide : {text!r}")
    mods = 0
    for p in parts[:-1]:
        if p not in MODIFIERS:
            raise ValueError(f"modificateur inconnu « {p} » dans {text!r} (ctrl, shift, alt, win)")
        mods |= MODIFIERS[p]
    key = parts[-1]
    if key in MODIFIERS:
        raise ValueError(f"{text!r} : il manque la touche après les modificateurs")
    if key not in NAMED_KEYS:
        raise ValueError(f"touche inconnue « {key} » dans {text!r}")
    return mods, NAMED_KEYS[key]


def format_hotkey(text: str) -> str:
    """Affichage lisible : « ctrl+shift+f9 » -> « Ctrl+Maj+F9 »."""
    names = {"ctrl": "Ctrl", "control": "Ctrl", "shift": "Maj", "maj": "Maj", "alt": "Alt", "win": "Win"}
    parts = [p.strip().lower() for p in str(text).replace(" ", "").split("+") if p.strip()]
    return "+".join(names.get(p, p.upper() if len(p) <= 3 else p.capitalize()) for p in parts)


class GlobalHotkeys:
    """Écoute des raccourcis globaux dans un thread dédié (file de messages Windows).

    Usage : `start()`, puis `poll()` à chaque image pour récupérer les actions déclenchées,
    et `stop()` à la fermeture. Sans Windows, rien n'est enregistré (poll() reste vide).
    """

    def __init__(self, bindings: dict[str, str]) -> None:
        # action -> combinaison texte (ex. {"recenter": "ctrl+shift+f9"}) ; vide = désactivé
        self.bindings = {a: c for a, c in bindings.items() if c and str(c).strip()}
        self.status: dict[str, bool] = {}  # action -> enregistrement réussi
        self.errors: dict[str, str] = {}   # action -> raison de l'échec
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._ready = threading.Event()

    def start(self) -> dict[str, bool]:
        """Enregistre les raccourcis ; renvoie {action: succès}."""
        parsed: dict[str, tuple[int, int]] = {}
        for action, combo in self.bindings.items():
            try:
                parsed[action] = parse_hotkey(combo)
            except ValueError as e:
                self.status[action] = False
                self.errors[action] = str(e)
        if platform.system() != "Windows":
            for action in parsed:
                self.status[action] = False
                self.errors[action] = "raccourcis globaux disponibles sous Windows seulement"
            return dict(self.status)
        if parsed:
            self._thread = threading.Thread(target=self._run, args=(parsed,), daemon=True)
            self._thread.start()
            self._ready.wait(timeout=2.0)
        return dict(self.status)

    def _run(self, parsed: dict[str, tuple[int, int]]) -> None:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
        user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT,
                                        wintypes.UINT, wintypes.UINT]

        self._thread_id = kernel32.GetCurrentThreadId()
        msg = wintypes.MSG()
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)  # crée la file de messages du thread

        ids: dict[int, str] = {}
        for i, (action, (mods, vk)) in enumerate(parsed.items(), start=1):
            ok = bool(user32.RegisterHotKey(None, i, mods | MOD_NOREPEAT, vk))
            self.status[action] = ok
            if ok:
                ids[i] = action
            else:
                self.errors[action] = "déjà utilisé par un autre programme"
        self._ready.set()

        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY and int(msg.wParam) in ids:
                    self._queue.put(ids[int(msg.wParam)])
        finally:
            for i in ids:
                user32.UnregisterHotKey(None, i)

    def poll(self) -> list[str]:
        """Actions déclenchées depuis le dernier appel (dans l'ordre)."""
        actions = []
        while True:
            try:
                actions.append(self._queue.get_nowait())
            except queue.Empty:
                return actions

    def stop(self) -> None:
        """Libère les raccourcis et arrête le thread."""
        if self._thread is not None and self._thread.is_alive() and self._thread_id:
            import ctypes
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            self._thread.join(timeout=1.0)
        self._thread = None
