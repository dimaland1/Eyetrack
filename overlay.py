"""Module d'overlay transparent et click-through du point de regard (Étape 6).

Affiche un réticule visuel fluide à l'écran sans interférer avec le jeu :
- Fenêtre plein écran transparente et 'Always on Top'.
- Totalement 'Click-Through' via les styles Win32 (WS_EX_TRANSPARENT | WS_EX_LAYERED).
  Tous les clics de souris et interactions clavier passent directement au jeu sous-jacent.
- Exécuté dans un thread léger dédié sans impacter le framerate du tracker.
"""

from __future__ import annotations
import ctypes
import platform
import queue
import threading
import time
import tkinter as tk


class GazeOverlay:
    """Gestionnaire de fenêtre overlay transparente et click-through."""

    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        self.width = width
        self.height = height
        self.is_running = False
        self.is_visible = False

        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._thread: threading.Thread | None = None
        self._root: tk.Tk | None = None
        self._canvas: tk.Canvas | None = None
        self._reticle_outer: int | None = None
        self._reticle_inner: int | None = None

        self._current_x = width // 2
        self._current_y = height // 2
        # Tkinter n'est pas thread-safe : les demandes d'affichage/masquage venant du thread
        # principal sont mémorisées ici et appliquées par le thread Tk (poll_position).
        self._visibility_dirty = False
        self.hwnd: int = 0  # handle Win32 de la fenêtre, connu une fois le thread Tk démarré

    def start(self, visible: bool = False) -> None:
        """Démarre le thread d'affichage de l'overlay."""
        if self.is_running:
            return

        self.is_running = True
        self.is_visible = visible
        self._thread = threading.Thread(target=self._run_tk_thread, daemon=True)
        self._thread.start()

    def _apply_click_through(self, root: tk.Tk) -> None:
        """Applique les styles Win32 pour rendre la fenêtre insensible aux clics (click-through)."""
        if platform.system() != "Windows":
            return

        root.update_idletasks()
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetParent(root.winfo_id())
            if hwnd == 0:
                hwnd = root.winfo_id()
            self.hwnd = hwnd  # handle Win32 de la fenêtre (diagnostic / tests)

            GWL_EXSTYLE = -20
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_LAYERED = 0x00080000
            WS_EX_TOOLWINDOW = 0x00000080
            WS_EX_TOPMOST = 0x00000008

            current_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            new_style = current_style | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW | WS_EX_TOPMOST
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
        except Exception as e:
            print(f"[overlay] Erreur styles Win32 : {e}")

    def _run_tk_thread(self) -> None:
        """Boucle Tkinter exécutée dans le thread dédié."""
        try:
            self._root = tk.Tk()
            self._root.overrideredirect(True)
            self._root.geometry(f"{self.width}x{self.height}+0+0")
            self._root.attributes("-topmost", True)

            # Couleur magique de transparence
            trans_color = "#000001"
            self._root.attributes("-transparentcolor", trans_color)
            self._root.config(bg=trans_color)

            self._canvas = tk.Canvas(
                self._root,
                width=self.width,
                height=self.height,
                bg=trans_color,
                highlightthickness=0,
            )
            self._canvas.pack(fill="both", expand=True)

            # Création du réticule (cercle extérieur cyan + point central)
            r_out = 16
            r_in = 4
            self._reticle_outer = self._canvas.create_oval(
                -50, -50, -50, -50,
                outline="#00E5FF",
                width=2,
            )
            self._reticle_inner = self._canvas.create_oval(
                -50, -50, -50, -50,
                fill="#FF9100",
                outline="",
            )

            self._apply_click_through(self._root)

            # Gestion de la visibilité initiale
            if not self.is_visible:
                self._root.withdraw()

            # Boucle périodique de rafraîchissement (60 Hz)
            def poll_position():
                if not self.is_running:
                    self._root.destroy()
                    return

                # Affichage / masquage demandé par toggle() (appliqué dans le thread Tk)
                if self._visibility_dirty:
                    self._visibility_dirty = False
                    if self.is_visible:
                        self._root.deiconify()
                        self._root.attributes("-topmost", True)
                        self._apply_click_through(self._root)
                    else:
                        self._root.withdraw()

                # Dépilement des dernières coordonnées
                updated = False
                while not self._queue.empty():
                    try:
                        self._current_x, self._current_y = self._queue.get_nowait()
                        updated = True
                    except queue.Empty:
                        break

                if updated and self.is_visible and self._canvas is not None and self._reticle_outer is not None:
                    cx, cy = self._current_x, self._current_y
                    r_o = 15
                    r_i = 3
                    self._canvas.coords(self._reticle_outer, cx - r_o, cy - r_o, cx + r_o, cy + r_o)
                    self._canvas.coords(self._reticle_inner, cx - r_i, cy - r_i, cx + r_i, cy + r_i)

                self._root.after(16, poll_position)

            self._root.after(16, poll_position)
            self._root.mainloop()

        except Exception as e:
            print(f"[overlay] Erreur thread overlay : {e}")
        finally:
            # Les objets Tk doivent être libérés par CE thread : sinon Python les détruit depuis
            # le thread principal à la fermeture -> « Tcl_AsyncDelete: async handler deleted by
            # the wrong thread » et code de sortie 3.
            self._canvas = None
            self._reticle_outer = None
            self._reticle_inner = None
            self._root = None
            import gc
            gc.collect()

    def update_gaze(self, x_px: float, y_px: float) -> None:
        """Met à jour les coordonnées du point de regard (thread-safe)."""
        if not self.is_running or not self.is_visible:
            return
        ix = int(round(x_px))
        iy = int(round(y_px))
        try:
            # Écrase les coordonnées anciennes pour garder la plus récente
            if self._queue.full():
                self._queue.get_nowait()
            self._queue.put_nowait((ix, iy))
        except (queue.Full, queue.Empty):
            pass

    def toggle(self) -> bool:
        """Active ou désactive l'affichage de l'overlay.

        Appelé depuis le thread principal : on ne touche pas à Tkinter ici (non thread-safe,
        risque de blocage), le thread Tk applique le changement à son prochain rafraîchissement.
        """
        self.is_visible = not self.is_visible
        self._visibility_dirty = True
        return self.is_visible

    def stop(self) -> None:
        """Arrête l'overlay et attend que son thread ait libéré Tkinter."""
        self.is_running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
