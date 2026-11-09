"""Panneau de contrôle (Tkinter) : état du tracker, commandes et réglages en direct.

Le panneau tourne dans son propre thread (comme l'overlay) : Tkinter n'est pas thread-safe,
donc il n'échange avec la boucle principale que par deux canaux simples :
- `publish(status)` : la boucle principale dépose l'état courant (dict) à chaque image ;
- `poll()` : la boucle principale récupère les commandes de l'utilisateur, sous la forme
  ("action", nom), ("profile", nom) ou ("set", champ_du_profil, valeur).

Les réglages modifient le profil actif en direct ; « Sauvegarder » les écrit dans son JSON.
Fermer le panneau arrête le tracker.
"""

from __future__ import annotations
import queue
import threading
import tkinter as tk
from tkinter import ttk

# Curseurs : (champ du profil, libellé, min, max, pas)
HEAD_SLIDERS = [
    ("head_yaw_gain", "Gain gauche / droite", 0.5, 4.0, 0.1),
    ("head_pitch_gain", "Gain haut / bas", 0.5, 3.0, 0.1),
    ("pose_min_cutoff", "Lissage au repos (bas = plus calme)", 0.2, 3.0, 0.1),
]
GAZE_SLIDERS = [
    ("gaze_yaw_gain", "Part du regard gauche / droite", 0.0, 1.5, 0.05),
    ("gaze_pitch_gain", "Part du regard haut / bas", 0.0, 1.5, 0.05),
    ("gaze_deadzone", "Zone morte au centre", 0.0, 0.3, 0.01),
    ("gaze_smoothing_s", "Lissage (s, haut = plus calme)", 0.05, 1.0, 0.05),
]
CHECKS = [
    ("recalib_on_click", "Recalibrer le regard sur les clics (bureau, curseur visible)"),
]
PANEL_FIELDS = [f for f, *_ in HEAD_SLIDERS + GAZE_SLIDERS] + [f for f, _ in CHECKS]

GREEN, RED, ORANGE, GREY = "#1a7f37", "#c62828", "#e65100", "#666666"


class ControlPanel:
    """Fenêtre de contrôle du tracker, exécutée dans un thread dédié."""

    def __init__(self, title: str = "Eyetrack - Panneau de contrôle") -> None:
        self.title = title
        self.is_running = False
        self._commands: queue.Queue = queue.Queue()
        self._status: dict = {}
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        # Widgets (créés et utilisés uniquement dans le thread Tk)
        self._root: tk.Tk | None = None
        self._labels: dict[str, tk.Label] = {}
        self._scales: dict[str, tk.Scale] = {}
        self._checks: dict[str, tk.BooleanVar] = {}
        self._buttons: dict[str, ttk.Button] = {}
        self._profile_box: ttk.Combobox | None = None
        self._hint: tk.Label | None = None
        self._shown_version = -1
        self._shown_profiles: tuple = ()
        self._updating = False  # vrai pendant une mise à jour des widgets par le programme

    # ------------------------------------------------------------------ interface (thread principal)
    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._run, name="panneau", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=3.0)

    def publish(self, status: dict) -> None:
        """Dépose l'état courant du tracker (remplacement atomique du dict)."""
        self._status = status

    def poll(self) -> list[tuple]:
        """Commandes envoyées par l'utilisateur depuis le dernier appel."""
        out = []
        while True:
            try:
                out.append(self._commands.get_nowait())
            except queue.Empty:
                return out

    def stop(self) -> None:
        self.is_running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)

    # ------------------------------------------------------------------ thread Tk
    def _send(self, *cmd) -> None:
        self._commands.put(tuple(cmd))

    def _run(self) -> None:
        try:
            self._build()
            self._ready.set()
            self._root.after(100, self._refresh)
            self._root.mainloop()
        except Exception as e:
            print(f"[panneau] Erreur : {e}")
        finally:
            self._ready.set()
            # Objets Tk libérés dans CE thread (cf. overlay.py : sinon Tcl_AsyncDelete à la sortie)
            self._labels.clear()
            self._scales.clear()
            self._checks.clear()
            self._buttons.clear()
            self._profile_box = None
            self._hint = None
            self._root = None
            import gc
            gc.collect()

    def _build(self) -> None:
        root = tk.Tk()
        self._root = root
        root.title(self.title)
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", lambda: self._send("action", "quit"))
        pad = {"padx": 8, "pady": 4}

        # --- État
        st = ttk.LabelFrame(root, text="État")
        st.grid(row=0, column=0, sticky="ew", **pad)
        rows = [("face", "Visage"), ("camera", "Caméra"), ("opentrack", "OpenTrack"),
                ("gaze", "Regard"), ("calib", "Calibration"), ("mode", "Caméra du jeu")]
        for i, (key, text) in enumerate(rows):
            tk.Label(st, text=text + " :", anchor="w", width=13).grid(row=i, column=0, sticky="w", padx=6)
            lbl = tk.Label(st, text="…", anchor="w", width=34)
            lbl.grid(row=i, column=1, sticky="w")
            self._labels[key] = lbl

        # --- Commandes
        cmd = ttk.LabelFrame(root, text="Commandes")
        cmd.grid(row=1, column=0, sticky="ew", **pad)
        buttons = [
            ("recenter", "Recentrer"), ("pause", "Pause"), ("gaze", "Couper le regard"),
            ("calibrate", "Calibrer le regard"), ("overlay", "Afficher l'overlay"), ("quit", "Quitter"),
        ]
        for i, (action, text) in enumerate(buttons):
            b = ttk.Button(cmd, text=text, width=19, command=lambda a=action: self._send("action", a))
            b.grid(row=i // 3, column=i % 3, padx=4, pady=3)
            self._buttons[action] = b

        # --- Profil
        pf = ttk.LabelFrame(root, text="Profil")
        pf.grid(row=2, column=0, sticky="ew", **pad)
        box = ttk.Combobox(pf, state="readonly", width=24)
        box.grid(row=0, column=0, padx=6, pady=4)
        box.bind("<<ComboboxSelected>>", lambda _e: self._send("profile", box.get()))
        self._profile_box = box
        ttk.Button(pf, text="Sauvegarder les réglages", command=lambda: self._send("action", "save")).grid(
            row=0, column=1, padx=6, pady=4)

        # --- Réglages
        def sliders(parent, spec):
            for i, (field, text, lo, hi, step) in enumerate(spec):
                tk.Label(parent, text=text, anchor="w", width=32).grid(row=i, column=0, sticky="w", padx=6)
                sc = tk.Scale(parent, from_=lo, to=hi, resolution=step, orient="horizontal", length=230,
                              showvalue=True, command=lambda v, f=field, s=step: self._on_scale(f, v, s))
                sc.grid(row=i, column=1, padx=6)
                self._scales[field] = sc

        head = ttk.LabelFrame(root, text="Tête")
        head.grid(row=3, column=0, sticky="ew", **pad)
        sliders(head, HEAD_SLIDERS)
        gaze = ttk.LabelFrame(root, text="Regard")
        gaze.grid(row=4, column=0, sticky="ew", **pad)
        sliders(gaze, GAZE_SLIDERS)
        for i, (field, text) in enumerate(CHECKS):
            # master explicite : sinon Tkinter prend la première fenêtre Tk du programme, celle de
            # l'overlay (autre thread) -> case affichée « − » et variable lue dans le mauvais thread
            var = tk.BooleanVar(master=root, value=False)
            ttk.Checkbutton(gaze, text=text, variable=var,
                            command=lambda f=field, v=var: self._on_check(f, v)).grid(
                row=len(GAZE_SLIDERS) + i, column=0, columnspan=2, sticky="w", padx=6, pady=2)
            self._checks[field] = var

        self._hint = tk.Label(root, text="", fg=GREY, justify="left")
        self._hint.grid(row=5, column=0, sticky="w", padx=10, pady=(0, 8))

    def _on_scale(self, field: str, value: str, step: float) -> None:
        if self._updating:
            return
        v = round(round(float(value) / step) * step, 4)
        self._send("set", field, v)

    def _on_check(self, field: str, var: tk.BooleanVar) -> None:
        if not self._updating:
            self._send("set", field, bool(var.get()))

    def _refresh(self) -> None:
        if not self.is_running:
            self._root.destroy()
            return
        s = self._status
        try:
            if s:
                self._update_status(s)
                self._sync_profile(s)
        except Exception as e:  # le panneau ne doit jamais faire tomber le tracker
            print(f"[panneau] Erreur d'affichage : {e}")
        self._root.after(150, self._refresh)

    def _set(self, key: str, text: str, color: str = "black") -> None:
        lbl = self._labels[key]
        if lbl.cget("text") != text or lbl.cget("fg") != color:
            lbl.config(text=text, fg=color)

    def _update_status(self, s: dict) -> None:
        if s.get("calibrating"):
            self._set("face", "calibration en cours…", ORANGE)
        elif s.get("face"):
            self._set("face", "clignement" if s.get("blinking") else "détecté", ORANGE if s.get("blinking") else GREEN)
        else:
            self._set("face", "perdu (valeur maintenue)", RED)
        self._set("camera", f"{s.get('fps', 0):.0f} images/s", GREEN if s.get("fps", 0) >= 24 else ORANGE)
        hz, gap = s.get("send_hz", 0.0), s.get("gap_ms", 0.0)
        if hz > 0:
            self._set("opentrack", f"{hz:.0f} envois/s, plus long trou {gap:.0f} ms",
                      GREEN if gap <= 20 else ORANGE)
        else:
            self._set("opentrack", "en attente du visage", GREY)
        self._set("gaze", "actif" if s.get("gaze_enabled") else "coupé", GREEN if s.get("gaze_enabled") else GREY)
        rls = s.get("rls", 0)
        self._set("calib", ("calibré" + (f" (+{rls} clics)" if rls else "")) if s.get("calibrated")
                  else "non calibré : bouton « Calibrer »", GREEN if s.get("calibrated") else RED)
        self._set("mode", "PAUSE : au centre" if s.get("paused") else "suit la tête",
                  ORANGE if s.get("paused") else GREEN)
        self._buttons["pause"].config(text="Reprendre" if s.get("paused") else "Pause")
        self._buttons["gaze"].config(text="Réactiver le regard" if not s.get("gaze_enabled") else "Couper le regard")
        self._buttons["overlay"].config(text="Masquer l'overlay" if s.get("overlay") else "Afficher l'overlay")
        hint = s.get("hotkey_hint", "")
        if self._hint.cget("text") != hint:
            self._hint.config(text=hint)

    def _sync_profile(self, s: dict) -> None:
        """Recharge les curseurs quand le profil change (pas pendant que l'utilisateur les bouge)."""
        profiles = tuple(s.get("profiles", ()))
        if profiles != self._shown_profiles:
            self._profile_box.config(values=list(profiles))
            self._shown_profiles = profiles
        version = s.get("profile_version", 0)
        if version == self._shown_version:
            return
        self._shown_version = version
        values = s.get("profile_values", {})
        self._updating = True
        try:
            self._profile_box.set(s.get("profile_name", ""))
            for field, sc in self._scales.items():
                if field in values:
                    sc.set(values[field])
            for field, var in self._checks.items():
                if field in values:
                    var.set(bool(values[field]))
            self._root.update_idletasks()  # les rappels des curseurs partent ici, encore gardés
        finally:
            self._updating = False
