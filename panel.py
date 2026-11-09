"""Panneau de contrôle (Tkinter) : état du tracker, commandes et réglages en direct.

Le panneau tourne dans son propre thread (comme l'overlay) : Tkinter n'est pas thread-safe,
donc il n'échange avec la boucle principale que par deux canaux simples :
- `publish(status)` : la boucle principale dépose l'état courant (dict) à chaque image ;
- `poll()` : la boucle principale récupère les commandes de l'utilisateur, sous la forme
  ("action", nom), ("profile", nom) ou ("set", champ_du_profil, valeur) ;
- aperçu caméra : quand `preview_enabled` est vrai (bouton « Voir la caméra »), la boucle
  principale dépose une vignette avec `publish_preview()` (~10 images/s). Éteint par défaut :
  en jeu, l'image ne sert à rien et ne coûte alors rien.

Les réglages modifient le profil actif en direct ; « Sauvegarder » les écrit dans son JSON.
Avec l'icône de la zone de notification, fermer le panneau le réduit seulement (le tracker
continue) ; sans elle, fermer le panneau arrête le tracker.
"""

from __future__ import annotations
import queue
import threading
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

# Curseurs : (champ du profil, libellé, min, max, pas)
# Toujours visibles : les trois réglages qu'on change vraiment d'un jeu à l'autre.
ESSENTIAL_SLIDERS = [
    ("head_yaw_gain", "Tête : amplification gauche / droite", 0.5, 4.0, 0.1),
    ("gaze_yaw_gain", "Regard : part dans la caméra", 0.0, 1.5, 0.05),
    ("gaze_smoothing_s", "Regard : lissage (s, haut = plus calme)", 0.05, 1.0, 0.05),
]
# Section « Réglages avancés », repliée par défaut
ADVANCED_SLIDERS = [
    ("head_pitch_gain", "Tête : amplification haut / bas", 0.5, 3.0, 0.1),
    ("pose_min_cutoff", "Tête : lissage au repos (bas = plus calme)", 0.2, 3.0, 0.1),
    ("gaze_pitch_gain", "Regard : part haut / bas", 0.0, 1.5, 0.05),
    ("gaze_deadzone", "Regard : zone morte au centre", 0.0, 0.3, 0.01),
]
CHECKS = [
    ("recalib_on_click", "Recalibrer le regard sur les clics (bureau, curseur visible)"),
]
PANEL_FIELDS = [f for f, *_ in ESSENTIAL_SLIDERS + ADVANCED_SLIDERS] + [f for f, _ in CHECKS]

GREEN, RED, ORANGE, GREY = "#1a7f37", "#c62828", "#e65100", "#666666"

PREVIEW_WIDTH = 320  # largeur de la vignette caméra (px)


def make_preview(frame_bgr: np.ndarray, face_detected: bool, iris_px: list[tuple[float, float]],
                 blinking: bool = False, width: int = PREVIEW_WIDTH) -> np.ndarray:
    """Vignette RGB de l'image caméra pour le panneau.

    Points jaunes sur les iris détectés (ils doivent tomber au centre des pupilles, sans être
    happés par un reflet des lunettes) ; cadre vert = visage suivi, orange = clignement,
    rouge = visage perdu.
    """
    h, w = frame_bgr.shape[:2]
    scale = width / float(w)
    small = cv2.resize(frame_bgr, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    if face_detected:
        for x, y in iris_px:
            cv2.circle(small, (int(round(x * scale)), int(round(y * scale))), 3, (0, 255, 255), -1)
        color = (0, 140, 255) if blinking else (40, 180, 40)
    else:
        color = (40, 40, 220)
    cv2.rectangle(small, (0, 0), (small.shape[1] - 1, small.shape[0] - 1), color, 4)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


class ControlPanel:
    """Fenêtre de contrôle du tracker, exécutée dans un thread dédié."""

    def __init__(self, title: str = "Eyetrack - Panneau de contrôle", close_to_tray: bool = False,
                 icon_path: str | None = None) -> None:
        self.title = title
        self.close_to_tray = close_to_tray  # croix = réduire (icône de la zone de notification)
        self.icon_path = icon_path
        self.is_running = False
        # Visibilité : demandée par la boucle principale (icône), appliquée par le thread Tk
        self.is_visible = True
        self._visibility_request: bool | None = None
        self._advanced: ttk.LabelFrame | None = None
        self._advanced_btn: ttk.Button | None = None
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
        # Aperçu caméra : demandé par le bouton (lu par la boucle principale), vignette déposée
        # par la boucle principale (largeur, hauteur, octets RGB, numéro)
        self.preview_enabled = False
        self._preview: tuple[int, int, bytes, int] | None = None
        self._preview_seq = 0
        self._preview_shown_seq = -1
        self._preview_box: ttk.LabelFrame | None = None
        self._preview_label: tk.Label | None = None
        self._preview_photo: tk.PhotoImage | None = None
        self._diag: tk.Label | None = None

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

    def show(self) -> None:
        self._visibility_request = True

    def hide(self) -> None:
        self._visibility_request = False

    def toggle_visible(self) -> bool:
        """Affiche ou masque le panneau ; renvoie le nouvel état demandé."""
        target = not self.is_visible
        self._visibility_request = target
        return target

    def publish_preview(self, rgb: np.ndarray) -> None:
        """Dépose une vignette RGB (uint8, h x w x 3) pour l'aperçu caméra."""
        h, w = rgb.shape[:2]
        self._preview_seq += 1
        self._preview = (w, h, np.ascontiguousarray(rgb, dtype=np.uint8).tobytes(), self._preview_seq)

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
            self._preview_box = None
            self._preview_label = None
            self._preview_photo = None
            self._diag = None
            self._advanced = None
            self._advanced_btn = None
            self._root = None
            import gc
            gc.collect()

    def _build(self) -> None:
        root = tk.Tk()
        self._root = root
        root.title(self.title)
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        if self.icon_path:
            try:
                root.iconbitmap(self.icon_path)
            except Exception:
                pass
        pad = {"padx": 8, "pady": 4}

        # --- État
        st = ttk.LabelFrame(root, text="État")
        st.grid(row=0, column=0, sticky="ew", **pad)
        rows = [("face", "Visage"), ("glare", "Reflets lunettes"), ("camera", "Caméra"), ("opentrack", "OpenTrack"),
                ("gaze", "Regard"), ("calib", "Calibration"), ("mode", "Caméra du jeu")]
        for i, (key, text) in enumerate(rows):
            tk.Label(st, text=text + " :", anchor="w", width=15).grid(row=i, column=0, sticky="w", padx=6)
            lbl = tk.Label(st, text="…", anchor="w", width=40)
            lbl.grid(row=i, column=1, sticky="w")
            self._labels[key] = lbl

        # --- Commandes
        cmd = ttk.LabelFrame(root, text="Commandes")
        cmd.grid(row=1, column=0, sticky="ew", **pad)
        buttons = [
            ("recenter", "Recentrer"), ("pause", "Pause"), ("gaze", "Couper le regard"),
            ("calibrate", "Calibrer le regard"), ("overlay", "Afficher l'overlay"), ("preview", "Voir la caméra"),
        ]
        for i, (action, text) in enumerate(buttons):
            if action == "preview":
                command = self._toggle_preview  # géré par le panneau lui-même
            else:
                command = lambda a=action: self._send("action", a)  # noqa: E731
            b = ttk.Button(cmd, text=text, width=19, command=command)
            b.grid(row=i // 3, column=i % 3, padx=4, pady=3)
            self._buttons[action] = b
        self._buttons["quit"] = ttk.Button(cmd, text="Quitter", width=19, command=lambda: self._send("action", "quit"))
        self._buttons["quit"].grid(row=2, column=2, padx=4, pady=3)

        # --- Aperçu caméra (colonne de droite, masquée par défaut)
        pv = ttk.LabelFrame(root, text="Caméra")
        pv.grid(row=0, column=1, rowspan=6, sticky="n", **pad)
        self._preview_label = tk.Label(pv, text="En attente de l'image…", width=PREVIEW_WIDTH // 8, height=12,
                                       bg="#202020", fg="#bbbbbb")
        self._preview_label.grid(row=0, column=0, padx=6, pady=6)
        self._diag = tk.Label(pv, text="", fg=GREY, justify="left", anchor="w")
        self._diag.grid(row=1, column=0, sticky="w", padx=6)
        tk.Label(pv, justify="left", anchor="w", fg=GREY, wraplength=PREVIEW_WIDTH,
                 text="Points jaunes : iris détectés. Ils doivent rester au centre des pupilles, "
                      "sans sauter sur un reflet des lunettes.\nCadre vert : visage suivi ; "
                      "orange : clignement ; rouge : visage perdu.").grid(row=2, column=0, sticky="w", padx=6, pady=(2, 8))
        self._preview_box = pv
        pv.grid_remove()

        # --- Profil
        pf = ttk.LabelFrame(root, text="Profil")
        pf.grid(row=2, column=0, sticky="ew", **pad)
        box = ttk.Combobox(pf, state="readonly", width=24)
        box.grid(row=0, column=0, padx=6, pady=4)
        box.bind("<<ComboboxSelected>>", lambda _e: self._send("profile", box.get()))
        self._profile_box = box
        ttk.Button(pf, text="Sauvegarder les réglages", command=lambda: self._send("action", "save")).grid(
            row=0, column=1, padx=6, pady=4)

        # --- Réglages (essentiels) et réglages avancés (repliés)
        def sliders(parent, spec):
            for i, (field, text, lo, hi, step) in enumerate(spec):
                tk.Label(parent, text=text, anchor="w", width=38).grid(row=i, column=0, sticky="w", padx=6)
                sc = tk.Scale(parent, from_=lo, to=hi, resolution=step, orient="horizontal", length=200,
                              showvalue=True, command=lambda v, f=field, s=step: self._on_scale(f, v, s))
                sc.grid(row=i, column=1, padx=6)
                self._scales[field] = sc

        rg = ttk.LabelFrame(root, text="Réglages")
        rg.grid(row=3, column=0, sticky="ew", **pad)
        sliders(rg, ESSENTIAL_SLIDERS)
        self._advanced_btn = ttk.Button(rg, text="Réglages avancés ▸", command=self._toggle_advanced)
        self._advanced_btn.grid(row=len(ESSENTIAL_SLIDERS), column=0, sticky="w", padx=6, pady=(2, 6))

        adv = ttk.LabelFrame(root, text="Réglages avancés")
        adv.grid(row=4, column=0, sticky="ew", **pad)
        sliders(adv, ADVANCED_SLIDERS)
        for i, (field, text) in enumerate(CHECKS):
            # master explicite : sinon Tkinter prend la première fenêtre Tk du programme, celle de
            # l'overlay (autre thread) -> case affichée « − » et variable lue dans le mauvais thread
            var = tk.BooleanVar(master=root, value=False)
            ttk.Checkbutton(adv, text=text, variable=var,
                            command=lambda f=field, v=var: self._on_check(f, v)).grid(
                row=len(ADVANCED_SLIDERS) + i, column=0, columnspan=2, sticky="w", padx=6, pady=2)
            self._checks[field] = var
        self._advanced = adv
        adv.grid_remove()

        self._hint = tk.Label(root, text="", fg=GREY, justify="left")
        self._hint.grid(row=5, column=0, sticky="w", padx=10, pady=(0, 8))

    def _toggle_advanced(self) -> None:
        self._advanced_open = not getattr(self, "_advanced_open", False)
        if self._advanced_open:
            self._advanced.grid()
            self._advanced_btn.config(text="Réglages avancés ▾")
        else:
            self._advanced.grid_remove()
            self._advanced_btn.config(text="Réglages avancés ▸")

    def _on_close(self) -> None:
        """Croix de la fenêtre : réduire dans la zone de notification, ou quitter sans icône."""
        if self.close_to_tray:
            self._apply_visible(False)
            self._send("action", "panel_hidden")
        else:
            self._send("action", "quit")

    def _apply_visible(self, visible: bool) -> None:
        if visible:
            self._root.deiconify()
            self._root.lift()
            self._root.attributes("-topmost", True)   # passer devant le jeu ou le bureau…
            self._root.after(300, lambda: self._root.attributes("-topmost", False))  # …sans y rester
            self._root.focus_force()
        else:
            self._root.withdraw()
        self.is_visible = visible

    def _on_scale(self, field: str, value: str, step: float) -> None:
        if self._updating:
            return
        v = round(round(float(value) / step) * step, 4)
        self._send("set", field, v)

    def _on_check(self, field: str, var: tk.BooleanVar) -> None:
        if not self._updating:
            self._send("set", field, bool(var.get()))

    def _toggle_preview(self) -> None:
        self.preview_enabled = not self.preview_enabled
        if self.preview_enabled:
            self._preview_box.grid()
        else:
            self._preview_box.grid_remove()
            self._preview_label.config(image="", text="En attente de l'image…", width=PREVIEW_WIDTH // 8, height=12)
            self._preview_photo = None
            self._preview_shown_seq = -1
        self._buttons["preview"].config(text="Masquer la caméra" if self.preview_enabled else "Voir la caméra")

    def _update_preview(self) -> None:
        pv = self._preview
        if not self.preview_enabled or pv is None or pv[3] == self._preview_shown_seq:
            return
        w, h, data, seq = pv
        # Image PPM brute : Tk l'affiche sans dépendance supplémentaire (master explicite, cf. CHECKS)
        photo = tk.PhotoImage(master=self._root, data=b"P6\n%d %d\n255\n" % (w, h) + data, format="PPM")
        self._preview_label.config(image=photo, text="", width=w, height=h)
        self._preview_photo = photo  # garder une référence, sinon l'image disparaît
        self._preview_shown_seq = seq

    def _refresh(self) -> None:
        if not self.is_running:
            self._root.destroy()
            return
        s = self._status
        try:
            req = self._visibility_request
            if req is not None:
                self._visibility_request = None
                if req != self.is_visible:
                    self._apply_visible(req)
            if s:
                self._update_status(s)
                self._sync_profile(s)
            self._update_preview()
        except Exception as e:  # le panneau ne doit jamais faire tomber le tracker
            print(f"[panneau] Erreur d'affichage : {e}")
        self._root.after(100, self._refresh)

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
        glare, glare_lbl = s.get("glare", 0.0), s.get("glare_label", "aucun")
        if not s.get("face"):
            self._set("glare", "—", GREY)
        else:
            advice = " : bougez la lampe ou l'écran" if glare_lbl == "gênant" else ""
            self._set("glare", f"{glare_lbl} ({glare * 100:.0f} % de l'iris){advice}",
                      {"aucun": GREEN, "léger": ORANGE}.get(glare_lbl, RED))
        rls = s.get("rls", 0)
        if s.get("calibrated"):
            q, qdeg = s.get("calib_quality", "inconnue"), s.get("calib_deg", float("nan"))
            text = "calibré" if q == "inconnue" else f"qualité {q} ({qdeg:.1f}°)"
            if rls:
                text += f", +{rls} clics"
            self._set("calib", text, {"bonne": GREEN, "inconnue": GREEN, "moyenne": ORANGE}.get(q, RED))
        else:
            self._set("calib", "non calibré : bouton « Calibrer le regard »", RED)
        self._set("mode", "PAUSE : au centre" if s.get("paused") else "suit la tête",
                  ORANGE if s.get("paused") else GREEN)
        self._buttons["pause"].config(text="Reprendre" if s.get("paused") else "Pause")
        self._buttons["gaze"].config(text="Réactiver le regard" if not s.get("gaze_enabled") else "Couper le regard")
        self._buttons["overlay"].config(text="Masquer l'overlay" if s.get("overlay") else "Afficher l'overlay")
        hint = s.get("hotkey_hint", "")
        if self._hint.cget("text") != hint:
            self._hint.config(text=hint)
        if self.preview_enabled:
            diag = s.get("diag", "")
            if self._diag.cget("text") != diag:
                self._diag.config(text=diag)

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
