"""Module d'acquisition vidéo depuis la webcam.

Gère la négociation du mode de capture (format, résolution, framerate réellement mesuré)
et fournit un horodatage précis pour chaque image capturée.
"""

from __future__ import annotations
import json
import os
import platform
import time
import cv2
import numpy as np

from paths import data_path

# Mémorise le mode de capture qui fonctionne (évite ~10-25 s de recherche à chaque lancement)
CAMERA_CACHE_FILE = data_path("camera_mode.json")
# Version des entrées mémorisées. Avant la version 2, l'ordre des réglages perdait le format
# MJPG sous DirectShow : la caméra retombait en non compressé, le 720p ne dépassait pas
# 10 images/s et le 640x480 était mémorisé à tort. Ces anciennes entrées sont ignorées.
CAMERA_CACHE_VERSION = 2
_BACKENDS = {"dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}
# Un mode de repli (résolution plus basse) ne remplace le mode demandé que s'il est nettement
# plus rapide. Dans une pièce sombre, tous les modes sont lents de la même façon (pose plus
# longue) : baisser la résolution n'apporterait rien.
FALLBACK_SPEEDUP = 1.5


class WebcamCapture:
    """Gestionnaire de capture vidéo pour webcam USB / intégrée."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 60,
        strict: bool = False,
    ) -> None:
        """Initialise et configure la webcam.

        Args:
            camera_index: Index de la caméra (généralement 0).
            width: Largeur demandée en pixels.
            height: Hauteur demandée en pixels.
            target_fps: Framerate demandé en images/seconde.
            strict: Utiliser exactement la résolution demandée (pas de repli ni de mémorisation),
                pour les diagnostics qui comparent les résolutions.
        """
        self.camera_index = camera_index
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = target_fps
        self.strict = strict

        self.cap: cv2.VideoCapture | None = None
        self.actual_width: int = 0
        self.actual_height: int = 0
        self.actual_fps: float = 0.0

        # Mesure de FPS réels
        self._frame_count: int = 0
        self._fps_start_time: float = time.perf_counter()
        self.measured_fps: float = 0.0

        self._open()

    @staticmethod
    def _fourcc_str(value: float) -> str:
        v = int(value)
        return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))

    def _configure(self, cap: cv2.VideoCapture, backend: int, width: int, height: int, mjpg: bool) -> None:
        """Applique framerate, résolution et format.

        Chaque changement reconstruit le graphe DirectShow (~2,5 s mesurés par réglage) :
        on n'envoie que le strict nécessaire.

        Avec DirectShow, le format doit être réglé EN DERNIER. OpenCV (cap_dshow.cpp) oublie le
        format demandé dès qu'un réglage a réussi, puis reconfigure la caméra sans lui au réglage
        suivant de la taille ou du framerate ; il reprend alors le premier format accepté de sa
        liste, où le non compressé (YUY2) passe avant MJPG. Mesuré sur une Logitech C920 avec le
        format réglé en premier : 720p à 10 images/s et 1080p à 5, contre 24 à 30 en MJPG.
        """
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        if backend == cv2.CAP_DSHOW:
            cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if mjpg:
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            return
        if mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, self.requested_fps)

    @staticmethod
    def _measure_fps(cap: cv2.VideoCapture, duration: float = 0.7) -> float:
        """Mesure le framerate réellement délivré par la caméra."""
        for _ in range(3):
            cap.read()
        n = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            ok, _ = cap.read()
            n += int(ok)
        return n / max(time.perf_counter() - t0, 1e-6)

    def _cache_key(self) -> str:
        return f"{self.camera_index}:{self.requested_width}x{self.requested_height}@{self.requested_fps}"

    def _load_cached_mode(self) -> tuple | None:
        try:
            with open(CAMERA_CACHE_FILE, "r", encoding="utf-8") as f:
                entry = json.load(f).get(self._cache_key())
            if entry and entry.get("v") == CAMERA_CACHE_VERSION:
                return (entry["label"], _BACKENDS[entry["backend"]], int(entry["w"]), int(entry["h"]),
                        bool(entry["mjpg"]))
        except (OSError, ValueError, KeyError, AttributeError):
            pass
        return None

    def _save_cached_mode(self, cand: tuple) -> None:
        label, backend, w, h, mjpg = cand
        names = {v: k for k, v in _BACKENDS.items()}
        try:
            data = {}
            if os.path.exists(CAMERA_CACHE_FILE):
                with open(CAMERA_CACHE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data[self._cache_key()] = {"v": CAMERA_CACHE_VERSION, "label": label,
                                       "backend": names.get(backend, "any"), "w": w, "h": h, "mjpg": mjpg}
            with open(CAMERA_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except (OSError, ValueError):
            pass

    def _open(self) -> None:
        """Ouvre la webcam en choisissant le mode qui délivre réellement le plus d'images.

        Sous Windows, OpenCV ouvre souvent les webcams en YUY2 (non compressé) : beaucoup
        sont alors limitées à ~10 fps en 720p à cause de la bande passante USB. On essaie
        donc MJPG (compressé) puis une résolution plus basse, et on garde le premier mode
        qui atteint un framerate correct. Le mode retenu est mémorisé dans camera_mode.json
        pour ne pas refaire la recherche à chaque démarrage (supprimer ce fichier pour la relancer).
        """
        w, h = self.requested_width, self.requested_height
        if platform.system() == "Windows":
            # Media Foundation (CAP_MSMF) volontairement absent : sur certaines machines son
            # ouverture bloque indéfiniment (constaté : > 40 s sans réponse).
            candidates = [
                ("DirectShow MJPG", cv2.CAP_DSHOW, w, h, True),
                ("DirectShow 640x480", cv2.CAP_DSHOW, 640, 480, True),
            ]
        else:
            candidates = [("défaut MJPG", cv2.CAP_ANY, w, h, True), ("défaut", cv2.CAP_ANY, w, h, False)]

        if self.strict:
            candidates = candidates[:1]
        cached = None if self.strict else self._load_cached_mode()
        if cached is not None:
            candidates = [cached] + [c for c in candidates if c[1:4] != cached[1:4]]

        good_enough = 0.8 * min(float(self.requested_fps), 30.0)
        best: tuple[float, tuple] | None = None
        # Une seule ouverture à la fois : ouvrir la caméra deux fois puis fermer la première
        # casse la seconde avec DirectShow.
        for pos, cand in enumerate(candidates):
            label, backend, cw, ch, mjpg = cand
            cap = cv2.VideoCapture(self.camera_index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            self._configure(cap, backend, cw, ch, mjpg)
            fps = self._measure_fps(cap)
            fmt = self._fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
            # MJPG demandé mais non obtenu (webcam sans MJPG à cette taille) : inutile de le
            # redemander au prochain lancement (2,5 s perdues)
            effective = (label, backend, cw, ch, mjpg and fmt == "MJPG")
            tag = "Mode mémorisé" if cand is cached else "Essai " + label
            print(f"[capture] {tag:30s} : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} {fmt} -> {fps:.1f} fps mesurés")
            # Les candidats sont classés par préférence : un suivant ne passe devant que s'il est
            # nettement plus rapide (mode demandé limité par l'USB, pas par la lumière).
            if best is None or (fps > 0 and fps >= FALLBACK_SPEEDUP * best[0]):
                best = (fps, effective)
            # Dernier candidat et c'est lui le mode retenu : inutile de le fermer pour le rouvrir
            last_and_best = pos == len(candidates) - 1 and best[1] is effective and fps > 0
            if fps >= good_enough or (cand is cached and fps > 0) or last_and_best:
                # Mode satisfaisant, ou mode mémorisé qui délivre des images : on le garde ouvert.
                # Un mode mémorisé lent l'est presque toujours à cause de la lumière (pose plus
                # longue dans une pièce sombre) : rechercher un autre mode coûterait ~10 s par
                # essai sans rien changer. Supprimer camera_mode.json pour relancer la recherche.
                if fps < good_enough:
                    print(f"[capture] Images/s faibles ({fps:.1f}) : pièce sombre ? "
                          f"(plus de lumière = plus d'images/s)")
                self.cap = cap
                break
            cap.release()

        if self.cap is None and best is not None:
            # Aucun mode « satisfaisant » : on rouvre le mode retenu
            _, backend, cw, ch, mjpg = best[1]
            print(f"[capture] Aucun mode n'atteint {good_enough:.0f} images/s : pièce sombre ? "
                  f"(plus de lumière = plus d'images/s). Mode retenu : {cw}x{ch}")
            self.cap = cv2.VideoCapture(self.camera_index, backend)
            if self.cap.isOpened():
                self._configure(self.cap, backend, cw, ch, mjpg)

        if self.cap is None or not self.cap.isOpened() or best is None:
            raise RuntimeError(f"Impossible d'ouvrir la webcam à l'index {self.camera_index}")

        if not self.strict:
            self._save_cached_mode(best[1])
        self.mode_label = best[1][0]
        # Lecture des valeurs réelles négociées avec le pilote
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = float(best[0])

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """Lit la frame suivante.

        Returns:
            Tuple (succès, image_bgr, timestamp_secondes).
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None, 0.0

        ret, frame = self.cap.read()
        # Horodatage à l'arrivée de la frame : read() bloque jusqu'à la prochaine image,
        # un timestamp pris avant inclurait ce temps d'attente variable (gigue du filtre).
        timestamp = time.perf_counter()
        if not ret or frame is None:
            return False, None, timestamp

        # Calcul du FPS glissant
        self._frame_count += 1
        elapsed = timestamp - self._fps_start_time
        if elapsed >= 1.0:
            self.measured_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start_time = timestamp

        return True, frame, timestamp

    def release(self) -> None:
        """Libère proprement la ressource caméra."""
        if getattr(self, "cap", None) is not None:
            self.cap.release()
            self.cap = None

    def __del__(self) -> None:
        self.release()
