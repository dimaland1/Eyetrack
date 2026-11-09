"""Module de sortie vers OpenTrack via UDP et d'affichage de debug visuel.

Gère :
- Le protocole OpenTrack standard "UDP over network" (6 doubles little-endian, port 4242).
- Le recentrage de la pose ('r').
- L'injection du décalage de regard dans la caméra de jeu.
- Le retour visuel HUD (FPS, latence en ms, état de calibration, repères 3D).
"""

from __future__ import annotations
from dataclasses import dataclass
import socket
import struct
import math
import ctypes
import platform
import sys
import threading
import time
import cv2
import numpy as np
from geometry import ScreenConfig


@dataclass
class OutputConfig:
    """Paramètres d'envoi UDP vers OpenTrack."""

    host: str = "127.0.0.1"
    port: int = 4242
    invert_x: bool = False
    invert_y: bool = False
    invert_z: bool = False
    invert_yaw: bool = False
    invert_pitch: bool = False
    invert_roll: bool = False
    # Sensibilité du regard (gains horizontal et vertical séparés)
    gaze_yaw_gain: float = 0.50
    gaze_pitch_gain: float = 0.65
    # Zone morte douce (0..1) : en dessous, la réponse est fortement atténuée mais jamais coupée
    gaze_deadzone: float = 0.10
    # Exposant de la courbe de réponse (1.0 = linéaire après la zone morte douce)
    gaze_curve_exp: float = 1.20
    # Gain de rotation de la tête (1.0 = 1:1). Comme les courbes TrackIR/OpenTrack : avec 2,5,
    # 20° de tête donnent 50° dans le jeu (rétroviseurs d'ETS2 sans quitter l'écran des yeux).
    head_yaw_gain: float = 1.5
    head_pitch_gain: float = 1.2
    # Lissage supplémentaire de la part « regard » de la caméra (constante de temps, s) :
    # le regard webcam est bruité (~3°) ; sans cela la caméra « danse » quand on fixe un coin.
    gaze_smoothing_s: float = 0.25
    # Envoi à OpenTrack indépendant de la webcam (30 images/s) : à 120 Hz avec prédiction, la
    # caméra du jeu avance par petits pas réguliers au lieu de 30 sauts par seconde.
    # 0 = un envoi par image webcam (ancien comportement).
    send_rate_hz: float = 120.0
    # Lissage très court de la sortie (s) : transforme les 30 sauts par seconde en mouvement
    # continu. Coûte ~10 ms de retard, que la prédiction ci-dessous rattrape.
    output_smoothing_s: float = 0.015
    # Avance de la prédiction (s) ; 0 = pas de prédiction. Mesuré en simulation (tête bruitée,
    # One Euro, rotation rapide de 40°) : 10 ms garde le retard d'avant (~50 ms) avec 3 fois
    # moins d'à-coups ; au-delà, la caméra dépasse sa cible à l'arrêt (7° pour 50 ms).
    max_prediction_s: float = 0.010
    # Durée du fondu de la pause (retour au centre / reprise), en secondes
    pause_fade_s: float = 0.4
    # Recentrage en douceur (constante de temps, s : ~95 % du chemin en 3 fois cette valeur)
    recenter_smoothing_s: float = 0.12


def soft_response_curve(val: float, deadzone: float, exp: float) -> float:
    """Courbe de réponse continue avec zone morte douce, bornée à [-1, 1].

    Contrairement à une zone morte « dure » (sortie nulle puis démarrage brusque),
    la forme hyperbolique sqrt(v² + d²) - d est quadratique près de zéro (le bruit
    est atténué sans être coupé) puis quasi linéaire au-delà de d : les
    micro-mouvements produisent une petite réponse et il n'y a aucun palier.
    """
    v = min(abs(float(val)), 1.0)
    d = max(float(deadzone), 0.0)
    if d > 0.0:
        mag = (math.sqrt(v * v + d * d) - d) / (math.sqrt(1.0 + d * d) - d)
    else:
        mag = v
    mag = min(max(mag, 0.0), 1.0) ** max(float(exp), 0.1)
    return math.copysign(mag, val)


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 sous lo, 1 au-dessus de hi, transition douce (smoothstep) entre les deux."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    u = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


# Vitesses au-delà desquelles une estimation est jugée aberrante (bornage de la prédiction)
MAX_ANGULAR_SPEED = 400.0  # °/s
MAX_LINEAR_SPEED = 150.0   # cm/s
# La prédiction n'agit que si la tête bouge vraiment : au repos, la vitesse estimée n'est que
# du bruit et l'extrapoler ajouterait du tremblement. Seuils (début, pleine prédiction) :
PREDICT_ANGULAR_RAMP = (4.0, 20.0)  # °/s
PREDICT_LINEAR_RAMP = (1.5, 6.0)    # cm/s


@dataclass
class _Sample:
    """Dernière pose filtrée reçue de la boucle principale."""

    t: float                      # horodatage de l'image (time.perf_counter)
    pose: np.ndarray              # x, y, z (cm), yaw, pitch, roll (°) au format MediaPipe
    vel: np.ndarray               # vitesse estimée (par seconde)
    gaze_px: tuple[float, float]
    gaze_enabled: bool
    tracking: bool                # False = visage perdu, valeur maintenue
    screen_config: ScreenConfig


class OpenTrackSender:
    """Émetteur UDP vers OpenTrack.

    Deux modes :
    - `start()` puis `update()` à chaque image : un thread envoie à `send_rate_hz` (120 Hz)
      en prédisant la pose entre deux images de la webcam ;
    - sans `start()` et avec send_rate_hz = 0 : un paquet par appel à `update()` / `send()`.
    """

    def __init__(self, config: OutputConfig | None = None) -> None:
        self.config = config or OutputConfig()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Références pour le recentrage ('r')
        self.ref_pose_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.ref_pose_ypr = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.ref_gaze_norm = np.array([0.5, 0.5], dtype=np.float64)
        self.is_recentered = False
        # Décalage caméra dû au regard, lissé (yaw, pitch en degrés)
        self._gaze_offset = np.zeros(2, dtype=np.float64)
        self._last_tick_t: float | None = None
        # Pose de sortie lissée (format MediaPipe, avant conversion)
        self._out_pose: np.ndarray | None = None
        # Pause : la caméra revient en douceur au centre et y reste
        self.paused = False
        self._pause_w = 1.0  # 1 = suivi normal, 0 = caméra au centre
        # Dernière pose reçue et thread d'envoi
        self._sample: _Sample | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._fine_timer = False  # horloge Windows à 1 ms demandée (timeBeginPeriod)
        self.last_values: tuple[float, ...] | None = None  # dernier paquet envoyé (diagnostic)
        # Fréquence d'envoi mesurée (affichée dans le HUD)
        self.measured_rate_hz = 0.0
        self.max_gap_ms = 0.0  # plus long intervalle entre deux paquets sur la dernière seconde
        self._rate_count = 0
        self._rate_t0 = time.perf_counter()
        self._last_emit_t: float | None = None
        self._gap_max = 0.0
        # Référence de recentrage réellement appliquée : rejoint la nouvelle référence en fondu
        # (recentrer pendant que la tête est tournée ne fait plus sauter la caméra d'un coup)
        self._ref_applied: np.ndarray | None = None

    # ------------------------------------------------------------------ commandes
    def recenter(
        self,
        current_pose_xyz: tuple[float, float, float],
        current_pose_ypr: tuple[float, float, float],
        current_gaze_px: tuple[float, float],
        screen_config: ScreenConfig,
    ) -> None:
        """Fixe la pose et le regard actuels comme nouvelle référence (zéro)."""
        self.ref_pose_xyz = np.array(current_pose_xyz, dtype=np.float64)
        self.ref_pose_ypr = np.array(current_pose_ypr, dtype=np.float64)
        nx, ny = screen_config.px_to_norm(current_gaze_px[0], current_gaze_px[1])
        self.ref_gaze_norm = np.array([nx, ny], dtype=np.float64)
        self.is_recentered = True
        print("[output] Pose et regard recentrés (zéro défini).")

    def set_paused(self, paused: bool) -> None:
        """Pause : la caméra revient en douceur au centre et y reste jusqu'à la reprise."""
        self.paused = bool(paused)

    def toggle_pause(self) -> bool:
        self.set_paused(not self.paused)
        return self.paused

    # ------------------------------------------------------------------ entrées
    def update(
        self,
        pose_xyz: tuple[float, float, float],
        pose_ypr: tuple[float, float, float],
        gaze_px: tuple[float, float],
        screen_config: ScreenConfig,
        gaze_enabled: bool = True,
        timestamp: float | None = None,
        tracking: bool = True,
    ) -> None:
        """Transmet la dernière pose filtrée (appelé à chaque image de la webcam).

        Avec le thread démarré, l'envoi se fait à `send_rate_hz` ; sinon (send_rate_hz = 0),
        un paquet est envoyé tout de suite.
        """
        t = time.perf_counter() if timestamp is None else float(timestamp)
        pose = np.array([*pose_xyz, *pose_ypr], dtype=np.float64)
        with self._lock:
            prev = self._sample
            vel = np.zeros(6, dtype=np.float64)
            if tracking and prev is not None and prev.tracking:
                dt = t - prev.t
                if 1e-3 < dt < 0.2:
                    # Vitesse entre deux images, légèrement lissée (la pose est déjà filtrée)
                    vel = prev.vel + 0.6 * ((pose - prev.pose) / dt - prev.vel)
                    vel[:3] = np.clip(vel[:3], -MAX_LINEAR_SPEED, MAX_LINEAR_SPEED)
                    vel[3:] = np.clip(vel[3:], -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
            self._sample = _Sample(t=t, pose=pose, vel=vel, gaze_px=(float(gaze_px[0]), float(gaze_px[1])),
                                   gaze_enabled=bool(gaze_enabled), tracking=bool(tracking),
                                   screen_config=screen_config)
        if not self._running and self.config.send_rate_hz <= 0:
            self._tick(time.perf_counter())

    def send(
        self,
        pose_xyz: tuple[float, float, float],
        pose_ypr: tuple[float, float, float],
        gaze_px: tuple[float, float],
        screen_config: ScreenConfig,
        gaze_enabled: bool = True,
    ) -> None:
        """Calcule la pose finale et envoie immédiatement un paquet."""
        now = time.perf_counter()
        self.update(pose_xyz, pose_ypr, gaze_px, screen_config, gaze_enabled, timestamp=now)
        if self._running or self.config.send_rate_hz > 0:
            self._tick(now)

    # ------------------------------------------------------------------ calcul
    def predicted_pose(self, now: float) -> np.ndarray | None:
        """Pose (format MediaPipe) extrapolée à l'instant `now` à partir de la dernière image."""
        with self._lock:
            s = self._sample
        if s is None:
            return None
        lead = min(max(now - s.t, 0.0), max(self.config.max_prediction_s, 0.0))
        if lead <= 0.0 or not s.tracking:
            return s.pose.copy()
        v = s.vel.copy()
        v[3:] *= _ramp(float(np.linalg.norm(v[3:])), *PREDICT_ANGULAR_RAMP)
        v[:3] *= _ramp(float(np.linalg.norm(v[:3])), *PREDICT_LINEAR_RAMP)
        return s.pose + v * lead

    def _compose(
        self,
        pose: np.ndarray,
        gaze_px: tuple[float, float],
        screen_config: ScreenConfig,
        gaze_enabled: bool,
        dt: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Pose MediaPipe -> 6 valeurs OpenTrack [x, y, z (cm), yaw, pitch, roll (°)]."""
        # Application du recentrage (en fondu ; immédiat avant le premier paquet)
        ref = np.concatenate([self.ref_pose_xyz, self.ref_pose_ypr])
        if self._ref_applied is None:
            self._ref_applied = ref.copy()
        elif dt > 0:
            a_ref = 1.0 - math.exp(-dt / max(self.config.recenter_smoothing_s, 1e-3))
            self._ref_applied += a_ref * (ref - self._ref_applied)
        rel_xyz = pose[:3] - self._ref_applied[:3]
        rel_ypr = pose[3:] - self._ref_applied[3:]

        # Conversion des conventions MediaPipe -> OpenTrack, constatées en jeu (BeamNG, ETS2) :
        # - X : se pencher à droite faisait glisser la caméra à gauche -> signe inversé.
        # - Z : MediaPipe augmente quand on s'approche de l'écran, OpenTrack l'interprète comme
        #   un recul (avancer le visage faisait reculer la caméra) -> signe inversé.
        # Ces sens sont verrouillés par tests/test_axes.py.
        x = -rel_xyz[0]
        y = rel_xyz[1]
        z = -rel_xyz[2]
        if self.config.invert_x:
            x = -x
        if self.config.invert_y:
            y = -y
        if self.config.invert_z:
            z = -z

        # - Pitch : positif quand la tête s'incline vers le BAS chez MediaPipe, vers le HAUT chez
        #   OpenTrack (tête baissée -> caméra qui montait) -> signe inversé.
        # - Roll : même signe chez MediaPipe et OpenTrack (vérifié en jeu une fois l'ordre des
        #   angles corrigé dans face.py ; le problème venait du mélange des axes, pas du signe).
        yaw = rel_ypr[0]
        pitch = -rel_ypr[1]
        roll = rel_ypr[2]
        if self.config.invert_yaw:
            yaw = -yaw
        if self.config.invert_pitch:
            pitch = -pitch
        if self.config.invert_roll:
            roll = -roll

        # Gain de rotation de la tête
        yaw *= self.config.head_yaw_gain
        pitch *= self.config.head_pitch_gain

        # Contribution du regard à l'orientation de caméra (cible lissée dans le temps ;
        # regard coupé = retour en douceur vers 0, sans à-coup)
        target = np.zeros(2, dtype=np.float64)
        if gaze_enabled:
            cur_nx, cur_ny = screen_config.px_to_norm(gaze_px[0], gaze_px[1])
            # Le regard est borné à l'écran : regarder hors écran ne doit pas faire exploser la rotation
            cur_nx = min(max(cur_nx, 0.0), 1.0)
            cur_ny = min(max(cur_ny, 0.0), 1.0)
            # Décalage relatif au point de référence (ou centre 0.5)
            dx = cur_nx - self.ref_gaze_norm[0]
            dy = cur_ny - self.ref_gaze_norm[1]

            gaze_dx = soft_response_curve(dx, self.config.gaze_deadzone, self.config.gaze_curve_exp)
            gaze_dy = soft_response_curve(dy, self.config.gaze_deadzone, self.config.gaze_curve_exp)

            # Conversion en angle (yaw : droite = positif ; pitch : regard vers le BAS de
            # l'écran = dy > 0 = pitch négatif, même convention que la tête)
            target[0] = gaze_dx * 35.0 * self.config.gaze_yaw_gain
            target[1] = -gaze_dy * 25.0 * self.config.gaze_pitch_gain

        tau = max(self.config.gaze_smoothing_s, 1e-3)
        a = 1.0 - math.exp(-dt / tau) if dt > 0 else 0.0
        self._gaze_offset += a * (target - self._gaze_offset)
        yaw += self._gaze_offset[0]
        pitch += self._gaze_offset[1]

        # Pause : fondu vers le centre (0) puis maintien ; reprise en fondu également
        goal = 0.0 if self.paused else 1.0
        step = dt / max(self.config.pause_fade_s, 1e-3)
        if goal > self._pause_w:
            self._pause_w = min(self._pause_w + step, goal)
        else:
            self._pause_w = max(self._pause_w - step, goal)
        w = self._pause_w * self._pause_w * (3.0 - 2.0 * self._pause_w)

        return (float(x * w), float(y * w), float(z * w), float(yaw * w), float(pitch * w), float(roll * w))

    def _tick(self, now: float) -> tuple[float, ...] | None:
        """Calcule et envoie un paquet pour l'instant `now` (appelé par le thread ou send())."""
        target = self.predicted_pose(now)
        if target is None:
            return None
        with self._lock:
            s = self._sample
        dt = 0.0 if self._last_tick_t is None else min(max(now - self._last_tick_t, 0.0), 0.2)
        self._last_tick_t = now

        # Lissage très court de la pose de sortie (arrondit les corrections de la prédiction)
        tau = self.config.output_smoothing_s
        if self._out_pose is None or tau <= 0.0 or dt <= 0.0:
            self._out_pose = target.copy()
        else:
            self._out_pose = self._out_pose + (1.0 - math.exp(-dt / tau)) * (target - self._out_pose)

        values = self._compose(self._out_pose, s.gaze_px, s.screen_config, s.gaze_enabled, dt)
        self._emit(values)
        return values

    def _emit(self, values: tuple[float, ...]) -> None:
        """Encodage binaire OpenTrack UDP : 6 doubles (48 octets little-endian)."""
        self.last_values = values
        payload = struct.pack("<6d", *values)
        try:
            self.sock.sendto(payload, (self.config.host, self.config.port))
        except Exception as e:
            print(f"[output] Erreur UDP : {e}")
        self._rate_count += 1
        now = time.perf_counter()
        if self._last_emit_t is not None:
            self._gap_max = max(self._gap_max, now - self._last_emit_t)
        self._last_emit_t = now
        if now - self._rate_t0 >= 1.0:
            self.measured_rate_hz = self._rate_count / (now - self._rate_t0)
            self.max_gap_ms = self._gap_max * 1000.0
            self._rate_count = 0
            self._gap_max = 0.0
            self._rate_t0 = now

    # ------------------------------------------------------------------ thread d'envoi
    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Démarre l'envoi régulier à `send_rate_hz` (sans effet si send_rate_hz = 0)."""
        if self._running or self.config.send_rate_hz <= 0:
            return
        # Régularité du thread : quand la boucle principale garde le GIL (code Python, extension C),
        # le thread d'envoi ne le récupère qu'au bout de l'intervalle de bascule, arrondi par
        # Windows à son horloge système (15,6 ms par défaut) -> trous de ~20 ms mesurés.
        # Horloge à 1 ms (comme la plupart des jeux) et bascule toutes les 2 ms : trous de ~10 ms.
        sys.setswitchinterval(min(sys.getswitchinterval(), 0.002))
        if platform.system() == "Windows" and not self._fine_timer:
            try:
                ctypes.windll.winmm.timeBeginPeriod(1)
                self._fine_timer = True
            except Exception:
                pass
        self._running = True
        self._thread = threading.Thread(target=self._run, name="opentrack-sender", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        period = 1.0 / max(self.config.send_rate_hz, 1.0)
        next_t = time.perf_counter()
        while self._running:
            try:
                self._tick(time.perf_counter())
            except Exception as e:  # le thread ne doit jamais mourir silencieusement
                print(f"[output] Erreur d'envoi : {e}")
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -period:
                next_t = time.perf_counter()  # gros retard : on se recale sans rafale

    def close(self) -> None:
        """Arrête le thread d'envoi et ferme la socket UDP."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self._fine_timer:
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass
            self._fine_timer = False
        self.sock.close()


class Visualizer:
    """Rendu de la fenêtre de contrôle avec métriques temps réel et repères visuels."""

    def __init__(self, window_name: str = "Eyetrack - Moniteur") -> None:
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def draw_head_axes(
        self,
        frame: np.ndarray,
        center_px: tuple[float, float],
        pose_ypr: tuple[float, float, float],
        axis_length: float = 60.0,
    ) -> None:
        """Dessine les 3 axes d'orientation 3D (X: Rouge, Y: Vert, Z: Bleu) sur le visage."""
        cx, cy = int(center_px[0]), int(center_px[1])
        yaw_rad = math.radians(pose_ypr[0])
        pitch_rad = math.radians(pose_ypr[1])
        roll_rad = math.radians(pose_ypr[2])

        # Calcul simplifié de projection des axes 3D
        # Axe X (Rouge - droite)
        x_end = (int(cx + axis_length * (math.cos(yaw_rad) * math.cos(roll_rad))),
                 int(cy + axis_length * (math.sin(pitch_rad) * math.sin(yaw_rad) - math.cos(pitch_rad) * math.sin(roll_rad))))
        # Axe Y (Vert - bas)
        y_end = (int(cx + axis_length * (-math.cos(yaw_rad) * math.sin(roll_rad))),
                 int(cy + axis_length * (math.cos(pitch_rad) * math.cos(roll_rad) + math.sin(pitch_rad) * math.sin(yaw_rad))))
        # Axe Z (Bleu - profondeur / nez vers l'avant)
        z_end = (int(cx + axis_length * math.sin(yaw_rad)),
                 int(cy - axis_length * math.sin(pitch_rad)))

        cv2.line(frame, (cx, cy), x_end, (0, 0, 255), 2)  # X: Rouge
        cv2.line(frame, (cx, cy), y_end, (0, 255, 0), 2)  # Y: Vert
        cv2.line(frame, (cx, cy), z_end, (255, 0, 0), 2)  # Z: Bleu

    def render(
        self,
        frame: np.ndarray,
        face_detected: bool,
        left_iris: tuple[float, float],
        right_iris: tuple[float, float],
        pose_xyz: tuple[float, float, float],
        pose_ypr: tuple[float, float, float],
        gaze_px: tuple[float, float],
        screen_config: ScreenConfig,
        gaze_mode: str,
        gaze_enabled: bool,
        is_calibrated: bool,
        measured_fps: float,
        step_latencies: dict[str, float],
        show_debug: bool = True,
        overlay_active: bool = False,
        rls_count: int = 0,
        profile_name: str = "",
        jitter: dict[str, float] | None = None,
        paused: bool = False,
        send_rate_hz: float = 0.0,
        send_gap_ms: float = 0.0,
        hotkey_hint: str = "",
    ) -> int:
        """Affiche le retour vidéo avec l'interface HUD complète.

        Returns:
            Code de la touche pressée (ex: ord('c'), ord('q')).
        """
        # Affichage agrandi si la caméra tourne en basse résolution (ex. 640x480 à 30 fps) :
        # le HUD est dessiné pour une largeur d'environ 1280 px.
        src_h, src_w = frame.shape[:2]
        scale = 1280.0 / src_w if src_w < 1280 else 1.0
        if scale != 1.0:
            display_frame = cv2.resize(frame, (int(src_w * scale), int(src_h * scale)), interpolation=cv2.INTER_LINEAR)
            left_iris = (left_iris[0] * scale, left_iris[1] * scale)
            right_iris = (right_iris[0] * scale, right_iris[1] * scale)
        else:
            display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # Tracé des iris si visage détecté
        if face_detected:
            cv2.circle(display_frame, (int(left_iris[0]), int(left_iris[1])), 4, (0, 255, 255), -1)
            cv2.circle(display_frame, (int(right_iris[0]), int(right_iris[1])), 4, (0, 255, 255), -1)
            # Axes 3D sur le point médian entre les yeux
            mid_x = (left_iris[0] + right_iris[0]) / 2.0
            mid_y = (left_iris[1] + right_iris[1]) / 2.0
            self.draw_head_axes(display_frame, (mid_x, mid_y), pose_ypr)

        # Panneau supérieur semi-transparent pour les métriques
        overlay = display_frame.copy()
        panel_h = 135 if show_debug else 75
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

        # Badges d'état
        mode_badge = f"Mode: {gaze_mode.upper()}"
        prof_badge = f"[{profile_name.upper()}]" if profile_name else ""
        gaze_status_text = "REGARD: ON" if gaze_enabled else "REGARD: OFF"
        gaze_status_color = (0, 255, 0) if gaze_enabled else (0, 165, 255)
        calib_text = "CALIBRE" if is_calibrated else "NON CALIBRE"
        calib_color = (0, 255, 0) if is_calibrated else (0, 0, 255)
        overlay_text = "OVERLAY: ON" if overlay_active else "OVERLAY: OFF"
        overlay_color = (0, 255, 200) if overlay_active else (120, 120, 120)

        cv2.putText(display_frame, f"FPS: {measured_fps:4.1f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(display_frame, mode_badge, (115, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
        if prof_badge:
            cv2.putText(display_frame, prof_badge, (225, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 2)
        cv2.putText(display_frame, gaze_status_text, (330, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, gaze_status_color, 2)
        cv2.putText(display_frame, calib_text, (460, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, calib_color, 2)
        cv2.putText(display_frame, overlay_text, (575, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, overlay_color, 2)
        if rls_count > 0:
            cv2.putText(display_frame, f"RLS: +{rls_count}", (705, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)

        # Affichage pose 6DoF
        yaw, pitch, roll = pose_ypr
        x, y, z = pose_xyz
        pose_str = f"Tete: Yaw={yaw:+5.1f} Pitch={pitch:+5.1f} Roll={roll:+5.1f} | Z={z:5.1f}cm"
        cv2.putText(display_frame, pose_str, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # Affichage position du regard écran
        gx, gy = gaze_px
        gaze_str = f"Regard ecran: X={gx:5.0f}px Y={gy:5.0f}px (Ecran {screen_config.width_px}x{screen_config.height_px})"
        cv2.putText(display_frame, gaze_str, (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

        # Métriques de latence détaillées par étape
        if show_debug and step_latencies:
            lat_str = (
                f"Latences (ms): Capt={step_latencies.get('capture', 0):.1f} | "
                f"Face={step_latencies.get('face', 0):.1f} | "
                f"Gaze={step_latencies.get('gaze', 0):.1f} | "
                f"Filtre={step_latencies.get('filter', 0):.1f} | "
                f"Total={step_latencies.get('total', 0):.1f}"
            )
            if send_rate_hz > 0:
                lat_str += f" | Envoi {send_rate_hz:.0f} Hz"
                if send_gap_ms > 0:
                    lat_str += f" (trou max {send_gap_ms:.0f} ms)"
            cv2.putText(display_frame, lat_str, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 255, 180), 1)
            if jitter:
                jit_str = (f"Agitation (1 s, si immobile) : tete {jitter['head_deg']:.2f} deg | "
                           f"regard {jitter['gaze_px']:.0f} px")
                cv2.putText(display_frame, jit_str, (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 200, 120), 1)

        # Mini-carte écran en bas à droite pour visualiser la position du regard
        mini_w, mini_h = 160, 90
        mini_x = w - mini_w - 15
        mini_y = h - mini_h - 15
        cv2.rectangle(display_frame, (mini_x, mini_y), (mini_x + mini_w, mini_y + mini_h), (40, 40, 40), -1)
        cv2.rectangle(display_frame, (mini_x, mini_y), (mini_x + mini_w, mini_y + mini_h), (120, 120, 120), 1)
        # Position normalisée du point de regard
        norm_gx = np.clip(gx / max(screen_config.width_px, 1), 0.0, 1.0)
        norm_gy = np.clip(gy / max(screen_config.height_px, 1), 0.0, 1.0)
        pt_x = int(mini_x + norm_gx * mini_w)
        pt_y = int(mini_y + norm_gy * mini_h)
        cv2.circle(display_frame, (pt_x, pt_y), 5, (0, 140, 255), -1)
        cv2.circle(display_frame, (pt_x, pt_y), 6, (0, 255, 255), 1)
        cv2.putText(display_frame, "Ecran", (mini_x + 5, mini_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)

        # Pause : la caméra du jeu est ramenée au centre
        if paused:
            cv2.putText(display_frame, "PAUSE - camera au centre", (15, panel_h + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

        # Légende des touches en bas (cv2.putText ne gère pas les accents)
        help_str = "[C] Calibrer  [G] Regard  [O] Overlay  [R] Recentrer  [Espace] Pause  [Q] Quitter"
        cv2.putText(display_frame, help_str, (15, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if hotkey_hint:
            cv2.putText(display_frame, hotkey_hint, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(self.window_name, display_frame)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        """Ferme la fenêtre de rendu."""
        cv2.destroyWindow(self.window_name)
