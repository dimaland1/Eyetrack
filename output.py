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


class OpenTrackSender:
    """Émetteur UDP vers OpenTrack."""

    def __init__(self, config: OutputConfig | None = None) -> None:
        self.config = config or OutputConfig()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Références pour le recentrage ('r')
        self.ref_pose_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.ref_pose_ypr = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.ref_gaze_norm = np.array([0.5, 0.5], dtype=np.float64)
        self.is_recentered = False

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

    def send(
        self,
        pose_xyz: tuple[float, float, float],
        pose_ypr: tuple[float, float, float],
        gaze_px: tuple[float, float],
        screen_config: ScreenConfig,
        gaze_enabled: bool = True,
    ) -> None:
        """Calcule la pose finale et transmet les 6 doubles little-endian via UDP.

        Format OpenTrack : [x, y, z (cm), yaw, pitch, roll (deg)].
        """
        raw_xyz = np.array(pose_xyz, dtype=np.float64)
        raw_ypr = np.array(pose_ypr, dtype=np.float64)

        # Application du recentrage
        rel_xyz = raw_xyz - self.ref_pose_xyz
        rel_ypr = raw_ypr - self.ref_pose_ypr

        # Inversion éventuelle des axes
        x = -rel_xyz[0] if self.config.invert_x else rel_xyz[0]
        y = -rel_xyz[1] if self.config.invert_y else rel_xyz[1]
        z = -rel_xyz[2] if self.config.invert_z else rel_xyz[2]

        yaw = -rel_ypr[0] if self.config.invert_yaw else rel_ypr[0]
        pitch = -rel_ypr[1] if self.config.invert_pitch else rel_ypr[1]
        roll = -rel_ypr[2] if self.config.invert_roll else rel_ypr[2]

        # Contribution du regard à l'orientation de caméra si activé
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

            # Conversion en angle de regard et pondération par gain séparé
            gaze_yaw_offset = gaze_dx * 35.0 * self.config.gaze_yaw_gain
            gaze_pitch_offset = gaze_dy * 25.0 * self.config.gaze_pitch_gain

            yaw += gaze_yaw_offset
            pitch += gaze_pitch_offset

        # Encodage binaire OpenTrack UDP : 6 doubles (6 * 8 = 48 octets little-endian)
        payload = struct.pack("<6d", float(x), float(y), float(z), float(yaw), float(pitch), float(roll))
        try:
            self.sock.sendto(payload, (self.config.host, self.config.port))
        except Exception as e:
            print(f"[output] Erreur UDP : {e}")

    def close(self) -> None:
        """Ferme la socket UDP."""
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

        # Légende des touches en bas
        help_str = "[C] Calibrer  [G] Regard ON/OFF  [O] Overlay ON/OFF  [R] Recentrer  [Q] Quitter"
        cv2.putText(display_frame, help_str, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow(self.window_name, display_frame)
        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        """Ferme la fenêtre de rendu."""
        cv2.destroyWindow(self.window_name)
