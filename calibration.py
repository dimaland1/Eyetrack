"""Module de calibration personnelle affine 5-9 points avec mouvement de tête.

Implémente un modèle affine régularisé Ridge compensant :
1. Le biais d'axe visuel (angle kappa oculaire).
2. Les gains horizontal et vertical distincts.
3. La fuite résiduelle de pose de tête (head-pose leakage).

Permet une calibration 5 points (rapide façon Beam) ou 9 points (dense).
"""

from __future__ import annotations
import math
import time
import cv2
import numpy as np
from geometry import ScreenConfig
from paths import data_path


CALIB_FORMAT_VERSION = 4  # v4 : features réseau = point d'impact du regard sur l'écran (mm)
# Écart-type minimal par feature : une feature qui n'a presque pas varié pendant la
# calibration (ex. position de la tête) ne doit pas être amplifiée démesurément.
MIN_FEATURE_STD = {
    "net": [20.0, 20.0],                                    # mm sur l'écran
    "iris": [0.005, 0.005, 0.005, 0.005, 0.005, 3.0, 3.0],  # ratios..., degrés tête
}
# Poids d'un clic souris en recalibration RLS, en « nombre d'échantillons » de calibration
RLS_EQUIV_SAMPLES = 15.0


class AffineGazeModel:
    """Modèle de calibration Ridge sur features standardisées.

    Modes :
    - 'net'  : features = point d'impact (x, y en mm) du rayon de regard sur le plan écran.
               Le réseau voit déjà la tête tournée et le rayon part de la position réelle de
               l'œil : rotations ET translations de la tête sont gérées par la géométrie. La
               calibration n'est qu'une correction affine (gains, décalages, axe visuel).
    - 'iris' : features = [rx, ry, rx², rx·ry, ry², yaw tête, pitch tête]. Le ratio d'iris est
               un regard œil-dans-tête : la pose de tête est ici nécessaire.

    Les features sont centrées-réduites avant le Ridge : la régularisation agit alors de la même
    façon sur toutes les features, quelles que soient leurs unités (ratios ~0.05, degrés, cm).
    """

    def __init__(self, mode: str = "net", alpha: float = 1e-3) -> None:
        """Initialise le modèle.

        Args:
            mode: 'net' ou 'iris'.
            alpha: Régularisation Ridge relative (fraction de N, sur features standardisées).
        """
        self.mode = mode
        self.alpha = float(alpha)
        self.weights: np.ndarray | None = None
        self.feat_mean: np.ndarray | None = None
        self.feat_std: np.ndarray | None = None
        self.is_fitted: bool = False
        self.P: np.ndarray | None = None

    def _raw_features(self, G: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Construit les features brutes (avant standardisation)."""
        G = np.atleast_2d(np.asarray(G, dtype=np.float64))
        H = np.atleast_2d(np.asarray(H, dtype=np.float64))
        if self.mode == "net":
            return G
        rx = G[:, 0:1]
        ry = G[:, 1:2]
        return np.hstack([rx, ry, rx**2, rx * ry, ry**2, H[:, 0:1], H[:, 1:2]])

    def _design(self, G: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Matrice de design standardisée [z_1 .. z_k, 1]."""
        raw = self._raw_features(G, H)
        z = (raw - self.feat_mean) / self.feat_std
        return np.hstack([z, np.ones((z.shape[0], 1), dtype=np.float64)])

    def fit(self, G: np.ndarray, H: np.ndarray, Y: np.ndarray) -> float:
        """Ajuste le modèle par moindres carrés régularisés Ridge.

        Args:
            G: Mesures de regard brut (N, d).
            H: Poses de tête (N, 2) [yaw, pitch] (utilisées en mode iris uniquement).
            Y: Coordonnées vraies des cibles écran en pixels (N, 2).

        Returns:
            RMSE résiduel d'entraînement en pixels.
        """
        raw = self._raw_features(G, H)
        self.feat_mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        # Une feature qui a peu varié (ex. tête immobile) ne doit pas être amplifiée
        floor = np.asarray(MIN_FEATURE_STD.get(self.mode, [1e-6] * raw.shape[1]), dtype=np.float64)
        self.feat_std = np.maximum(std, floor[: raw.shape[1]])

        phi = self._design(G, H)
        n, k = phi.shape
        reg = self.alpha * n * np.eye(k, dtype=np.float64)
        reg[-1, -1] = 0.0  # biais (dernière colonne) non pénalisé

        A = phi.T @ phi + reg
        self.weights = np.linalg.solve(A, phi.T @ Y)
        # Covariance RLS : on « dilue » la confiance du fit initial pour qu'un clic pèse
        # comme ~RLS_EQUIV_SAMPLES échantillons de calibration (sinon 1 clic = 1/N, inefficace).
        self.P = np.linalg.inv(A) * (n / RLS_EQUIV_SAMPLES)
        self.is_fitted = True

        pred = phi @ self.weights
        return float(np.sqrt(np.mean(np.sum((Y - pred) ** 2, axis=1))))

    def update_rls(
        self,
        gaze_raw: np.ndarray | list[float] | tuple[float, ...],
        head_pose: tuple[float, float],
        target_xy: tuple[float, float],
        max_error_px: float = 220.0,
        lambda_factor: float = 0.998,
    ) -> tuple[bool, float]:
        """Met à jour les poids par Recursive Least Squares (RLS) sur un clic souris.

        Returns:
            Tuple (succès, distance_au_clic_px).
        """
        if not self.is_fitted or self.weights is None or self.P is None:
            return False, 0.0

        phi = self._design(gaze_raw, head_pose)
        pred = phi @ self.weights
        err = np.asarray(target_xy, dtype=np.float64).reshape(1, 2) - pred
        dist_px = float(np.linalg.norm(err))
        if dist_px > max_error_px:
            return False, dist_px

        phi_col = phi.T
        denom = float(lambda_factor + (phi @ self.P @ phi_col)[0, 0])
        k = (self.P @ phi_col) / max(denom, 1e-9)
        self.weights = self.weights + k @ err
        self.P = (self.P - k @ (phi @ self.P)) / lambda_factor
        return True, dist_px

    def predict(
        self,
        gaze_raw: np.ndarray | list[float] | tuple[float, ...],
        head_pose: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[float, float]:
        """Prédit la position du regard écran (x, y) en pixels."""
        if not self.is_fitted or self.weights is None:
            raise RuntimeError("Le modèle de calibration n'est pas encore ajusté.")
        pred = self._design(gaze_raw, head_pose) @ self.weights
        return float(pred[0, 0]), float(pred[0, 1])

    def save(self, filepath: str) -> None:
        """Sauvegarde les poids, la standardisation et la covariance RLS dans un .npz."""
        if not self.is_fitted or self.weights is None:
            raise RuntimeError("Impossible de sauvegarder un modèle non entraîné.")
        np.savez(
            filepath,
            version=CALIB_FORMAT_VERSION,
            weights=self.weights,
            feat_mean=self.feat_mean,
            feat_std=self.feat_std,
            P=self.P,
            alpha=self.alpha,
            mode=self.mode,
        )

    def load(self, filepath: str) -> bool:
        """Charge un modèle. Les calibrations de l'ancien format sont refusées."""
        try:
            data = np.load(filepath)
            if "version" not in data.files or int(data["version"]) < CALIB_FORMAT_VERSION:
                print(f"[calibration] '{filepath}' utilise l'ancien format de calibration : "
                      "recalibrez avec la touche [C].")
                return False
            self.mode = str(data["mode"])
            self.weights = data["weights"]
            self.feat_mean = data["feat_mean"]
            self.feat_std = data["feat_std"]
            self.P = data["P"]
            self.alpha = float(data["alpha"])
            self.is_fitted = True
            return True
        except Exception as e:
            print(f"[calibration] Erreur chargement ({filepath}) : {e}")
            return False


class CalibrationManager:
    """Gestionnaire de la calibration interactive avec guidage de mouvement de tête."""

    def __init__(
        self,
        screen_config: ScreenConfig,
        mode: str = "net",
        num_points: int = 9,
    ) -> None:
        self.screen_config = screen_config
        self.mode = mode
        self.num_points = num_points
        self.model = AffineGazeModel(mode=mode)
        self.is_calibrated = False
        self.rls_update_count = 0

        self._init_grid(num_points)

    def _init_grid(self, num_points: int) -> None:
        """Configure la grille de points (5 points en croix ou 9 points en grille)."""
        if num_points == 5:
            # 5 points en croix (Beam style) : Centre, Gauche, Droite, Haut, Bas
            self.points_norm = [
                (0.50, 0.50),  # Centre
                (0.15, 0.50),  # Gauche
                (0.85, 0.50),  # Droite
                (0.50, 0.15),  # Haut
                (0.50, 0.85),  # Bas
            ]
        else:
            # 9 points complets (3x3)
            self.points_norm = [
                (0.15, 0.15), (0.50, 0.15), (0.85, 0.15),
                (0.15, 0.50), (0.50, 0.50), (0.85, 0.50),
                (0.15, 0.85), (0.50, 0.85), (0.85, 0.85),
            ]

    def run_interactive_calibration(
        self,
        capture_fn,
        face_detector,
        gaze_estimator,
        normalizer=None,
        samples_per_point: int = 45,
        settle_frames: int = 15,
    ) -> bool:
        """Procédure interactive avec consigne de mouvement doux de tête."""
        w = self.screen_config.width_px
        h = self.screen_config.height_px

        window_name = "Calibration Personnelle (ESC pour annuler)"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        collected_G: list[np.ndarray] = []
        collected_H: list[np.ndarray] = []
        collected_Y: list[np.ndarray] = []
        # Données brutes enregistrées pour analyse hors ligne (calibration_samples_<mode>.npz)
        from gaze_model import IrisGazeEstimator
        iris_probe = IrisGazeEstimator()
        collected_extra: list[np.ndarray] = []
        t_calib_start = time.perf_counter()

        total_points = len(self.points_norm)

        for pt_idx, (nx, ny) in enumerate(self.points_norm):
            target_x, target_y = self.screen_config.norm_to_px(nx, ny)
            point_G: list[np.ndarray] = []
            point_H: list[np.ndarray] = []
            point_extra: list[np.ndarray] = []

            frames_seen = 0
            initial_head_pose: tuple[float, float] | None = None
            head_varied = False

            while len(point_G) < samples_per_point:
                ret, frame, _ = capture_fn()
                if not ret or frame is None:
                    continue

                face_data = face_detector.process(frame)
                norm_data = normalizer.process(frame, face_data) if normalizer is not None else None
                gaze_data = gaze_estimator.estimate(face_data, norm_data=norm_data)

                # Écran de calibration
                calib_screen = np.zeros((h, w, 3), dtype=np.uint8)
                frames_seen += 1

                # Animation cible
                progress = len(point_G) / float(samples_per_point)
                radius = int(24 + 8 * math.sin(frames_seen * 0.25))

                # Instructions principales
                msg1 = f"Point {pt_idx + 1}/{total_points} - Fixez le repere orange"
                msg2 = "Conseil : bougez doucement la tete (inclinaison / rotation) en fixant la cible"
                cv2.putText(calib_screen, msg1, (w // 2 - 280, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 220, 220), 2)
                cv2.putText(calib_screen, msg2, (w // 2 - 420, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 200, 255), 1)

                # Dessin de la cible
                cv2.circle(calib_screen, (target_x, target_y), radius, (0, 140, 255), 2)
                cv2.circle(calib_screen, (target_x, target_y), 6, (0, 220, 255), -1)

                # Jauge de progression sous la cible
                bar_w = 70
                fill_w = int(bar_w * progress)
                cv2.rectangle(calib_screen, (target_x - bar_w // 2, target_y + 35), (target_x - bar_w // 2 + fill_w, target_y + 40), (0, 255, 120), -1)

                # Widget boussole de tête en haut à droite
                if face_data.detected:
                    hy, hp = face_data.pose_ypr[0], face_data.pose_ypr[1]
                    if initial_head_pose is None and frames_seen > settle_frames:
                        initial_head_pose = (hy, hp)

                    # Vérification de mouvement de tête
                    if initial_head_pose is not None:
                        d_head = math.sqrt((hy - initial_head_pose[0])**2 + (hp - initial_head_pose[1])**2)
                        if d_head > 3.0:  # Déplacement tête > 3 degrés
                            head_varied = True

                    # Mini widget pose tête
                    bx, by = w - 120, 70
                    cv2.rectangle(calib_screen, (bx - 40, by - 40), (bx + 40, by + 40), (40, 40, 40), -1)
                    cv2.rectangle(calib_screen, (bx - 40, by - 40), (bx + 40, by + 40), (100, 100, 100), 1)
                    dot_x = int(bx + np.clip(hy * 2.0, -35, 35))
                    dot_y = int(by + np.clip(hp * 2.0, -35, 35))
                    dot_col = (0, 255, 0) if head_varied else (0, 165, 255)
                    cv2.circle(calib_screen, (dot_x, dot_y), 5, dot_col, -1)
                    cv2.putText(calib_screen, "Tete", (bx - 15, by + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                cv2.imshow(window_name, calib_screen)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    cv2.destroyWindow(window_name)
                    print("[calibration] Calibration annulée.")
                    return False

                if face_data.detected and frames_seen > settle_frames:
                    point_G.append(gaze_data.raw_features)
                    point_H.append(np.array([face_data.pose_ypr[0], face_data.pose_ypr[1]]))
                    iris_r = iris_probe.estimate(face_data).iris_ratios
                    origin = norm_data.origin_3d_mm if norm_data is not None else np.zeros(3)
                    point_extra.append(np.array([
                        time.perf_counter() - t_calib_start,
                        iris_r[0], iris_r[1],
                        *face_data.pose_xyz, *face_data.pose_ypr,
                        *origin,
                        gaze_data.pitch_yaw_deg[1], gaze_data.pitch_yaw_deg[0],
                        face_data.iris_diameter_px,
                    ], dtype=np.float64))

            # Enregistrement des paires collectées pour ce point cible
            for g_feat, h_pose, extra in zip(point_G, point_H, point_extra):
                collected_G.append(g_feat)
                collected_H.append(h_pose)
                collected_Y.append(np.array([target_x, target_y], dtype=np.float64))
                collected_extra.append(extra)

        cv2.destroyWindow(window_name)

        # Ajustement du modèle Affine Ridge
        G_mat = np.array(collected_G)
        H_mat = np.array(collected_H)
        Y_mat = np.array(collected_Y)

        try:
            np.savez(
                data_path(f"calibration_samples_{self.mode}.npz"),
                G=G_mat, H=H_mat, Y=Y_mat, extra=np.array(collected_extra),
                extra_cols=np.array(["t", "iris_rx", "iris_ry", "pose_x", "pose_y", "pose_z",
                                     "pose_yaw", "pose_pitch", "pose_roll", "origin_x", "origin_y",
                                     "origin_z", "net_yaw", "net_pitch", "iris_px"]),
                screen=np.array([self.screen_config.width_px, self.screen_config.height_px,
                                 self.screen_config.diagonal_inches, self.screen_config.distance_cm]),
            )
        except Exception as e:  # l'enregistrement d'analyse ne doit jamais bloquer la calibration
            print(f"[calibration] (échantillons non enregistrés : {e})")

        rmse_px = self.model.fit(G_mat, H_mat, Y_mat)
        deg_err = rmse_px * self.screen_config.mm_per_px_x / (self.screen_config.distance_cm * 10.0) * (180.0 / math.pi)

        print(f"[calibration] Calibration ({total_points} points) terminée.")
        print(f"              -> RMSE résiduel : {rmse_px:.1f} px (~{deg_err:.2f}°)")
        self._print_diagnostic(G_mat, H_mat, Y_mat)
        self.is_calibrated = True
        return True

    def _print_diagnostic(self, G: np.ndarray, H: np.ndarray, Y: np.ndarray) -> None:
        """Affiche si le regard mesuré suit vraiment les cibles, et l'erreur par point."""
        sc = self.screen_config
        dist_mm = sc.distance_cm * 10.0
        targets = np.unique(Y, axis=0)
        means = np.array([G[np.all(Y == t, axis=1)].mean(axis=0) for t in targets])
        head = np.array([H[np.all(Y == t, axis=1)].mean(axis=0) for t in targets])

        left = targets[:, 0] < sc.width_px * 0.3
        right = targets[:, 0] > sc.width_px * 0.7
        top = targets[:, 1] < sc.height_px * 0.3
        bottom = targets[:, 1] > sc.height_px * 0.7

        def span(col: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
            return float(col[b].mean() - col[a].mean()) if a.any() and b.any() else float("nan")

        # Amplitude attendue (angle visuel entre les cibles extrêmes)
        exp_x = 2.0 * math.degrees(math.atan(0.35 * sc.width_mm / dist_mm))
        exp_y = 2.0 * math.degrees(math.atan(0.35 * sc.height_mm / dist_mm))

        print("[calibration] Diagnostic :")
        if self.mode == "net":
            # Features en mm sur l'écran : écart attendu entre cibles extrêmes = 70 % de l'écran
            mx, my = span(means[:, 0], left, right), span(means[:, 1], top, bottom)
            exp_mx, exp_my = 0.7 * sc.width_mm, 0.7 * sc.height_mm
            print(f"              Regard réseau gauche->droite : {mx:+.0f} mm  (attendu ≈ +{exp_mx:.0f} mm)")
            print(f"              Regard réseau haut->bas      : {my:+.0f} mm  (attendu ≈ +{exp_my:.0f} mm)")
            ratio = min(abs(mx) / exp_mx, abs(my) / exp_my) if np.isfinite(mx) and np.isfinite(my) else 1.0
            if ratio < 0.3:
                print("              /!\\ Le regard estimé bouge très peu entre les cibles : le réseau suit mal")
                print("                  vos yeux (éclairage de face, reflets de lunettes, visage trop loin ?).")
        else:
            mx, my = span(means[:, 0], left, right), span(means[:, 1], top, bottom)
            print(f"              Ratio iris gauche->droite : {mx:+.3f}   haut->bas : {my:+.3f}")
        hx, hy = span(head[:, 0], left, right), span(head[:, 1], top, bottom)
        print(f"              Tête gauche->droite : {hx:+.1f}°   haut->bas : {hy:+.1f}°")

        # Erreur moyenne par cible
        errs = []
        for t in targets:
            mask = np.all(Y == t, axis=1)
            preds = np.array([self.model.predict(g, (h[0], h[1])) for g, h in zip(G[mask], H[mask])])
            errs.append(float(np.linalg.norm(preds.mean(axis=0) - t)))
        worst = int(np.argmax(errs))
        print(f"              Erreur par point (px) : " + ", ".join(f"{e:.0f}" for e in errs)
              + f"  | pire : ({targets[worst][0]:.0f}, {targets[worst][1]:.0f})")

    def predict(
        self,
        raw_features: np.ndarray,
        head_pose: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[float, float]:
        """Prédit le point de regard écran compensé de la pose de tête."""
        if not self.is_calibrated:
            return float(self.screen_config.width_px / 2), float(self.screen_config.height_px / 2)
        return self.model.predict(raw_features, head_pose=head_pose)

    def handle_mouse_click(
        self,
        gaze_raw: np.ndarray,
        head_pose: tuple[float, float],
        click_px: tuple[float, float],
        max_error_px: float = 220.0,
    ) -> bool:
        """Applique une recalibration continue (RLS) lors d'un clic souris utilisateur."""
        if not self.is_calibrated:
            return False

        updated, dist = self.model.update_rls(
            gaze_raw=gaze_raw,
            head_pose=head_pose,
            target_xy=click_px,
            max_error_px=max_error_px,
        )
        if updated:
            self.rls_update_count += 1
        return updated

    def save(self, filepath: str = "calibration.npz") -> None:
        """Sauvegarde l'état de calibration dans un fichier."""
        self.model.save(filepath)

    def load(self, filepath: str = "calibration.npz") -> bool:
        """Charge l'état de calibration depuis un fichier en vérifiant la compatibilité du mode."""
        import os
        if not os.path.exists(filepath):
            return False

        loaded_model = AffineGazeModel(mode=self.mode)
        if not loaded_model.load(filepath):
            return False

        if loaded_model.mode != self.mode:
            print(f"[calibration] '{filepath}' ({loaded_model.mode}) n'est pas compatible avec le mode '{self.mode}'.")
            return False

        self.model = loaded_model
        self.is_calibrated = True
        return True
