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
from face import GLARE_STRONG, glare_on_iris
from geometry import ScreenConfig
from i18n import tr
from paths import data_path
import winapp


CALIB_FORMAT_VERSION = 5  # v5 : features réseau = point d'impact (mm) + ratios d'iris
# Régularisation Ridge par mode (validation croisée : 1e-2 le mieux en net, 1e-3 en iris)
RIDGE_ALPHA = {"net": 1e-2, "iris": 1e-3}
# Écart-type minimal par feature : une feature qui n'a presque pas varié pendant la
# calibration (ex. position de la tête) ne doit pas être amplifiée démesurément.
MIN_FEATURE_STD = {
    "net": [20.0, 20.0, 0.005, 0.005],                      # mm sur l'écran, ratios d'iris
    "iris": [0.005, 0.005, 0.005, 0.005, 0.005, 3.0, 3.0],  # ratios..., degrés tête
}
# Poids d'un clic souris en recalibration RLS, en « nombre d'échantillons » de calibration
RLS_EQUIV_SAMPLES = 15.0
# Compensation géométrique de la fuite de pose de tête du réseau MobileOne (entraîné sur Gaze360) :
# le réseau réagit à 100 % aux rotations de la tête mais seulement partiellement aux contre-rotations
# des yeux dans les orbites (réflexe vestibulo-oculaire). Ces distances effectives (mm) annulent le
# glissement parasite du point d'impact quand la tête tourne à regard fixe.
HEAD_YAW_LEAKAGE_MM = 500.0
HEAD_PITCH_LEAKAGE_MM = 320.0
# Part de bruit intra-cible conservée en mode 'net' après rejet MAD (évite le biais d'atténuation
# Errors-in-Variables qui sous-pondérait le réseau, plus bruité d'une frame à l'autre que l'iris).
TARGET_SHRINK_GAMMA = 0.15


class AffineGazeModel:
    """Modèle de calibration Ridge sur features standardisées.

    Modes :
    - 'net'  : features = point d'impact (x, y en mm) du rayon de regard sur le plan écran
               compensé de la fuite de rotation de tête (VOR), + ratios d'iris (x, y).
               L'ajustement calcule les centroïdes robustes (médiane/MAD à 75 %) par cible pour
               éliminer les micro-saccades et le biais d'atténuation entre réseau et iris, avec
               une régularisation directionnelle découplant l'axe vertical du bruit horizontal.
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
        self.quality_deg: float = float("nan")  # note de la calibration (°, validation croisée)

    def _raw_features(self, G: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Construit les features brutes (avant standardisation)."""
        G = np.atleast_2d(np.asarray(G, dtype=np.float64))
        H = np.atleast_2d(np.asarray(H, dtype=np.float64))
        if self.mode == "net":
            out = G.copy()
            if H.shape[1] >= 2 and out.shape[1] >= 2:
                if len(H) == 1 and len(out) > 1:
                    H = np.tile(H, (len(out), 1))
                if len(H) == len(out):
                    out[:, 0] -= HEAD_YAW_LEAKAGE_MM * np.tan(np.radians(H[:, 0]))
                    out[:, 1] += HEAD_PITCH_LEAKAGE_MM * np.tan(np.radians(H[:, 1]))
            return out
        rx = G[:, 0:1]
        ry = G[:, 1:2]
        return np.hstack([rx, ry, rx**2, rx * ry, ry**2, H[:, 0:1], H[:, 1:2]])

    @staticmethod
    def _trim_cluster(F: np.ndarray, keep_pct: float = 75.0) -> np.ndarray:
        """Conserve les échantillons les plus proches de la médiane (rejet MAD des saccades)."""
        if len(F) < 8:
            return F
        med = np.median(F, axis=0)
        mad = np.maximum(np.median(np.abs(F - med), axis=0), 1e-6)
        dist = np.linalg.norm((F - med) / mad, axis=1)
        return F[dist <= np.percentile(dist, keep_pct)]

    def _denoise_targets(self, raw: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """Filtre les outliers par cible et réduit le bruit intra-cible en mode 'net'."""
        targets = np.unique(Y, axis=0)
        n_unique = len(targets)
        if self.mode != "net" or n_unique < 4 or len(raw) < n_unique * 4:
            return raw, Y, n_unique

        gamma = TARGET_SHRINK_GAMMA if n_unique >= 6 else 0.50
        raw_out: list[np.ndarray] = []
        y_out: list[np.ndarray] = []
        for t in targets:
            mask = np.all(Y == t, axis=1)
            ft = self._trim_cluster(raw[mask])
            mean_t = ft.mean(axis=0)
            raw_out.append((1.0 - gamma) * mean_t + gamma * ft)
            y_out.append(np.tile(t, (len(ft), 1)))
        return np.vstack(raw_out), np.vstack(y_out), n_unique

    def _design(self, G: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Matrice de design standardisée [z_1 .. z_k, 1]."""
        raw = self._raw_features(G, H)
        z = (raw - self.feat_mean) / self.feat_std
        return np.hstack([z, np.ones((z.shape[0], 1), dtype=np.float64)])

    def fit(self, G: np.ndarray, H: np.ndarray, Y: np.ndarray) -> float:
        """Ajuste le modèle par moindres carrés régularisés Ridge.

        Args:
            G: Mesures de regard brut (N, d).
            H: Poses de tête (N, 2) [yaw, pitch].
            Y: Coordonnées vraies des cibles écran en pixels (N, 2).

        Returns:
            RMSE résiduel d'entraînement en pixels.
        """
        raw_all = self._raw_features(G, H)
        Y_arr = np.asarray(Y, dtype=np.float64)
        raw_fit, Y_fit, n_unique = self._denoise_targets(raw_all, Y_arr)

        self.feat_mean = raw_fit.mean(axis=0)
        std = raw_fit.std(axis=0)
        # Une feature qui a peu varié (ex. tête immobile) ne doit pas être amplifiée
        floor = np.asarray(MIN_FEATURE_STD.get(self.mode, [1e-6] * raw_fit.shape[1]), dtype=np.float64)
        self.feat_std = np.maximum(std, floor[: raw_fit.shape[1]])

        z_fit = (raw_fit - self.feat_mean) / self.feat_std
        phi_fit = np.hstack([z_fit, np.ones((z_fit.shape[0], 1), dtype=np.float64)])
        n, k = phi_fit.shape

        if self.mode == "net" and k == 5:
            # Régularisation directionnelle : évite que les termes croisés (ex. ratio vertical
            # d'iris pollué par la paupière supérieure en bas d'écran) ne dévient l'autre axe.
            if n_unique >= 6:
                pen_x = [1.0, 1.5, 1.0, 1.5, 0.0]
                pen_y = [10.0, 1.0, 10.0, 2.5, 0.0]
            else:
                pen_x = [5.0, 25.0, 5.0, 25.0, 0.0]
                pen_y = [25.0, 5.0, 25.0, 10.0, 0.0]
            Ax = phi_fit.T @ phi_fit + self.alpha * n * np.diag(pen_x)
            Ay = phi_fit.T @ phi_fit + self.alpha * n * np.diag(pen_y)
            wx = np.linalg.solve(Ax, phi_fit.T @ Y_fit[:, 0])
            wy = np.linalg.solve(Ay, phi_fit.T @ Y_fit[:, 1])
            self.weights = np.column_stack([wx, wy])
            A = 0.5 * (Ax + Ay)
        else:
            reg = self.alpha * n * np.eye(k, dtype=np.float64)
            reg[-1, -1] = 0.0  # biais (dernière colonne) non pénalisé
            A = phi_fit.T @ phi_fit + reg
            self.weights = np.linalg.solve(A, phi_fit.T @ Y_fit)

        # Covariance RLS : on « dilue » la confiance du fit initial pour qu'un clic pèse
        # comme ~RLS_EQUIV_SAMPLES échantillons de calibration (sinon 1 clic = 1/N, inefficace).
        self.P = np.linalg.inv(A) * (n / RLS_EQUIV_SAMPLES)
        self.is_fitted = True

        phi_all = self._design(G, H)
        pred = phi_all @ self.weights
        return float(np.sqrt(np.mean(np.sum((Y_arr - pred) ** 2, axis=1))))

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
            quality_deg=self.quality_deg,
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
            self.quality_deg = float(data["quality_deg"]) if "quality_deg" in data.files else float("nan")
            self.is_fitted = True
            return True
        except Exception as e:
            print(f"[calibration] Erreur chargement ({filepath}) : {e}")
            return False


# Note de qualité : erreur angulaire moyenne de chaque point prédit SANS lui (validation croisée
# « un point sorti ») -- plus honnête que l'erreur résiduelle, qui flatte toujours le modèle.
# Seuils calés sur une vraie calibration 9 points avec lunettes (nov. 2026, bien vécue en jeu) :
# 4,8° de moyenne, points de 1,4° (centre) à 8,0° (coins, extrapolés quand on les retire).
# Un point raté (regard sur une autre cible) y donne 18 à 38° et une moyenne de 7,3 à 14°.
QUALITY_GOOD_DEG = 5.5   # « bonne »
QUALITY_OK_DEG = 7.0     # « moyenne » ; au-delà : « à refaire »
# Un point est à refaire s'il dépasse 10° ET le double de la médiane des points : un point raté
# gonfle aussi un peu ses voisins, un seuil relatif évite de les accuser à tort.
POINT_BAD_DEG = 10.0
POINT_BAD_RATIO = 2.0
POINT_WARN_DEG = 6.0     # orange au-delà de 6° et de 1,4 fois la médiane
POINT_WARN_RATIO = 1.4
# Avec 5 points (croix), retirer un bras de la croix oblige à extrapoler tout un côté : la même
# calibration y donne ~1,7 fois plus d'erreur. Note approximative, 9 points conseillés.
FEW_POINTS_FACTOR = 1.7


def quality_label(mean_deg: float, num_points: int = 9) -> str:
    """Note lisible d'une calibration à partir de son erreur moyenne (°)."""
    if not np.isfinite(mean_deg):
        return "inconnue"
    k = FEW_POINTS_FACTOR if num_points < 9 else 1.0
    if mean_deg <= QUALITY_GOOD_DEG * k:
        return "bonne"
    if mean_deg <= QUALITY_OK_DEG * k:
        return "moyenne"
    return "à refaire"


def classify_points(errs: list[float]) -> list[str]:
    """État de chaque point : "ok", "warn" (orange) ou "bad" (rouge, à refaire)."""
    if not errs:
        return []
    med = float(np.median(errs))
    out = []
    for e in errs:
        if e > max(POINT_BAD_DEG, POINT_BAD_RATIO * med):
            out.append("bad")
        elif e > max(POINT_WARN_DEG, POINT_WARN_RATIO * med):
            out.append("warn")
        else:
            out.append("ok")
    return out


def points_to_redo(errs: list[float], glare: list[float] | None = None) -> list[int]:
    """Points proposés à [R] : les rouges et ceux pris avec un reflet, sinon le moins bon."""
    bad = {i for i, c in enumerate(classify_points(errs)) if c == "bad"}
    if glare is not None:
        bad |= {i for i, g in enumerate(glare) if g > GLARE_STRONG}
    return sorted(bad) if bad else ([int(np.argmax(errs))] if errs else [])


def evaluate_points(
    mode: str,
    points: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    screen_config: ScreenConfig,
) -> tuple[list[float], list[np.ndarray], float]:
    """Erreur de chaque point de calibration, prédit par un modèle appris SANS ce point.

    Args:
        points: liste de (G (n, d), H (n, 2), cible (2,) en px) par point de calibration.

    Returns:
        (erreur en degrés par point, position moyenne prédite par point, erreur moyenne).
    """
    alpha = RIDGE_ALPHA.get(mode, 1e-3)
    errs: list[float] = []
    preds: list[np.ndarray] = []
    for i, (G_i, H_i, t_i) in enumerate(points):
        others = [p for j, p in enumerate(points) if j != i]
        m = AffineGazeModel(mode=mode, alpha=alpha)
        m.fit(np.vstack([p[0] for p in others]), np.vstack([p[1] for p in others]),
              np.vstack([np.tile(p[2], (len(p[0]), 1)) for p in others]))
        raw_i = m._trim_cluster(m._raw_features(G_i, H_i))
        z_i = (raw_i - m.feat_mean) / m.feat_std
        phi_i = np.hstack([z_i, np.ones((len(z_i), 1), dtype=np.float64)])
        mean_pred = (phi_i @ m.weights).mean(axis=0)
        errs.append(screen_config.visual_angle_deg((float(t_i[0]), float(t_i[1])),
                                                   (float(mean_pred[0]), float(mean_pred[1]))))
        preds.append(mean_pred)
    return errs, preds, float(np.mean(errs)) if errs else float("nan")


def draw_review(screen_config: ScreenConfig, points: list[dict], errs: list[float],
                preds: list[np.ndarray], mean_deg: float) -> np.ndarray:
    """Image de l'écran de résultats de la calibration (texte sans accents : cv2.putText)."""
    w, h = screen_config.width_px, screen_config.height_px
    img = np.zeros((h, w, 3), dtype=np.uint8)
    label = quality_label(mean_deg, len(points))
    color = {"bonne": (60, 200, 60), "moyenne": (0, 165, 255)}.get(label, (60, 60, 230))
    ascii_label = tr({"bonne": "BONNE", "moyenne": "MOYENNE", "à refaire": "A REFAIRE"}.get(label, "?"))
    title = tr("Calibration terminee - qualite : {label} (erreur moyenne {deg} deg)",
               label=ascii_label, deg=f"{mean_deg:.1f}")
    cv2.putText(img, title, (w // 2 - 430, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)
    cv2.putText(img, tr("Cercle : cible ; croix : ou le regard a ete estime (chaque point predit sans lui)"),
                (w // 2 - 430, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 190, 190), 1)

    states = classify_points(errs)
    for i, (p, e, pr) in enumerate(zip(points, errs, preds)):
        tx, ty = int(p["target"][0]), int(p["target"][1])
        c = {"ok": (60, 200, 60), "warn": (0, 165, 255)}.get(states[i], (60, 60, 230))
        px = int(np.clip(pr[0], 0, w - 1))
        py = int(np.clip(pr[1], 0, h - 1))
        cv2.line(img, (tx, ty), (px, py), (110, 110, 110), 1)
        cv2.drawMarker(img, (px, py), c, cv2.MARKER_CROSS, 16, 2)
        cv2.circle(img, (tx, ty), 18, c, 3)
        cv2.putText(img, f"{i + 1}", (tx - 7, ty + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        cv2.putText(img, f"{e:.1f} deg", (tx + 24, ty - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1)
        if p.get("glare", 0.0) > GLARE_STRONG:
            cv2.putText(img, tr("reflet"), (tx + 24, ty + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 230), 1)

    glare = [p.get("glare", 0.0) for p in points]
    redo = points_to_redo(errs, glare)
    flagged = any(c == "bad" for c in states) or any(g > GLARE_STRONG for g in glare)
    redo_txt = (tr("[R] Refaire les points {pts} (rouges ou avec reflet)", pts=", ".join(str(i + 1) for i in redo))
                if flagged else tr("[R] Refaire le moins bon point ({n})", n=redo[0] + 1))
    lines = [tr("[Entree] Valider"), redo_txt, tr("[Echap] Annuler (garder l'ancienne calibration)")]
    if len(points) < 9:
        lines.insert(0, tr("5 points : note approximative (9 points conseilles)"))
    glare_pts = [i + 1 for i, g in enumerate(glare) if g > GLARE_STRONG]
    warning = ""
    if glare_pts:
        warning = tr("Reflets sur les lunettes aux points {pts} : deplacez la lampe ou inclinez l'ecran avant de "
                     "les refaire", pts=", ".join(map(str, glare_pts)))
        lines.insert(0, warning)
    # Consignes dans la bande libre entre la rangée du milieu et celle du bas (cibles à 50 % et
    # 90 % de la hauteur), sur un fond sombre pour rester lisibles par-dessus les traits.
    line_h = 34
    text_w = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0][0] for line in lines)
    y0 = int(h * 0.70) - (line_h * (len(lines) - 1)) // 2
    x0 = max(20, w // 2 - text_w // 2)
    cv2.rectangle(img, (x0 - 20, y0 - 30), (x0 + text_w + 20, y0 + line_h * (len(lines) - 1) + 16), (25, 25, 25), -1)
    for k, line in enumerate(lines):
        col = (60, 60, 230) if warning and line == warning else (230, 230, 230)
        cv2.putText(img, line, (x0, y0 + line_h * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 1)
    return img


def draw_intro(w: int, h: int, frame: np.ndarray | None, face_detected: bool, glare: float,
               num_points: int = 9) -> np.ndarray:
    """Écran d'accueil du premier lancement : explications, image de la caméra, Commencer / Plus tard.

    Texte sans accents (cv2.putText). `frame` : image caméra (déjà en miroir) ou None.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Tailles prévues pour 1920x1080, agrandies ou réduites selon l'écran (lisible en 1440p/4K)
    sc = max(0.7, min(h / 1080.0, w / 1920.0))
    x0 = max(20, w // 2 - int(560 * sc))
    y = 0.0

    def put(text: str, dy: float, size: float, color: tuple, thick: int = 1, x: int | None = None) -> None:
        nonlocal y
        y += dy * sc
        cv2.putText(img, text, (x0 if x is None else x, int(y)), font, size * sc, color,
                    max(1, round(thick * sc)), cv2.LINE_AA)

    put(tr("Bienvenue dans Eyetrack"), 110, 1.4, (0, 200, 255), 3)
    put(tr("Premiere utilisation : on calibre votre regard (environ 1 minute)"), 55, 0.8, (230, 230, 230), 2)
    steps = [
        tr("1. Installez-vous comme pour jouer, face a la camera (gardez vos lunettes si vous en portez)."),
        tr("2. Un repere orange apparait a {n} endroits : fixez-le jusqu'a ce qu'il change de place.", n=num_points),
        tr("3. Bougez doucement la tete en le fixant : le regard est appris dans toutes les positions."),
        tr("4. A la fin, une note s'affiche : [Entree] valide, [R] refait les points rates."),
    ]
    for k, line in enumerate(steps):
        put(line, 60 if k == 0 else 42, 0.65, (210, 210, 210))

    # Image de la caméra : on vérifie d'un coup d'œil qu'on est bien cadré
    y_cam = int(y + 35 * sc)
    cam_w = max(1, min(int(420 * sc), w - 2 * x0))
    cam_h = int(cam_w * (frame.shape[0] / frame.shape[1] if frame is not None else 0.75))
    color = (60, 200, 60) if face_detected else (60, 60, 230)
    if frame is not None and y_cam + cam_h < h:
        img[y_cam:y_cam + cam_h, x0:x0 + cam_w] = cv2.resize(frame, (cam_w, cam_h))
    cv2.rectangle(img, (x0 - 3, y_cam - 3), (x0 + cam_w + 2, y_cam + cam_h + 2), color, max(2, round(3 * sc)))
    status = tr("Visage detecte") if face_detected else tr("Visage non detecte : placez-vous face a la camera")
    sx = x0 + cam_w + int(30 * sc)
    y = float(y_cam)
    put(status, 30, 0.75, color, 2, x=sx)
    if glare > GLARE_STRONG:
        put(tr("Reflet sur les lunettes : deplacez la lampe ou inclinez un peu l'ecran"), 40, 0.6, (60, 60, 230), 1,
            x=sx)

    y = float(y_cam + cam_h)
    put(tr("[Entree] Commencer la calibration"), 75, 0.95, (60, 200, 60), 2)
    put(tr("[Echap] Plus tard (bouton \"Calibrer le regard\" du panneau)"), 45, 0.7, (190, 190, 190))
    return img


def open_calibration_window(window_name: str, w: int, h: int) -> bool:
    """Ouvre la fenêtre de calibration en plein écran, au-dessus de tout, avec le clavier.

    Sans cela, elle pouvait s'ouvrir derrière le jeu, le navigateur ou le panneau, et Échap ou
    Entrée partaient dans une autre fenêtre. Renvoie vrai si elle est bien au premier plan.
    """
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, np.zeros((h, w, 3), np.uint8))  # fenêtre affichée avant d'être levée
    cv2.waitKey(1)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except (cv2.error, AttributeError):  # OpenCV ancien ou sans interface Windows
        pass
    return winapp.bring_to_front(window_name)


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
        self.model = AffineGazeModel(mode=mode, alpha=RIDGE_ALPHA.get(mode, 1e-3))
        self.is_calibrated = False
        self.rls_update_count = 0
        self.last_point_errors_deg: list[float] = []

        self._init_grid(num_points)

    def _init_grid(self, num_points: int) -> None:
        """Configure la grille de points (5 points en croix ou 9 points en grille).

        Cibles à 10 % des bords (au lieu de 15 %) : le regard est appris plus près des coins,
        là où il sert en jeu (rétroviseurs, angles morts), au lieu d'y être extrapolé.
        """
        lo, hi = 0.10, 0.90
        if num_points == 5:
            # 5 points en croix : Centre, Gauche, Droite, Haut, Bas
            self.points_norm = [(0.50, 0.50), (lo, 0.50), (hi, 0.50), (0.50, lo), (0.50, hi)]
        else:
            # 9 points complets (3x3)
            self.points_norm = [(x, y) for y in (lo, 0.50, hi) for x in (lo, 0.50, hi)]

    @property
    def quality_deg(self) -> float:
        """Erreur moyenne (°) en validation croisée de la calibration active (NaN si inconnue)."""
        return float(getattr(self.model, "quality_deg", float("nan")))

    @property
    def quality_label(self) -> str:
        return quality_label(self.quality_deg, len(self.points_norm))

    def run_interactive_calibration(
        self,
        capture_fn,
        face_detector,
        gaze_estimator,
        normalizer=None,
        samples_per_point: int = 45,
        settle_frames: int = 15,
        review: bool = True,
        intro: bool = False,
    ) -> bool:
        """Calibration guidée : collecte point par point, note de qualité, points à refaire.

        Après les points, un écran de résultats montre l'erreur de chaque point (prédit SANS
        lui, donc honnête) et la note globale : [Entrée] valide, [R] refait les points en
        rouge (ou le moins bon), [Échap] annule en gardant l'ancienne calibration.
        Avec review=False (outils, tests), la calibration est validée directement.
        Avec intro=True (premier lancement), un écran d'accueil explique la calibration avant
        de commencer ; [Échap] la remet à plus tard (renvoie False).
        """
        w = self.screen_config.width_px
        h = self.screen_config.height_px

        window_name = tr("Calibration (Echap pour annuler)")
        open_calibration_window(window_name, w, h)

        from gaze_model import IrisGazeEstimator
        from filters import BlinkDetector
        ctx = {
            "capture_fn": capture_fn, "face_detector": face_detector, "gaze_estimator": gaze_estimator,
            "normalizer": normalizer, "samples_per_point": samples_per_point, "settle_frames": settle_frames,
            "iris_probe": IrisGazeEstimator(), "blink": BlinkDetector(), "t0": time.perf_counter(),
            "window": window_name, "w": w, "h": h,
        }
        if intro and not self._intro_screen(ctx):
            cv2.destroyWindow(window_name)
            print("[calibration] Calibration remise à plus tard (bouton « Calibrer le regard » du panneau).")
            return False

        total_points = len(self.points_norm)
        points: list[dict | None] = [None] * total_points
        to_collect = list(range(total_points))
        rounds = 0
        while True:
            rounds += 1
            for pt_idx in to_collect:
                res = self._collect_point(pt_idx, ctx)
                if res is None:
                    cv2.destroyWindow(window_name)
                    print("[calibration] Calibration annulée (ancienne calibration conservée).")
                    return False
                points[pt_idx] = res

            G_mat = np.vstack([p["G"] for p in points])
            H_mat = np.vstack([p["H"] for p in points])
            Y_mat = np.vstack([np.tile(p["target"], (len(p["G"]), 1)) for p in points])
            new_model = AffineGazeModel(mode=self.mode, alpha=RIDGE_ALPHA.get(self.mode, 1e-3))
            rmse_px = new_model.fit(G_mat, H_mat, Y_mat)
            errs, preds, mean_deg = evaluate_points(
                self.mode, [(p["G"], p["H"], p["target"]) for p in points], self.screen_config)
            new_model.quality_deg = mean_deg

            if not review:
                break
            choice, redo = self._review_screen(points, errs, preds, mean_deg, ctx)
            if choice == "accept":
                break
            if choice == "cancel":
                cv2.destroyWindow(window_name)
                print("[calibration] Calibration annulée (ancienne calibration conservée).")
                return False
            to_collect = redo
            print(f"[calibration] Points refaits : {', '.join(str(i + 1) for i in redo)}")

        cv2.destroyWindow(window_name)
        self.model = new_model
        self.is_calibrated = True
        self.rls_update_count = 0
        self.last_point_errors_deg = errs

        # Données brutes enregistrées pour analyse hors ligne (calibration_samples_<mode>.npz)
        try:
            np.savez(
                data_path(f"calibration_samples_{self.mode}.npz"),
                G=G_mat, H=H_mat, Y=Y_mat, extra=np.vstack([p["extra"] for p in points]),
                extra_cols=np.array(["t", "iris_rx", "iris_ry", "pose_x", "pose_y", "pose_z",
                                     "pose_yaw", "pose_pitch", "pose_roll", "origin_x", "origin_y",
                                     "origin_z", "net_yaw", "net_pitch", "iris_px", "eye_open", "glare"]),
                screen=np.array([self.screen_config.width_px, self.screen_config.height_px,
                                 self.screen_config.diagonal_inches, self.screen_config.distance_cm]),
            )
        except Exception as e:  # l'enregistrement d'analyse ne doit jamais bloquer la calibration
            print(f"[calibration] (échantillons non enregistrés : {e})")

        deg_err = rmse_px * self.screen_config.mm_per_px_x / (self.screen_config.distance_cm * 10.0) * (180.0 / math.pi)
        print(f"[calibration] Calibration ({total_points} points, {rounds} passage(s)) validée "
              f"({ctx['blink'].blink_count} clignements ignorés, "
              f"{sum(p['skipped'] for p in points)} images écartées).")
        print(f"              -> Qualité : {quality_label(mean_deg, total_points)} ({mean_deg:.1f}° en moyenne, chaque point "
              f"prédit sans lui) | RMSE résiduel : {rmse_px:.1f} px (~{deg_err:.2f}°)")
        print("              -> Erreur par point (°) : " + ", ".join(f"{e:.1f}" for e in errs))
        glare_pts = [i + 1 for i, p in enumerate(points) if p["glare"] > GLARE_STRONG]
        if glare_pts:
            print(f"              /!\\ Reflets sur les lunettes aux points {', '.join(map(str, glare_pts))}")
        self._print_diagnostic(G_mat, H_mat, Y_mat)
        return True

    def _intro_screen(self, ctx: dict) -> bool:
        """Écran d'accueil (premier lancement) : vrai pour commencer, faux pour plus tard."""
        w, h, n = ctx["w"], ctx["h"], len(self.points_norm)
        cv2.imshow(ctx["window"], draw_intro(w, h, None, False, 0.0, n))
        glare_live = 0.0
        while True:
            ret, frame, _ = ctx["capture_fn"]()
            if ret and frame is not None:
                face_data = ctx["face_detector"].process(frame)
                glare_live += 0.3 * (glare_on_iris(frame, face_data) - glare_live)
                cv2.imshow(ctx["window"], draw_intro(w, h, frame, face_data.detected, glare_live, n))
            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10, 32):           # Entrée / Espace
                return True
            if key == 27 or key == ord("q"):  # Échap
                return False

    def _collect_point(self, pt_idx: int, ctx: dict) -> dict | None:
        """Affiche la cible `pt_idx` et collecte ses échantillons ; None si annulé (Échap)."""
        w, h = ctx["w"], ctx["h"]
        nx, ny = self.points_norm[pt_idx]
        target_x, target_y = self.screen_config.norm_to_px(nx, ny)
        total_points = len(self.points_norm)
        samples_per_point, settle_frames = ctx["samples_per_point"], ctx["settle_frames"]
        blink, iris_probe = ctx["blink"], ctx["iris_probe"]

        point_G: list[np.ndarray] = []
        point_H: list[np.ndarray] = []
        point_extra: list[np.ndarray] = []
        glare_vals: list[float] = []
        glare_live = 0.0
        skipped = 0
        frames_seen = 0
        initial_head_pose: tuple[float, float] | None = None
        head_varied = False

        while len(point_G) < samples_per_point:
            ret, frame, _ = ctx["capture_fn"]()
            if not ret or frame is None:
                continue

            face_data = ctx["face_detector"].process(frame)
            normalizer = ctx["normalizer"]
            norm_data = normalizer.process(frame, face_data) if normalizer is not None else None
            gaze_data = ctx["gaze_estimator"].estimate(face_data, norm_data=norm_data)
            glare = glare_on_iris(frame, face_data)
            glare_live += 0.3 * (glare - glare_live)

            calib_screen = np.zeros((h, w, 3), dtype=np.uint8)
            frames_seen += 1
            progress = len(point_G) / float(samples_per_point)
            radius = int(24 + 8 * math.sin(frames_seen * 0.25))

            msg1 = tr("Point {i}/{n} - Fixez le repere orange", i=pt_idx + 1, n=total_points)
            msg2 = tr("Conseil : bougez doucement la tete (inclinaison / rotation) en fixant la cible")
            cv2.putText(calib_screen, msg1, (w // 2 - 280, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 220, 220), 2)
            cv2.putText(calib_screen, msg2, (w // 2 - 420, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 200, 255), 1)
            if glare_live > GLARE_STRONG:
                cv2.putText(calib_screen, tr("Reflet sur les lunettes : deplacez la lampe ou inclinez un peu l'ecran"),
                            (w // 2 - 440, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 240), 2)
            if not face_data.detected and frames_seen > settle_frames:
                cv2.putText(calib_screen, tr("Visage non detecte : restez face a la camera"),
                            (w // 2 - 290, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 240), 2)

            cv2.circle(calib_screen, (target_x, target_y), radius, (0, 140, 255), 2)
            cv2.circle(calib_screen, (target_x, target_y), 6, (0, 220, 255), -1)
            bar_w = 70
            fill_w = int(bar_w * progress)
            cv2.rectangle(calib_screen, (target_x - bar_w // 2, target_y + 35),
                          (target_x - bar_w // 2 + fill_w, target_y + 40), (0, 255, 120), -1)

            if face_data.detected:
                hy, hp = face_data.pose_ypr[0], face_data.pose_ypr[1]
                if initial_head_pose is None and frames_seen > settle_frames:
                    initial_head_pose = (hy, hp)
                if initial_head_pose is not None:
                    d_head = math.sqrt((hy - initial_head_pose[0]) ** 2 + (hp - initial_head_pose[1]) ** 2)
                    if d_head > 3.0:
                        head_varied = True
                bx, by = w - 120, 70
                cv2.rectangle(calib_screen, (bx - 40, by - 40), (bx + 40, by + 40), (40, 40, 40), -1)
                cv2.rectangle(calib_screen, (bx - 40, by - 40), (bx + 40, by + 40), (100, 100, 100), 1)
                dot_x = int(bx + np.clip(hy * 2.0, -35, 35))
                dot_y = int(by + np.clip(hp * 2.0, -35, 35))
                cv2.circle(calib_screen, (dot_x, dot_y), 5, (0, 255, 0) if head_varied else (0, 165, 255), -1)
                cv2.putText(calib_screen, tr("Tete"), (bx - 15, by + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow(ctx["window"], calib_screen)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                return None

            blinking = face_data.detected and blink.update(face_data.eye_openness, time.perf_counter())
            if blinking and frames_seen > settle_frames:
                skipped += 1
            if face_data.detected and frames_seen > settle_frames and not blinking:
                point_G.append(np.asarray(gaze_data.raw_features, dtype=np.float64))
                point_H.append(np.array([face_data.pose_ypr[0], face_data.pose_ypr[1]], dtype=np.float64))
                glare_vals.append(glare)
                iris_r = iris_probe.estimate(face_data).iris_ratios
                origin = norm_data.origin_3d_mm if norm_data is not None else np.zeros(3)
                point_extra.append(np.array([
                    time.perf_counter() - ctx["t0"],
                    iris_r[0], iris_r[1],
                    *face_data.pose_xyz, *face_data.pose_ypr,
                    *origin,
                    gaze_data.pitch_yaw_deg[1], gaze_data.pitch_yaw_deg[0],
                    face_data.iris_diameter_px,
                    face_data.eye_openness,
                    glare,
                ], dtype=np.float64))

        return {
            "G": np.array(point_G), "H": np.array(point_H), "extra": np.array(point_extra),
            "target": np.array([target_x, target_y], dtype=np.float64),
            "glare": float(np.mean(glare_vals)) if glare_vals else 0.0, "skipped": skipped,
        }

    def _review_screen(self, points: list[dict], errs: list[float], preds: list[np.ndarray],
                       mean_deg: float, ctx: dict) -> tuple[str, list[int]]:
        """Écran de résultats : renvoie ("accept" | "redo" | "cancel", points à refaire)."""
        img = draw_review(self.screen_config, points, errs, preds, mean_deg)
        redo = points_to_redo(errs, [p["glare"] for p in points])
        while True:
            cv2.imshow(ctx["window"], img)
            key = cv2.waitKey(50) & 0xFF
            if key in (13, 10, 32):           # Entrée / Espace
                return "accept", []
            if key in (ord("r"), ord("R")):
                return "redo", redo
            if key == 27 or key == ord("q"):  # Échap
                return "cancel", []

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

        # Écart entre cibles extrêmes (fraction de l'écran), d'après la grille réellement utilisée
        frac_x = float(targets[:, 0].max() - targets[:, 0].min()) / max(sc.width_px, 1)
        frac_y = float(targets[:, 1].max() - targets[:, 1].min()) / max(sc.height_px, 1)

        print("[calibration] Diagnostic :")
        if self.mode == "net":
            # Features en mm sur l'écran : écart attendu entre cibles extrêmes
            mx, my = span(means[:, 0], left, right), span(means[:, 1], top, bottom)
            exp_mx, exp_my = frac_x * sc.width_mm, frac_y * sc.height_mm
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

        loaded_model = AffineGazeModel(mode=self.mode, alpha=RIDGE_ALPHA.get(self.mode, 1e-3))
        if not loaded_model.load(filepath):
            return False

        if loaded_model.mode != self.mode:
            print(f"[calibration] '{filepath}' ({loaded_model.mode}) n'est pas compatible avec le mode '{self.mode}'.")
            return False

        self.model = loaded_model
        self.is_calibrated = True
        return True
