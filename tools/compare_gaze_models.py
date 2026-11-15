"""Compare les modèles de regard sur les MÊMES images de la webcam (MobileNetV2, MobileOne-S0, ResNet-18).

Usage (depuis le dossier du projet, Eyetrack quitté pour libérer la caméra) :
    python tools/compare_gaze_models.py                  séance puis analyse
    python tools/compare_gaze_models.py --analyse FICHIER.npz   refaire l'analyse d'une séance

Séance (~2 min) : écran d'accueil (cadrage), puis 2 séries des 9 cibles de la calibration, la
seconde dans un autre ordre. Pendant la séance, aucun réseau ne tourne : l'outil garde en mémoire
ce que le réseau verrait à chaque image (visage normalisé, points du visage, repère). Ensuite,
chaque réseau présent dans models/ traite toute la séquence, dans l'ordre, avec son propre filtre
temporel, exactement comme s'il tournait seul dans le tracker. (Les trois réseaux en direct
prendraient ~110 ms par image : la séance tomberait à ~8 images/s, loin des conditions de jeu.)
Les images restent en mémoire et ne sont jamais enregistrées ; seules les mesures le sont.
Même chaîne que le tracker : MediaPipe, normalisation, calibration Ridge de l'application.

Mesures, pour chaque modèle :
- note de l'application : erreur moyenne de chaque cible prédite sans elle, dans chaque série ;
- série croisée : calibré sur une série, testé sur l'autre (le cas réel : calibrer, puis jouer) ;
- tremblement : agitation du point de regard d'une image à l'autre sur une cible fixe (avant le
  filtre du tracker) ;
- temps de calcul du réseau.
Les données brutes vont dans data/compare_models_<date>.npz, le rapport dans le .txt du même nom.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration import (AffineGazeModel, CalibrationManager, RIDGE_ALPHA,  # noqa: E402
                         evaluate_points, open_calibration_window)
from filters import BlinkDetector  # noqa: E402
from gaze_model import GazeFeatures, IrisGazeEstimator, NetGazeEstimator  # noqa: E402
from geometry import ScreenConfig, get_screen_resolution  # noqa: E402
from paths import USER_MODELS_DIR, data_path  # noqa: E402

MODEL_FILES = ["mobilenetv2_gaze.onnx", "mobileone_s0_gaze.onnx", "resnet18_gaze.onnx"]
# Par modèle : point d'impact (x, y en mm), ratios d'iris (x, y), angles du réseau (pitch, yaw)
FEATS_PER_MODEL = 6
# Seconde série : même grille, autre ordre (le regard n'arrive pas sur une cible depuis la même voisine)
PASS2_ORDER = [4, 0, 8, 2, 6, 1, 7, 3, 5]
EXTRA_COLS = ["t", "iris_rx", "iris_ry", "pose_x", "pose_y", "pose_z", "pose_yaw", "pose_pitch", "pose_roll",
              "origin_x", "origin_y", "origin_z", "net_yaw", "net_pitch", "iris_px", "eye_open", "glare"]


class RecordingEstimator:
    """Remplace le réseau pendant la séance : garde, pour chaque image, ce que le réseau verrait.

    Renvoie comme features le numéro de l'image : la calibration range ainsi, pour chaque
    échantillon retenu, le numéro qui permet de retrouver ensuite la sortie de chaque réseau.
    """

    def __init__(self) -> None:
        # Par image : (visage normalisé, points du visage en px, rotation, origine) ou None sans visage
        self.frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None] = []
        self.t_first: float | None = None
        self.t_last: float | None = None

    def estimate(self, face_data, norm_data=None) -> GazeFeatures:
        now = time.perf_counter()
        self.t_first = self.t_first or now
        self.t_last = now
        idx = len(self.frames)
        valid = (face_data.detected and face_data.landmarks_px is not None and norm_data is not None
                 and norm_data.is_valid and norm_data.face_patch is not None)
        self.frames.append((norm_data.face_patch.copy(), np.array(face_data.landmarks_px, dtype=np.float64),
                            np.array(norm_data.rotation_matrix, dtype=np.float64),
                            np.array(norm_data.origin_3d_mm, dtype=np.float64)) if valid else None)
        return GazeFeatures(raw_features=np.array([float(idx)]), confidence=1.0 if valid else 0.0)

    def frames_per_s(self) -> float:
        if not self.t_first or not self.t_last or self.t_last <= self.t_first:
            return float("nan")
        return (len(self.frames) - 1) / (self.t_last - self.t_first)


def run_networks(frames: list, nets: list, progress=None) -> tuple[np.ndarray, np.ndarray]:
    """Passe la séquence enregistrée dans chaque réseau, dans l'ordre (un filtre temporel chacun).

    Renvoie (features (n images, FEATS_PER_MODEL x n réseaux), temps de calcul médian en ms).
    Une image sans visage ne passe pas dans le réseau, comme dans le tracker (filtre inchangé).
    """
    from face import FaceData
    from normalize import NormalizationData

    out = np.full((len(frames), FEATS_PER_MODEL * len(nets)), np.nan)
    infer_ms = np.full(len(nets), np.nan)
    for m, net in enumerate(nets):
        times = []
        for i, rec in enumerate(frames):
            if progress is not None and i % 50 == 0:
                progress(m, i)
            if rec is None:
                continue
            patch, landmarks, rotation, origin = rec
            fd = FaceData(detected=True, landmarks_px=landmarks)
            nd = NormalizationData(face_patch=patch, rotation_matrix=rotation, origin_3d_mm=origin, is_valid=True)
            t0 = time.perf_counter()
            g = net.estimate(fd, norm_data=nd)
            times.append(time.perf_counter() - t0)
            if g.confidence > 0:
                out[i, m * FEATS_PER_MODEL:(m + 1) * FEATS_PER_MODEL] = np.r_[
                    np.asarray(g.raw_features, dtype=np.float64)[:4], g.pitch_yaw_deg]
        if times:
            infer_ms[m] = 1000.0 * float(np.median(times))
    return out, infer_ms


# --- Séance ---

def _message(window: str, w: int, h: int, lines: list[str], wait_keys: bool = True) -> bool:
    """Écran noir avec du texte ; vrai sur Entrée / Espace, faux sur Échap."""
    img = np.zeros((h, w, 3), np.uint8)
    top = max(60, (h - 40 * len(lines)) // 2)
    for i, line in enumerate(lines):
        size = 0.9 if i == 0 else 0.65
        color = (0, 220, 255) if i == 0 else (220, 220, 220)
        cv2.putText(img, line, (80, top + i * 40), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2 if i == 0 else 1)
    while True:
        cv2.imshow(window, img)
        key = cv2.waitKey(50) & 0xFF
        if not wait_keys:
            return True
        if key in (13, 10, 32):
            return True
        if key in (27, ord("q")):
            return False


def run_session(args: argparse.Namespace) -> Path | None:
    from capture import WebcamCapture
    from face import FaceDetector
    from normalize import FaceNormalizer
    from profile import load_profile

    paths = [USER_MODELS_DIR / f for f in MODEL_FILES if (USER_MODELS_DIR / f).is_file()]
    if len(paths) < 2:
        sys.exit(f"Il faut au moins deux modèles de regard dans {USER_MODELS_DIR} "
                 f"(trouvés : {', '.join(p.name for p in paths) or 'aucun'}).")
    profile = load_profile(args.profile)
    sw, sh = get_screen_resolution()
    screen = ScreenConfig(width_px=sw, height_px=sh, diagonal_inches=profile.screen_diag,
                          distance_cm=profile.distance_cm)
    print(f"Modèles comparés : {', '.join(p.name for p in paths)}")
    print(f"Écran {sw}x{sh}, {profile.screen_diag}\" à {profile.distance_cm} cm (profil {profile.name})")

    nets = [NetGazeEstimator(model_path=str(p)) for p in paths]  # chargés tout de suite : erreur avant la séance
    recorder = RecordingEstimator()
    # Mêmes réglages que le tracker (main.py) : 1280x720 demandé, mode mémorisé dans camera_mode.json
    capture = WebcamCapture(camera_index=args.camera, width=1280, height=720, target_fps=60)
    print(f"Caméra : {capture.actual_width}x{capture.actual_height} @ {capture.actual_fps:.0f} fps")
    face_detector = FaceDetector()
    normalizer = FaceNormalizer(target_size=448, virtual_focal=profile.patch_focal)

    def read_mirrored():
        ok, frame, ts = capture.read()
        if ok and frame is not None:
            frame = cv2.flip(frame, 1)  # même miroir que le tracker
        return ok, frame, ts

    mgr = CalibrationManager(screen_config=screen, mode="net", num_points=9)
    window = "Eyetrack - comparaison des modeles (Echap pour arreter)"
    open_calibration_window(window, sw, sh)
    ctx = {
        "capture_fn": read_mirrored, "face_detector": face_detector, "gaze_estimator": recorder,
        "normalizer": normalizer, "samples_per_point": args.samples_per_point, "settle_frames": 15,
        "iris_probe": IrisGazeEstimator(), "blink": BlinkDetector(), "t0": time.perf_counter(),
        "window": window, "w": sw, "h": sh,
    }
    collected: list[tuple[int, int, dict]] = []
    try:
        if not mgr._intro_screen(ctx):
            print("Séance annulée.")
            return None
        for pass_id, order in enumerate([list(range(9)), PASS2_ORDER]):
            if pass_id == 1 and not _message(window, sw, sh, [
                    "Serie 1 terminee.",
                    "Detendez-vous quelques secondes, sans bouger la camera ni l'ecran.",
                    "[Entree] : serie 2 (les memes 9 cibles, dans un autre ordre)    [Echap] : arreter ici"]):
                break
            for idx in order:
                res = mgr._collect_point(idx, ctx)
                if res is None:
                    break
                collected.append((pass_id, idx, res))
            else:
                continue
            break  # Échap pendant une série
        complete_passes = sum(1 for p in (0, 1) if sum(1 for c in collected if c[0] == p) == 9)
        if complete_passes == 0:
            print("Séance arrêtée avant la fin de la première série : rien n'est enregistré.")
            return None
        collected = [c for c in collected if sum(1 for d in collected if d[0] == c[0]) == 9]
    finally:
        capture.release()

    names = [n.model_name for n in nets]
    total = len(recorder.frames)

    def progress(m: int, i: int) -> None:
        _message(window, sw, sh, [f"Calcul des reseaux : {names[m]} ({m + 1}/{len(nets)})",
                                  f"Image {i}/{total}"], wait_keys=False)

    print(f"Séance : {total} images à {recorder.frames_per_s():.0f} images/s ; calcul des réseaux…")
    feats, infer_ms = run_networks(recorder.frames, nets, progress)
    recorder.frames.clear()  # images libérées : rien n'en est enregistré
    out = Path(data_path(time.strftime("compare_models_%Y%m%d_%H%M%S.npz")))
    save_session(out, names, feats, collected,
                 screen=[sw, sh, profile.screen_diag, profile.distance_cm],
                 camera=[capture.actual_width, capture.actual_height, capture.actual_fps],
                 infer_ms=infer_ms, session_fps=recorder.frames_per_s())
    report = analyse(out)
    out.with_suffix(".txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Données : {out}\nRapport : {out.with_suffix('.txt')}")
    lines = [_ascii(ln) for ln in report.splitlines() if ln.strip()][:16]
    _message(window, sw, sh, lines + ["", "[Entree] : fermer"])
    cv2.destroyAllWindows()
    return out


def save_session(out: Path, names: list[str], feats: np.ndarray, collected: list, screen, camera,
                 infer_ms: np.ndarray, session_fps: float) -> None:
    """Enregistre les mesures de la séance (aucune image) ; G = sorties des réseaux par échantillon."""
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [len(res["G"]) for _, _, res in collected]
    np.savez(
        out,
        models=np.array(names),
        # La colonne 0 de G (collecte) est le numéro de l'image : on y lit la sortie de chaque réseau
        G=np.vstack([feats[res["G"][:, 0].astype(int)] for _, _, res in collected]),
        H=np.vstack([res["H"] for _, _, res in collected]),
        Y=np.vstack([np.tile(res["target"], (n, 1)) for (_, _, res), n in zip(collected, rows)]),
        point=np.concatenate([np.full(n, idx) for (_, idx, _), n in zip(collected, rows)]),
        pass_id=np.concatenate([np.full(n, p) for (p, _, _), n in zip(collected, rows)]),
        extra=np.vstack([res["extra"] for _, _, res in collected]),
        extra_cols=np.array(EXTRA_COLS),
        glare=np.array([res["glare"] for _, _, res in collected]),
        screen=np.array(screen, dtype=np.float64),
        camera=np.array(camera, dtype=np.float64),
        infer_ms=np.asarray(infer_ms, dtype=np.float64),
        session_fps=np.array(session_fps),
        date=np.array(time.strftime("%Y-%m-%d %H:%M")),
    )


def _ascii(text: str) -> str:
    """Texte affichable par OpenCV (pas d'accents ni de « ° »)."""
    import unicodedata
    text = text.replace("°", " deg").replace("…", "...")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


# --- Analyse (hors ligne) ---

def _point_sets(G: np.ndarray, H: np.ndarray, Y: np.ndarray, point: np.ndarray, pass_id: np.ndarray,
                which: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """Échantillons d'une série, regroupés par cible (dans l'ordre de la grille)."""
    sets = []
    for idx in range(9):
        m = (pass_id == which) & (point == idx) & np.isfinite(G).all(axis=1)
        if m.sum() >= 5:
            sets.append((G[m], H[m], Y[m][0], idx))
    return sets


def _fit(sets) -> AffineGazeModel:
    model = AffineGazeModel(mode="net", alpha=RIDGE_ALPHA["net"])
    model.fit(np.vstack([s[0] for s in sets]), np.vstack([s[1] for s in sets]),
              np.vstack([np.tile(s[2], (len(s[0]), 1)) for s in sets]))
    return model


def _predict_frames(model: AffineGazeModel, G: np.ndarray, H: np.ndarray) -> np.ndarray:
    return model._design(G, H) @ model.weights


def cross_pass(sets_train, sets_test, screen: ScreenConfig) -> tuple[list[float], list[float]]:
    """Calibré sur une série, testé sur l'autre : (erreur par cible, tremblement par cible) en degrés.

    L'erreur prend la moyenne robuste des images de la cible (comme la note de l'application) ;
    le tremblement est l'écart quadratique moyen des images autour de leur médiane.
    """
    model = _fit(sets_train)
    errs, jitter = [], []
    for G_t, H_t, target, _ in sets_test:
        raw = model._trim_cluster(model._raw_features(G_t, H_t))
        z = (raw - model.feat_mean) / model.feat_std
        mean_pred = (np.hstack([z, np.ones((len(z), 1))]) @ model.weights).mean(axis=0)
        errs.append(screen.visual_angle_deg((float(target[0]), float(target[1])),
                                            (float(mean_pred[0]), float(mean_pred[1]))))
        frames = _predict_frames(model, G_t, H_t)
        center = np.median(frames, axis=0)
        dev = [screen.visual_angle_deg((float(center[0]), float(center[1])), (float(p[0]), float(p[1])))
               for p in frames]
        jitter.append(float(np.sqrt(np.mean(np.square(dev)))))
    return errs, jitter


def _bootstrap_ci(diff: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _deg(x: float) -> str:
    return "  -  " if not np.isfinite(x) else f"{x:.1f}°".replace(".", ",")


def analyse(path: Path | str) -> str:
    d = np.load(path, allow_pickle=False)
    names = [str(n) for n in d["models"]]
    sw, sh, diag, dist = (float(v) for v in d["screen"])
    screen = ScreenConfig(width_px=int(sw), height_px=int(sh), diagonal_inches=diag, distance_cm=dist)
    H, Y, point, pass_id = d["H"], d["Y"], d["point"], d["pass_id"]
    passes = sorted(set(int(p) for p in pass_id))
    results = {}
    for m, name in enumerate(names):
        G = d["G"][:, m * FEATS_PER_MODEL: m * FEATS_PER_MODEL + 4]
        sets = {p: _point_sets(G, H, Y, point, pass_id, p) for p in passes}
        loo = {p: evaluate_points("net", [(s[0], s[1], s[2]) for s in sets[p]], screen)[2]
               for p in passes if len(sets[p]) >= 6}
        cross_errs, cross_jit = [], []
        if len(passes) == 2 and all(len(sets[p]) >= 6 for p in passes):
            for a, b in ((0, 1), (1, 0)):
                e, j = cross_pass(sets[a], sets[b], screen)
                cross_errs += e
                cross_jit += j
        else:  # une seule série : tremblement mesuré avec le modèle de la série elle-même
            p = passes[0]
            _, cross_jit = cross_pass(sets[p], sets[p], screen)
        results[name] = {"loo": loo, "cross": np.array(cross_errs), "jitter": float(np.median(cross_jit)),
                         "ms": float(d["infer_ms"][m])}

    cam = d["camera"]
    lines = [
        f"Comparaison des modèles de regard - séance du {d['date']}",
        f"{len(passes)} série(s) de 9 cibles, {len(Y)} images ; écran {int(sw)}x{int(sh)} {diag:g}\" à {dist:g} cm ; "
        f"caméra {int(cam[0])}x{int(cam[1])}, {float(d['session_fps']):.0f} images/s pendant la séance",
        "",
        f"{'Modèle':<26}{'Note app (séries)':<20}{'Série croisée':<15}{'Tremblement':<13}{'Calcul':<8}",
    ]
    for name, r in results.items():
        loo = " / ".join(_deg(r["loo"][p]) for p in passes if p in r["loo"])
        cross = _deg(float(r["cross"].mean())) if len(r["cross"]) else "  -  "
        lines.append(f"{name:<26}{loo:<20}{cross:<15}{_deg(r['jitter']):<13}{r['ms']:.0f} ms")
    lines += ["",
              "Note app : erreur de chaque cible prédite sans elle (ce qu'affiche Eyetrack après la calibration).",
              "Série croisée : calibré sur une série, testé sur l'autre. C'est le chiffre qui compte.",
              "Tremblement : agitation du point de regard sur une cible fixe, avant le filtre (plus bas = plus calme)."]
    ref = names[0]
    if len(results[ref]["cross"]):
        lines += ["", f"Écarts par cible (série croisée), par rapport à {ref} (négatif = meilleur que lui) :"]
        for name in names[1:]:
            diff = results[name]["cross"] - results[ref]["cross"]
            lo, hi = _bootstrap_ci(diff)
            wins = int((diff < 0).sum())
            verdict = ("pas de différence nette" if lo < 0 < hi
                       else ("meilleur" if hi < 0 else "moins bon"))
            lines.append(f"  {name:<24}{_deg(float(diff.mean())):>7}  (IC 95 % {_deg(lo)} à {_deg(hi)}) ; "
                         f"meilleur sur {wins}/{len(diff)} cibles -> {verdict}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare les modèles de regard sur les mêmes images")
    parser.add_argument("--analyse", metavar="FICHIER", help="Refaire l'analyse d'une séance enregistrée (.npz)")
    parser.add_argument("--camera", type=int, default=0, help="Index de la caméra (défaut : 0)")
    parser.add_argument("--profile", default="default", help="Profil (écran, distance) : défaut « default »")
    parser.add_argument("--samples-per-point", type=int, default=45, help="Images par cible (défaut : 45, comme la calibration)")
    args = parser.parse_args()
    if args.analyse:
        print(analyse(args.analyse))
        return
    run_session(args)


if __name__ == "__main__":
    main()
