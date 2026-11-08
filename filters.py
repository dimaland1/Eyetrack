"""Filtre One Euro pour le lissage adaptatif du regard et de la pose de tête.

Référence : Casiez et al., CHI 2012 ("1 € Filter: A Simple Speed-based Low-pass Filter").
Permet un lissage fort lors des fixations (élimine les tremblements) et une
latence quasi-nulle lors des mouvements rapides / saccades.
"""

from __future__ import annotations
import math
import numpy as np


class OneEuroFilter1D:
    """Filtre One Euro pour un signal scalaire 1D."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        """Initialise le filtre 1D.

        Args:
            min_cutoff: Fréquence de coupure minimale en Hz (lissage à basse vitesse).
            beta: Coefficient d'adaptation à la vitesse (plus il est grand, plus la latence baisse).
            d_cutoff: Fréquence de coupure pour la dérivée en Hz.
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        """Calcule le coefficient de lissage exponentiel."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, timestamp: float) -> float:
        """Filtre la valeur x à l'instant timestamp (en secondes)."""
        if self._t_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = timestamp
            return x

        dt = timestamp - self._t_prev
        if dt <= 1e-6:
            # Timestamp identique ou trop proche, on renvoie l'état précédent
            return self._x_prev if self._x_prev is not None else x

        # Estimation de la dérivée filtrée
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Fréquence de coupure adaptative
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)

        # Filtrage du signal principal
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = timestamp

        return x_hat

    def reset(self) -> None:
        """Réinitialise l'état interne du filtre."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class OneEuroFilter:
    """Filtre One Euro multi-dimensionnel pour vecteurs numpy ou listes de scalaires."""

    def __init__(
        self,
        dim: int,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        median_window: int = 1,
    ) -> None:
        """Initialise le filtre multi-dimensionnel.

        Args:
            dim: Dimension du vecteur à filtrer (ex: 2 pour le regard, 6 pour la pose).
            min_cutoff: Fréquence de coupure minimale en Hz.
            beta: Coefficient d'adaptation à la vitesse.
            d_cutoff: Fréquence de coupure pour la dérivée en Hz.
            median_window: Taille du pré-filtre médian glissant (1 = désactivé). Retire les
                pics isolés du réseau avant le One Euro : sans lui, chaque pic fait croire
                à un mouvement rapide et le filtre s'ouvre (jitter).
        """
        self.dim = dim
        self.median_window = max(1, int(median_window))
        self._history: list[np.ndarray] = []
        self.filters = [
            OneEuroFilter1D(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
            for _ in range(dim)
        ]

    def filter(self, values: np.ndarray | list[float] | tuple[float, ...], timestamp: float) -> np.ndarray:
        """Filtre un vecteur à l'instant timestamp (en secondes)."""
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) != self.dim:
            raise ValueError(f"Dimension incorrecte: attendue {self.dim}, reçue {len(arr)}")

        if self.median_window > 1:
            self._history.append(arr.copy())
            if len(self._history) > self.median_window:
                self._history.pop(0)
            arr = np.median(np.array(self._history), axis=0)

        out = np.zeros(self.dim, dtype=np.float64)
        for i in range(self.dim):
            out[i] = self.filters[i].filter(float(arr[i]), timestamp)
        return out

    def set_params(self, min_cutoff: float | None = None, beta: float | None = None, d_cutoff: float | None = None,
                   median_window: int | None = None) -> None:
        """Met à jour les paramètres de tous les filtres 1D (sans perdre leur état)."""
        if median_window is not None:
            self.median_window = max(1, int(median_window))
        for f in self.filters:
            if min_cutoff is not None:
                f.min_cutoff = float(min_cutoff)
            if beta is not None:
                f.beta = float(beta)
            if d_cutoff is not None:
                f.d_cutoff = float(d_cutoff)

    def reset(self) -> None:
        """Réinitialise tous les filtres 1D."""
        self._history.clear()
        for f in self.filters:
            f.reset()


class LossBlender:
    """Gère les pertes de suivi du visage sans à-coup.

    - Pendant une perte, la dernière valeur de sortie est maintenue (``held``).
    - À la réacquisition après une perte suffisamment longue, la sortie part de
      la valeur maintenue et rejoint la nouvelle mesure en ``blend_time`` secondes
      (interpolation smoothstep), au lieu de sauter d'un coup.
    """

    def __init__(self, dim: int, blend_time: float = 0.35, min_loss: float = 0.1) -> None:
        self.dim = dim
        self.blend_time = float(blend_time)
        self.min_loss = float(min_loss)
        self.held: np.ndarray | None = None
        self._lost_since: float | None = None
        self._blend_from: np.ndarray | None = None
        self._blend_start: float = 0.0

    def mark_lost(self, timestamp: float) -> None:
        """Signale que le visage n'est pas détecté sur cette frame."""
        if self._lost_since is None:
            self._lost_since = timestamp

    def lost_duration(self, timestamp: float) -> float:
        """Durée (s) de la perte en cours, 0 si le visage est suivi."""
        return 0.0 if self._lost_since is None else max(0.0, timestamp - self._lost_since)

    def update(self, value: np.ndarray | list[float] | tuple[float, ...], timestamp: float) -> np.ndarray:
        """Reçoit une nouvelle mesure valide et renvoie la sortie lissée à la réacquisition."""
        arr = np.asarray(value, dtype=np.float64)
        if self._lost_since is not None:
            if self.held is not None and self.lost_duration(timestamp) >= self.min_loss:
                self._blend_from = self.held.copy()
                self._blend_start = timestamp
            self._lost_since = None

        out = arr
        if self._blend_from is not None:
            b = (timestamp - self._blend_start) / max(self.blend_time, 1e-3)
            if b >= 1.0:
                self._blend_from = None
            else:
                s = b * b * (3.0 - 2.0 * b)
                out = self._blend_from + (arr - self._blend_from) * s

        self.held = out.copy()
        return out


class BlinkDetector:
    """Détecte les clignements à partir de l'ouverture des yeux (FaceData.eye_openness).

    Pendant un clignement, les points des paupières et de l'iris s'écrasent : le regard
    estimé saute (constaté : ~6 % des échantillons d'une calibration, par salves de ~0,3 s).
    On compare l'ouverture à une référence « œil ouvert » qui s'adapte lentement (lunettes,
    distance, fatigue) ; sous ``close_ratio`` × référence, le regard est déclaré non fiable,
    ainsi que pendant ``hold_after_s`` après la réouverture (le temps que l'œil se stabilise).
    Le seuil reste bas (0,55) : regarder le bas de l'écran ferme un peu les paupières sans
    être un clignement.
    """

    def __init__(self, close_ratio: float = 0.55, hold_after_s: float = 0.15, adapt: float = 0.02) -> None:
        self.close_ratio = float(close_ratio)
        self.hold_after_s = float(hold_after_s)
        self.adapt = float(adapt)
        self.baseline: float | None = None
        self._closed = False
        self._reopen_t: float | None = None
        self.blink_count = 0

    def update(self, openness: float, timestamp: float) -> bool:
        """Renvoie True si le regard n'est pas fiable sur cette image (clignement en cours)."""
        if openness <= 0.0:
            return True
        if self.baseline is None:
            self.baseline = openness
            return False
        # Référence « œil ouvert » : monte vite, descend lentement, ignore les yeux mi-clos
        if openness > self.baseline:
            self.baseline += 0.1 * (openness - self.baseline)
        elif openness > 0.8 * self.baseline:
            self.baseline += self.adapt * (openness - self.baseline)

        if openness < self.close_ratio * self.baseline:
            if not self._closed:
                self.blink_count += 1
            self._closed = True
            self._reopen_t = None
            return True
        if self._closed:
            self._closed = False
            self._reopen_t = timestamp
        if self._reopen_t is not None and timestamp - self._reopen_t < self.hold_after_s:
            return True
        return False
