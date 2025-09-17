# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from math import radians, sin, cos, atan, isfinite
from typing import Iterable, Dict, Any, Optional, Tuple, List
from collections import OrderedDict

import numpy as np
from scipy.signal import savgol_filter, medfilt, butter, filtfilt


# -----------------
# Utils internes
# -----------------
def _window_pts_from_m(s_m: np.ndarray, target_m: float, poly: int, min_size: int = 3) -> int:
    """Convertit une fenêtre métrique en nombre de points impairs et bornés.
    s_m : abscisses cumulées (m)
    target_m : taille désirée (m)
    poly : ordre du polynôme SG
    """
    n = s_m.size
    if n < min_size:
        return n
    if target_m <= 0 or not np.isfinite(target_m):
        # fallback: fenêtre minimale valide
        k = max(min_size, (poly + 2) | 1)  # ensure > poly and odd
        return min(k, n if n % 2 == 1 else n - 1)
    # estimer pas médian
    steps = np.diff(s_m)
    d = float(np.median(steps[steps > 0]) if steps.size else target_m)
    k = int(max(min_size, round(target_m / max(d, 1e-6))))
    # impaire et > poly
    if k % 2 == 0:
        k += 1
    if k <= poly:
        k = poly + 2 if (poly + 2) % 2 == 1 else poly + 3
    # borne à N
    k = min(k, n if n % 2 == 1 else n - 1)
    return max(min_size, k)


# ---------------------------------------------------------------------------
# Config pipeline altitude
# ---------------------------------------------------------------------------

@dataclass
class AltitudePipelineConfig:
    """
    Paramètres pour le calcul de profil/pente.

    Attributes
    ----------
    mode : str
        "track_first" -> on calcule d'abord la distance curviligne le long
        de la trace puis on dérive l'altitude par rapport à cette abscisse.
    sg_window_m : float
        Taille cible (en mètres) de la fenêtre Savitzky–Golay pour la pente.
    sg_poly : int
        Ordre du polynôme Savitzky–Golay.
    """
    mode: str = "track_first"
    sg_window_m: float = 5.0
    sg_poly: int = 3
    pre_median_3: bool = False

    # Options post-traitement pente
    slope_clip_pct: float | None = None  # ex: 15.0 => clip à ±15%
    slope_bw_cutoff_m: float | None = None  # longueur d'onde coupure (m)
    slope_bw_order: int = 2

    # Cache LRU (désactivé si 0)
    cache_lru_points: int = 0  # nombre max de clés (points) mémorisées
    cache_round_decimals: int = 5  # arrondi lat/lon pour la clé


# ---------------------------------------------------------------------------
# Service Altitude
# ---------------------------------------------------------------------------

class AltitudeService:
    """
    Service d'altitude : interroge un provider pour l'altitude aux points donnés,
    et calcule profil + pente le long d'une trace.

    Le service est "duck-typed" vis-à-vis du provider : on tente successivement
    `altitude_at(lats, lons)`, `elevation_at(lats, lons)`, puis `sample(lat, lon)`
    (en vectorisant si besoin).
    """

    def __init__(self, provider: Any, cfg: Optional[AltitudePipelineConfig] = None):
        self.provider = provider
        self.cfg = cfg or AltitudePipelineConfig()
        # Optionnel : ressource interne à fermer dans close()
        self._vrt_l93 = None  # compat leftover si utilisé ailleurs
        # Cache LRU optionnel
        self._cache = OrderedDict() if getattr(self.cfg, "cache_lru_points", 0) else None

    # -----------------------------
    # Appel provider, tolérant API
    # -----------------------------
    def _provider_altitude_at(self, lats: Iterable[float], lons: Iterable[float]) -> np.ndarray:
        lats = np.atleast_1d(np.asarray(lats, dtype=float))
        lons = np.atleast_1d(np.asarray(lons, dtype=float))

        if self._cache is None:
            # chemin simple, sans cache
            fn = getattr(self.provider, "altitude_at", None) or getattr(self.provider, "elevation_at", None)
            if callable(fn):
                return np.asarray(fn(lats, lons), dtype=np.float32)
            fn = getattr(self.provider, "sample", None)
            if callable(fn):
                return np.asarray(fn(lats, lons), dtype=np.float32)
            raise AttributeError("Le provider ne propose ni `altitude_at`, ni `elevation_at`, ni `sample`.")

        # Chemin cache LRU
        out = np.full(lats.shape, np.nan, dtype=np.float32)
        misses_idx: list[int] = []
        keys: list[tuple[float, float]] = []
        rnd = int(getattr(self.cfg, "cache_round_decimals", 5))

        for i, (la, lo) in enumerate(zip(lats, lons)):
            k = (round(float(la), rnd), round(float(lo), rnd))
            keys.append(k)
            if k in self._cache:
                out[i] = self._cache[k]
            else:
                misses_idx.append(i)

        if misses_idx:
            mi = np.asarray(misses_idx, dtype=int)
            fetched = None
            fn = getattr(self.provider, "altitude_at", None) or getattr(self.provider, "elevation_at", None)
            if callable(fn):
                fetched = np.asarray(fn(lats[mi], lons[mi]), dtype=np.float32)
            else:
                fn = getattr(self.provider, "sample", None)
                if callable(fn):
                    fetched = np.asarray(fn(lats[mi], lons[mi]), dtype=np.float32)
            if fetched is None:
                raise AttributeError("Le provider ne propose ni `altitude_at`, ni `elevation_at`, ni `sample`.")
            out[mi] = fetched

            cap = int(getattr(self.cfg, "cache_lru_points", 0))
            for i2, val in zip(mi, fetched):
                k = keys[i2]
                self._cache[k] = float(val)
                self._cache.move_to_end(k)
                if cap and len(self._cache) > cap:
                    self._cache.popitem(last=False)

        return out

    # -----------------------------
    # API publique : altitude brute
    # -----------------------------
    def altitude_at(self, lats: Iterable[float], lons: Iterable[float]) -> np.ndarray:
        """
        Retourne les altitudes pour les tableaux `lats` et `lons`.
        """
        lats = np.atleast_1d(np.asarray(lats, dtype=float))
        lons = np.atleast_1d(np.asarray(lons, dtype=float))
        z = self._provider_altitude_at(lats, lons)
        return z

    # -----------------------------
    # Distances géodésiques (m)
    # -----------------------------
    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Distance haversine (m). Suffisant pour l’échantillonnage routier ici.
        """
        R = 6_371_000.0  # rayon terrestre moyen (m)
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2.0) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.sqrt(a))
        return float(R * c)

    def _curvilinear_abscissa(self, lats: Iterable[float], lons: Iterable[float]) -> np.ndarray:
        """
        Abscisse curviligne cumulée (m) le long de la polyline.
        """
        lats = np.atleast_1d(np.asarray(lats, dtype=float))
        lons = np.atleast_1d(np.asarray(lons, dtype=float))
        n = len(lats)
        s = np.zeros(n, dtype=float)
        if n <= 1:
            return s
        for i in range(1, n):
            s[i] = s[i - 1] + self._haversine_m(lats[i - 1], lons[i - 1], lats[i], lons[i])
        return s

    # -----------------------------
    # API publique : profil + pente
    # -----------------------------
    def profile_and_grade(
        self, lats: Iterable[float], lons: Iterable[float]
    ) -> Dict[str, Any]:
        """
        Calcule :
          - z : altitude aux points,
          - grade : dérivée dz/ds (m/m),
          - theta : arctan(grade) (rad),
          - meta : infos auxiliaires.

        En mode `track_first`, on fait un Savitzky–Golay **borné** (fenêtre impaire,
        > polyorder et <= N). Si la trace est trop courte, on bascule en gradient simple.
        """
        lats = np.atleast_1d(np.asarray(lats, dtype=float))
        lons = np.atleast_1d(np.asarray(lons, dtype=float))

        # Altitudes
        z = self.altitude_at(lats, lons).astype(np.float32)

        # Abscisse curviligne (m)
        s = self._curvilinear_abscissa(lats, lons).astype(float)

        # Pré-lissage optionnel de l'altitude pour réduire les outliers ponctuels
        # Active si cfg.pre_median_3 == True ; médiane glissante de taille 3.
        if getattr(self.cfg, "pre_median_3", False) and len(z) >= 3:
            z_smooth = medfilt(z, kernel_size=3).astype(np.float32)
        else:
            z_smooth = z

        mode_used = "track_first"  # (autres modes à venir)

        # ----- Calcul de la pente
        if mode_used == "track_first":
            # Pas moyen : median(diff(s)), clampé à >= 1m pour stabilité
            step = float(np.maximum(np.median(np.diff(s)) if len(s) > 1 else 1.0, 1.0))
            n = int(len(z_smooth))

            # Fallback immédiat si trop court
            if n < 3 or self.cfg.sg_poly >= n:
                grade = np.gradient(z_smooth, s, edge_order=1)
            else:
                # Fenêtre cible à partir d'une taille métrique
                win_pts = _window_pts_from_m(s, self.cfg.sg_window_m, self.cfg.sg_poly)

                # fenêtre minimale > polyorder et impaire
                min_win = int(self.cfg.sg_poly + 2)
                if (min_win % 2) == 0:
                    min_win += 1
                if win_pts < min_win:
                    win_pts = min_win

                # clamp à la longueur du signal (impair)
                if win_pts > n:
                    win_pts = n if (n % 2) == 1 else n - 1

                # encore trop court ? gradient
                if win_pts <= self.cfg.sg_poly or win_pts < 3:
                    grade = np.gradient(z_smooth, s, edge_order=1)
                else:
                    grade = savgol_filter(
                        z_smooth,
                        window_length=win_pts,
                        polyorder=self.cfg.sg_poly,
                        deriv=1,
                        delta=step,
                    )
        else:
            # Par défaut : gradient simple si d’autres modes arrivent ici
            grade = np.gradient(z, s, edge_order=1)

        # --- Post-traitements sur la pente
        # Clip pente en pourcentage si demandé
        clip_pct = getattr(self.cfg, "slope_clip_pct", None)
        if clip_pct is not None and np.isfinite(clip_pct) and clip_pct > 0:
            # grade est en m/m ; convertissons le seuil en m/m
            thr = float(clip_pct) / 100.0
            grade = np.clip(grade, -thr, thr)

        # Filtre passe-bas Butterworth optionnel (cutoff en longueur d'onde, m)
        cutoff_m = getattr(self.cfg, "slope_bw_cutoff_m", None)
        order = int(getattr(self.cfg, "slope_bw_order", 2))
        if cutoff_m is not None and np.isfinite(cutoff_m) and cutoff_m > 0 and order >= 1 and len(grade) > (order * 3 + 1):
            # Convertir longueur d'onde (m) -> fréquence normalisée (0..1)
            step_m = float(np.maximum(np.median(np.diff(s)) if len(s) > 1 else 1.0, 1.0))
            wn = 2.0 * step_m / float(cutoff_m)  # cycles/sample normalisé par Nyquist
            wn = float(np.clip(wn, 1e-6, 0.999999))
            b, a = butter(order, wn, btype="lowpass")
            try:
                grade = filtfilt(b, a, grade)
            except Exception:
                # fallback robuste
                pass

        # Angles (rad)
        theta = np.array([atan(g) if isfinite(g) else np.nan for g in grade], dtype=float)

        meta = {
            "mode": mode_used,
            "sg_window_m": float(self.cfg.sg_window_m),
            "sg_poly": int(self.cfg.sg_poly),
            "pre_median_3": bool(getattr(self.cfg, "pre_median_3", False)),
            "slope_clip_pct": clip_pct if clip_pct is not None else None,
            "slope_bw_cutoff_m": float(cutoff_m) if cutoff_m is not None else None,
            "slope_bw_order": int(order),
        }

        return {
            "z": z,
            "grade": np.asarray(grade, dtype=float),
            "theta": theta,
            "meta": meta,
        }

    # -----------------------------
    # Nettoyage
    # -----------------------------
    def close(self):
        """Libère les ressources internes (ex. VRT) et celles du provider."""
        try:
            vrt = getattr(self, "_vrt_l93", None)
            if vrt is not None:
                close = getattr(vrt, "close", None)
                if callable(close):
                    close()
                self._vrt_l93 = None
        except Exception:
            pass

        try:
            close = getattr(self.provider, "close", None)
            if callable(close):
                close()
        except Exception:
            pass