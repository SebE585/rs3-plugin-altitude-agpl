# src/rs3_plugin_altitude_agpl/plugin.py
# -*- coding: utf-8 -*-
"""
AltitudeStage — point d'entrée plugin pour RoadSimulator3.

Entry point (pyproject.toml):
[project.entry-points."rs3.plugins"]
altitude = "rs3_plugin_altitude_agpl.plugin:AltitudeStage"
"""
from __future__ import annotations

import os
import inspect
from typing import Optional, Dict, Any

import numpy as np
from scipy.signal import medfilt, savgol_filter, butter, filtfilt

ALT_DEBUG = bool(os.environ.get("RS3_ALT_DEBUG"))

from .factory import build_service_from_yaml, build_service_from_dict


class AltitudeStage:
    """
    Stage RS3: ajoute au DataFrame les colonnes minimales pour le rapport:
      - altitude_m : altitude (m)
      - slope_percent : pente (%)

    Entrée: DataFrame avec colonnes 'lat','lon' (WGS84).
    """

    # RS3_ALT_Z_SMOOTH supports: none|median|savgol|median+sg
    def _load_runtime_options(self) -> None:
        """
        Lit les options runtime (lissage altitude/pente, resampling, clipping) depuis les variables d'environnement.
        Appelée depuis __init__ pour tous les cas (config dict ou chemin YAML).
        """
        # Options de post-traitement pente (configurables via env)
        # - RS3_ALT_SLOPE_MEDFILT : taille (impair) du filtre médian (points)
        # - RS3_ALT_SLOPE_CLIP_PCT : clipping en % (absolu)
        try:
            self._slope_medfilt = int(os.environ.get("RS3_ALT_SLOPE_MEDFILT", "3"))
        except Exception:
            self._slope_medfilt = 3
        if self._slope_medfilt % 2 == 0:
            self._slope_medfilt += 1  # impose impair
        try:
            self._slope_clip = float(os.environ.get("RS3_ALT_SLOPE_CLIP_PCT", "12.0"))
        except Exception:
            self._slope_clip = 12.0

        # Altitude pre-smoothing options
        self._z_smooth = str(os.environ.get("RS3_ALT_Z_SMOOTH", "median+sg")).strip().lower()  # none|median|savgol|median+sg
        # Median kernel (points, odd) when RS3_ALT_Z_SMOOTH=median
        try:
            self._z_medfilt = int(os.environ.get("RS3_ALT_Z_MEDFILT", "3"))
        except Exception:
            self._z_medfilt = 3
        if self._z_medfilt % 2 == 0:
            self._z_medfilt += 1
        # Savitzky–Golay params (meters + poly) when RS3_ALT_Z_SMOOTH=savgol
        try:
            self._z_sg_win_m = float(os.environ.get("RS3_ALT_Z_SG_WIN_M", "35"))
        except Exception:
            self._z_sg_win_m = 35.0
        try:
            self._z_sg_poly = int(os.environ.get("RS3_ALT_Z_SG_POLY", "2"))
        except Exception:
            self._z_sg_poly = 2

        # Derivative (slope) computation via SG (meters + poly)
        try:
            self._deriv_sg_win_m = float(os.environ.get("RS3_ALT_SG_DERIV_WIN_M", "120"))
        except Exception:
            self._deriv_sg_win_m = 120.0
        try:
            self._deriv_sg_poly = int(os.environ.get("RS3_ALT_SG_DERIV_POLY", "1"))
        except Exception:
            self._deriv_sg_poly = 1

        # Post-smoothing of slope (grade) with Savitzky–Golay (meters + poly)
        try:
            self._slope_sg_win_m = float(os.environ.get("RS3_ALT_SLOPE_SG_WIN_M", "0"))
        except Exception:
            self._slope_sg_win_m = 0.0
        try:
            self._slope_sg_poly = int(os.environ.get("RS3_ALT_SLOPE_SG_POLY", "2"))
        except Exception:
            self._slope_sg_poly = 2

        # Optional additional low-pass (Butterworth) on slope, expressed as cutoff wavelength in meters.
        # If cutoff_m > 0, we apply filtfilt with order (default 2).
        try:
            self._slope_bw_cut_m = float(os.environ.get("RS3_ALT_SLOPE_BW_CUTOFF_M", "0"))
        except Exception:
            self._slope_bw_cut_m = 0.0
        try:
            self._slope_bw_order = int(os.environ.get("RS3_ALT_SLOPE_BW_ORDER", "2"))
        except Exception:
            self._slope_bw_order = 2
        if self._slope_bw_order < 1:
            self._slope_bw_order = 1

        # Slope computation method: default SG derivative, optional OLS
        self._slope_method = str(os.environ.get("RS3_ALT_SLOPE_METHOD", "ols")).strip().lower()  # ""|"ols"
        try:
            self._slope_ols_win_m = float(os.environ.get("RS3_ALT_SLOPE_OLS_WIN_M", "120"))
        except Exception:
            self._slope_ols_win_m = 120.0

        # Kalman (pente quasi-constante) options
        self._slope_kf_enabled = False  # becomes True if method=="kalman"
        try:
            self._kf_q_z = float(os.environ.get("RS3_ALT_KF_QZ", "1e-3"))      # process noise on z
        except Exception:
            self._kf_q_z = 1e-3
        try:
            self._kf_q_slope = float(os.environ.get("RS3_ALT_KF_QSLOPE", "1e-5"))  # process noise on slope
        except Exception:
            self._kf_q_slope = 1e-5
        try:
            self._kf_r = float(os.environ.get("RS3_ALT_KF_R", "1e-1"))         # measurement noise on z
        except Exception:
            self._kf_r = 1e-1

        # Uniform resampling step (meters) for z(s) before smoothing/derivative
        # Helps stabilize SG & gradient when sampling is irregular.
        try:
            self._resample_m = float(os.environ.get("RS3_ALT_RESAMPLE_M", "3.0"))
        except Exception:
            self._resample_m = 3.0
        if not np.isfinite(self._resample_m) or self._resample_m <= 0:
            self._resample_m = 0.0  # disables resampling

    name = "altitude"

    @staticmethod
    def _ols_slopes_on_uniform_grid(s_u: np.ndarray, z_u: np.ndarray, win_m: float, du: float) -> np.ndarray:
        """Compute slope (dz/ds) on a *uniform* curvilinear grid using centered OLS windows.
        Returns an array same size as s_u. du must be > 0.
        """
        n = int(len(s_u))
        if n < 3 or not np.isfinite(du) or du <= 0:
            return np.gradient(z_u, s_u, edge_order=1)
        # window size in points (odd)
        w = int(max(3, round(float(win_m) / float(du))))
        if (w % 2) == 0:
            w += 1
        half = w // 2
        slopes = np.full(n, np.nan, dtype=float)
        for i in range(n):
            i0 = max(0, i - half)
            i1 = min(n, i + half + 1)
            x = s_u[i0:i1]
            y = z_u[i0:i1]
            if len(x) < 3:
                continue
            x0 = x - x.mean()
            denom = float(np.dot(x0, x0))
            if denom <= 0:
                continue
            a = float(np.dot(x0, y - y.mean()) / denom)
            slopes[i] = a  # dz/ds
        # fill NaNs by simple gradient fallback
        if np.isnan(slopes).any():
            grad = np.gradient(z_u, s_u, edge_order=1)
            idx = np.isnan(slopes)
            slopes[idx] = grad[idx]
        return slopes.astype(np.float32)

    @staticmethod
    def _kalman_slope_quasi_const(z_u: np.ndarray, du: float, q_z: float, q_slope: float, r_meas: float) -> np.ndarray:
        """Kalman 1D le long de s avec état x=[z, slope]^T.
        Discretisation: z_{k+1} = z_k + slope_k * du + w_z ; slope_{k+1} = slope_k + w_s
        Mesure: y_k = z_k + v. Retourne slope (dz/ds) sur la même grille que z_u.
        """
        z_u = np.asarray(z_u, dtype=float)
        n = int(z_u.size)
        if n == 0 or not np.isfinite(du) or du <= 0:
            return np.zeros_like(z_u, dtype=np.float32)

        # Matrices du modèle
        A = np.array([[1.0, float(du)],
                      [0.0, 1.0]], dtype=float)
        H = np.array([[1.0, 0.0]], dtype=float)  # observe z only
        Q = np.array([[float(q_z), 0.0],
                      [0.0, float(q_slope)]], dtype=float)
        R = np.array([[float(r_meas)]], dtype=float)

        # Initialisation
        x = np.array([float(z_u[0]), 0.0], dtype=float)
        P = np.eye(2, dtype=float)
        slopes = np.empty(n, dtype=float)

        I = np.eye(2, dtype=float)
        for k in range(n):
            # Predict
            x = A @ x
            P = A @ P @ A.T + Q
            # Update avec mesure z_u[k]
            y = np.array([float(z_u[k])])
            S = H @ P @ H.T + R  # scalaire 1x1
            K = (P @ H.T) @ np.linalg.inv(S)  # 2x1
            x = x + (K @ (y - (H @ x))).reshape(-1)
            P = (I - K @ H) @ P
            slopes[k] = x[1]

        return slopes.astype(np.float32)

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        from math import radians, sin, cos
        R = 6_371_000.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2.0) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.sqrt(a))
        return float(R * c)

    @classmethod
    def _curvilinear_abscissa(cls, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        n = len(lats)
        s = np.zeros(n, dtype=float)
        if n <= 1:
            return s
        for i in range(1, n):
            s[i] = s[i - 1] + cls._haversine_m(lats[i - 1], lons[i - 1], lats[i], lons[i])
        return s

    def __init__(self, config_path: Optional[str] = None, **kwargs: Dict[str, Any]) -> None:
        """
        Le stage accepte la configuration sous trois formes :
          - config (dict): un mapping Python (même structure que le YAML)
          - config (str): chemin d'un fichier YAML
          - config_path (str) : alias historique (chemin YAML)
          - None : on prend RS3_ALTITUDE_CFG ou 'service/config/altitude_srtm.yaml'
        """
        self._load_runtime_options()

        cfg_in = kwargs.get("config", None)

        # 0) Si un service HTTP est explicitement fourni, on le privilégie
        base_url = kwargs.get("base_url") or os.environ.get("RS3_ALT_BASE_URL")
        if isinstance(base_url, str) and base_url.strip():
            cfg_http = {
                "altitude": {
                    "source": os.environ.get("RS3_ALT_SOURCE", "srtm"),
                    "service": {"base_url": base_url.strip()},
                }
            }
            self._service = build_service_from_dict(cfg_http)
            if ALT_DEBUG:
                print(f"[AltitudeStage] init from base_url={base_url.strip()} — file={__file__}")
            return

        # 1) Config inline (dict)
        if isinstance(cfg_in, dict):
            # Les options runtime (lissage/clip/resampling) ont déjà été chargées
            # via self._load_runtime_options().
            self._service = build_service_from_dict(cfg_in)
            if ALT_DEBUG:
                print(f"[AltitudeStage] init from dict — file={__file__}")
            return

        # 2) Chemin explicite passé via "config" ou "config_path"
        path: Optional[str] = None
        if isinstance(cfg_in, str) and cfg_in.strip():
            path = cfg_in
        elif isinstance(config_path, str) and config_path.strip():
            path = config_path
        else:
            # 3) Fallback: variable d'env ou chemin par défaut du repo plugin
            path = os.environ.get("RS3_ALTITUDE_CFG", "service/config/altitude_srtm.yaml")

        try:
            self._service = build_service_from_yaml(path)
        except FileNotFoundError as e:
            # Message clair + piste de contournement via RS3_ALT_BASE_URL
            raise FileNotFoundError(
                f"AltitudeStage: fichier YAML introuvable: {path}. "
                f"Spécifie RS3_ALTITUDE_CFG ou fournis base_url (ENV RS3_ALT_BASE_URL)."
            ) from e
        if ALT_DEBUG:
            print(f"[AltitudeStage] init from yaml={path} — file={__file__}")
            print("[AltitudeStage] runtime opts:", {
                "z_smooth": self._z_smooth,
                "z_medfilt": self._z_medfilt,
                "z_sg_win_m": self._z_sg_win_m,
                "z_sg_poly": self._z_sg_poly,
                "deriv_sg_win_m": self._deriv_sg_win_m,
                "deriv_sg_poly": self._deriv_sg_poly,
                "resample_m": self._resample_m,
                "slope_medfilt": self._slope_medfilt,
                "slope_clip": self._slope_clip,
                "slope_sg_win_m": self._slope_sg_win_m,
                "slope_sg_poly": self._slope_sg_poly,
                "slope_bw_cut_m": self._slope_bw_cut_m,
                "slope_bw_order": self._slope_bw_order,
            })

    # === API RS3 : appelé par le pipeline avec un Context ===
    def run(self, ctx):
        """
        RS3 appelle les stages avec un objet Context (portant typiquement `df`).
        On lit ctx.df, on applique le calcul, puis on ré-injecte le df modifié.
        """
        df = getattr(ctx, "df", None)
        if df is None:
            return ctx  # rien à faire / compat

        new_df = self.apply(df)
        ctx.df = new_df
        return ctx

    # === API utilitaire : pour usage direct sur un DataFrame ===
    def apply(self, df):
        if ALT_DEBUG:
            try:
                print(f"[AltitudeStage] apply() from {__file__}; source lines: "+str(len(inspect.getsource(AltitudeStage.apply).splitlines())))
            except Exception:
                print(f"[AltitudeStage] apply() from {__file__}")
        # NOP si colonnes manquantes
        if "lat" not in df.columns or "lon" not in df.columns:
            return df

        lats_full = df["lat"].to_numpy(dtype=float, copy=False)
        lons_full = df["lon"].to_numpy(dtype=float, copy=False)
        n = len(lats_full)
        if n == 0:
            return df

        # 1) Dédoublonnage strict des points consécutifs (évite ds≈0 → pentes infinies)
        keep = np.ones(n, dtype=bool)
        if n > 1:
            same = (np.isclose(lats_full[1:], lats_full[:-1], rtol=0.0, atol=1e-12) &
                    np.isclose(lons_full[1:], lons_full[:-1], rtol=0.0, atol=1e-12))
            keep[1:] = ~same
        lats = lats_full[keep]
        lons = lons_full[keep]

        # Si tous les points sont identiques, retourner alt/pente constantes
        if len(lats) == 1:
            out_alt = np.full(n, np.nan, dtype=np.float32)
            out_slope = np.zeros(n, dtype=np.float32)
            # On récupère tout de même l'altitude au point unique
            out = self._service.profile_and_grade(lats, lons)
            if "z" in out and len(out["z"]) == 1:
                out_alt[:] = np.float32(out["z"][0])
            df = df.copy()
            df["altitude_m"] = out_alt
            df["slope_percent"] = out_slope
            return df

        # 2) Profil & pente sur la version dédoublonnée
        #    On récupère l'altitude via le service puis on applique un
        #    lissage optionnel de l'altitude avant de dériver. Pour limiter
        #    les effets d'un échantillonnage irrégulier, on peut
        #    ré-échantillonner z(s) sur une abscisse curviligne uniforme.
        out = self._service.profile_and_grade(lats, lons)
        alt = np.asarray(out.get("z"), dtype=np.float32)

        # Abscisse curviligne (m)
        s = self._curvilinear_abscissa(lats, lons).astype(float)
        if s.size >= 2:
            step_obs = float(np.maximum(np.median(np.diff(s)), 1.0))
        else:
            step_obs = 1.0

        # ---------- Option: ré-échantillonnage uniforme ----------
        use_uniform = (self._resample_m is not None) and (self._resample_m > 0.0) and (len(s) >= 2)
        if use_uniform:
            # grille régulière de s
            s0, s1 = float(s[0]), float(s[-1])
            if s1 <= s0:
                s_u = s.copy()
                z_u = alt.copy()
                du = step_obs
            else:
                du = float(self._resample_m)
                # éviter une grille trop fine : borne à 0.5 m mini
                du = max(0.5, du)
                s_u = np.arange(s0, s1 + 1e-9, du, dtype=float)
                # interpolation linéaire de z(s) -> z_u
                z_u = np.interp(s_u, s, alt).astype(np.float32)
        else:
            s_u = s
            z_u = alt
            du = step_obs

        # ---------- Lissage de l'altitude (sur la grille choisie) ----------
        z_u_s = z_u.copy()
        # Helper to apply median in samples
        def _apply_median(a: np.ndarray, k_pts: int) -> np.ndarray:
            k = int(max(1, k_pts))
            if (k % 2) == 0:
                k += 1
            if a.size >= 3:
                k_eff = min(k, (a.size if (a.size % 2) == 1 else a.size - 1))
                if k_eff >= 3:
                    return medfilt(a, kernel_size=k_eff).astype(np.float32)
            return a.astype(np.float32)

        # Convert window in meters to points on current grid
        def _win_pts_m_to_samples(win_m: float, du_local: float, poly: int, series_len: int) -> int:
            if not np.isfinite(du_local) or du_local <= 0:
                du_local = float(np.median(np.diff(s_u))) if s_u.size > 1 else 1.0
            win_pts = int(max(5, round(float(win_m) / float(du_local))))
            min_win = int(poly + 2)
            if (min_win % 2) == 0:
                min_win += 1
            if win_pts < min_win:
                win_pts = min_win
            if win_pts > series_len:
                win_pts = series_len if (series_len % 2) == 1 else series_len - 1
            return max(win_pts, 3)

        mode = str(self._z_smooth or "").lower().strip()
        if mode in ("median", "median+sg"):
            z_u_s = _apply_median(z_u_s, int(self._z_medfilt))
        if mode in ("savgol", "median+sg"):
            win_pts = _win_pts_m_to_samples(self._z_sg_win_m, du, int(self._z_sg_poly), z_u_s.size)
            if win_pts >= 3 and win_pts > int(self._z_sg_poly):
                z_u_s = savgol_filter(
                    z_u_s.astype(float),
                    window_length=win_pts,
                    polyorder=int(self._z_sg_poly),
                    deriv=0
                ).astype(np.float32)
        # if mode == "none" or unknown: keep z_u_s as-is

        # ---------- Dérivée (pente) sur la grille choisie ----------
        if self._slope_method == "ols":
            # Local linear regression (OLS) on uniform grid
            grade_u = self._ols_slopes_on_uniform_grid(s_u, z_u_s, win_m=self._slope_ols_win_m, du=du)
        elif self._slope_method == "kalman":
            # Kalman sur l'abscisse curviligne: état [z, slope]
            self._slope_kf_enabled = True
            grade_u = self._kalman_slope_quasi_const(
                z_u=z_u_s,
                du=float(du),
                q_z=float(self._kf_q_z),
                q_slope=float(self._kf_q_slope),
                r_meas=float(self._kf_r),
            )
        else:
            grade_u = None
            if len(z_u_s) >= 3 and self._deriv_sg_poly < len(z_u_s):
                win_pts = int(max(5, round(self._deriv_sg_win_m / du)))
                min_win = int(self._deriv_sg_poly + 2)
                if (min_win % 2) == 0:
                    min_win += 1
                if win_pts < min_win:
                    win_pts = min_win
                if win_pts > len(z_u_s):
                    win_pts = len(z_u_s) if (len(z_u_s) % 2) == 1 else len(z_u_s) - 1
                if win_pts >= 3 and win_pts > self._deriv_sg_poly:
                    grade_u = savgol_filter(
                        z_u_s,
                        window_length=win_pts,
                        polyorder=self._deriv_sg_poly,
                        deriv=1,
                        delta=du
                    )
            if grade_u is None:
                # gradient simple en dernier recours
                grade_u = np.gradient(z_u_s, s_u, edge_order=1)

        # Capture a copy of the raw slope in percent on the uniform grid (for diagnostics)
        slope_raw_pct_u = (np.asarray(grade_u, dtype=float) * 100.0) if isinstance(grade_u, np.ndarray) else None

        # --- Post-smoothing of slope (grade) with Savitzky–Golay on the same grid (s_u) ---
        if isinstance(grade_u, np.ndarray) and grade_u.size >= 3 and self._slope_sg_win_m and self._slope_sg_win_m > 0:
            # Convert slope SG window from meters to points on the current grid spacing (du)
            du_for_slope = float(du) if np.isfinite(du) and du > 0 else (float(np.median(np.diff(s_u))) if len(s_u) > 1 else 1.0)
            win_pts = int(max(5, round(float(self._slope_sg_win_m) / du_for_slope)))
            min_win = int(self._slope_sg_poly + 2)
            if (min_win % 2) == 0:
                min_win += 1
            if win_pts < min_win:
                win_pts = min_win
            if win_pts > len(grade_u):
                win_pts = len(grade_u) if (len(grade_u) % 2) == 1 else len(grade_u) - 1
            if win_pts >= 3 and win_pts > self._slope_sg_poly:
                grade_u = savgol_filter(
                    grade_u.astype(float),
                    window_length=win_pts,
                    polyorder=int(self._slope_sg_poly),
                    deriv=0
                ).astype(np.float32)

        # --- Optional Butterworth low-pass on slope (spatial domain) ---
        # Interprets RS3_ALT_SLOPE_BW_CUTOFF_M as cutoff wavelength (meters).
        if isinstance(grade_u, np.ndarray) and grade_u.size >= 5 and np.isfinite(du) and du > 0 and self._slope_bw_cut_m and self._slope_bw_cut_m > 0:
            # Spatial sampling frequency (samples per meter)
            fs = 1.0 / float(du)
            # Cutoff frequency in cycles per meter (1 / wavelength)
            fc = 1.0 / float(self._slope_bw_cut_m)
            # Normalize to Nyquist (radial not required for butter(); use [0,1] with 1 --> Nyquist)
            wn = (fc) / (fs * 0.5)  # = 2*du / cutoff_m
            # Guardrails
            wn = float(np.clip(wn, 1e-6, 0.9999))
            try:
                b, a = butter(N=int(self._slope_bw_order), Wn=wn, btype="low", analog=False, output="ba")
                grade_u = filtfilt(b, a, grade_u.astype(float)).astype(np.float32)
            except Exception:
                # If filtering fails (e.g., ill-conditioned), keep previous grade_u
                pass

        # ---------- Ramener sur les points d'origine ----------
        if use_uniform and len(s_u) >= 2:
            # Interpolation linéaire de la pente vers s (puis %)
            grade = np.interp(s, s_u, grade_u).astype(np.float32)
            alt_s = np.interp(s, s_u, z_u_s).astype(np.float32)
        else:
            grade = np.asarray(grade_u, dtype=np.float32)
            alt_s = z_u_s.astype(np.float32)

        # Compute the raw slope percent aligned to s for diagnostics when ALT_DEBUG is enabled
        slope_raw_pct = None
        if ALT_DEBUG and slope_raw_pct_u is not None:
            if use_uniform and len(s_u) >= 2:
                slope_raw_pct = np.interp(s, s_u, slope_raw_pct_u).astype(np.float32)
            else:
                slope_raw_pct = slope_raw_pct_u.astype(np.float32)

        # 3) Reprojection des valeurs vers la longueur d'origine
        # On propage chaque valeur calculée vers les indices supprimés (forward-fill).
        out_alt_full = np.empty(n, dtype=np.float32)
        out_slope_full = np.empty(n, dtype=np.float32)

        idx_kept = np.flatnonzero(keep)
        out_alt_full[:] = np.nan
        out_slope_full[:] = np.nan
        out_alt_full[idx_kept] = alt_s
        out_slope_full[idx_kept] = grade * 100.0  # en %

        # Forward-fill sur positions supprimées
        # (les premiers NaN éventuels sont remplis par la première valeur valide)
        def _ffill_inplace(a: np.ndarray) -> None:
            isn = np.isnan(a)
            if not isn.any():
                return
            # première valeur valide
            first_valid = np.flatnonzero(~isn)
            if first_valid.size:
                a[: first_valid[0]] = a[first_valid[0]]
            # ffill
            for i in range(1, a.size):
                if np.isnan(a[i]):
                    a[i] = a[i - 1]

        _ffill_inplace(out_alt_full)
        _ffill_inplace(out_slope_full)

        # 4) Post-traitement pente : médiane glissante (impair), puis clipping
        k = int(max(1, self._slope_medfilt))
        if k > 1 and (k % 2) == 1:
            # éviter kernel plus grand que la série
            k_eff = min(k, (n if n % 2 == 1 else n - 1)) if n >= 3 else 1
            if k_eff >= 3:
                out_slope_full = medfilt(out_slope_full, kernel_size=k_eff).astype(np.float32)

        clip_abs = float(abs(self._slope_clip))
        if np.isfinite(clip_abs) and clip_abs > 0:
            np.clip(out_slope_full, -clip_abs, clip_abs, out=out_slope_full)

        # 5) Injection colonnes attendues par exporter.py
        df = df.copy()
        df["altitude_m"] = out_alt_full
        df["slope_percent"] = out_slope_full
        if ALT_DEBUG:
            try:
                # Basic diagnostics to verify slope changes have an effect
                stats = {
                    "alt_mean": float(np.nanmean(alt_s)) if isinstance(alt_s, np.ndarray) else None,
                    "alt_std": float(np.nanstd(alt_s)) if isinstance(alt_s, np.ndarray) else None,
                    "slope_raw_mean": float(np.nanmean(slope_raw_pct)) if isinstance(slope_raw_pct, np.ndarray) else None,
                    "slope_raw_std": float(np.nanstd(slope_raw_pct)) if isinstance(slope_raw_pct, np.ndarray) else None,
                    "slope_final_mean": float(np.nanmean(out_slope_full)),
                    "slope_final_std": float(np.nanstd(out_slope_full)),
                }
                print("[AltitudeStage] stats:", stats)
            except Exception:
                pass
            try:
                print("[AltitudeStage] effective:", {
                    "n": int(n),
                    "du(m)": float(du),
                    "z_smooth": self._z_smooth,
                    "z_win_m": float(self._z_sg_win_m),
                    "deriv_win_m": float(self._deriv_sg_win_m),
                    "slope_sg_win_m": float(self._slope_sg_win_m),
                    "slope_bw_cut_m": float(self._slope_bw_cut_m),
                    "slope_clip_%": float(self._slope_clip),
                    "slope_bw_order": self._slope_bw_order,
                    "slope_method": self._slope_method,
                    "slope_ols_win_m": float(self._slope_ols_win_m),
                    "kf_enabled": bool(self._slope_kf_enabled),
                    "kf_q_z": float(self._kf_q_z),
                    "kf_q_slope": float(self._kf_q_slope),
                    "kf_r": float(self._kf_r),
                })
            except Exception:
                pass
        if ALT_DEBUG:
            try:
                # raw altitude (service) aligned to full length
                _alt_raw_full = np.empty(n, dtype=np.float32); _alt_raw_full[:] = np.nan
                _alt_raw_full[idx_kept] = alt.astype(np.float32)
                _ffill_inplace(_alt_raw_full)
                df["z_raw_dbg"] = _alt_raw_full
                # smoothed altitude aligned to full length
                _alt_s_full = np.empty(n, dtype=np.float32); _alt_s_full[:] = np.nan
                _alt_s_full[idx_kept] = alt_s
                _ffill_inplace(_alt_s_full)
                df["z_smooth_dbg"] = _alt_s_full
                # raw slope percent (pre post-filters) aligned if available
                if slope_raw_pct is not None:
                    _slope_raw_full = np.empty(n, dtype=np.float32); _slope_raw_full[:] = np.nan
                    _slope_raw_full[idx_kept] = slope_raw_pct
                    _ffill_inplace(_slope_raw_full)
                    df["slope_raw_pct_dbg"] = _slope_raw_full
            except Exception:
                pass
        return df