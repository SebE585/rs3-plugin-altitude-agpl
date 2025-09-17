# -*- coding: utf-8 -*-
"""
Interpolateur 10 Hz pour la pente (grade) et l'angle (theta) le long d'une trace.
- Source d'altitude fournie par le plugin (RGEALTI/SRTM) via un YAML.
- Entrées: CSV (lat,lon) ou listes --lat/--lon.
- Vitesse constante (--speed-kph), par défaut 50 km/h.
- Sortie: impression console et/ou CSV (--out).
"""
from __future__ import annotations
import argparse
import sys
from typing import List, Tuple, Optional

import numpy as np

from rs3_plugin_altitude_agpl.factory import build_service_from_yaml

def _ensure_arrays(lats, lons) -> Tuple[np.ndarray, np.ndarray]:
    la = np.atleast_1d(np.asarray(lats, dtype=float))
    lo = np.atleast_1d(np.asarray(lons, dtype=float))
    if la.size != lo.size:
        raise ValueError("lat/lon doivent avoir la même longueur.")
    if la.size < 2:
        raise ValueError("Au moins 2 points sont nécessaires.")
    return la, lo

def _load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    df = pd.read_csv(path)
    if not {"lat", "lon"} <= set(df.columns):
        raise ValueError("Le CSV doit contenir les colonnes lat, lon.")
    return df["lat"].to_numpy(), df["lon"].to_numpy()

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Interpole grade/theta à 10 Hz le long d'une trace.")
    p.add_argument("-c", "--config", required=True, help="YAML de configuration altitude (IGN/SRTM).")
    g_in = p.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--csv", help="CSV d'entrée avec colonnes lat,lon")
    g_in.add_argument("--lat", nargs="+", type=float, help="Liste de latitudes")
    p.add_argument("--lon", nargs="+", type=float, help="Liste de longitudes (si --lat utilisé)")
    p.add_argument("--speed-kph", type=float, default=50.0, help="Vitesse constante pour projeter en temps (km/h). Défaut=50")
    p.add_argument("--out", help="Chemin CSV de sortie (sinon affichage console)")
    args = p.parse_args(argv)

    # Charge service (provider + pipeline)
    svc = build_service_from_yaml(args.config)

    # Charge les points
    if args.csv:
        lats, lons = _load_csv(args.csv)
    else:
        if args.lon is None:
            p.error("--lon est requis quand --lat est utilisé")
        lats = np.array(args.lat, dtype=float)
        lons = np.array(args.lon, dtype=float)

    lats, lons = _ensure_arrays(lats, lons)

    # Profil et pente (grade m/m, theta rad)
    out = svc.profile_and_grade(lats, lons)
    z = np.asarray(out["z"], dtype=float)            # altitudes aux points d'origine
    grade = np.asarray(out["grade"], dtype=float)    # m/m
    theta = np.asarray(out["theta"], dtype=float)    # rad

    # Abscisse curviligne s (m)
    s = svc._curvilinear_abscissa(lats, lons).astype(float)

    # Conversion distance -> temps avec vitesse constante
    v = max(args.speed_kph, 1e-3) * (1000.0 / 3600.0)  # m/s
    T = s[-1] / v if s[-1] > 0 else 0.0

    # Grille temporelle 10 Hz
    if T <= 0:
        print("[WARN] Longueur totale ~0 m, rien à interpoler.")
        svc.close()
        return 0

    t = np.arange(0.0, T + 1e-9, 0.1)  # 0.1 s -> 10 Hz
    s_t = v * t                        # abscisse équivalente à t

    # Interpolation sur s
    # (np.interp suppose s croissant et renvoie les bords via clamp)
    z_t = np.interp(s_t, s, z)
    g_t = np.interp(s_t, s, grade)
    th_t = np.interp(s_t, s, theta)

    # Sortie
    if args.out:
        import pandas as pd
        df = pd.DataFrame({
            "t_s": t,
            "s_m": s_t,
            "z_m": z_t,
            "grade_m_per_m": g_t,
            "theta_rad": th_t,
        })
        df.to_csv(args.out, index=False)
        print(f"[OK] Écrit: {args.out}  (N={len(df)})")
    else:
        # Aperçu console (premières lignes)
        print("   t(s)     s(m)      z(m)     grade(m/m)    theta(rad)")
        for i in range(min(10, t.size)):
            print(f"{t[i]:7.2f}  {s_t[i]:7.2f}  {z_t[i]:8.2f}  {g_t[i]:12.5f}  {th_t[i]:10.5f}")
        if t.size > 10:
            print(f"... ({t.size} échantillons à 10 Hz)")

    svc.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())