# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import requests

from .altitude_service import AltitudeService, AltitudePipelineConfig
from .providers import IGNRGEAltiProvider, SRTMProvider


class HttpAltitudeProvider:
    """Client HTTP minimal pour l'API Flask altitude.
    Expose `altitude_at(lats, lons)` en utilisant `/profile` (batch).
    """
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.timeout = float(timeout)

    def altitude_at(self, lats, lons):
        payload = [{"lat": float(la), "lon": float(lo)} for la, lo in zip(lats, lons)]
        url = f"{self.base_url}/profile"
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        alts = data.get("altitude")
        if alts is None:
            raise RuntimeError(f"/profile response sans 'altitude': {data}")
        return alts


# --------------------------------------------------------------------------------------
# Construction du provider depuis YAML/env
# --------------------------------------------------------------------------------------
def _provider_from_cfg(cfg: dict[str, any]):
    alt = cfg.get("altitude", {}) if cfg else {}
    src = os.environ.get("RS3_ALT_PROVIDER", str(alt.get("source", "srtm")))

    nodata = float(alt.get("nodata", -9999.0))

    # Support d'un provider HTTP via service Flask
    svc = alt.get("service") if isinstance(alt.get("service"), dict) else None
    base_url = None
    if svc:
        base_url = svc.get("base_url") or os.environ.get("RS3_ALT_BASE_URL")
    else:
        base_url = os.environ.get("RS3_ALT_BASE_URL")
    if isinstance(base_url, str) and base_url.strip():
        timeout = float(alt.get("timeout_s", 30.0)) if isinstance(alt, dict) else 30.0
        return HttpAltitudeProvider(base_url.strip(), timeout=timeout)

    if src == "ign":
        cog = alt.get("ign", {}).get("cog_path") or os.environ.get("RS3_RGEALTI_COG")
        if not cog:
            raise ValueError("Chemin RGE COG manquant (altitude.ign.cog_path ou RS3_RGEALTI_COG).")
        return IGNRGEAltiProvider(cog, nodata=nodata)

    if src == "srtm":
        tif = alt.get("srtm", {}).get("raster_path") or os.environ.get("RS3_SRTM_PATH")
        if not tif:
            raise ValueError("Chemin SRTM manquant (altitude.srtm.raster_path ou RS3_SRTM_PATH).")
        return SRTMProvider(tif, nodata=nodata)

    # Par défaut
    raise ValueError(f"altitude.source inconnu: {src!r}")


def _pipeline_from_cfg(pipe: dict[str, any]) -> AltitudePipelineConfig:
    """
    Construit AltitudePipelineConfig. Ignore les clés inconnues.
    """
    return AltitudePipelineConfig(
        mode=str(pipe.get("mode", "track_first")),
        sg_window_m=float(pipe.get("sg_window_m", 5.0)),
        sg_poly=int(pipe.get("sg_poly", 3)),
        pre_median_3=bool(pipe.get("pre_median_3", False)),
        slope_clip_pct=pipe.get("slope_clip_pct", None),
        slope_bw_cutoff_m=pipe.get("slope_bw_cutoff_m", None),
        slope_bw_order=int(pipe.get("slope_bw_order", 2)),
        cache_lru_points=int(pipe.get("cache_lru_points", 0)),
        cache_round_decimals=int(pipe.get("cache_round_decimals", 5)),
    )


def build_service_from_yaml(yaml_path: str) -> AltitudeService:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    provider = _provider_from_cfg(cfg)
    pipe_cfg = _pipeline_from_cfg(cfg.get("altitude", {}).get("pipeline", {}))
    return AltitudeService(provider, pipe_cfg)


def build_service_from_dict(cfg: dict[str, any]) -> AltitudeService:
    """Construit un AltitudeService depuis un dict Python.

    Attendu par le plugin RS3 via l'entry point `altitude`.
    Le schéma est identique à celui du YAML (`altitude: {source: ..., pipeline: {...}}`).
    Les variables d'environnement (ex: RS3_ALT_PROVIDER) restent prises en compte.
    """
    cfg = cfg or {}
    provider = _provider_from_cfg(cfg)
    pipe_cfg = _pipeline_from_cfg(cfg.get("altitude", {}).get("pipeline", {}))
    return AltitudeService(provider, pipe_cfg)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _cli() -> int:
    parser = argparse.ArgumentParser(
        prog="rs3_plugin_altitude_agpl.factory",
        description="Outils altitude (IGN/SRTM) — sample & grade",
    )
    parser.add_argument("-c", "--config", dest="config", help="Chemin YAML de config altitude.", required=True)

    sub = parser.add_subparsers(dest="cmd", required=True)

    # sample: une coordonnée WGS84 -> altitude
    p_samp = sub.add_parser("sample", help="Échantillonne l'altitude à une position WGS84.")
    p_samp.add_argument("--lat", type=float, required=True)
    p_samp.add_argument("--lon", type=float, required=True)

    # grade: profil+pente depuis CSV lat/lon
    p_grade = sub.add_parser("grade", help="Calcule profil+pente depuis un CSV lat/lon.")
    p_grade.add_argument("--csv", required=True, help="CSV avec colonnes lat,lon")
    p_grade.add_argument("--out", required=False, help="CSV de sortie (z,grade,theta)")

    args = parser.parse_args()

    if args.cmd == "sample":
        svc = build_service_from_yaml(args.config)
        val = svc.altitude_at([args.lat], [args.lon])[0]
        print(f"altitude_m={val:.3f}")
        return 0

    if args.cmd == "grade":
        svc = build_service_from_yaml(args.config)
        df = pd.read_csv(args.csv)
        out = svc.profile_and_grade(df["lat"].values, df["lon"].values)
        res = pd.DataFrame(
            {
                "z_m": out["z"],
                "grade_m_per_m": out["grade"],
                "theta_rad": out["theta"],
            }
        )
        if args.out:
            res.to_csv(args.out, index=False)
            print(f"[OK] écrit: {args.out}  (N={len(res)})")
        else:
            print(res.head().to_string(index=False))
        return 0

    parser.error("Commande inconnue.")
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())