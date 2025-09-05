import os
import logging
import yaml
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from rs3_contracts.api import Result, Stage, ContextSpec  # type: ignore
import requests
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_BASE = os.environ.get("RS3_ALTITUDE_BASE", "http://localhost:5004").rstrip("/")
DEFAULT_URL  = os.environ.get("RS3_ALT_API_URL", f"{DEFAULT_BASE}/enrich_terrain")

class AltitudePlugin:
    name = "rs3-plugin-altitude-agpl"
    license = "AGPL-3.0-only"
    kind = "enricher"   # le loader l’ajoute dans la liste des enrichers

    def provides_schema_fragments(self) -> List[Dict[str, Any]]:
        frag_path = Path(__file__).resolve().parent / "schema_fragments" / "altitude.yaml"
        with open(frag_path, "r", encoding="utf-8") as f:
            return [yaml.safe_load(f)]

    def apply(self, df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
        """Alimente altitude_m via l’API SRTM (format attendu par ton service)."""
        if df.empty or not {"lat", "lon"}.issubset(df.columns):
            return df

        cfg = (config or {})
        # ---- Resolve base/api URL from several possible locations (flat, nested, plugin.altitude)
        api_url = DEFAULT_URL
        base_url = DEFAULT_BASE
        if isinstance(cfg, dict):
            # plugin-namespace first
            alt_cfg = (cfg.get("plugins", {}) or {}).get("altitude", {}) or cfg.get("altitude", {}) or {}
            # flat overrides
            api_url = alt_cfg.get("api_url", cfg.get("api_url", api_url))
            base_url = alt_cfg.get("base_url", alt_cfg.get("base", cfg.get("base_url", base_url)))
            # legacy nested form: {"dataset": {"altitude": {"api_url": "..."}}}
            api_url = cfg.get("dataset", {}).get("altitude", {}).get("api_url", api_url)
        # if only base_url is provided, append default path
        if api_url == DEFAULT_URL and base_url:
            api_url = f"{base_url.rstrip('/')}/enrich_terrain"

        # ---- Resolve timeout (seconds) → tuple(connect, read)
        t_read = float(
            cfg.get("timeout") if isinstance(cfg, dict) and "timeout" in cfg else
            cfg.get("timeout_s", 30.0) if isinstance(cfg, dict) else 30.0
        )
        t_conn = min(10.0, t_read)
        timeout: Tuple[float, float] = (t_conn, t_read)

        # tiny debug to help field diagnosis without being too verbose
        log.info("[ALT] endpoint=%s timeout=%ss (connect=%ss, read=%ss)", api_url, t_read, t_conn, t_read)

        # distance_m (cumul) si absente
        if "distance_m" not in df.columns:
            try:
                from core.terrain.distance import compute_cumulative_distance
                df = compute_cumulative_distance(df)
            except Exception:
                # fallback simple (0..n)
                df["distance_m"] = np.arange(len(df), dtype=float)

        try:
            sample = df[["lat","lon"]].head(1).to_dict(orient="records")[0]
            log.debug("[ALT] payload head %s", sample)
        except Exception:
            pass

        payload = df[["lat", "lon", "distance_m"]].to_dict(orient="records")

        try:
            r = requests.post(api_url, json=payload, timeout=timeout)
            # expose short server reply for easier schema debugging
            if r.status_code >= 400:
                body = (r.text or "")[:300].replace("\n", " ")
                log.warning("[ALT] API %s → %s: %s", api_url, r.status_code, body)
            r.raise_for_status()
            data = r.json()
        except requests.Timeout as e:
            log.warning("[ALT] timeout API %s après %ss — altitude_m laissée vide.", api_url, t_read)
            if "altitude_m" not in df.columns:
                df["altitude_m"] = pd.Series([np.nan] * len(df), index=df.index, dtype="float32")
            return df
        except requests.RequestException as e:
            body = ""
            try:
                body = (getattr(e.response, "text", "") or "")[:300].replace("\n", " ")
            except Exception:
                pass
            log.warning("[ALT] appel API %s en échec: %s — %s — altitude_m laissée vide.", api_url, e, body)
            if "altitude_m" not in df.columns:
                df["altitude_m"] = pd.Series([np.nan] * len(df), index=df.index, dtype="float32")
            return df

        if not isinstance(data, list) or len(data) != len(df):
            log.warning("[ALT] réponse inattendue (len=%s vs %s) — altitude_m laissée vide.",
                        getattr(data, "__len__", lambda: "n/a")(), len(df))
            if "altitude_m" not in df.columns:
                df["altitude_m"] = pd.Series([np.nan] * len(df), index=df.index, dtype="float32")
            return df

        try:
            alt = [pt.get("altitude", np.nan) for pt in data]
            df["altitude_m"] = pd.Series(alt, index=df.index, dtype="float32")
            # (optionnel) colonnes bonus si servies
            if isinstance(data, list) and data:
                if "altitude_smoothed" in data[0]:
                    df["altitude_smoothed"] = pd.Series([pt.get("altitude_smoothed", np.nan) for pt in data], index=df.index, dtype="float32")
                if "slope_percent" in data[0]:
                    df["slope_percent"] = pd.Series([pt.get("slope_percent", np.nan) for pt in data], index=df.index, dtype="float32")
        except Exception as e:
            log.warning("[ALT] parsing JSON altitude en échec: %s", e)
            if "altitude_m" not in df.columns:
                df["altitude_m"] = pd.Series([np.nan] * len(df), index=df.index, dtype="float32")

        log.info("[ALT] altitude_m valorisée sur %d points (source=%s).", int(df["altitude_m"].notna().sum()), api_url)
        return df

class AltitudeStage(Stage):
    """
    Adaptateur 'stage' pour la pipeline RS3.
    - Injecte les fragments de schéma du plugin (altitude.yaml) dans ctx.artifacts['schema_fragments'].
    - Appelle AltitudePlugin.apply() pour enrichir le DataFrame avec 'altitude_m' (et colonnes bonus si dispo).
    """
    name = "AltitudeStage"

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self._plugin = AltitudePlugin()

    def run(self, ctx: ContextSpec) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result(False, "df vide")

        # 1) Enregistre les fragments de schéma attendus par Exporter
        try:
            frags = self._plugin.provides_schema_fragments()
            if frags:
                lst = ctx.artifacts.get("schema_fragments")
                if not isinstance(lst, list):
                    lst = []
                lst.extend(frags)
                ctx.artifacts["schema_fragments"] = lst
        except Exception as e:
            # Ne bloque pas la pipeline si le schéma n'est pas dispo
            log.warning("[ALT] provides_schema_fragments() a échoué: %s", e)

        # 2) Enrichissement altitude via le service
        try:
            # Permet aux utilisateurs de préciser la config sous cfg['plugins']['altitude']
            plugin_cfg = None
            if isinstance(ctx.cfg, dict):
                plugin_cfg = (ctx.cfg.get("plugins", {}) or {}).get("altitude", None) or ctx.cfg

            new_df = self._plugin.apply(df, config=plugin_cfg)
            if new_df is not None and not new_df.empty:
                ctx.df = new_df
        except Exception as e:
            log.warning("[ALT] enrichissement altitude en échec: %s", e)
            # Ne pas échouer la pipeline : on laisse df intact

        return Result(True, "altitude stage completed (non-blocking)")

def discover_stages(cfg: Optional[dict] = None) -> List[Stage]:
    """
    Point d'entrée de découverte (entry point 'rs3.plugins').
    Le loader RS3 attend une **liste d'instances de stages** possédant `.run(...)`.
    """
    return [AltitudeStage(config=cfg)]