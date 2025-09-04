import os
import logging
import yaml
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from core2.contracts import Result  # type: ignore
from core2.context import Context  # type: ignore

log = logging.getLogger(__name__)

DEFAULT_URL = os.environ.get("RS3_ALT_API_URL", "http://localhost:5004/enrich_terrain")

class AltitudePlugin:
    name = "rs3-plugin-altitude-agpl"
    license = "AGPL-3.0-only"
    kind = "enricher"   # le loader l’ajoute dans la liste des enrichers

    def provides_schema_fragments(self) -> List[Dict[str, Any]]:
        frag_path = Path(__file__).with_suffix("").parent / "schema_fragments" / "altitude.yaml"
        with open(frag_path, "r", encoding="utf-8") as f:
            return [yaml.safe_load(f)]

    def apply(self, df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
        """Alimente altitude_m via l’API SRTM (format attendu par ton service)."""
        if df.empty or not {"lat", "lon"}.issubset(df.columns):
            return df

        api_url = (config or {}).get("dataset", {}).get("altitude", {}).get("api_url", DEFAULT_URL)

        # distance_m (cumul) si absente
        if "distance_m" not in df.columns:
            try:
                from core.terrain.distance import compute_cumulative_distance
                df = compute_cumulative_distance(df)
            except Exception:
                # fallback simple (0..n)
                import numpy as np
                df["distance_m"] = np.arange(len(df), dtype=float)

        payload = df[["lat", "lon", "distance_m"]].to_dict(orient="records")

        import requests
        try:
            r = requests.post(api_url, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            log.warning("[ALT] appel API %s en échec: %s — altitude_m laissée vide.", api_url, e)
            df.setdefault("altitude_m", pd.NA)
            return df

        if not isinstance(data, list) or len(data) != len(df):
            log.warning("[ALT] réponse inattendue (len=%s vs %s) — altitude_m laissée vide.",
                        getattr(data, "__len__", lambda: "n/a")(), len(df))
            df.setdefault("altitude_m", pd.NA)
            return df

        try:
            df["altitude_m"] = pd.Series([pt["altitude"] for pt in data], index=df.index, dtype="float32")
            # (optionnel) colonnes bonus si servies
            if "altitude_smoothed" in data[0]:
                df["altitude_smoothed"] = pd.Series([pt["altitude_smoothed"] for pt in data], index=df.index, dtype="float32")
            if "slope_percent" in data[0]:
                df["slope_percent"] = pd.Series([pt["slope_percent"] for pt in data], index=df.index, dtype="float32")
        except Exception as e:
            log.warning("[ALT] parsing JSON altitude en échec: %s", e)
            df.setdefault("altitude_m", pd.NA)

        log.info("[ALT] altitude_m valorisée sur %d points.", int(df["altitude_m"].notna().sum()))
        return df

class AltitudeStage:
    """
    Adaptateur 'stage' pour la pipeline RS3.
    - Injecte les fragments de schéma du plugin (altitude.yaml) dans ctx.artifacts['schema_fragments'].
    - Appelle AltitudePlugin.apply() pour enrichir le DataFrame avec 'altitude_m' (et colonnes bonus si dispo).
    """
    name = "AltitudeStage"

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self._plugin = AltitudePlugin()

    def run(self, ctx: Context) -> Result:
        df = ctx.df
        if df is None or df.empty:
            return Result(ok=False, message="df vide")

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

        return Result()

def discover_stages(cfg: Optional[dict] = None) -> List[object]:
    """
    Point d'entrée de découverte (entry point 'rs3.plugins' ou fallback d'import direct).
    Retourne la liste des stages à insérer dans la pipeline.
    """
    return [AltitudeStage(config=cfg)]