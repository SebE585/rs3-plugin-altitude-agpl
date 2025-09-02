import os
import logging
import yaml
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

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