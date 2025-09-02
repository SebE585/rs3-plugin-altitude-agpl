

# enrichments/altitude_enricher.py
import os
import logging
from typing import Optional, List, Dict, Any

import requests
import pandas as pd

from core.terrain.distance import compute_cumulative_distance

logger = logging.getLogger(__name__)

# Default API endpoint; can be overridden by CONFIG or ENV
DEFAULT_SRTM_API_URL = "http://localhost:5004/enrich_terrain"


def _resolve_api_url(config: Optional[dict]) -> str:
    """Resolve API URL from config/env with sane defaults."""
    # Order of precedence: config.dataset.altitude.api_url → ENV RS3_ALT_API_URL → default
    if config:
        api_url = (
            config.get("dataset", {})
                  .get("altitude", {})
                  .get("api_url")
        )
        if api_url:
            return api_url
    return os.environ.get("RS3_ALT_API_URL", DEFAULT_SRTM_API_URL)


def enrich_terrain_via_api(df: pd.DataFrame, config: Optional[dict] = None, timeout: float = 30.0) -> pd.DataFrame:
    """
    Enrichit un DataFrame GPS avec les données d'altitude et de pente,
    en appelant une API SRTM externe.

    L'API attend un JSON avec les champs 'lat', 'lon', 'distance_m' et renvoie
    par point un objet contenant :
      - altitude : en mètres
      - altitude_smoothed : altitude lissée
      - slope_percent : pente (%) locale

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les colonnes 'lat' et 'lon'. Si 'distance_m' est absente,
        elle est calculée automatiquement (métrique cumulée). 
    config : dict | None
        Peut contenir dataset.altitude.api_url pour surcharger l'URL.
    timeout : float
        Timeout HTTP (secondes) pour l'appel API.

    Retours
    -------
    pd.DataFrame
        DataFrame d’origine enrichi avec les colonnes :
          - 'altitude' (m)
          - 'altitude_smoothed' (m)
          - 'slope_percent' (%)
        + 'altitude_m' (alias de 'altitude' pour compat dataset v1.0)

    Exceptions
    ----------
    RuntimeError : en cas d’erreur réseau/HTTP
    ValueError : si la réponse JSON est invalide ou de longueur inattendue
    """
    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError("Le DataFrame doit contenir les colonnes 'lat' et 'lon'.")

    # Calcul de la distance cumulée si absente
    if "distance_m" not in df.columns:
        df = compute_cumulative_distance(df)

    api_url = _resolve_api_url(config)

    payload: List[Dict[str, Any]] = df[["lat", "lon", "distance_m"]].to_dict(orient="records")

    try:
        resp = requests.post(api_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise RuntimeError(f"[TERRAIN API] Erreur d'appel API SRTM: {e}")

    if not isinstance(data, list) or len(data) != len(df):
        raise ValueError(
            f"[TERRAIN API] Réponse JSON inattendue: type={type(data)} len={getattr(data, '__len__', lambda: 'n/a')()} vs df={len(df)}"
        )

    try:
        df["altitude"] = [pt["altitude"] for pt in data]
        df["altitude_smoothed"] = [pt["altitude_smoothed"] for pt in data]
        df["slope_percent"] = [pt["slope_percent"] for pt in data]
    except (KeyError, TypeError, IndexError) as e:
        raise ValueError(f"[TERRAIN API] Réponse JSON invalide ou incomplète: {e}")

    # Alias dataset v1.0
    df["altitude_m"] = df["altitude"].astype("float32")

    logger.debug("Altitude enrichie via API (%s) → %d points", api_url, len(df))
    return df


def enrich_altitude(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Wrapper v1.0 utilisé par le pipeline.
    - Si provider = 'none' → on retourne df tel quel.
    - Sinon → appel API SRTM via `enrich_terrain_via_api`.

    Config attendue:
      config["dataset"]["altitude"]["provider"] ∈ {"srtm", "ign", "demo", "none"}
      config["dataset"]["altitude"]["api_url"] (optionnel)
    """
    provider = (config or {}).get("dataset", {}).get("altitude", {}).get("provider")
    if provider is None:
        # Optionnellement, lire depuis l'env
        provider = os.environ.get("RS3_ALT_PROVIDER", "none")

    if str(provider).lower() == "none":
        return df

    # Pour v1.0, ign ≈ srtm côté API (même endpoint). 'demo' peut aussi passer ici si l'API la gère.
    return enrich_terrain_via_api(df, config=config)