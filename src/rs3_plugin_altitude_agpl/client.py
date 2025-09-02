import requests
import pandas as pd
from core.terrain.distance import compute_cumulative_distance

SRTM_API_URL = "http://localhost:5004/enrich_terrain"

def enrich_terrain_via_api(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit un DataFrame GPS avec les données d'altitude et de pente,
    en appelant une API SRTM externe.

    L'API attend un JSON avec les champs 'lat', 'lon', 'distance_m' et renvoie :
    - altitude : en mètres
    - altitude_smoothed : altitude lissée
    - slope_percent : pente (%) locale

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant les colonnes 'lat' et 'lon'. Si 'distance_m' est absente,
        elle est calculée automatiquement.

    Retours
    -------
    pd.DataFrame
        DataFrame d’origine enrichi avec les colonnes :
        - 'altitude'
        - 'altitude_smoothed'
        - 'slope_percent'

    Exceptions
    ----------
    requests.RequestException : en cas d’erreur réseau ou HTTP
    ValueError : si la réponse JSON est invalide ou incomplète
    """
    if not {'lat', 'lon'}.issubset(df.columns):
        raise ValueError("Le DataFrame doit contenir les colonnes 'lat' et 'lon'.")

    if 'distance_m' not in df.columns:
        df = compute_cumulative_distance(df)

    payload = df[['lat', 'lon', 'distance_m']].to_dict(orient='records')

    try:
        response = requests.post(SRTM_API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        raise RuntimeError(f"[TERRAIN API] Erreur d'appel API SRTM : {e}")

    try:
        df['altitude'] = [pt['altitude'] for pt in data]
        df['altitude_smoothed'] = [pt['altitude_smoothed'] for pt in data]
        df['slope_percent'] = [pt['slope_percent'] for pt in data]
    except (KeyError, TypeError, IndexError) as e:
        raise ValueError(f"[TERRAIN API] Réponse JSON invalide ou incomplète : {e}")

    return df