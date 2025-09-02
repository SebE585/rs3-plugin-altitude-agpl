import pandas as pd
import logging
from scipy.ndimage import gaussian_filter1d
from core.terrain.api import enrich_terrain_via_api
from core.terrain.local import enrich_with_terrain
from core.terrain.slope import compute_slope_from_altitude

logger = logging.getLogger(__name__)

@deprecated
def enrich_terrain(df: pd.DataFrame, mnt_path: str = None) -> pd.DataFrame:
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    """
    Enrichit un DataFrame GPS avec les informations topographiques issues :
    - soit d'une API SRTM distante,
    - soit d'un MNT local (GeoTIFF, en fallback),
    puis ajoute :
    - une version lissée de l'altitude,
    - une pente locale (en %), clipée à ±30 %.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant au minimum les colonnes 'lat' et 'lon'.

    mnt_path : str, optionnel
        Chemin local vers un fichier GeoTIFF en cas d’échec de l’API.

    Retours
    -------
    pd.DataFrame
        Le DataFrame enrichi avec les colonnes suivantes (si absentes) :
        - 'altitude'
        - 'altitude_smoothed'
        - 'slope_percent'

    Exceptions
    ----------
    RuntimeError : si l'API échoue et qu'aucun MNT n’est fourni.
    ValueError : si les colonnes 'lat'/'lon' sont manquantes.
    """
    if not {'lat', 'lon'}.issubset(df.columns):
        raise ValueError("Le DataFrame doit contenir les colonnes 'lat' et 'lon'.")

    try:
        logger.info(f"[TERRAIN] Enrichissement via API SRTM ({len(df)} points)...")
        df = enrich_terrain_via_api(df)
    except Exception as e:
        logger.warning(f"[TERRAIN] API indisponible : {e}")
        if mnt_path:
            logger.info(f"[TERRAIN] Fallback MNT local activé : {mnt_path}")
            df = enrich_with_terrain(df, mnt_path)
        else:
            raise RuntimeError("Enrichissement terrain échoué : ni API ni MNT local disponible.")

    # 🔁 Post-traitement inertiel si nécessaire
    if 'altitude_smoothed' not in df.columns:
        df['altitude_smoothed'] = gaussian_filter1d(df['altitude'].astype(float), sigma=3)

    if 'slope_percent' not in df.columns:
        slope = compute_slope_from_altitude(df, altitude_col='altitude_smoothed')
        df['slope_percent'] = pd.Series(slope).clip(-30, 30).fillna(0).astype(float)

    logger.info("[TERRAIN] Altitude lissée et pente calculée (clip ±30%) ajoutées.")
    return df