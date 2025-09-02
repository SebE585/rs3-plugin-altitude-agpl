import os
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from core.terrain.distance import compute_cumulative_distance
from core.terrain.slope import compute_slope_from_altitude

def enrich_with_terrain(df: pd.DataFrame, mnt_path: str) -> pd.DataFrame:
    """
    Enrichit un DataFrame de points GPS avec les données d’altitude et de pente
    en utilisant un MNT local (GeoTIFF).

    Étapes :
    - Extraction de l'altitude depuis le raster GeoTIFF.
    - Lissage de l'altitude (filtre gaussien).
    - Calcul de la pente (différences relatives lissées, clip ±30 %).
    - Ajout des colonnes : 'altitude', 'altitude_smoothed', 'slope_percent'.

    Paramètres
    ----------
    df : pd.DataFrame
        Un DataFrame contenant les colonnes 'lat' et 'lon'.
    
    mnt_path : str
        Chemin absolu vers un fichier GeoTIFF (MNT).

    Retours
    -------
    pd.DataFrame
        Le DataFrame d’origine enrichi des colonnes :
        - 'altitude'
        - 'altitude_smoothed'
        - 'slope_percent'

    Exceptions
    ----------
    FileNotFoundError : si le fichier MNT n’existe pas.
    ValueError : si les colonnes nécessaires sont absentes.
    """
    if not os.path.exists(mnt_path):
        raise FileNotFoundError(f"MNT file not found: {mnt_path}")

    if not {'lat', 'lon'}.issubset(df.columns):
        raise ValueError("Le DataFrame doit contenir les colonnes 'lat' et 'lon'.")

    # ⛰️ Lecture MNT
    with rasterio.open(mnt_path) as mnt:
        transformer = Transformer.from_crs("EPSG:4326", mnt.crs, always_xy=True)
        coords = [transformer.transform(lon, lat) for lat, lon in zip(df['lat'], df['lon'])]
        altitudes = [val[0] if val[0] is not None else 0.0 for val in mnt.sample(coords)]
        df['altitude'] = altitudes

    # 🛣️ Calcul de la distance cumulée si absente
    if 'distance_m' not in df.columns:
        df = compute_cumulative_distance(df)

    # 🎚️ Lissage de l'altitude
    df['altitude_smoothed'] = gaussian_filter1d(df['altitude'].astype(float), sigma=3)

    # 📈 Calcul de pente clipée à ±30 %
    slope = compute_slope_from_altitude(df, 'altitude_smoothed', 'distance_m')
    slope_clipped = pd.Series(slope).clip(-30, 30).fillna(0).astype(float).to_numpy()

    # 🪄 Lissage final de la pente
    df['slope_percent'] = uniform_filter1d(slope_clipped, size=15, mode='nearest')

    return df