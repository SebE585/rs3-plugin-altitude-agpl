import pandas as pd
from geopy.distance import geodesic

def compute_cumulative_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la distance cumulée (en mètres) entre les points GPS successifs d’un DataFrame.

    La distance est calculée géodésiquement (WGS84) à partir des colonnes 'lat' et 'lon'.
    Une nouvelle colonne 'distance_m' est ajoutée au DataFrame, contenant la distance totale
    depuis le premier point.

    Paramètres
    ----------
    df : pd.DataFrame
        Un DataFrame contenant au moins les colonnes 'lat' et 'lon'.

    Retours
    -------
    pd.DataFrame
        Le DataFrame d’origine avec une colonne supplémentaire 'distance_m'.
    
    Exceptions
    ----------
    ValueError : si les colonnes 'lat' ou 'lon' sont absentes.
    """
    if 'lat' not in df.columns or 'lon' not in df.columns:
        raise ValueError("Le DataFrame doit contenir les colonnes 'lat' et 'lon'.")

    distances = [0.0]
    total = 0.0
    lat_lon_pairs = zip(df['lat'].values, df['lon'].values)

    prev_point = next(lat_lon_pairs)
    for curr_point in lat_lon_pairs:
        d = geodesic(prev_point, curr_point).meters
        total += d
        distances.append(total)
        prev_point = curr_point

    df['distance_m'] = distances
    return df