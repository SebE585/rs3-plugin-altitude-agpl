import numpy as np
import pandas as pd

def compute_slope_from_altitude(
    df: pd.DataFrame,
    altitude_col: str = 'altitude',
    distance_col: str = 'distance_m'
) -> np.ndarray:
    """
    Calcule la pente locale (en pourcentage) entre chaque paire de points GPS consécutifs,
    à partir de l'altitude et de la distance cumulée.

    Formule :
        slope_percent = 100 × Δaltitude / Δdistance

    Le premier point est défini comme NaN, car aucun point précédent n'existe pour comparer.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant au minimum les colonnes `altitude_col` et `distance_col`.

    altitude_col : str
        Nom de la colonne contenant l'altitude (en mètres).

    distance_col : str
        Nom de la colonne contenant la distance cumulée (en mètres).

    Retours
    -------
    np.ndarray
        Tableau de même longueur que le DataFrame, contenant la pente (en %) pour chaque point.
        Le premier élément est toujours NaN.

    Exceptions
    ----------
    KeyError : si les colonnes spécifiées sont absentes du DataFrame.
    """
    if altitude_col not in df.columns or distance_col not in df.columns:
        raise KeyError(f"Colonnes manquantes : '{altitude_col}' ou '{distance_col}'.")

    altitude = df[altitude_col].values
    distance = df[distance_col].values

    delta_altitude = np.diff(altitude)
    delta_distance = np.diff(distance).astype(float)

    # Évite division par 0
    delta_distance[delta_distance == 0] = np.nan

    slope_percent = (delta_altitude / delta_distance) * 100
    slope_percent = np.insert(slope_percent, 0, np.nan)

    return slope_percent