# app.py
from flask import Flask, request, jsonify
import pandas as pd
import os
from geopy.distance import geodesic
import rasterio
from pyproj import Transformer
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
import traceback

app = Flask(__name__)

def compute_cumulative_distance(df: pd.DataFrame) -> pd.DataFrame:
    distances = [0.0]
    for i in range(1, len(df)):
        p1 = (df.loc[i - 1, 'lat'], df.loc[i - 1, 'lon'])
        p2 = (df.loc[i, 'lat'], df.loc[i, 'lon'])
        d = geodesic(p1, p2).meters
        distances.append(distances[-1] + d)
    df['distance_m'] = distances
    print(f"[INFO] Distance cumulée calculée pour {len(df)} points.")
    return df

def enrich_with_terrain(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    mnt_path = config.get("mnt_path", "/data/srtm/normandy_l93.tif")

    if not os.path.exists(mnt_path):
        print(f"[WARNING] Fichier MNT introuvable : {mnt_path}")
        df['altitude'] = 0.0
        df['slope_percent'] = 0.0
        return df

    print(f"[INFO] Chargement du MNT : {mnt_path}")

    with rasterio.open(mnt_path) as mnt:
        transformer = Transformer.from_crs("EPSG:4326", mnt.crs, always_xy=True)
        coords = [transformer.transform(lon, lat) for lat, lon in zip(df['lat'], df['lon'])]
        altitudes = [val[0] if val[0] is not None else 0.0 for val in mnt.sample(coords)]
        df['altitude'] = altitudes

    if 'distance_m' not in df.columns:
        df = compute_cumulative_distance(df)

    df['altitude_smoothed'] = gaussian_filter1d(df['altitude'].astype(float), sigma=3)

    alt_diff = df['altitude_smoothed'].diff().fillna(0)
    dist_diff = df['distance_m'].diff().replace(0, pd.NA).fillna(1)

    slope = alt_diff / dist_diff * 100
    slope = slope.clip(lower=-30, upper=30)

    slope_float = slope.fillna(0).astype(float)
    slope_smooth = uniform_filter1d(slope_float, size=15, mode='nearest')

    df['slope_percent'] = slope_smooth

    print("[INFO] Altitude interpolée, lissée et pente calculée (clipping -30% à 30%).")
    print("[INFO] Pente lissée avec une fenêtre de taille 15.")

    return df

@app.route('/enrich_terrain', methods=['POST'])
@deprecated
def enrich_terrain_api():
    logger.warning("⚠️ Appel d'une fonction marquée @deprecated.")
    try:
        coords = request.get_json()
        if not coords or not isinstance(coords, list):
            return jsonify({"error": "Invalid JSON payload"}), 400

        df = pd.DataFrame(coords)

        if 'distance_m' not in df.columns:
            df = compute_cumulative_distance(df)

        config = {"mnt_path": os.environ.get("MNT_PATH", "/data/srtm/normandy_l93.tif")}
        df = enrich_with_terrain(df, config)

        return jsonify(df.to_dict(orient='records'))

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5004)
