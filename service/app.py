# app.py
from flask import Flask, request, jsonify
import pandas as pd
import os
from geopy.distance import geodesic
import rasterio
from pyproj import Transformer
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
import traceback
import argparse
import yaml
import numpy as np

app = Flask(__name__)

# ---- Global config (loaded from YAML/ENV)
CONFIG = {
    # Generic fallback path (kept for backward compatibility)
    "mnt_path": os.environ.get("MNT_PATH", "/data/srtm/normandy_l93.tif"),
    # Named providers (env-vars are optional)
    "providers": {
        "rge1": os.environ.get("RS3_RGEALTI_1M_COG"),
        "rge5": os.environ.get("RS3_RGEALTI_5M_COG"),
        "srtm30": os.environ.get("RS3_SRTM30_COG"),
        "srtm90": os.environ.get("RS3_SRTM90_COG"),
    },
}

def load_config_from_file(path: str | None) -> None:
    """Load/merge config from a YAML file into CONFIG."""
    global CONFIG
    if not path:
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Accept both flat and nested structures
        if isinstance(data, dict):
            # service/config style: altitude: { srtm: { raster_path: ... } }
            alt = (data.get("altitude") or {}) if isinstance(data.get("altitude"), dict) else {}
            srtm = (alt.get("srtm") or {}) if isinstance(alt.get("srtm"), dict) else {}
            raster = srtm.get("raster_path")
            # also accept top-level mnt_path
            mnt = data.get("mnt_path")
            CONFIG["mnt_path"] = str(mnt or raster or CONFIG["mnt_path"])

            # Merge provider-specific raster paths if present in YAML
            # Accept formats like:
            # altitude:
            #   rge1: { raster_path: "..." }
            #   rge5: { raster_path: "..." }
            #   srtm30: { raster_path: "..." }
            #   srtm90: { raster_path: "..." }
            provs = CONFIG.get("providers", {}).copy()
            for key in ("rge1", "rge5", "srtm30", "srtm90"):
                try:
                    v = (alt.get(key) or {})
                    if isinstance(v, dict) and v.get("raster_path"):
                        provs[key] = str(v["raster_path"])
                except Exception:
                    pass
            # Also accept top-level 'providers' dict with same keys
            if isinstance(data.get("providers"), dict):
                for key in ("rge1", "rge5", "srtm30", "srtm90"):
                    if isinstance(data["providers"].get(key), dict) and data["providers"][key].get("raster_path"):
                        provs[key] = str(data["providers"][key]["raster_path"])
                    elif isinstance(data["providers"].get(key), str):
                        provs[key] = str(data["providers"][key])
            CONFIG["providers"] = provs
    except FileNotFoundError:
        print(f"[WARN] Config file not found: {path}")
    except Exception as e:
        print(f"[WARN] Failed to load config '{path}': {e}")

def resolve_mnt_path(requested: str | None):
    """
    Resolve the best available MNT path given a requested provider.
    Priority order with fallback: rge1 -> rge5 -> srtm30 -> srtm90 -> CONFIG.mnt_path
    Returns (provider_used, path) or (None, None) if nothing exists on disk.
    """
    providers = CONFIG.get("providers", {}) or {}
    order = ["rge1", "rge5", "srtm30", "srtm90"]
    req = (requested or "").strip().lower() or None

    # Build candidate list (requested first if given), then the rest of the order
    if req and req in order:
        candidates = [req] + [p for p in order if p != req]
    else:
        candidates = order[:]

    # Append generic fallback if set
    generic = CONFIG.get("mnt_path")
    paths = []
    for p in candidates:
        path = providers.get(p)
        if path:
            paths.append((p, path))
    if generic:
        paths.append(("custom", generic))

    for p, path in paths:
        if path and os.path.exists(path):
            return p, path

    return None, None


# --- Helper for single-point sampling with provider resolution
def query_one(provider_hint: str | None, lat: float, lon: float):
    """Resolve provider, sample altitude at (lat, lon) and return (altitude, provider_used).
    Raises RuntimeError if no MNT is available."""
    provider_used, mnt_path = resolve_mnt_path(provider_hint)
    if not mnt_path:
        raise RuntimeError("No MNT available")
    with rasterio.open(mnt_path) as mnt:
        transformer = Transformer.from_crs("EPSG:4326", mnt.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        val = next(mnt.sample([(x, y)]), [None])
        alt = float(val[0]) if val and val[0] is not None else None
    return alt, provider_used

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

def ema1d(values: np.ndarray, alpha: float) -> np.ndarray:
    """Simple EMA over a 1D array. NaNs are forward-filled before EMA, and NaNs at start are set to first finite."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x
    # forward-fill NaNs
    mask = np.isfinite(x)
    if not mask.any():
        return np.zeros_like(x)
    first = np.argmax(mask)
    x[:first] = x[mask][0]
    for i in range(first + 1, x.size):
        if not np.isfinite(x[i]):
            x[i] = x[i - 1]
    y = np.empty_like(x)
    y[first] = x[first]
    for i in range(first + 1, x.size):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    # restore original NaNs (keep EMA where we filled to maintain continuity)
    return y

def sample_altitudes(df: pd.DataFrame, mnt_path: str) -> list[float | None]:
    """Batch sample altitudes from raster for a DataFrame with lat, lon columns."""
    with rasterio.open(mnt_path) as mnt:
        transformer = Transformer.from_crs("EPSG:4326", mnt.crs, always_xy=True)
        xy = [transformer.transform(lon, lat) for lat, lon in zip(df['lat'], df['lon'])]
        vals = list(mnt.sample(xy))
        return [float(v[0]) if v and v[0] is not None else None for v in vals]

def slope_window_percent(distance_m: np.ndarray,
                         altitude_m: np.ndarray,
                         window_m: float,
                         clip_pct: float = 30.0) -> np.ndarray:
    """
    Compute slope (%) for each point using a centered distance window of size `window_m`.
    For point i, we pick indices l and r such that:
      distance_m[l] >= distance_m[i] - window_m/2
      distance_m[r] <= distance_m[i] + window_m/2
    slope[i] = 100 * (altitude[r] - altitude[l]) / (distance[r] - distance[l])
    If denominator <= 0.5 m or missing altitudes, returns NaN. Result is clipped to ±clip_pct.
    """
    n = len(distance_m)
    half = 0.5 * float(window_m)
    dist = np.asarray(distance_m, dtype=np.float64)
    alt = np.asarray(altitude_m, dtype=np.float64)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    l = 0
    r = 0
    for i in range(n):
        center = dist[i]
        # move left bound
        left_bound = center - half
        while l + 1 <= i and dist[l] < left_bound:
            l += 1
        # move right bound
        right_bound = center + half
        if r < i:
            r = i
        while r + 1 < n and dist[r + 1] <= right_bound:
            r += 1
        if r <= l:
            continue
        dz_left = alt[l]
        dz_right = alt[r]
        if not (np.isfinite(dz_left) and np.isfinite(dz_right)):
            continue
        dd = dist[r] - dist[l]
        if dd <= 0.5:  # avoid tiny denominators
            continue
        out[i] = 100.0 * (dz_right - dz_left) / dd
    # clip extreme values
    if np.isfinite(clip_pct) and clip_pct > 0:
        out = np.clip(out, -abs(clip_pct), abs(clip_pct))
    return out

def enrich_with_terrain(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    mnt_path = (config or {}).get("mnt_path") or CONFIG.get("mnt_path", "/data/srtm/normandy_l93.tif")

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

@app.route('/health', methods=['GET'])
def health():
    provs = CONFIG.get("providers", {}) or {}
    availability = {k: (bool(v) and os.path.exists(v)) for k, v in provs.items()}
    return jsonify({
        "status": "ok",
        "mnt_path": CONFIG.get("mnt_path"),
        "providers": provs,
        "available": availability
    })

@app.route('/sample', methods=['GET'])
def sample_point():
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
    except Exception:
        return jsonify({"error": "Provide numeric lat & lon query parameters"}), 400

    provider = request.args.get("provider")
    try:
        alt, provider_used = query_one(provider, lat, lon)
        return jsonify({"lat": lat, "lon": lon, "altitude": alt, "provider_used": provider_used})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"lat": lat, "lon": lon, "altitude": None, "error": str(e)}), 500


# --- Batch sampling endpoint
@app.route('/sample_batch', methods=['POST'])
def sample_batch():
    """
    Batch altitude sampling.
    Body JSON can be either:
      {"points": [{"lat": ..., "lon": ...}, ...], "provider": "rge1|rge5|srtm30|srtm90"}
    or directly a list: [{"lat": ..., "lon": ...}, ...]
    Optional query param `provider` also accepted (overrides body).
    Returns: [{lat, lon, altitude, provider_used}, ...]
    """
    try:
        payload = request.get_json()
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    # Accept two shapes: dict with 'points' or a raw list
    points = None
    body_provider = None
    if isinstance(payload, dict):
        points = payload.get("points")
        body_provider = payload.get("provider")
    elif isinstance(payload, list):
        points = payload
    if not points or not isinstance(points, list):
        return jsonify({"error": "JSON must be a list of points or an object with a 'points' array"}), 400

    # Optional provider can come from query string or body (query overrides)
    provider = request.args.get("provider") or body_provider

    out = []
    for item in points:
        try:
            lat = float(item["lat"])  # may raise
            lon = float(item["lon"])  # may raise
            alt, used = query_one(provider, lat, lon)
            out.append({"lat": lat, "lon": lon, "altitude": alt, "provider_used": used})
        except Exception as e:
            # Attach error for this point but keep processing others
            try:
                lat_v = float(item.get("lat")) if isinstance(item, dict) and item.get("lat") is not None else None
                lon_v = float(item.get("lon")) if isinstance(item, dict) and item.get("lon") is not None else None
            except Exception:
                lat_v, lon_v = None, None
            out.append({"lat": lat_v, "lon": lon_v, "altitude": None, "error": str(e)})

    return jsonify(out)

@app.route('/profile', methods=['POST'])
def profile_points():
    try:
        coords = request.get_json()
        if not coords or not isinstance(coords, list):
            return jsonify({"error": "JSON body must be a list of {lat, lon}"}), 400
        df = pd.DataFrame(coords)
        if not {"lat", "lon"}.issubset(df.columns):
            return jsonify({"error": "Each item must include 'lat' and 'lon'"}), 400
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    provider = request.args.get("provider")
    provider_used, mnt_path = resolve_mnt_path(provider)

    if not mnt_path:
        return jsonify({"error": "No MNT available"}), 500

    try:
        with rasterio.open(mnt_path) as mnt:
            transformer = Transformer.from_crs("EPSG:4326", mnt.crs, always_xy=True)
            xy = [transformer.transform(lon, lat) for lat, lon in zip(df['lat'], df['lon'])]
            vals = list(mnt.sample(xy))
            alts = [float(v[0]) if v and v[0] is not None else None for v in vals]
        return jsonify({"altitude": alts, "provider_used": provider_used})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/slope', methods=['POST'])
def slope_api():
    """
    Compute altitude and slope on a stream of points.
    Body: list of {timestamp?, lat, lon}
    Query params:
      - window_m (float, default 30)
      - ema (bool, default true)
      - ema_alpha (float, default 0.25)
      - clip (float percent, default 30)
      - provider (str: rge1|rge5|srtm30|srtm90, optional)
    Returns per point: {timestamp, lat, lon, altitude, slope_pct, slope_pct_ema?, window_m, ema_alpha, clip_pct}
    """
    try:
        coords = request.get_json()
        if not coords or not isinstance(coords, list):
            return jsonify({"error": "JSON body must be a list of {lat, lon}"}), 400
        df = pd.DataFrame(coords)
        if not {"lat", "lon"}.issubset(df.columns):
            return jsonify({"error": "Each item must include 'lat' and 'lon'"}), 400
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    # Parameters
    try:
        window_m = float(request.args.get("window_m", 30.0))
    except Exception:
        window_m = 30.0
    ema_flag = request.args.get("ema", "true").lower() in {"1", "true", "yes", "y"}
    try:
        ema_alpha = float(request.args.get("ema_alpha", 0.25))
    except Exception:
        ema_alpha = 0.25
    try:
        clip_pct = float(request.args.get("clip", 30.0))
    except Exception:
        clip_pct = 30.0

    provider = request.args.get("provider")
    provider_used, mnt_path = resolve_mnt_path(provider)

    if not mnt_path:
        return jsonify({"error": "No MNT available"}), 500

    # Distance cumulative
    if 'distance_m' not in df.columns:
        df = compute_cumulative_distance(df)

    # Altitudes
    try:
        alts = sample_altitudes(df, mnt_path)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Sampling failed: {e}"}), 500
    df["altitude"] = alts

    # Slope by distance window
    slopes = slope_window_percent(df["distance_m"].to_numpy(),
                                  df["altitude"].to_numpy(),
                                  window_m=window_m,
                                  clip_pct=clip_pct)
    df["slope_pct"] = slopes

    # Optional EMA
    if ema_flag:
        df["slope_pct_ema"] = ema1d(slopes, alpha=ema_alpha)

    # Build response
    cols = ["timestamp", "lat", "lon", "altitude", "slope_pct"]
    if ema_flag:
        cols.append("slope_pct_ema")
    df["window_m"] = window_m
    df["ema_alpha"] = ema_alpha if ema_flag else None
    df["clip_pct"] = clip_pct
    cols += ["window_m", "ema_alpha", "clip_pct"]

    df["provider_used"] = provider_used
    cols.append("provider_used")

    return jsonify(df[cols].to_dict(orient="records"))

@app.route('/enrich_terrain', methods=['POST'])
def enrich_terrain_api():
    try:
        coords = request.get_json()
        if not coords or not isinstance(coords, list):
            return jsonify({"error": "Invalid JSON payload"}), 400

        df = pd.DataFrame(coords)

        if 'distance_m' not in df.columns:
            df = compute_cumulative_distance(df)

        config = {"mnt_path": CONFIG.get("mnt_path")}
        df = enrich_with_terrain(df, config)

        return jsonify(df.to_dict(orient='records'))

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="YAML config file (can include altitude.srtm.raster_path or mnt_path)")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5004)))
    args = parser.parse_args()

    load_config_from_file(args.config)
    print(f"[INFO] Service starting with MNT: {CONFIG.get('mnt_path')}")

    app.run(host=args.host, port=args.port)
