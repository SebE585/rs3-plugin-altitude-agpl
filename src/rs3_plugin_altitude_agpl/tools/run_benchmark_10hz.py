#!/usr/bin/env python3
import os, math, argparse, statistics as stats
import pandas as pd
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
import folium
import json

# --- smoothing
def ema1d(arr, alpha=0.2):
    """Simple exponential moving average (EMA) on a 1D array.
    - Keeps NaNs as gaps (does not update the state on NaNs).
    - Returns an array of same length.
    """
    out = np.full(len(arr), np.nan, dtype=float)
    s = None
    for i, v in enumerate(arr):
        if math.isfinite(v):
            s = v if s is None else (alpha * v + (1.0 - alpha) * s)
            out[i] = s
        else:
            # preserve gap; do not carry the previous value into NaN stretches
            out[i] = np.nan
    return out

def speeds_mps_from_ts_lat_lon(timestamps, lats, lons, default_dt=0.1):
    """Compute instantaneous speed (m/s) using consecutive haversine distance and timestamps.
    timestamps: array-like of pandas Timestamps or strings; if missing/invalid, fallback to default_dt seconds (10 Hz).
    Returns a float numpy array of same length as lats/lons.
    """
    n = len(lats)
    out = np.zeros(n, dtype=float)
    # parse timestamps to seconds
    try:
        ts = pd.to_datetime(timestamps)
    except Exception:
        ts = pd.Series([pd.NaT] * n)
    R = 6371000.0
    for i in range(1, n):
        dt = (ts.iloc[i] - ts.iloc[i-1]).total_seconds() if (pd.notna(ts.iloc[i]) and pd.notna(ts.iloc[i-1])) else default_dt
        if not (dt and dt > 0):
            dt = default_dt
        d = haversine_m(lats[i-1], lons[i-1], lats[i], lons[i])
        out[i] = d / dt
    return out

def write_outliers_geojson(lats, lons, timestamps, series_a, series_b, threshold_abs, out_path):
    """Write a GeoJSON of points where |series_a - series_b| > threshold_abs.
    Returns the number of outliers written.
    """
    feats = []
    for la, lo, t, a, b in zip(lats, lons, timestamps, series_a, series_b):
        if math.isfinite(a) and math.isfinite(b) and math.isfinite(la) and math.isfinite(lo):
            d = abs(a - b)
            if d > threshold_abs:
                feats.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(lo), float(la)]},
                    "properties": {
                        "timestamp": None if t is None else str(t),
                        "delta_abs_%": round(float(d), 3),
                        "a_%": round(float(a), 3),
                        "b_%": round(float(b), 3)
                    }
                })
    with open(out_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    return len(feats)

# ---------- utils
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def cumdist_m(lats, lons):
    d = [0.0]
    for i in range(1, len(lats)):
        d.append(d[-1] + haversine_m(lats[i-1], lons[i-1], lats[i], lons[i]))
    return np.array(d)

def sample_raster(path, lats, lons):
    if not path:
        return np.full(len(lats), np.nan, dtype=float)
    try:
        with rasterio.open(path) as ds:
            # rasterio attend (x=lon, y=lat)
            pts = list(zip(lons, lats))
            vals = [v[0] for v in ds.sample(pts)]
            nodata = ds.nodata
            out = []
            for v in vals:
                if v is None:
                    out.append(np.nan)
                else:
                    if nodata is not None and v == nodata:
                        out.append(np.nan)
                    else:
                        out.append(float(v))
            return np.array(out, dtype=float)
    except RasterioIOError:
        return np.full(len(lats), np.nan, dtype=float)

def slope_window_percent(dist_m, z, win_m=20.0, clip=30.0):
    """Pente en % par point via différence centrée sur une fenêtre de distance.
       slope[i] = 100*(z[j2]-z[j1])/(dist[j2]-dist[j1]) avec j1/j2 extrêmes dans [i-win/2, i+win/2] en distance."""
    n = len(z)
    out = np.full(n, np.nan, dtype=float)
    half = win_m / 2.0
    j1 = 0
    for i in range(n):
        # avancer j1 pour être à >= dist[i]-half
        while j1 < i and dist_m[i] - dist_m[j1] > half:
            j1 += 1
        # reculer j2 depuis la fin pour être à <= dist[i]+half
        j2 = i
        while j2 + 1 < n and dist_m[j2 + 1] - dist_m[i] <= half:
            j2 += 1
        if j2 > j1 and math.isfinite(z[j1]) and math.isfinite(z[j2]):
            dz = z[j2] - z[j1]
            ds = dist_m[j2] - dist_m[j1]
            if ds > 0:
                s = 100.0 * dz / ds
                # clip robuste (évite extrêmes aberrants)
                s = max(-clip, min(clip, s))
                out[i] = s
    return out

def robust_stats(a, b, clip=30.0):
    D = [x - y for (x, y) in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if not D:
        return dict(n=0, MAE=np.nan, RMSE=np.nan, P95=np.nan, MAX=np.nan)
    Dclip = [max(-clip, min(clip, d)) for d in D]
    n = len(Dclip)
    mae = sum(abs(d) for d in Dclip) / n
    rmse = math.sqrt(sum(d*d for d in Dclip) / n)
    p95 = np.percentile([abs(d) for d in Dclip], 95)
    mxx = max(abs(d) for d in Dclip)
    return dict(n=n, MAE=mae, RMSE=rmse, P95=p95, MAX=mxx)

def color_for_delta(val, vmax=30.0):
    # 0% = vert → 15% = jaune → 30% = rouge
    x = max(0.0, min(1.0, val / vmax))
    # simple gradient g->y->r via 2 segments
    if x <= 0.5:
        # green (0,180,0) -> yellow (255,255,0)
        t = x / 0.5
        r = int(0 + t * (255 - 0))
        g = int(180 + t * (255 - 180))
        b = 0
    else:
        # yellow (255,255,0) -> red (255,0,0)
        t = (x - 0.5) / 0.5
        r = 255
        g = int(255 - t * 255)
        b = 0
    return f"#{r:02x}{g:02x}{b:02x}"

# ---------- main
def main():
    ap = argparse.ArgumentParser(description="Benchmark pentes 10Hz multi-fenêtres + carte")
    ap.add_argument("--points", required=True, help="CSV avec colonnes timestamp,lat,lon")
    ap.add_argument("--windows", default="10,20,30", help="fenêtres distance en mètres séparées par des virgules")
    ap.add_argument("--outdir", default="bench_out", help="dossier de sortie")
    ap.add_argument("--clip", type=float, default=30.0, help="clip absolu (%%) pour les stats et la carte")
    ap.add_argument("--no-map", action="store_true", help="désactive la génération des cartes Folium (accélère fortement)")
    ap.add_argument("--map-max", type=int, default=20000, help="nb max de points affichés sur chaque carte (échantillonnage si >)")
    ap.add_argument("--ema", action="store_true", help="applique un lissage EMA aux séries de pentes (colonnes *_ema_%)")
    ap.add_argument("--ema-alpha", type=float, default=0.2, help="alpha du lissage EMA (0<alpha<=1), défaut=0.2")
    ap.add_argument("--export-outliers", action="store_true", help="Exporte un GeoJSON des écarts de pente > seuil pour une paire choisie")
    ap.add_argument("--outliers-pair", default="rge5_ema", choices=["rge5","srtm30","srtm90","rge5_ema","srtm30_ema","srtm90_ema"], help="Paire comparée à RGE1 (optionnellement avec EMA)")
    ap.add_argument("--outliers-thresh", type=float, default=5.0, help="Seuil absolu d'écart de pente en %% pour l'export GeoJSON (défaut=5)")
    ap.add_argument("--export-speed-bins", action="store_true", help="Exporte un CSV de stats |Δ| par classes de vitesse pour la même paire que --outliers-pair")
    ap.add_argument("--speed-bins", default="0,2,5,10,15,25,40", help="Borniers de vitesses m/s séparés par des virgules (ex: '0,2,5,10,15,25,40')")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.points)
    if not {"lat","lon"}.issubset(df.columns):
        raise SystemExit("Le CSV doit contenir les colonnes: lat, lon (timestamp facultatif).")

    lats = df["lat"].astype(float).to_numpy()
    lons = df["lon"].astype(float).to_numpy()
    ts   = df["timestamp"].astype(str).to_numpy() if "timestamp" in df.columns else np.array([None]*len(lats))
    dist = cumdist_m(lats, lons)
    speeds = speeds_mps_from_ts_lat_lon(ts, lats, lons, default_dt=0.1)

    # Altitudes
    prov_paths = {
        "rge1"  : os.getenv("RS3_RGEALTI_1M_COG"),
        "rge5"  : os.getenv("RS3_RGEALTI_5M_COG"),
        "srtm30": os.getenv("RS3_SRTM30_COG"),
        "srtm90": os.getenv("RS3_SRTM90_COG"),
    }
    alts = {k: sample_raster(v, lats, lons) for k, v in prov_paths.items()}

    # Boucle fenêtres
    wins = [float(x) for x in args.windows.split(",")]
    all_stats_rows = []
    for W in wins:
        slopes = {k: slope_window_percent(dist, z, win_m=W, clip=args.clip) for k, z in alts.items()}

        # Optionnel: lissage EMA des pentes
        slopes_ema = {}
        if args.ema:
            for k, arr in slopes.items():
                slopes_ema[k] = ema1d(arr, alpha=args.ema_alpha)

        # Sauvegarde CSV par fenêtre
        out_cols = {
            "timestamp": ts,
            "lat": lats, "lon": lons,
            "z_rge1": alts["rge1"], "z_rge5": alts["rge5"], "z_srtm30": alts["srtm30"], "z_srtm90": alts["srtm90"],
            "slope_rge1_%": slopes["rge1"], "slope_rge5_%": slopes["rge5"],
            "slope_srtm30_%": slopes["srtm30"], "slope_srtm90_%": slopes["srtm90"],
        }
        if args.ema:
            out_cols.update({
                "slope_rge1_ema_%": slopes_ema["rge1"],
                "slope_rge5_ema_%": slopes_ema["rge5"],
                "slope_srtm30_ema_%": slopes_ema["srtm30"],
                "slope_srtm90_ema_%": slopes_ema["srtm90"],
            })
        pd.DataFrame(out_cols).to_csv(os.path.join(args.outdir, f"bench_slopes_{int(W)}m.csv"), index=False)

        # Optionnel: export GeoJSON des outliers pour une paire choisie
        if args.export_outliers:
            pair = args.outliers_pair
            use_ema = pair.endswith("_ema")
            key = pair.replace("_ema", "")
            A = slopes_ema["rge1"] if (args.ema and use_ema) else slopes["rge1"]
            B = (slopes_ema[key] if (args.ema and use_ema) else slopes[key])
            out_geo = os.path.join(args.outdir, f"outliers_rge1_{pair}_{int(W)}m.geojson")
            n_out = write_outliers_geojson(lats, lons, ts, A, B, args.outliers_thresh, out_geo)
            print(f"[OUTLIERS] {pair} (W={int(W)}m) -> seuil>{args.outliers_thresh:.1f}% | n={n_out} | {out_geo}")

        # Optionnel: stats |Δ| par classe de vitesse
        if args.export_speed_bins:
            pair = args.outliers_pair
            use_ema = pair.endswith("_ema")
            key = pair.replace("_ema", "")
            A = slopes_ema["rge1"] if (args.ema and use_ema) else slopes["rge1"]
            B = (slopes_ema[key] if (args.ema and use_ema) else slopes[key])
            delta = np.abs(A - B)
            # bins
            try:
                bins = [float(x) for x in args.speed_bins.split(",")]
            except Exception:
                bins = [0,2,5,10,15,25,40]
            bins = sorted(bins)
            cats = pd.cut(pd.Series(speeds), bins=bins, include_lowest=False)
            df_bins = pd.DataFrame({"speed_mps": cats, "delta_abs_pct": delta})
            df_bins = df_bins.dropna(subset=["speed_mps","delta_abs_pct"])
            grp = df_bins.groupby("speed_mps")
            out_df = grp["delta_abs_pct"].agg(count="count", median="median", mean="mean")
            # P95 per bin
            p95 = grp["delta_abs_pct"].quantile(0.95).rename("p95")
            out_df = out_df.join(p95)
            out_csv = os.path.join(args.outdir, f"speed_bins_rge1_{pair}_{int(W)}m.csv")
            out_df.to_csv(out_csv)
            print(f"[SPEED-BINS] {pair} (W={int(W)}m) -> {out_csv}")

        # Stats vs RGE1
        for other in ("rge5","srtm30","srtm90"):
            st = robust_stats(slopes["rge1"], slopes[other], clip=args.clip)
            st.update(dict(window_m=W, pair=f"rge1_vs_{other}"))
            all_stats_rows.append(st)

            # Carte Leaflet (Δ pente vs RGE1) — optionnelle et échantillonnée
            if not args.no_map:
                A = slopes["rge1"]; B = slopes[other]
                deltas = np.array([abs(a-b) if (math.isfinite(a) and math.isfinite(b)) else np.nan for a,b in zip(A,B)])
                finite_mask = np.isfinite(deltas)
                if not np.any(finite_mask):
                    continue
                lats_f = lats[finite_mask]
                lons_f = lons[finite_mask]
                deltas_f = deltas[finite_mask]

                # Échantillonnage si trop de points (rendue Folium très coûteuse sinon)
                npts = len(deltas_f)
                if npts > args.map_max:
                    # indices réguliers pour préserver l'uniformité spatiale approximative
                    idx = np.linspace(0, npts - 1, args.map_max, dtype=int)
                    lats_f = lats_f[idx]
                    lons_f = lons_f[idx]
                    deltas_f = deltas_f[idx]

                # centre carte sur la médiane
                lat0 = float(np.nanmedian(lats_f)); lon0 = float(np.nanmedian(lons_f))
                m = folium.Map(location=[lat0, lon0], zoom_start=13, control_scale=True, tiles="OpenStreetMap")
                for la, lo, d in zip(lats_f, lons_f, deltas_f):
                    folium.CircleMarker(
                        location=[la, lo],
                        radius=2.5,
                        weight=0,
                        fill=True,
                        fill_opacity=0.8,
                        color=color_for_delta(min(d, args.clip), vmax=args.clip),
                        popup=f"Δ|pente|={d:.2f}%"
                    ).add_to(m)
                html_path = os.path.join(args.outdir, f"map_delta_{other}_{int(W)}m.html")
                m.save(html_path)

        if args.ema:
            for other in ("rge5", "srtm30", "srtm90"):
                st_ema = robust_stats(slopes_ema["rge1"], slopes_ema[other], clip=args.clip)
                st_ema.update(dict(window_m=W, pair=f"rge1_vs_{other}_ema"))
                all_stats_rows.append(st_ema)

    # Agrégat stats
    stats_csv = os.path.join(args.outdir, "summary_stats.csv")
    pd.DataFrame(all_stats_rows)[["pair","window_m","n","MAE","RMSE","P95","MAX"]].to_csv(stats_csv, index=False)

    # Affichage console résumé
    for row in all_stats_rows:
        print(f"{row['pair']} (distwin {row['window_m']} m) -> n={row['n']} | MAE={row['MAE']:.2f}% | "
              f"RMSE={row['RMSE']:.2f}% | P95|Δ|={row['P95']:.2f}% | max|Δ| (clippé)={row['MAX']:.2f}%")

    if args.no_map:
        print("[INFO] Cartes Folium désactivées (--no-map).")
    else:
        print(f"[INFO] Limite d'affichage cartes: {args.map_max} points par carte.")

    print(f"\nOK -> {args.outdir}")

if __name__ == "__main__":
    main()