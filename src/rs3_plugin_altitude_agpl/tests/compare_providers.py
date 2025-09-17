# -*- coding: utf-8 -*-
import os
import csv
import math
import numpy as np

# Optional dependencies used for direct raster sampling
try:
    import rasterio
    from rasterio.warp import transform
except Exception as e:
    rasterio = None

# ---- Inputs -----------------------------------------------------------------

CSV = os.path.join(os.path.dirname(__file__), "data", "points_from_yaml.csv")

# Defaults point to the "cog" folder you showed in your terminal
RGE1_PATH   = os.environ.get("RS3_RGEALTI_1M_COG",   os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "elevation", "cog", "HauteNormandie_WGS84_COG.tif")))
RGE5_PATH   = os.environ.get("RS3_RGEALTI_5M_COG",   os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "elevation", "cog", "dem_5m_cog.tif")))
SRTM30_PATH = os.environ.get("RS3_SRTM30_COG",       os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "elevation", "cog", "srtm30_normandie_cog.tif")))
SRTM90_PATH = os.environ.get("RS3_SRTM90_COG",       os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "elevation", "cog", "srtm90_normandie_cog.tif")))

# ---- Minimal sampler ---------------------------------------------------------

class RasterSampler:
    """
    Tiny wrapper around rasterio to sample a DEM at (lat, lon) coordinates.
    - Accepts any DEM CRS (will reproject from EPSG:4326).
    - Uses bilinear resampling via read() with indexes=1 and rasterio.sample.
    - Maps nodata to NaN.
    """
    def __init__(self, path, nodata=None):
        if rasterio is None:
            raise RuntimeError("rasterio is required to use RasterSampler. Please `pip install rasterio`.")
        self.ds = rasterio.open(path)
        self.path = path
        self.nodata = self.ds.nodata if nodata is None else nodata
        # normalize nodata (some files use -9999, some None)
        self.has_nodata = self.nodata is not None

    def sample(self, lats, lons):
        # Transform from EPSG:4326 (lon, lat) to dataset CRS coordinates
        # rasterio.transform expects (x, y) = (lon, lat)
        xs, ys = transform("EPSG:4326", self.ds.crs, lons, lats)
        coords = list(zip(xs, ys))
        vals = []
        for val in self.ds.sample(coords):
            z = float(val[0])
            if self.has_nodata and (z == self.nodata):
                vals.append(float("nan"))
            else:
                # Some rasters encode sea as very negative; keep as is (comparison remains valid)
                vals.append(z)
        return np.array(vals, dtype="float64")

    def close(self):
        if self.ds:
            self.ds.close()

# ---- Helpers ----------------------------------------------------------------

def read_points(csv_path):
    lats, lons, labels = [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            lats.append(float(row["lat"]))
            lons.append(float(row["lon"]))
            labels.append(row.get("label", ""))
    return np.array(lats, dtype="float64"), np.array(lons, dtype="float64"), labels

def stats(name, ref, other):
    mask = ~np.isnan(ref) & ~np.isnan(other)
    if not mask.any():
        return f"[{name}] Aucun point comparable."
    d = ref[mask] - other[mask]
    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d**2)))
    mmax = float(np.max(np.abs(d)))
    return f"[{name}] Points comparables: {mask.sum()}/{len(mask)} | MAE={mae:.2f} m | RMSE={rmse:.2f} m | Max |Δ|={mmax:.2f} m"

def human(v):
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    return f"{v:.2f}"

def check_exists(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable pour {label}: {path}")

# ---- Main -------------------------------------------------------------------

def main():
    # Check inputs
    check_exists(CSV, "CSV de points")
    for p, lab in [
        (RGE1_PATH, "RGE 1 m"),
        (RGE5_PATH, "RGE 5 m"),
        (SRTM30_PATH, "SRTM30"),
        (SRTM90_PATH, "SRTM90"),
    ]:
        check_exists(p, lab)

    lats, lons, labels = read_points(CSV)

    # Open rasters
    rge1 = RasterSampler(RGE1_PATH, nodata=-9999.0)  # RGE 1 m: souvent -9999
    # Force nodata for RGE 5 m as well (some builds keep -9999 but the tag may be missing)
    rge5 = RasterSampler(RGE5_PATH, nodata=-9999.0)
    s30  = RasterSampler(SRTM30_PATH)                # SRTM30 COG
    s90  = RasterSampler(SRTM90_PATH)                # SRTM90 COG

    # Quick debug to verify coverage/CRS at runtime
    try:
        print(f"[DEBUG] RGE1  CRS={rge1.ds.crs} bounds={rge1.ds.bounds} nodata={rge1.nodata}")
        print(f"[DEBUG] RGE5  CRS={rge5.ds.crs} bounds={rge5.ds.bounds} nodata={rge5.nodata}")
        print(f"[DEBUG] SRTM30 CRS={s30.ds.crs} bounds={s30.ds.bounds} nodata={s30.nodata}")
        print(f"[DEBUG] SRTM90 CRS={s90.ds.crs} bounds={s90.ds.bounds} nodata={s90.nodata}")
    except Exception:
        pass

    # Sample
    z1 = rge1.sample(lats, lons)
    z5 = rge5.sample(lats, lons)
    z30 = s30.sample(lats, lons)
    z90 = s90.sample(lats, lons)

    # Print header
    print("\nLabel                             lat        lon        RGE1(m)   RGE5(m)   SRTM30(m)  SRTM90(m)   Δ1-5    Δ1-30   Δ1-90")
    print("-" * 120)
    for lab, la, lo, a, b, c, d in zip(labels, lats, lons, z1, z5, z30, z90):
        dd15  = float("nan") if (math.isnan(a) or math.isnan(b)) else (a - b)
        dd130 = float("nan") if (math.isnan(a) or math.isnan(c)) else (a - c)
        dd190 = float("nan") if (math.isnan(a) or math.isnan(d)) else (a - d)
        print(f"{lab:32s}  {la:9.6f}  {lo:9.6f}  {human(a):>8s}   {human(b):>8s}   {human(c):>8s}   {human(d):>8s}   "
              f"{'NaN' if math.isnan(dd15) else f'{dd15:+.2f}':>6s}  "
              f"{'NaN' if math.isnan(dd130) else f'{dd130:+.2f}':>6s}  "
              f"{'NaN' if math.isnan(dd190) else f'{dd190:+.2f}':>6s}")

    # Stats vs RGE 1 m
    print("\n== Statistiques vs RGE 1 m ==")
    print(stats("RGE5",  z1, z5))
    print(stats("SRTM30", z1, z30))
    print(stats("SRTM90", z1, z90))

    # Close
    rge1.close(); rge5.close(); s30.close(); s90.close()

if __name__ == "__main__":
    raise SystemExit(main())