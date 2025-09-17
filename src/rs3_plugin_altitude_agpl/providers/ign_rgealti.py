# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence, Optional
import importlib
import numpy as np
import warnings
import os

from .base import DatasetNotOpen, AltitudeOutOfBounds

# Imports via importlib pour éviter les hard-deps au chargement
rasterio = importlib.import_module("rasterio")
WarpedVRT = importlib.import_module("rasterio.vrt").WarpedVRT
Resampling = importlib.import_module("rasterio.enums").Resampling
CRS = importlib.import_module("rasterio.crs").CRS


class IGNRGEAltiProvider:
    """
    Provider IGN RGE ALTI (COG).

    - Ouvre le COG (souvent en EPSG:2154 / Lambert-93).
    - Si le CRS source != EPSG:4326 (WGS84), crée un WarpedVRT (bilinear) vers WGS84.
    - Échantillonne en WGS84 avec ds.sample([(lon, lat), ...]).
    """

    def __init__(self, cog_path: str, nodata: float = -9999.0):
        self.path = cog_path
        self._src = rasterio.open(self.path)
        self._vrt = None
        # Debug facultatif (activez RS3_ALT_DEBUG=1)
        self._debug = bool(int(os.environ.get("RS3_ALT_DEBUG", "0"))) if "os" in globals() else False
        # nodata préférée = celle du fichier s’il existe, sinon valeur passée
        self._nodata: Optional[float] = (
            self._src.nodata if self._src.nodata is not None else nodata
        )

        # Reprojection on-the-fly vers WGS84 si nécessaire
        try:
            if self._src.crs and self._src.crs != CRS.from_epsg(4326):
                self._vrt = WarpedVRT(
                    self._src,
                    dst_crs=CRS.from_epsg(4326),
                    resampling=Resampling.bilinear,
                )
                if getattr(self, "_debug", False):
                    try:
                        print(f"[IGN RGE] src.crs={self._src.crs}, vrt={(self._vrt is not None)}")
                        b = (self._dataset().bounds if self._dataset() is not None else None)
                        print(f"[IGN RGE] ds.bounds={b}")
                    except Exception:
                        pass
        except Exception:
            # si WarpedVRT indisponible, on reste sur _src (échantillonnage échouera si CRS != 4326)
            self._vrt = None

    @property
    def nodata(self) -> Optional[float]:
        return self._nodata

    def _dataset(self):
        return self._vrt if self._vrt is not None else self._src

    def sample(self, lats: Sequence[float], lons: Sequence[float]) -> np.ndarray:
        ds = self._dataset()
        if ds is None:
            raise DatasetNotOpen("IGN RGEALTI dataset is not opened.")
        # rasterio attend (x=lon, y=lat) en WGS84
        lats = np.atleast_1d(np.asarray(lats, dtype=float))
        lons = np.atleast_1d(np.asarray(lons, dtype=float))
        vals = np.full(lats.shape, np.nan, dtype=np.float32)

        try:
            left, bottom, right, top = ds.bounds
        except Exception:
            left = bottom = right = top = None

        if left is not None:
            inb = (lons >= left) & (lons <= right) & (lats >= bottom) & (lats <= top)
        else:
            # si bounds non dispo, on tente tout
            inb = np.ones_like(lats, dtype=bool)

        idx = np.nonzero(inb)[0]
        if idx.size:
            pts = list(zip(lons[idx], lats[idx]))
            fetched = np.fromiter((v[0] for v in ds.sample(pts)), dtype=np.float32, count=len(pts))
            # map NoData -> NaN
            if self._nodata is not None and not np.isnan(self._nodata):
                fetched = np.where(np.isclose(fetched, self._nodata), np.nan, fetched)
            vals[idx] = fetched

        # Si tout est out-of-bounds, remonter une erreur explicite
        if not np.any(inb):
            raise AltitudeOutOfBounds("All requested points are outside dataset bounds (CRS/bounds mismatch).")

        return vals

    def get_rasterio_dataset(self):
        return self._dataset()

    def close(self) -> None:
        try:
            if self._vrt is not None:
                self._vrt.close()
        finally:
            self._vrt = None
        try:
            if self._src is not None:
                self._src.close()
        finally:
            self._src = None