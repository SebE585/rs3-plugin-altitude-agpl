# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence, Optional
import importlib
import numpy as np

from .base import DatasetNotOpen, AltitudeOutOfBounds

# Imports rasterio via importlib
rasterio = importlib.import_module("rasterio")
WarpedVRT = importlib.import_module("rasterio.vrt").WarpedVRT
Resampling = importlib.import_module("rasterio.enums").Resampling
CRS = importlib.import_module("rasterio.crs").CRS


class SRTMProvider:
    """
    Provider SRTM.
    - Accepte un GeoTIFF/VRT en EPSG:4326 (WGS84) OU dans un autre CRS (ex: EPSG:2154).
    - Si le CRS source != EPSG:4326, reprojection à la volée via WarpedVRT (bilinear).
    - Chemin via paramètre `raster_path`.
    """

    def __init__(self, raster_path: str, nodata: float = -32768.0):
        self.path = raster_path
        self._src = rasterio.open(self.path)
        self._vrt = None
        self._nodata: Optional[float] = self._src.nodata if self._src.nodata is not None else nodata

        try:
            if self._src.crs and self._src.crs != CRS.from_epsg(4326):
                self._vrt = WarpedVRT(
                    self._src,
                    dst_crs=CRS.from_epsg(4326),
                    resampling=Resampling.bilinear,
                )
        except Exception:
            # Si WarpedVRT indispo, on laisse le src direct (éventuelle erreur au sample si CRS != 4326)
            self._vrt = None

    @property
    def nodata(self) -> Optional[float]:
        return self._nodata

    def _dataset(self):
        return self._vrt if self._vrt is not None else self._src

    def sample(self, lats: Sequence[float], lons: Sequence[float]) -> np.ndarray:
        ds = self._dataset()
        if ds is None:
            raise DatasetNotOpen("SRTM dataset is not opened.")
        pts = list(zip(lons, lats))  # rasterio attend (x=lon, y=lat)
        vals = np.fromiter((v[0] for v in ds.sample(pts)), dtype=np.float32, count=len(pts))
        if vals.size == 0:
            raise AltitudeOutOfBounds("Requested points are outside raster bounds.")
        if self._nodata is not None and not np.isnan(self._nodata):
            vals = np.where(np.isclose(vals, self._nodata), np.nan, vals)
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