# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Sequence, Protocol, Optional, Any
import numpy as np


# --- Exceptions harmonisées ---
class AltitudeError(Exception):
    """Erreur générique provider d'altitude."""


class AltitudeOutOfBounds(AltitudeError):
    """Requête hors emprise du raster."""


class DatasetNotOpen(AltitudeError):
    """Dataset sous-jacent non ouvert/fermé."""


class AltitudeProvider(Protocol):
    """
    Contrat minimal pour tout provider d'altitude.

    Implémentez cette interface dans IGNRGEAltiProvider, SRTMProvider, etc.
    Le protocole ne dépend pas de rasterio (imports lourds) pour rester léger.
    """

    nodata: Optional[float]

    def sample(self, lats: Sequence[float], lons: Sequence[float]) -> np.ndarray:
        """Retourne un array float32 (m), NaN pour NoData/hors emprise."""
        ...

    # Optionnel : certains providers exposent un dataset (ex: rasterio)
    def get_rasterio_dataset(self) -> Any:  # pragma: no cover - méthode facultative
        """Expose le dataset sous-jacent (lecture seule) si pertinent, sinon None."""
        return None

    def close(self) -> None:
        """Libère les ressources associées au provider (datasets, handles...)."""
        ...