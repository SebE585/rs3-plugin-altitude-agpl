# -*- coding: utf-8 -*-
import os
import pytest
import numpy as np

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def RGE_present():
    p = os.environ.get("RS3_RGEALTI_COG")
    return bool(p) and os.path.exists(p)

def test__window_pts_from_m_basic():
    import numpy as np
    from rs3_plugin_altitude_agpl.altitude_service import _window_pts_from_m
    # s de 0 à 100 m, pas ~5 m => 21 points
    s = np.linspace(0, 100, 21)
    # target 20 m, poly 3 => au moins 5, impaire
    k = _window_pts_from_m(s, 20.0, 3)
    assert k % 2 == 1 and k >= 5
    # target très grand => clamp à N impair
    k2 = _window_pts_from_m(s, 10_000.0, 3)
    assert k2 <= len(s) and (k2 % 2 == 1)
    # target <=0 => fallback min impaire > poly
    k3 = _window_pts_from_m(s, -1.0, 4)
    assert k3 % 2 == 1 and k3 >= 6


@pytest.mark.skipif(not RGE_present(), reason="RS3_RGEALTI_COG non défini ou fichier absent")
def test_altitude_service_fallback_two_points():
    """Avec seulement 2 points, la pente doit être calculée via gradient (fallback), sans crash."""
    from rs3_plugin_altitude_agpl.providers import IGNRGEAltiProvider
    from rs3_plugin_altitude_agpl import AltitudeService, AltitudePipelineConfig

    prov = IGNRGEAltiProvider(os.environ["RS3_RGEALTI_COG"], nodata=-9999.0)
    svc = AltitudeService(prov, AltitudePipelineConfig(mode="track_first", sg_window_m=21, sg_poly=3))

    # Deux points en Haute-Normandie (ex: Louviers -> Pacy)
    lats = [49.201000, 49.016000]
    lons = [1.167000, 1.382000]

    z = svc.altitude_at(lats, lons)
    assert z.shape == (2,)
    assert np.isfinite(z).all()

    out = svc.profile_and_grade(lats, lons)
    assert set(out.keys()) == {"z", "grade", "theta", "meta"}
    assert len(out["z"]) == len(lats)
    assert len(out["grade"]) == len(lats)
    # gradient sur 2 points => grade[0] et grade[1] doivent être finis
    assert np.isfinite(out["grade"]).all()


@pytest.mark.skipif(not RGE_present(), reason="RS3_RGEALTI_COG non défini ou fichier absent")
def test_altitude_service_savgol_six_points():
    """Avec 6 points, Savitzky–Golay doit passer (fenêtre bornée) et retourner des valeurs finies."""
    from rs3_plugin_altitude_agpl.providers import IGNRGEAltiProvider
    from rs3_plugin_altitude_agpl import AltitudeService, AltitudePipelineConfig

    prov = IGNRGEAltiProvider(os.environ["RS3_RGEALTI_COG"], nodata=-9999.0)
    svc = AltitudeService(prov, AltitudePipelineConfig(mode="track_first", sg_window_m=21, sg_poly=3))

    # Petite polyline (~6 points) autour de Louviers -> Val-de-Reuil -> Poses (sur voies)
    lats = [49.2052, 49.1995, 49.1918, 49.1856, 49.1789, 49.1730]
    lons = [ 1.1705,  1.1752,  1.1820,  1.1895,  1.1960,  1.2025]

    z = svc.altitude_at(lats, lons)
    assert z.shape == (6,)
    assert np.isfinite(z).all()

    out = svc.profile_and_grade(lats, lons)
    assert set(out.keys()) == {"z", "grade", "theta", "meta"}
    assert len(out["z"]) == len(lats)
    assert len(out["grade"]) == len(lats)
    assert np.isfinite(out["grade"]).all()
    assert np.isfinite(out["theta"]).all()