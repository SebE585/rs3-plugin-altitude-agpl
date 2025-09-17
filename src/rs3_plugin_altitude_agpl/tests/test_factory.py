# -*- coding: utf-8 -*-
import types
import tempfile
import textwrap
import importlib
import pytest

def test_build_service_from_yaml_with_mocks(monkeypatch):
    # On crée deux faux providers qui respectent l'interface (pas d'ouverture de fichiers).
    class FakeIGN:
        def __init__(self, path, nodata=-9999.0): self.path = path; self.nodata = nodata
        def sample(self, lats, lons): return __import__("numpy").zeros(len(lats), dtype="f4")
        def get_rasterio_dataset(self): raise RuntimeError("not used in this test")
        def close(self): pass

    class FakeSRTM:
        def __init__(self, path): self.path = path
        def sample(self, lats, lons): return __import__("numpy").ones(len(lats), dtype="f4")
        def get_rasterio_dataset(self): raise RuntimeError("not used in this test")
        def close(self): pass

    from rs3_plugin_altitude_agpl import factory
    # Monkeypatch les providers dans factory
    monkeypatch.setattr(factory, "IGNRGEAltiProvider", FakeIGN, raising=True)
    monkeypatch.setattr(factory, "SRTMProvider", FakeSRTM, raising=True)

    # YAML minimal
    yaml_text = textwrap.dedent("""
    altitude:
      source: ign
      ign:
        cog_path: /fake/haute_normandie_cog.tif
      nodata: -9999.0

    altitude_pipeline:
      mode: track_first
      sg_window_m: 21
      sg_poly: 3
      res_m: 5
      sample_step_m: 1.0
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=True) as tf:
        tf.write(yaml_text)
        tf.flush()
        svc = factory.build_service_from_yaml(tf.name)

    # Sanity: l'instance existe et expose les méthodes publiques
    assert hasattr(svc, "altitude_at")
    assert hasattr(svc, "profile_and_grade")