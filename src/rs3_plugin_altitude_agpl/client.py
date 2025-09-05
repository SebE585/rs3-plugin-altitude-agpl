import os
import json
from typing import Iterable, List, Dict, Any, Tuple
import requests
import pandas as pd
from core.terrain.distance import compute_cumulative_distance

# ============================================================
#  Client Altitude – mode "investigation"
#  - Timeout et base_url configurables (kwargs/env)
#  - Logs détaillés en cas d'erreur (status + body)
#  - Essais de plusieurs schémas de payload pour découvrir
#    celui attendu par l'API, sauf si ALT_PAYLOAD_MODE est fixé.
#  - Affiche un échantillon du payload envoyé.
# ============================================================

DEFAULT_BASE_URL = os.environ.get("RS3_ALTITUDE_BASE", "http://localhost:5004").rstrip("/")
DEFAULT_TIMEOUT_READ = float(os.environ.get("ALT_READ_TIMEOUT", "30.0"))
DEFAULT_TIMEOUT_CONNECT = float(os.environ.get("ALT_CONNECT_TIMEOUT", str(min(10.0, DEFAULT_TIMEOUT_READ))))
DEFAULT_TIMEOUT: Tuple[float, float] = (DEFAULT_TIMEOUT_CONNECT, DEFAULT_TIMEOUT_READ)

ENRICH_PATH = "/enrich_terrain"

# Liste des stratégies de sérialisation testées (dans l'ordre)
# Chacune est une (nom, callable(df) -> payload_dict, content_type)
def _payload_records(df: pd.DataFrame) -> Tuple[Dict[str, Any], str]:
  # format actuel (liste de dicts) – API serveur initiale supposée
  recs = df[["lat", "lon", "distance_m"]].to_dict(orient="records")
  return recs, "application/json"

def _payload_points_records(df: pd.DataFrame) -> Tuple[Dict[str, Any], str]:
  # points imbriqués
  recs = df[["lat", "lon", "distance_m"]].to_dict(orient="records")
  return {"points": recs}, "application/json"

def _payload_xy_records(df: pd.DataFrame) -> Tuple[Dict[str, Any], str]:
  # x=lon, y=lat (certains services l'attendent)
  recs = [{"x": float(lon), "y": float(lat), "d": float(dist)} for lat, lon, dist in df[["lat", "lon", "distance_m"]].itertuples(index=False)]
  return {"points": recs}, "application/json"

def _payload_geojson_linestring(df: pd.DataFrame) -> Tuple[Dict[str, Any], str]:
  coords = [[float(lon), float(lat)] for lat, lon in df[["lat", "lon"]].itertuples(index=False)]
  return {"type": "LineString", "coordinates": coords}, "application/json"

PAYLOAD_BUILDERS = [
  ("records", _payload_records),
  ("points_records", _payload_points_records),
  ("xy_records", _payload_xy_records),
  ("geojson", _payload_geojson_linestring),
]

def _post_with_debug(url: str, payload: Any, timeout: Tuple[float, float], content_type: str = "application/json") -> Any:
  headers = {"Content-Type": content_type, "Accept": "application/json"}
  try:
    print(f"[ALT] POST {url} timeout={timeout} content_type={content_type}")
    # échantillon payload
    try:
      if isinstance(payload, dict):
        sample = next(iter(payload.values())) if payload else None
      elif isinstance(payload, list):
        sample = payload[0] if payload else None
      else:
        sample = str(payload)[:200]
      print(f"[ALT] payload sample: {str(sample)[:200]}")
    except Exception:
      pass

    r = requests.post(url, json=payload if content_type == "application/json" else None,
                      data=None if content_type == "application/json" else payload,
                      headers=headers, timeout=timeout)
    status = r.status_code
    txt = r.text[:500].replace("\n", " ")
    print(f"[ALT] status={status} body~={txt}")
    r.raise_for_status()
    return r.json()
  except requests.Timeout as ex:
    raise RuntimeError(f"[ALT] Timeout while calling {url} with timeout={timeout}") from ex
  except requests.RequestException as ex:
    # remonter avec le corps de réponse pour découvrir le schéma attendu
    body = getattr(ex.response, "text", "") if hasattr(ex, "response") and ex.response is not None else ""
    body = (body or "")[:800].replace("\n", " ")
    raise RuntimeError(f"[ALT] HTTP error on {url}: {ex} — body={body}")

def enrich_terrain_via_api(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
  """
  Enrichit un DataFrame GPS avec les données d'altitude et de pente via API externe.

  Colonnes en entrée requises : 'lat', 'lon'
  Si 'distance_m' est absente, elle est calculée automatiquement.

  Paramètres (optionnels via kwargs ou env) :
    - base_url (str)                    (env: RS3_ALTITUDE_BASE)
    - timeout / timeout_s (float, sec)  (env: ALT_READ_TIMEOUT)
    - connect_timeout (float, sec)      (env: ALT_CONNECT_TIMEOUT)
    - payload_mode (str) in {'records','points_records','xy_records','geojson'}
        (env: ALT_PAYLOAD_MODE) — si absent, on essaie en cascade les 4.
  """
  if not {"lat", "lon"}.issubset(df.columns):
    raise ValueError("Le DataFrame doit contenir les colonnes 'lat' et 'lon'.")

  if "distance_m" not in df.columns:
    df = compute_cumulative_distance(df)

  base_url = (kwargs.get("base_url") or os.environ.get("RS3_ALTITUDE_BASE") or DEFAULT_BASE_URL).rstrip("/")

  # Timeout lecture
  t_read = float(
    kwargs.get("timeout")
    or kwargs.get("timeout_s")
    or kwargs.get("read_timeout")
    or os.environ.get("ALT_READ_TIMEOUT", DEFAULT_TIMEOUT_READ)
  )
  t_conn = float(
    kwargs.get("connect_timeout")
    or os.environ.get("ALT_CONNECT_TIMEOUT", min(10.0, t_read))
  )
  timeout = (t_conn, t_read)

  payload_mode = (kwargs.get("payload_mode") or os.environ.get("ALT_PAYLOAD_MODE") or "").strip()

  url = f"{base_url}{ENRICH_PATH}"
  print(f"[ALT] Using base_url={base_url}, timeout={timeout}s, payload_mode={payload_mode or 'auto'}")

  # Choisir les builders à tester
  builders = PAYLOAD_BUILDERS
  if payload_mode:
    # garder uniquement celui demandé
    builders = [b for b in PAYLOAD_BUILDERS if b[0] == payload_mode]
    if not builders:
      raise ValueError(f"[ALT] payload_mode inconnu: {payload_mode}")

  last_error = None
  for name, builder in builders:
    try:
      payload, ctype = builder(df)
      print(f"[ALT] Trying payload builder: {name}")
      data = _post_with_debug(url, payload, timeout, content_type=ctype)
      # Mapping attendu: liste d'objets renvoyés ou dict avec clés
      # On tente des clés standard; sinon, on lève pour inspection.
      if isinstance(data, dict):
        # cas où le serveur renvoie {"altitude":[...], "altitude_smoothed":[...], "slope_percent":[...]}
        if all(k in data for k in ("altitude", "altitude_smoothed", "slope_percent")):
          df["altitude"] = data["altitude"]
          df["altitude_smoothed"] = data["altitude_smoothed"]
          df["slope_percent"] = data["slope_percent"]
          return df
        # cas où le serveur renvoie {"points":[{...}]}
        if "points" in data and isinstance(data["points"], list):
          data = data["points"]
        else:
          # pas de format reconnu – on laisse filer pour inspection
          raise ValueError(f"Réponse dict non reconnue (clés={list(data.keys())[:8]})")

      if isinstance(data, list):
        # liste parallèle avec champs explicites
        try:
          df["altitude"] = [pt["altitude"] for pt in data]
          df["altitude_smoothed"] = [pt.get("altitude_smoothed", pt.get("altitude_smooth", None)) for pt in data]
          df["slope_percent"] = [pt.get("slope_percent", pt.get("slope", None)) for pt in data]
          return df
        except Exception as e:
          # Essayer un format compact: liste de scalaires ou de triplets
          # - [alt]
          # - [[alt, alt_smoothed, slope], ...]
          if all(isinstance(x, (int, float, type(None))) for x in data):
            df["altitude"] = data
            return df
          if all(isinstance(x, (list, tuple)) for x in data):
            # Essayons d'interpréter 3 colonnes
            alt = []
            alt_sm = []
            slope = []
            for row in data:
              alt.append(row[0] if len(row) > 0 else None)
              alt_sm.append(row[1] if len(row) > 1 else None)
              slope.append(row[2] if len(row) > 2 else None)
            df["altitude"] = alt
            df["altitude_smoothed"] = alt_sm
            df["slope_percent"] = slope
            return df
          raise

      # Si on arrive ici, format non géré → on lève pour passer au builder suivant
      raise ValueError(f"Format de réponse non géré: {type(data).__name__}")
    except Exception as ex:
      last_error = ex
      print(f"[ALT] Builder '{name}' failed: {ex}")
      continue

  # Aucun builder n'a fonctionné
  raise RuntimeError(f"[ALT] Impossible de parser la réponse de l'API. Dernière erreur: {last_error}")