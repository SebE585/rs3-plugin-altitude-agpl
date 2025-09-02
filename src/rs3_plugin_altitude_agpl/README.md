# Module core/terrain

## Présentation

Le module `core/terrain` gère l'enrichissement des trajectoires GPS avec des données terrain telles que l'altitude et la pente. Il permet d'ajouter des informations d'élévation précises à partir de fichiers MNT locaux ou de sources externes, facilitant ainsi l'analyse topographique des trajets.

## Structure

- `api.py` : Accès à des données d'altitude via des API externes.
- `local.py` : Chargement et traitement des fichiers MNT locaux (raster).
- `utils.py` : Fonctions utilitaires pour le traitement des données terrain.

## Fonctionnalités principales

- Chargement des altitudes à partir de MNT raster locaux ou d'API externes.
- Calcul du pourcentage de pente à partir des données d'altitude.
- Clip des valeurs de pente pour rester dans des plages réalistes.
- Validation des coordonnées GPS pour garantir leur cohérence.

## Exemple d’utilisation

```python
from core.terrain.local import enrich_with_mnt_local
from core.terrain.utils import compute_slope_percent

# Enrichissement des données GPS avec l'altitude locale
df = enrich_with_mnt_local(df)

# Calcul du pourcentage de pente
df = compute_slope_percent(df)
```

## Prérequis

- Bibliothèques Python : `rasterio`, `numpy`, `pandas`
- Fichiers MNT locaux au format GeoTIFF disponibles pour la région d'intérêt

## Tests

Des tests unitaires couvrent l'enrichissement en altitude et le calcul de la pente pour assurer la fiabilité des traitements.

## Roadmap

Améliorations envisagées :

- Mise en cache des altitudes pour optimiser les performances.
- Support de plusieurs sources de MNT pour une meilleure couverture.
- Traitement parallèle pour accélérer le calcul sur de grands jeux de données.
