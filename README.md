

# rs3-plugin-altitude-agpl

Plugin **Altitude** pour [RoadSimulator3](https://github.com/SebE585/RoadSimulator3).  
Sous licence **AGPL-3.0-only**.

---

## 📌 Description

Ce plugin ajoute l’enrichissement **altitude** (`altitude_m`) aux trajectoires simulées RS3 en interrogeant des sources externes (ex. **SRTM**, **IGN**).  
Il s’intègre automatiquement au pipeline via le mécanisme de plugins (`entry_points`).

---

## 🚀 Installation

```bash
git clone https://github.com/SebE585/rs3-plugin-altitude-agpl.git
cd rs3-plugin-altitude-agpl
pip install -e .
```

Vérifier que le plugin est bien détecté :

```bash
python -c "import importlib.metadata as m; print(m.entry_points().select(group='rs3.plugins'))"
```

---

## ⚙️ Utilisation

Une fois installé, il est automatiquement chargé par RS3 :

```bash
python -m runner.run_simulation
```

Sortie attendue :

```
INFO:core.plugins.loader:[PLUGINS] chargé: rs3-plugin-altitude-agpl (AGPL-3.0-only)
INFO:rs3_plugin_altitude_agpl.plugin:[ALT] altitude_m valorisée sur 81750 points.
```

Les colonnes ajoutées au dataset :

- `altitude_m` : altitude orthométrique (mètres).
- `distance_m` : distance cumulée (mètres).
- `slope_percent` : pente instantanée (%).

---

## 📂 Arborescence

```
src/rs3_plugin_altitude_agpl/
 ├── __init__.py
 ├── plugin.py
 ├── altitude_enricher.py
 ├── core_distance.py
 ├── config/defaults.yaml
 └── schema_fragments/altitude.yaml
```

---

## 🔑 Licence

Ce projet est distribué sous licence **AGPL-3.0-only**.  
Voir le fichier [LICENSE](LICENSE).