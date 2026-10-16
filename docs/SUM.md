# SUM — Manuel utilisateur et procédure d'exploitation

**Projet :** Earth-ops · **Version :** 0.1

## 1. Installation

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest                      # 45 tests, doivent tous passer
```

Ou avec Docker :

```bash
docker build -t earth-ops .
docker run --rm -v "$PWD/data:/app/data" earth-ops run data/products/*
```

## 2. Procédure nominale (un cycle d'exploitation)

| Étape | Commande | Résultat attendu |
|---|---|---|
| 1. Ingestion | `python -m earth_ops ingest --bbox 1.30 43.52 1.46 43.63 --start 2025-01-01 --end 2025-12-31 --max-cloud 30 --limit 20` | Un dossier par produit dans `data/products/`, chacun avec `metadata.json` |
| 2. Traitement | `python -m earth_ops run data/products/* --chunks 1024` | `data/output/<produit>/NDVI.tif` + `quicklook.png`, code retour 0 |
| 3. Contrôle | consulter `data/kpi/runs.jsonl` | Une ligne par exécution, statut `NOMINAL` ou `DEGRADED` |
| 4. Surveillance | `python -m earth_ops anomalies` | Tableau trié par score d'anomalie |
| 5. Série temporelle | `python -m earth_ops series data/products/*` | `data/series/` : CSV, courbe NDVI, composite médian |
| 6. Mosaïque nationale | `python -m earth_ops mosaic` | `data/mosaic/ndvi_mosaic.tif` + `.png` : France, Lambert-93, 500 m (≈ 10 min, réseau uniquement) |

Sans réseau, l'étape 1 peut être remplacée par `python -m earth_ops simulate --count 10`.

## 3. Statuts d'une exécution

| Statut | Signification | Produit livré | Code retour CLI |
|---|---|---|---|
| `NOMINAL` | Produit valide, taux de pixels valides ≥ seuil (0,5) | Oui | 0 |
| `DEGRADED` | Produit livré mais trop nuageux pour être fiable | Oui | 0 |
| `REJECTED` | Entrée invalide, refusée par la validation | Non | 1 |
| `FAILED` | Erreur inattendue pendant le traitement | Non | 1 |

## 4. Codes d'erreur et actions opérateur

| Code | Cause | Action |
|---|---|---|
| `E_MISSING_FILE` | Bande ou `metadata.json` absent (ingestion interrompue) | Supprimer le dossier du produit et relancer `ingest` |
| `E_METADATA` | `metadata.json` illisible, incomplet, `scale`/`offset` non finis, date invalide | Vérifier la source STAC ; relancer l'ingestion |
| `E_CORRUPT` | GeoTIFF illisible | Supprimer le produit et le ré-ingérer |
| `E_GEOMETRY` | Bandes sur des grilles différentes | Ré-ingérer ; signaler si persistant (anomalie de rééchantillonnage) |
| `E_CRS` | CRS non projeté | Vérifier la source ; le traitement suppose une grille UTM |
| `E_DTYPE` | Type de données inattendu | Vérifier la source (changement de format amont ?) |
| `E_EMPTY` | > 99 % de nodata sur l'emprise (bord de fauchée) | Normal pour certaines orbites : ignorer le produit |
| `E_RADIOMETRY` | Réflectance médiane négative : `scale`/`offset` déclarés incohérents avec les données | Ne pas corriger à la main. Vérifier `boa_offset_applied` dans `metadata.json` et la source ; signaler au fournisseur (cf. SVR, NC-01/NC-02) |
| `E_PROCESSING` | Exception inattendue (mémoire, disque, bug) | Lire le message dans le KPI et les logs ; relancer avec `--chunks 1024` si mémoire insuffisante |

## 5. Supervision

- **Logs** : sortie standard, une ligne par début/fin d'étape, préfixée par l'identifiant d'exécution (`run_id`).
- **KPI** (`data/kpi/runs.jsonl`), un objet JSON par exécution :

```json
{"run_id": "3f9c2a1b", "product_id": "S2A_31TCJ_20250614_0_L2A", "started_at": "2026-10-16T09:00:00+00:00",
 "status": "NOMINAL", "stages_s": {"validate": 0.01, "process": 0.42, "quicklook": 0.15},
 "total_s": 0.58, "chunks": 1024, "pixels": 1574770, "throughput_mpx_s": 2.7,
 "valid_ratio": 0.97, "cloud_ratio": 0.01, "ndvi_mean": 0.41, "ndvi_std": 0.19, "ndvi_min": -0.62, "ndvi_max": 0.93}
```

- **Anomalies** : `anomalies` entraîne une Isolation Forest sur l'historique (au moins 8 exécutions terminées). Un drapeau `YES` n'est pas une panne : c'est une exécution à examiner (durée inhabituelle, scène atypique, configuration erronée). Par construction (contamination 0,1), environ 10 % des exécutions sont signalées même quand tout va bien : se fier d'abord au classement par score.

## 6. Paramètres

| Paramètre | Défaut | Effet |
|---|---|---|
| `--chunks` | aucun (`run`), 1024 (`series`) | Taille des blocs Dask en pixels. Aucun : scène entière en mémoire |
| `--min-valid` | 0,5 | Seuil `NOMINAL`/`DEGRADED` sur le taux de pixels valides |
| `--max-cloud` | 40 | Couverture nuageuse maximale de la scène (catalogue) à l'ingestion |