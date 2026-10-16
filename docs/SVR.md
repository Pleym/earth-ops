# SVR — Rapport de validation logicielle

**Projet :** Earth-ops · **Version :** 0.1 · **Date :** 2026-10-16

**Environnement :** macOS, Apple M3 (8 cœurs), 16 Go · Python 3.12.9 · numpy 2.5.4 ·
xarray 2026.9.0 · dask 2026.8.0 · rasterio 1.5.2 / GDAL 3.12.2 · rioxarray 0.23.0 ·
scikit-learn 1.9.1

## 1. Synthèse

| Élément | Résultat |
|---|---|
| Tests automatisés | **45 / 45 PASS** (`pytest`), couverture 83 % |
| Démonstration données réelles | 20 produits Sentinel-2 L2A (tuile 31TCJ, 2025) + mosaïque France (§ 3.6) |
| Exigences | 37 / 37 vérifiées |
| Non-conformités | NC-01 (corrigée, a créé REQ-VAL-08) · NC-02 (données source, rejet attendu) |
| Revue de code indépendante | 10 constats : 9 corrigés avec test de non-régression, 1 documenté (§ 6) |

La couverture de `ingest.py` (45 %) est faible car l'accès réseau n'est pas testé unitairement :
l'ingestion est vérifiée par démonstration (§ 3.1), comme la recherche catalogue de la mosaïque
(§ 3.6). Les workers du benchmark tournent dans des sous-processus non mesurés par la couverture.

## 2. Matrice de traçabilité

| Exigence | Preuve | Résultat |
|---|---|---|
| REQ-ING-01 | § 3.1 : 30 items trouvés, une seule tuile (MGRS-31TCJ), 20 dates retenues sur l'année | PASS |
| REQ-ING-02 | § 3.1 : 3 bandes de 1316 × 1247 px par produit, 100 Mo au total ; inspection de `ingest_item` | PASS |
| REQ-ING-03 | § 3.1 : 2ᵉ ingestion → 20 « already archived », 0 octet téléchargé | PASS |
| REQ-ING-04 | Inspection `ingest_item` (metadata en dernier, écriture atomique) et `ingest` (erreur par produit journalisée) | PASS |
| REQ-ING-05 | `test_req_ing_05_unsafe_product_id_is_refused` (×3) ; inspection `_href` | PASS |
| REQ-VAL-01 | `test_req_val_01_nominal_product_is_accepted` | PASS |
| REQ-VAL-02 | `test_req_val_02_missing_band_is_rejected` ; § 3.3 | PASS |
| REQ-VAL-03 | `test_req_val_03_incomplete_metadata_is_rejected` (×4), `test_req_val_03_invalid_metadata_values_are_rejected` (×3) | PASS |
| REQ-VAL-04 | `test_req_val_04_corrupted_file_is_rejected` (×2 : en-tête, blocs tronqués) ; § 3.3 | PASS |
| REQ-VAL-05 | `test_req_val_05_band_shape_mismatch…`, `…_band_crs_mismatch…`, `…_geographic_crs…` | PASS |
| REQ-VAL-06 | `test_req_val_06_wrong_dtype_is_rejected` | PASS |
| REQ-VAL-07 | `test_req_val_07_empty_product_is_rejected` | PASS |
| REQ-VAL-08 | `test_req_val_08_offset_inconsistent_with_data_is_rejected` ; § 5 | PASS |
| REQ-PRO-01 | `test_req_pro_01_ndvi_matches_reference_formula` | PASS |
| REQ-PRO-02 | `test_req_pro_02_nodata_and_zero_denominator_give_nan` | PASS |
| REQ-PRO-03 | `test_req_pro_03_chunked_result_equals_in_memory_result` ; § 4 (écart 0) | PASS |
| REQ-PRO-04 | `test_req_pro_04_series_is_stacked_by_date`, `…_series_rejects_misaligned_grids` ; § 3.5 | PASS |
| REQ-PRO-05 | `test_req_ops_01_nominal_run_produces_ndvi_product_and_quicklook` | PASS |
| REQ-PRO-06 | `test_req_pro_06_mosaic_keeps_max_ndvi_and_survives_bad_scenes`, `test_req_pro_06_mosaic_end_to_end_on_local_scenes` ; § 3.6 | PASS |
| REQ-QUA-01 | `test_req_qua_01_clouds_and_out_of_range_values_are_masked` | PASS |
| REQ-QUA-02 | `test_req_qua_02_statistics_on_known_product` | PASS |
| REQ-QUA-03 | `test_req_qua_03_status_depends_on_valid_ratio` | PASS |
| REQ-QUA-04 | `test_req_qua_04_all_masked_product_gives_null_stats` | PASS |
| REQ-OPS-01 | `test_req_ops_01_…` ; § 3.2 | PASS |
| REQ-OPS-02 | `test_req_ops_02_invalid_product_is_rejected_without_crash` ; § 3.3 | PASS |
| REQ-OPS-03 | `test_req_ops_03_unexpected_error_is_reported_as_failed` | PASS |
| REQ-OPS-04 | `test_req_ops_04_cloudy_product_is_degraded` | PASS |
| REQ-OPS-05 | `test_req_ops_05_chunked_and_in_memory_runs_write_identical_products` (valeurs et taille de fichier), `test_req_ops_05_failed_write_leaves_no_partial_product` ; inspection `_deliver` | PASS |
| REQ-OPS-06 | `test_cli_end_to_end` ; § 3.3 (code retour 1) | PASS |
| REQ-MON-01 | `test_req_mon_01_every_run_is_recorded_with_stage_durations` | PASS |
| REQ-MON-02 | `test_req_mon_02_stage_duration_is_recorded_even_on_error` | PASS |
| REQ-ANO-01 | `test_req_ano_01_slow_run_is_flagged_as_most_anomalous` ; § 3.4 | PASS |
| REQ-ANO-02 | `test_req_ano_02_failed_runs_are_excluded_and_input_untouched` | PASS |
| REQ-ANO-03 | `test_req_ano_03_too_few_runs_raise` | PASS |
| REQ-PERF-01 | `test_req_perf_01_benchmark_reports_time_memory_and_exactness` ; § 4 | PASS |
| REQ-PERF-02 | § 4 : 1 484 Mo (NumPy) vs 546 Mo (bloc 1024, ×2,7) et 382 Mo (bloc 256, ×3,9) | PASS |
| REQ-PERF-03 | § 4 : écart max 0,0000, 0 différence de masque, pour les 4 tailles de bloc | PASS |

## 3. Démonstration sur données réelles

AOI : `1.30 43.52 1.46 43.63` (sud-ouest de Toulouse), période 2025, couverture nuageuse < 30 %.

### 3.1 Ingestion

30 items trouvés dans le catalogue, tous sur MGRS-31TCJ ; 20 dates retenues, réparties de janvier
à décembre. 20 produits archivés en 77 s (100 Mo). Une seconde exécution identique ignore les 20
produits (7 s, recherche catalogue seule).

### 3.2 Traitement nominal (`run data/products/* --chunks 1024`)

18 `NOMINAL`, 2 `REJECTED` (`E_RADIOMETRY`, NC-02). Durée de traitement ≈ 0,2 s par produit
(1,64 Mpx). Taux de pixels valides de 0,84 à 1,00.

### 3.3 Injection de pannes

| Produit | Panne | Statut | Code |
|---|---|---|---|
| `MISSING_B08` | B08.tif supprimé | `REJECTED` | `E_MISSING_FILE` |
| `CORRUPT_SCL` | SCL.tif remplacé par des octets invalides | `REJECTED` | `E_CORRUPT` |

Code retour de la CLI : 1. Aucun produit livré.

### 3.4 Exécution mal configurée et détection d'anomalies

Le produit `S2C_31TCJ_20250613_1_L2A` est retraité avec des blocs Dask de 32 px (configuration
erronée) : 8,55 s au lieu de ≈ 0,22 s. Isolation Forest sur les 19 exécutions terminées
(contamination 0,1) :

| Rang | Exécution | Durée (s) | Taux valide | Score | Drapeau |
|---|---|---|---|---|---|
| 1 | `S2C_…20250613` (blocs 32 px) | 8,55 | 1,00 | 0,712 | YES |
| 2 | `S2A_…20251212` | 0,20 | 0,84 | 0,649 | YES |
| 3 | `S2A_…20250804` | 0,20 | 0,87 | 0,616 | |

L'exécution mal configurée est la plus atypique. Le 2ᵉ drapeau est une scène de décembre
partiellement nuageuse (22 %) : atypique mais légitime, c'est à l'opérateur de trancher.

### 3.5 Série temporelle

18 produits valides (2 exclus par la validation) empilés en cube `(time, y, x)` ; NDVI moyen de
0,34 (août) à 0,50 (mai), cohérent avec le cycle cultural du Sud-Ouest. Composite médian écrit
en GeoTIFF + quicklook.

### 3.6 Mosaïque France (REQ-PRO-06)

`python -m earth_ops mosaic` : France métropolitaine, mai-juillet 2025, 500 m, Lambert-93.

| Version | Sélection des scènes | Scènes | Durée | Résultat |
|---|---|---|---|---|
| v1 | 2 moins nuageuses par tuile (< 30 %) | 340 | 5 min 55 | Raccords visibles aux limites de tuiles |
| v2 | 4 moins nuageuses par tuile (< 30 %) | 680 | 7 min 54 | Raccords réduits, encore visibles dans l'ouest |
| v3 | la plus claire par tuile et par mois (< 60 %) | 510 | 8 min 45 | Raccords disparus |

170 tuiles MGRS, grille de 2483 × 2239 px dont 69 % observés (le reste est la mer, masquée
via SCL). Une scène rejetée par le contrôle radiométrique en v1 (même incohérence de métadonnées
que NC-02), sans interruption de la mosaïque. Aucun fichier intermédiaire stocké : seuls la
carte GeoTIFF (6,6 Mo) et le PNG sont écrits.

**Analyse des raccords.** Choisir les scènes uniquement sur la couverture nuageuse donnait à
certaines tuiles des dates de juillet seulement (après moisson, NDVI plus bas) et à leurs
voisines des dates de mai : des carrés apparaissaient aux limites de tuiles. La stratification
par mois garantit à chaque tuile une date de printemps quand il en existe une ; le composite de
valeur maximale retient alors le pic de végétation partout.

## 4. Benchmark (REQ-PERF)

Scène simulée de 5490 × 5490 px (30,1 Mpx), lecture → NDVI → masque → GeoTIFF int16 compressé,
médiane de 3 répétitions, chaque configuration dans un processus séparé. Baseline (interpréteur
et bibliothèques) : 174 Mo. Données brutes : [`benchmark_results.json`](benchmark_results.json).

| Méthode | Bloc | Temps (s) | Mémoire crête (Mo) | Débit (Mpx/s) | Écart max | Masque |
|---|---|---|---|---|---|---|
| NumPy | – | 2,37 | 1 484 | 12,7 | 0,0000 | 0 |
| Dask | 256 | 2,24 | 382 | 13,4 | 0,0000 | 0 |
| Dask | 512 | 1,71 | 409 | 17,7 | 0,0000 | 0 |
| Dask | 1024 | 1,58 | 546 | 19,0 | 0,0000 | 0 |
| Dask | 2048 | 1,54 | 1 100 | 19,5 | 0,0000 | 0 |

**Analyse.** La mémoire crête croît avec la taille de bloc, à peu près comme *nombre de threads ×
taille de bloc × tableaux intermédiaires*. Le temps décroît jusqu'à 1024 px puis plafonne : en
dessous, le coût d'ordonnancement domine ; au-dessus, il n'y a plus de gain mais la mémoire
remonte. Configuration retenue en exploitation : **1024 px**.

*Note de configuration :* ces mesures précèdent la mise en commun du cœur NumPy
(`masked_ndvi_array`, § 6). Les opérations sont inchangées pour ces données (aucune somme de
réflectances négative) et REQ-PERF-03 reste vérifiée par le test automatisé.

## 5. Non-conformités

### NC-01 — Offset radiométrique appliqué deux fois (corrigée)

- **Détection** : en exploitation, par les KPI et non par les tests. 15 produits sur 20 étaient
  `DEGRADED` malgré 0 % de nuages ; réflectance rouge médiane de −0,03.
- **Cause racine** : `raster:bands` du catalogue annonce `offset = -0.1`, alors que l'offset a déjà
  été retiré des DN (`earthsearch:boa_offset_applied: true`).
- **Impact** : NDVI biaisé sur tous les produits concernés.
- **Correction** : (1) l'ingestion lit le drapeau et fixe l'offset à 0 ; (2) nouvelle exigence
  REQ-VAL-08 et code `E_RADIOMETRY` ; (3) archive ré-ingérée et retraitée. L'historique KPI
  d'avant correction est conservé (`data/kpi/runs_before_offset_fix.jsonl`).
- **Leçon** : la validation d'entrée doit vérifier la plausibilité physique, pas seulement le format.

### NC-02 — Drapeau fournisseur erroné sur 2 produits Sentinel-2C (rejet attendu)

`S2C_31TCJ_20250203_0_L2A` et `S2C_31TCJ_20250305_0_L2A` déclarent `boa_offset_applied: false`
alors que leurs DN n'ont pas d'offset (réflectance médiane −0,038 et −0,025). Ils sont rejetés
par REQ-VAL-08, ce qui est le comportement attendu. Aucune correction automatique : deviner
l'offset à partir des données risquerait de masquer de vraies erreurs.

## 6. Revue de code indépendante

Revue par un relecteur indépendant (agent de revue Python), qui a reproduit chaque constat avec
un script d'injection de fautes. Aucun constat critique ou majeur. Les implémentations NumPy et
Xarray/Dask ont été confirmées identiques au bit près sur 12 combinaisons (3 couples
scale/offset × 4 tailles de bloc, toutes classes SCL).

| # | Gravité | Constat | Action | Preuve |
|---|---|---|---|---|
| 1 | Moyenne | Somme rouge + PIR négative : NDVI de signe inversé conservé | Seuil sur la somme signée, dans les deux implémentations | `test_req_pro_02` (3ᵉ pixel) |
| 2 | Moyenne | Blocs compressés corrompus : `FAILED` au lieu de `REJECTED`, série interrompue | `E_CORRUPT` sur la lecture d'échantillon | `test_req_val_04` (fichier tronqué) |
| 3 | Moyenne | Blocs Dask non multiples de 256 : fichier ×7, temps ×10 | Blocs réalignés sur les tuiles GeoTIFF à l'écriture | `test_req_ops_05` (taille) |
| 4 | Faible | Fichier partiel laissé après échec ; composite non atomique | `_deliver` : fichier temporaire, renommage, nettoyage | `test_req_ops_05_failed_write…` |
| 5 | Faible | Métadonnées : `NaN`, date invalide, JSON non objet acceptés | Contrôles dans `load_metadata` | `test_req_val_03` (×5 cas ajoutés) |
| 6 | Faible (sécurité) | Identifiant de produit distant utilisé comme chemin (`../`) | Liste blanche d'identifiant, URL HTTPS uniquement | `test_req_ing_05` |
| 7 | Faible | `metadata.json` non écrit atomiquement | Fichier temporaire + renommage | Inspection |
| 8 | Faible | Benchmark : nettoyage sauté en cas d'échec, `rmtree` trop large | `try/finally`, suppression du seul produit simulé | Inspection |
| 9 | Faible | `contamination=0.1` signale 10 % des exécutions par construction | Documenté dans le SUM (§ 5) ; classement par score conservé | — |
| 10 | Faible | Dockerfile : utilisateur non-root sans droit d'écriture sur `data/` | `chown` de `/app/data` | Inspection (image non construite) |

## 7. Limites de la validation

- Pas de comparaison à un NDVI de référence externe : le recoupement NumPy/Dask vérifie la
  cohérence interne, pas la justesse absolue.
- Mosaïque France : pas de découpage à la frontière (les pays voisins apparaissent), et le
  composite de valeur maximale surestime volontairement le NDVI (il privilégie le pic de végétation).
- Une seule AOI, une seule machine, ordonnanceur Dask local ; benchmark sur scène simulée.
- Image Docker non construite ni testée dans cette version (espace disque insuffisant sur le
  poste de développement).
