# SRS — Spécification des exigences logicielles

**Projet :** Earth-ops · **Version :** 0.1 · **Statut :** validé (voir [SVR](SVR.md))

> Document inspiré de la structure ECSS-E-ST-40C (SRS → tests → SVR). Le projet s'inspire des
> pratiques ECSS sans revendiquer de conformité à la norme.

## 1. Objet

Chaîne de traitement « mini segment sol » pour des produits Sentinel-2 L2A : ingestion depuis un
catalogue STAC public, validation des entrées, calcul NDVI (Xarray, Dask optionnel), contrôle
qualité, supervision des exécutions (KPI) et détection d'exécutions atypiques.

## 2. Méthodes de vérification

| Code | Méthode | Signification |
|---|---|---|
| T | Test | Test automatisé `pytest` (le nom du test reprend l'ID de l'exigence) |
| D | Démonstration | Exécution réelle documentée dans le SVR |
| I | Inspection | Relecture du code |
| A | Analyse | Analyse de mesures (benchmark) |

## 3. Exigences

### Ingestion (ING)

| ID | Exigence | Vérif. |
|---|---|---|
| REQ-ING-01 | Le système doit rechercher les produits Sentinel-2 L2A d'une emprise (bbox) et d'une période dans un catalogue STAC, en retenant une seule tuile MGRS et une acquisition par date. | D |
| REQ-ING-02 | Seules les bandes B04, B08 et SCL doivent être téléchargées, uniquement sur l'emprise demandée ; SCL (20 m) doit être rééchantillonnée au plus proche voisin sur la grille 10 m. | D, I |
| REQ-ING-03 | Un produit déjà archivé ne doit pas être téléchargé à nouveau. | D |
| REQ-ING-04 | `metadata.json` doit être écrit en dernier et de façon atomique : sa présence marque un produit complet. L'échec d'un produit ne doit pas interrompre le lot. | I |
| REQ-ING-05 | Les données venant du catalogue doivent être contrôlées avant usage : identifiant de produit sans séparateur de chemin (pas de `../`), URL d'actifs en HTTPS uniquement. | T, I |

### Validation des entrées (VAL)

| ID | Exigence | Vérif. |
|---|---|---|
| REQ-VAL-01 | Un produit complet et cohérent doit être accepté. | T |
| REQ-VAL-02 | Un fichier manquant doit entraîner le rejet `E_MISSING_FILE`. | T |
| REQ-VAL-03 | Des métadonnées illisibles, qui ne sont pas un objet JSON, incomplètes, avec `scale`/`offset` non finis ou une date non ISO 8601 doivent entraîner `E_METADATA`. | T |
| REQ-VAL-04 | Un raster illisible (en-tête ou blocs compressés) doit entraîner `E_CORRUPT`. | T |
| REQ-VAL-05 | Les trois bandes doivent partager CRS, géotransformation et dimensions (`E_GEOMETRY`) ; le CRS doit être projeté (`E_CRS`). | T |
| REQ-VAL-06 | Chaque bande doit avoir une seule couche du type attendu (B04/B08 `uint16`, SCL `uint8`), sinon `E_DTYPE`. | T |
| REQ-VAL-07 | Un produit dont plus de 99 % de B04 vaut nodata doit entraîner `E_EMPTY`. | T |
| REQ-VAL-08 | Un produit dont la réflectance B04 médiane (avec le `scale`/`offset` déclarés) est négative doit entraîner `E_RADIOMETRY`. *Ajoutée suite à la non-conformité NC-01 (voir SVR).* | T, D |

### Traitement (PRO)

| ID | Exigence | Vérif. |
|---|---|---|
| REQ-PRO-01 | Le NDVI doit être calculé en float32 à partir des réflectances `DN × scale + offset`, avec `scale`/`offset` lus dans les métadonnées du produit. | T |
| REQ-PRO-02 | Les pixels nodata (DN = 0) et ceux où rouge + PIR < 1e-4 (somme quasi nulle ou négative) doivent donner NaN. | T |
| REQ-PRO-03 | Le résultat calculé par blocs Dask doit être identique au calcul en mémoire. | T |
| REQ-PRO-04 | Une série de produits doit être empilée en cube `(time, y, x)` trié par date ; des grilles différentes doivent être refusées. | T |
| REQ-PRO-05 | Le produit livré doit être un GeoTIFF int16 (NDVI × 10000, nodata −32768) accompagné d'un quicklook PNG avec coordonnées projetées. | T |
| REQ-PRO-06 | Une mosaïque nationale (France métropolitaine par défaut) doit être produite en Lambert-93 par composite de valeur maximale de la scène la plus claire de chaque tuile pour chaque mois, en lisant les aperçus des COG (sans télécharger les tuiles complètes). Une scène en échec ne doit pas interrompre la mosaïque. | T, D |

### Contrôle qualité (QUA)

| ID | Exigence | Vérif. |
|---|---|---|
| REQ-QUA-01 | Les pixels SCL 0, 1, 3, 8, 9, 10 et les NDVI hors [−1, 1] doivent être masqués. | T |
| REQ-QUA-02 | Chaque produit doit avoir : taux de pixels valides, taux de nuages, moyenne, écart-type, min et max du NDVI. | T |
| REQ-QUA-03 | Le statut doit être `NOMINAL` si le taux de pixels valides ≥ 0,5 (paramétrable), sinon `DEGRADED`. | T |
| REQ-QUA-04 | Un produit entièrement masqué doit produire des statistiques nulles sans faire échouer le traitement. | T |

### Exploitation (OPS)

| ID | Exigence | Vérif. |
|---|---|---|
| REQ-OPS-01 | Une exécution nominale doit livrer le produit NDVI et son quicklook. | T |
| REQ-OPS-02 | Un produit invalide doit être `REJECTED` sans plantage et sans produit livré. | T |
| REQ-OPS-03 | Une erreur inattendue doit donner le statut `FAILED` (`E_PROCESSING`) avec le message d'erreur, sans interrompre le lot. | T |
| REQ-OPS-04 | Un produit trop nuageux doit être `DEGRADED`. | T |
| REQ-OPS-05 | Le produit doit être écrit sous un nom temporaire puis renommé : un produit livré est toujours complet, et une écriture qui échoue ne laisse aucun fichier partiel. Les sorties en mémoire et par blocs doivent être identiques, sans surcoût disque quelle que soit la taille de bloc. | T, I |
| REQ-OPS-06 | La CLI doit retourner un code non nul si au moins un produit est `REJECTED` ou `FAILED`. | T |

### Supervision (MON)

| ID | Exigence | Vérif. |
|---|---|---|
| REQ-MON-01 | Chaque exécution, quel que soit son statut, doit ajouter un enregistrement KPI (JSON Lines) : identifiant, statut, durée de chaque étape, durée totale, débit, statistiques qualité. | T |
| REQ-MON-02 | La durée d'une étape doit être enregistrée même si l'étape échoue. | T |

### Détection d'anomalies (ANO)

| ID | Exigence | Vérif. |
|---|---|---|
| REQ-ANO-01 | Une Isolation Forest sur (durée totale, taux valide, NDVI moyen) doit attribuer à chaque exécution un score d'anomalie et un drapeau. | T, D |
| REQ-ANO-02 | Seules les exécutions terminées (`NOMINAL`/`DEGRADED`) doivent être analysées ; l'historique d'entrée ne doit pas être modifié. | T |
| REQ-ANO-03 | Il faut au moins 8 exécutions terminées, sinon une erreur explicite. | T |

### Performance (PERF)

| ID | Exigence | Vérif. |
|---|---|---|
| REQ-PERF-01 | Un benchmark doit comparer NumPy (scène entière) et Xarray+Dask (plusieurs tailles de blocs) sur le même produit : temps, mémoire crête, débit et écart numérique. | T, A |
| REQ-PERF-02 | Sur une scène de 5490 × 5490 px, la mémoire crête du traitement par blocs (bloc ≤ 1024 px) doit être au moins 2 fois inférieure à celle du traitement NumPy. | A |
| REQ-PERF-03 | L'écart entre les deux implémentations doit être ≤ 1e-4 en NDVI (un pas de quantification), sans différence de masque. | T, A |
