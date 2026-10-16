"""Anomaly detection on run KPIs with an Isolation Forest (unsupervised, no labels needed)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from earth_ops.quality import DEGRADED, NOMINAL

FEATURES = ("total_s", "valid_ratio", "ndvi_mean")
MIN_RUNS = 8


def detect_anomalies(runs: list[dict], contamination: float = 0.1, seed: int = 0) -> list[dict]:
    """Score completed runs; returns new records with `anomaly_score` and `is_anomaly`."""
    usable = [r for r in runs if r.get("status") in (NOMINAL, DEGRADED)
              and all(r.get(f) is not None for f in FEATURES)]
    if len(usable) < MIN_RUNS:
        raise ValueError(f"need at least {MIN_RUNS} completed runs, got {len(usable)}")
    features = np.array([[r[f] for f in FEATURES] for r in usable], dtype=float)
    model = IsolationForest(contamination=contamination, random_state=seed).fit(features)
    scores = -model.score_samples(features)  # higher = easier to isolate = more anomalous
    flags = model.predict(features) == -1
    return [{**r, "anomaly_score": round(float(s), 4), "is_anomaly": bool(f)}
            for r, s, f in zip(usable, scores, flags, strict=True)]


def format_report(flagged: list[dict]) -> str:
    lines = ["| run_id | product_id | status | total_s | valid_ratio | ndvi_mean | score "
             "| anomaly |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(flagged, key=lambda r: r["anomaly_score"], reverse=True):
        lines.append(f"| {r['run_id']} | {r['product_id']} | {r['status']} | {r['total_s']:.2f} "
                     f"| {r['valid_ratio']:.2f} | {r['ndvi_mean']:.3f} | {r['anomaly_score']:.3f} "
                     f"| {'YES' if r['is_anomaly'] else ''} |")
    return "\n".join(lines)
