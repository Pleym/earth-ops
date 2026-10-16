"""Operational monitoring: logs, per-stage timings and one KPI record per run (JSON Lines)."""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("earth_ops")
DEFAULT_KPI_FILE = Path("data/kpi/runs.jsonl")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    logging.getLogger("rasterio").setLevel(logging.WARNING)  # per-request session chatter


class RunMonitor:
    """Times the stages of one run and appends its KPI record to the KPI file."""

    def __init__(self, product_id: str, kpi_file: Path) -> None:
        self.product_id = product_id
        self.kpi_file = Path(kpi_file)
        self.run_id = uuid.uuid4().hex[:8]
        self.started_at = datetime.now(UTC)
        self.stages_s: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        log.info("[%s] %s | %s started", self.run_id, self.product_id, name)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages_s[name] = round(time.perf_counter() - t0, 4)
            log.info("[%s] %s | %s ended after %.2f s", self.run_id, self.product_id, name,
                     self.stages_s[name])

    def record(self, status: str, **kpis: object) -> dict:
        record = {
            "run_id": self.run_id, "product_id": self.product_id,
            "started_at": self.started_at.isoformat(timespec="seconds"), "status": status,
            "stages_s": dict(self.stages_s), "total_s": round(time.perf_counter() - self._t0, 4),
            **kpis,
        }
        self.kpi_file.parent.mkdir(parents=True, exist_ok=True)
        with self.kpi_file.open("a") as f:
            f.write(json.dumps(record) + "\n")
        level = logging.INFO if status == "NOMINAL" else logging.WARNING
        log.log(level, "[%s] %s | %s %s", self.run_id, self.product_id, status,
                kpis.get("error", ""))
        return record


def load_runs(kpi_file: Path) -> list[dict]:
    path = Path(kpi_file)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
