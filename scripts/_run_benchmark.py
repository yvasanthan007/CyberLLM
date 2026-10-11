"""Run the UNSW-NB15 benchmark and write a readable summary.

Progress is streamed to a log file so a long run is observable rather than
apparently hung.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cyberllm.core.logging_config import configure_logging  # noqa: E402
from cyberllm.detection.benchmark import (  # noqa: E402
    BenchmarkConfig, run_benchmark, summarise,
)

LOG = Path(__file__).with_name("_benchmark_run.log")
logging.basicConfig(
    level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(name)s %(message)s",
)

data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else r"d:\CyberLLM\data\raw\unsw_nb15")
out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else r"d:\CyberLLM\artifacts\reports\unsw_nb15")
max_rows = int(sys.argv[3]) if len(sys.argv) > 3 else None
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

handler = logging.FileHandler(LOG, encoding="utf-8", mode="w")
logging.getLogger().addHandler(handler)

configure_logging("INFO")

config = BenchmarkConfig(
    data_dir=data_dir, out_dir=out_dir, max_rows=max_rows, seed=seed,
    register_model=True,
)
started = time.monotonic()
report = run_benchmark(config)
Path(__file__).with_name("_benchmark_summary.txt").write_text(
    summarise(report), encoding="utf-8"
)
Path(__file__).with_name("_benchmark_report.json").write_text(
    json.dumps(report, indent=2, default=str), encoding="utf-8"
)
logging.getLogger().info("elapsed %.1fs", time.monotonic() - started)
