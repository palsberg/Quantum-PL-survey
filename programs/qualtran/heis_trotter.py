"""Heisenberg XXX via Qualtran-based Trotterization."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from . import common


def run_simulation(config: Dict[str, Any]) -> np.ndarray:
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    field = float(params.get("field", 0.2))
    steps = int(params.get("trotter_steps", 32))
    angle = float(params.get("init_angle", 0.0))
    return common.heis_trotter_state(num_sites, J, field, total_time, steps, angle)


if __name__ == "__main__":
    cfg = {
        "num_sites": 3,
        "time": 0.3,
        "params": {"J": 1.0, "field": 0.2, "trotter_steps": 40},
    }
    print(run_simulation(cfg))
