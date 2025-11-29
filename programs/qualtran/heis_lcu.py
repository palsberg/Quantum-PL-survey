"""Heisenberg XXX via Qualtran-native 2nd-order Taylor LCU block."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .common import heis_lcu_state


def run_simulation(config: Dict[str, Any]) -> np.ndarray:
    num_sites = int(config["num_sites"])
    time = float(config["time"])
    params = config.get("params", {})
    J = float(params["J"])
    field = float(params["field"])
    precision = float(params["lcu_precision"])
    return heis_lcu_state(num_sites, J, field, time, precision)


if __name__ == "__main__":
    cfg = {
        "num_sites": 3,
        "time": 0.2,
        "params": {"J": 1.0, "field": 0.3, "lcu_precision": 1e-2},
    }
    print(run_simulation(cfg))
