"""TFIM via Qualtran-native 2nd-order Taylor LCU block."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .common import tfim_lcu_state


def run_simulation(config: Dict[str, Any]) -> np.ndarray:
    num_sites = int(config["num_sites"])
    time = float(config["time"])
    params = config.get("params", {})
    J = float(params["J"])
    h = float(params["h"])
    precision = float(params["lcu_precision"])
    return tfim_lcu_state(num_sites, J, h, time, precision)


if __name__ == "__main__":
    cfg = {
        "num_sites": 3,
        "time": 0.2,
        "params": {"J": 1.0, "h": 0.7, "lcu_precision": 5e-3},
    }
    print(run_simulation(cfg))
