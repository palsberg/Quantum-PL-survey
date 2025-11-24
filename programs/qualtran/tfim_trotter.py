"""TFIM via Qualtran's Ising Trotter bloqs."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from . import common


def run_simulation(config: Dict[str, Any]) -> np.ndarray:
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    h = float(params.get("h", 1.0))
    steps = int(params.get("trotter_steps", 32))
    order = 1
    angle = float(params.get("init_angle", np.pi / 8))
    return common.tfim_trotter_state(num_sites, J, h, total_time, steps, order, angle)


if __name__ == "__main__":
    cfg = {
        "num_sites": 3,
        "time": 0.4,
        "params": {"J": 1.0, "h": 0.6, "trotter_steps": 64, "trotter_order": 1},
    }
    state = run_simulation(cfg)
    print(state)
