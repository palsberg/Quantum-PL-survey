"""TFIM via Qrisp's built-in Trotterized evolution."""

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
    order = int(params.get("trotter_order", 1))
    angle = float(params.get("init_angle", np.pi / 8))

    def builder():
        return common.build_tfim_operator(num_sites, J, h)

    return common.trotter_state(builder, num_sites, total_time, steps, order=order, init_angle=angle)


if __name__ == "__main__":
    cfg = {
        "num_sites": 3,
        "time": 0.3,
        "params": {"J": 1.0, "h": 0.7, "trotter_steps": 48, "trotter_order": 2},
    }
    state = run_simulation(cfg)
    print(state)

