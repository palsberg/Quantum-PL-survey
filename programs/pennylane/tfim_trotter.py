"""TFIM via Lie–Trotterization in PennyLane."""

from __future__ import annotations

from typing import Any, Dict

from . import common


def run_simulation(config: Dict[str, Any]):
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    h = float(params.get("h", 1.0))
    steps = int(params.get("trotter_steps", 2))
    return common.tfim_trotter_state(num_sites, J, h, total_time, steps)


if __name__ == "__main__":
    cfg = {"num_sites": 3, "time": 0.3, "params": {"J": 1.0, "h": 0.7, "trotter_steps": 4}}
    print(run_simulation(cfg))

