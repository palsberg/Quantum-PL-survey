"""Heisenberg XXX via Lie–Trotterization in PyQuil."""

from __future__ import annotations

from typing import Any, Dict

from . import common


def run_simulation(config: Dict[str, Any]):
    num_sites = int(config["num_sites"])
    total_time = float(config["time"])
    params = config.get("params", {})
    J = float(params.get("J", 1.0))
    field = float(params.get("field", 0.2))
    steps = int(params.get("trotter_steps", 2))

    prog, _ = common.trotterize_heisenberg_xxx(num_sites, J, field, total_time, steps)
    return common.simulate_statevector(prog, num_sites)


if __name__ == "__main__":
    cfg = {"num_sites": 3, "time": 0.3, "params": {"J": 1.0, "field": 0.3, "trotter_steps": 4}}
    print(run_simulation(cfg))

